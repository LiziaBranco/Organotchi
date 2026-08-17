"""
Sweep classical segmentation parameters on one representative tile and save
visual comparisons, to pick good values before committing to a full
segmentation_metadata.json run.

For each swept parameter, this produces one image: the raw tile followed by
one mask panel per candidate value (all other parameters held at the current
BASELINE), each panel captioned with the value tried and a quick
count/area/circularity readout. This is a one-factor-at-a-time sweep, not a
full grid search -- there's no ground truth for this single image to score
against automatically, so the point is to make the *visual* comparison fast,
not to replace it.

Run from data_analysis/ (matching segmentation.py's convention):

    python fit_experiment_data.py
    python fit_experiment_data.py --project 2026_04_30-Zeynep-CP --day 39 --batch CP03_1 --tile Tile000002.jpg
    python fit_experiment_data.py --sweep threshold,min_circularity
    python fit_experiment_data.py --set min_area=50000 --set max_area=2000000
    python fit_experiment_data.py --set threshold=otsu
    python fit_experiment_data.py --set center_fraction=0.9
    python fit_experiment_data.py --set close_kernel_size=40
    python fit_experiment_data.py --diagnose   # which stage is dropping organoids? see below

Edit BASELINE / SWEEPS below to change what gets tried by default.

The classical segmentation pipeline (see ``utils_segmentation.py``) currently
uses only four filters, for the time being: ``threshold``, ``min_area``,
``max_area``, ``min_circularity`` (plus ``foreground_polarity``,
``organoid_mode``, ``center_fraction``, and ``close_kernel_size``, which
aren't filters in the same sense but are needed to run at all / are optional
cheap preprocessing). There is no halo growth or touching-object split in
the pipeline right now.

``threshold`` accepts either a fixed number or the literal string ``"otsu"``,
which resolves to that image's own Otsu threshold (``skimage.filters.threshold_otsu``)
instead of a fixed cutoff -- useful when illumination genuinely drifts across
a timecourse (confirmed on AnnaRaic: a fixed threshold detected organoids on
Day 0 but missed them entirely on Day 1). ``--diagnose`` prints the resolved
per-image value when this is set; ``--sweep threshold`` includes it as one of
the candidates alongside the fixed values.

``center_fraction`` (default ``None`` = disabled) excludes everything
outside a circle concentric with the image, of this fraction of its
**width** -- e.g. ``0.9`` keeps the central 90% of the width. Width, not
the shorter dimension, because in these stitched multi-well images the well
is wider than it is tall and its physical diameter in pixels is close to
the image's width -- using width means values close to 1.0 track the well's
actual edge, rather than leaving a wide unaccounted margin on the image's
long axis. Approximates the physical well boundary well enough to drop a
well-edge ring before it ever reaches contour extraction, without detecting
the well's actual edge per image (confirmed valuable on Zeynep-CP's
late-timepoint multi-organoid images, where the well ring was otherwise a
large, low-circularity candidate contour sitting among the real organoids).

``close_kernel_size`` (default ``None`` = disabled; an int is a square
kernel size in pixels) runs a morphological closing pass (dilate then erode)
right after ``center_fraction``, before ``min_area``/``max_area``. Bridges
small gaps in an otherwise near-complete object boundary -- e.g. a cystic
organoid's rim that didn't fully threshold all the way around (uneven
staining/focus across part of it), leaving an open C-shape rather than a
closed loop. An open contour's filled draw only fills the thin sliver it
traces, not the disk it would enclose if closed, so the object is either
dropped outright (low circularity) or saved as a fragment of its true
extent; closing it into one loop first means area/circularity filtering and
the final filled draw both see -- and correctly fill -- the whole object, no
separate hole-filling step needed. Keep the kernel just big enough to
bridge the gaps actually seen, not larger: a kernel that can bridge a gap in
one organoid's rim can just as easily bridge the *spacing* between two
separate, adjacent organoids.

`--diagnose` is a different mode from the sweep: instead of trying several
values, it runs once at BASELINE and reports (a) exactly which file gets
read for this tile -- so a stale cached .jpg silently shadowing a newer raw
.tif shows up here, not just as an unexplained bad result -- (b) a raw pixel
histogram, and (c) one panel per pipeline stage, in the same order the real
pipeline applies them (raw threshold -> center-fraction mask -> morphological
closing -> min-area filter -> max-area filter -> candidate contours ->
circularity filter -> area filter -> final accepted objects), each captioned
with a count/area/circularity readout and the count delta from the previous
stage -- so you can see exactly *which* stage drops organoids, not just that
they end up missing somewhere between thresholding and the final result.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils_segmentation import (  # noqa: E402
    segment_classical_brightfield_image,
    _collect_tile_images,
    _find_object_contours,
    _contour_shape_metrics,
    _comparison_resample_filter,
    _load_image_preview_for_comparison,
    _convert_tif_to_jpg_for_segmentation,
)

# Same relative convention as experiment_info.py's `base_dir` -- this script
# is meant to be run with data_analysis/ as the working directory, same as
# segmentation.py. Not imported from experiment_info.py directly, since that
# module runs an interactive project-selection prompt as an import side
# effect; this script is meant to run non-interactively from CLI args.
DEFAULT_BASE_DIR = "../../../AGMayer/Lizia/Organoid_Segmentation"

# 1.25x calibration, matching experiment_info.py's pixel_to_micron table --
# override with --pixel-to-micron-x/-y if the project uses a different one.
DEFAULT_PIXEL_TO_MICRON_X = 2.8422
DEFAULT_PIXEL_TO_MICRON_Y = 2.8394

PANEL_MAX_SIDE = 480
CAPTION_HEIGHT = 54
TITLE_HEIGHT = 30
PADDING = 12
GAP = 10

# Current best-guess parameters; every sweep below varies exactly one of
# these while holding the rest fixed. Keep this in sync with whatever you've
# settled on for the project you're actually running.
BASELINE = dict(
    organoid_mode="multiple",
    threshold=100,
    foreground_polarity="dark",
    min_area=100000,
    max_area=None,
    min_circularity=0.02,
    center_fraction=0.78,
    close_kernel_size=90,
)

# One entry per parameter worth exploring; edit freely. Candidate lists
# deliberately bracket BASELINE's own value so you can see both directions.
SWEEPS = {
    "threshold": [100, 115, 128, 145, 160, "otsu"],
    "min_circularity": [0.05, 0.2, 0.3, 0.45, 0.6],
    "min_area": [30000, 50000, 80000, 120000],
    "max_area": [1000000, 2000000, 4000000, 6000000],
    "center_fraction": [None, 0.98, 0.95, 0.9, 0.85],
    "close_kernel_size": [None, 2, 5, 10, 15],
}


def _coerce_value(raw_value):
    """Parse a --set key=value CLI string into int/float/bool/str."""
    lowered = raw_value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(raw_value)
    except ValueError:
        pass
    try:
        return float(raw_value)
    except ValueError:
        pass
    return raw_value


def _resolve_tile_path(base_dir, project, day, batch, tile):
    raw_data_dir = os.path.join(base_dir, project, "raw_data", f"Day {day}", batch)
    if not os.path.isdir(raw_data_dir):
        raise FileNotFoundError(f"Raw data folder not found: {raw_data_dir}")

    tile_stem = os.path.splitext(tile)[0]
    for tile_path in _collect_tile_images(raw_data_dir):
        if os.path.splitext(os.path.basename(tile_path))[0] == tile_stem:
            return tile_path

    raise FileNotFoundError(f"Tile '{tile}' not found under {raw_data_dir}")


def _mask_stats(mask, pixel_to_micron_x, pixel_to_micron_y):
    """Quick numeric readout for a panel caption: object count, total area,
    mean circularity -- computed the same way the real pipeline does."""
    contours = _find_object_contours(mask)
    if not contours:
        return {"count": 0, "area_um2": 0.0, "mean_circularity": float("nan")}

    areas, circularities = [], []
    for contour in contours:
        area, circularity = _contour_shape_metrics(contour, pixel_to_micron_x, pixel_to_micron_y)
        areas.append(area)
        circularities.append(circularity)

    return {
        "count": len(contours),
        "area_um2": sum(areas),
        "mean_circularity": float(np.mean(circularities)),
    }


def _downsample_mask_for_display(mask, max_side=PANEL_MAX_SIDE):
    preview = Image.fromarray(np.where(mask > 0, 255, 0).astype(np.uint8))
    preview.thumbnail((max_side, max_side), resample=_comparison_resample_filter(binary=True))
    return preview.convert("RGB")


def _contours_to_mask(shape, contours, reference_mask=None):
    """Render a contour list (as captured via ``debug_capture``) back into a
    binary mask, so it can go through the same stats/display helpers used
    for actual mask arrays.

    If ``reference_mask`` is given, only pixels that are actually
    foreground there are kept -- i.e. show exactly what's still there, not a
    premature fill. Every intermediate "candidate contours" stage is drawn
    *before* the real pipeline's own final ``drawContours(..., FILLED)``
    step has legitimately earned that solid look (that only really happens
    once, for the truly final accepted objects) -- naively filling every
    contour here would make a well-boundary ring or a hollow/thin core
    appear solid several stages before anything actually changed it, which
    is exactly the "did something get filled in between these two steps?"
    confusion this is meant to avoid. Omit ``reference_mask`` only for the
    one panel that should legitimately show the filled result (the real
    returned ``final_mask``, not a contour-list reconstruction, is used for
    that instead)."""
    mask = np.zeros(shape, dtype=np.uint8)
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    if reference_mask is not None:
        mask[reference_mask == 0] = 0
    return mask


def _format_value(value):
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _format_caption(label, stats):
    circ = stats["mean_circularity"]
    circ_text = f"{circ:.2f}" if circ == circ else "n/a"  # NaN check
    return [label, f"count={stats['count']}  area={stats['area_um2']:.0f}um2  circ={circ_text}"]


def _compose_sweep_row(raw_preview, panels_with_captions, row_title):
    """raw preview + N mask panels, each with a 2-line caption underneath."""
    panel_w = max(raw_preview.width, *(p.width for p, _ in panels_with_captions))
    panel_h = max(raw_preview.height, *(p.height for p, _ in panels_with_captions))

    n_panels = 1 + len(panels_with_captions)
    canvas_width = n_panels * panel_w + (n_panels - 1) * GAP + 2 * PADDING
    canvas_height = TITLE_HEIGHT + panel_h + CAPTION_HEIGHT + 2 * PADDING
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((PADDING, 6), row_title, fill="black")

    x = PADDING
    top = PADDING + TITLE_HEIGHT
    canvas.paste(raw_preview, (x + (panel_w - raw_preview.width) // 2, top))
    draw.text((x, top + panel_h + 4), "raw tile", fill="black")
    x += panel_w + GAP

    for panel_img, caption_lines in panels_with_captions:
        canvas.paste(panel_img, (x + (panel_w - panel_img.width) // 2, top))
        for i, line in enumerate(caption_lines):
            draw.text((x, top + panel_h + 4 + i * 14), line, fill="black")
        x += panel_w + GAP

    return canvas


def run_sweep(tile_path, pixel_to_micron_x, pixel_to_micron_y, output_dir, sweep_names=None, dry_run=False):
    os.makedirs(output_dir, exist_ok=True)
    raw_preview = _load_image_preview_for_comparison(tile_path, max_side=PANEL_MAX_SIDE, binary=False)

    sweep_names = sweep_names or list(SWEEPS.keys())
    unknown = set(sweep_names) - set(SWEEPS.keys())
    if unknown:
        raise ValueError(f"Unknown sweep parameter(s): {sorted(unknown)}. Known: {sorted(SWEEPS.keys())}")

    summary_rows = []

    for param_name in sweep_names:
        candidates = SWEEPS[param_name]
        print(f"--- sweeping {param_name}: {candidates}")
        panels_with_captions = []

        for value in candidates:
            run_params = {**BASELINE, param_name: value}
            organoid_mode = run_params.pop("organoid_mode")

            if dry_run:
                print(f"    {param_name}={value}: (dry run, not executed)")
                continue

            try:
                mask, count = segment_classical_brightfield_image(
                    tile_path, pixel_to_micron_x, pixel_to_micron_y,
                    organoid_mode=organoid_mode, **run_params,
                )
            except Exception as exc:
                print(f"    {param_name}={value}: FAILED ({exc})")
                summary_rows.append({"parameter": param_name, "value": value, "error": str(exc)})
                continue

            stats = _mask_stats(mask, pixel_to_micron_x, pixel_to_micron_y)
            print(f"    {param_name}={value}: count={stats['count']} area={stats['area_um2']:.0f}um2")
            summary_rows.append({
                "parameter": param_name, "value": value,
                "count": stats["count"], "area_um2": stats["area_um2"],
                "mean_circularity": stats["mean_circularity"], "error": "",
            })

            label = f"{param_name}={_format_value(value)}"
            panel = _downsample_mask_for_display(mask)
            panels_with_captions.append((panel, _format_caption(label, stats)))

        if dry_run or not panels_with_captions:
            continue

        row_image = _compose_sweep_row(raw_preview, panels_with_captions, row_title=f"sweep: {param_name}")
        out_path = os.path.join(output_dir, f"{param_name}.jpg")
        row_image.save(out_path, "JPEG", quality=90)
        print(f"    saved {out_path}")

    if not dry_run:
        summary_path = os.path.join(output_dir, "sweep_summary.csv")
        with open(summary_path, "w", newline="") as f:
            fieldnames = ["parameter", "value", "count", "area_um2", "mean_circularity", "error"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print(f"\nWrote {summary_path}")


# Ordered (debug_capture key, panel label) pairs describing every pipeline
# sub-stage, in the exact order the real pipeline applies them.
_DIAGNOSE_STEPS = [
    ("threshold", "raw threshold"),
    ("after_center_fraction_mask", "after center-fraction mask"),
    ("after_closing", "after morphological closing"),
    ("after_min_area_filter", "after min-area filter"),
    ("after_max_area_filter", "after max-area filter"),
    ("raw_contours", "candidate contours (pre-filter)"),
    ("after_circularity_filter", "after circularity filter"),
    ("after_area_filter", "after area filter"),
]

_CONTOUR_LIST_KEYS = {"raw_contours", "after_circularity_filter", "after_area_filter", "accepted_contours"}


def run_diagnose(tile_path, pixel_to_micron_x, pixel_to_micron_y, output_dir):
    """Pinpoint exactly which pipeline stage is dropping organoids, by
    running the real pipeline once with ``debug_capture`` and rendering one
    panel per stage, in the same order the pipeline applies them: raw
    threshold -> min-area filter -> max-area filter -> candidate contours ->
    circularity filter -> area filter -> final accepted objects
    (single-organoid mode additionally keeps only the largest survivor at
    that last step).

    Also prints exactly which file gets read (resolved the same way the
    real pipeline resolves it -- so a stale cached .jpg silently shadowing a
    newer .tif shows up here too, not just in a full segmentation run) and a
    raw pixel histogram. Each panel is captioned with count/area/circularity
    and the count delta from the previous stage, so the exact stage where
    organoids get lost is visible directly, not just "before vs after".
    """
    os.makedirs(output_dir, exist_ok=True)

    resolved_path = _convert_tif_to_jpg_for_segmentation(tile_path)
    print(f"Tile resolved via _collect_tile_images: {tile_path}")
    if resolved_path != tile_path:
        print(f"  -> converted to: {resolved_path}")

    with Image.open(resolved_path) as img:
        gray = np.array(img.convert("L"))

    print(f"\nRaw pixel stats (post-load, pre-threshold): "
          f"min={gray.min()} max={gray.max()} mean={gray.mean():.1f}")
    hist, _ = np.histogram(gray, bins=[0, 32, 64, 96, 128, 160, 192, 224, 256])
    bin_labels = ["0-31", "32-63", "64-95", "96-127", "128-159", "160-191", "192-223", "224-255"]
    print("Histogram (pixel count per gray-level bucket):")
    for label, count in zip(bin_labels, hist):
        print(f"  {label:>9}: {count}")

    run_params = dict(BASELINE)
    organoid_mode = run_params.pop("organoid_mode")

    debug_capture = {}
    final_mask, final_count = segment_classical_brightfield_image(
        resolved_path, pixel_to_micron_x, pixel_to_micron_y, organoid_mode=organoid_mode,
        debug_capture=debug_capture, **run_params,
    )

    if run_params.get("threshold") == "otsu":
        print(f"\nOtsu-resolved threshold for this image: {debug_capture['threshold_value']:.1f}")

    # What the "candidate contours"/"after ... filter" stages were actually
    # extracted from -- i.e. the real pixel pattern they should be drawn
    # against, not a naive fill. This is `_create_classical_binary_mask`'s
    # own final output (after max-area filtering, or the latest earlier
    # stage that actually ran, down to the raw threshold if none of
    # center_fraction/close_kernel_size/min_area/max_area were set).
    contour_reference_mask = debug_capture.get(
        "after_max_area_filter", debug_capture.get(
            "after_min_area_filter", debug_capture.get(
                "after_closing", debug_capture.get(
                    "after_center_fraction_mask", debug_capture["threshold"]
                )
            )
        )
    )

    steps = []  # (label, mask) pairs, in pipeline order
    for key, label in _DIAGNOSE_STEPS:
        if key not in debug_capture:
            continue
        value = debug_capture[key]
        mask = _contours_to_mask(gray.shape, value, reference_mask=contour_reference_mask) if key in _CONTOUR_LIST_KEYS else value
        steps.append((label, mask))
    steps.append(("final accepted objects", final_mask))

    print(f"\nStep-by-step ({organoid_mode}-organoid mode):")
    panels_with_captions = []
    prev_count = None
    threshold_was_empty = False
    for label, mask in steps:
        stats = _mask_stats(mask, pixel_to_micron_x, pixel_to_micron_y)
        circ = stats["mean_circularity"]
        circ_text = f"{circ:.2f}" if circ == circ else "n/a"  # NaN check
        delta = ""
        if prev_count is not None and stats["count"] != prev_count:
            delta = f"  ({stats['count'] - prev_count:+d} vs previous step)"
        print(f"  [{label}] count={stats['count']} area={stats['area_um2']:.0f}um2 circ={circ_text}{delta}")
        if label == "raw threshold" and stats["count"] == 0:
            threshold_was_empty = True
        prev_count = stats["count"]
        panel = _downsample_mask_for_display(mask)
        panels_with_captions.append((panel, _format_caption(label, stats)))

    if threshold_was_empty:
        print("\n-> Thresholding itself found NOTHING. Check the histogram above: is there a gray-level "
              "bucket that should be the organoid cores but isn't crossing `threshold`? If pixel values "
              "look washed out/flat, suspect upstream data (stale .jpg cache, or the source file itself).")

    raw_preview = _load_image_preview_for_comparison(tile_path, max_side=PANEL_MAX_SIDE, binary=False)
    row_image = _compose_sweep_row(
        raw_preview, panels_with_captions,
        row_title="diagnose: pixel data -> threshold -> ... -> filter (step by step)",
    )
    out_path = os.path.join(output_dir, "diagnose.jpg")
    row_image.save(out_path, "JPEG", quality=90)
    print(f"\nSaved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="2026_04_30-Zeynep-CP")
    parser.add_argument("--day", default="39")
    parser.add_argument("--batch", default="CP03_1")
    parser.add_argument("--tile", default="Tile000002.jpg")
    parser.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--pixel-to-micron-x", type=float, default=DEFAULT_PIXEL_TO_MICRON_X)
    parser.add_argument("--pixel-to-micron-y", type=float, default=DEFAULT_PIXEL_TO_MICRON_Y)
    parser.add_argument("--output-dir", default=None,
                         help="Default: <base_dir>/<project>/processed_data/Day <day>/<batch>/parameter_fitting/<tile>/")
    parser.add_argument("--sweep", default=None, help="Comma-separated subset of SWEEPS to run (default: all).")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                         help="Override a BASELINE value for this run, e.g. --set min_area=50000")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be tried without running anything.")
    parser.add_argument("--diagnose", action="store_true",
                         help="Instead of sweeping, report exactly which file gets read, a raw pixel "
                              "histogram, and one panel per pipeline stage at BASELINE (raw threshold, "
                              "min-area filter, max-area filter, candidate contours, circularity filter, "
                              "area filter, final result), each with a count delta from the previous "
                              "stage, to pinpoint exactly which stage organoids are lost at.")
    args = parser.parse_args()

    for override in args.set:
        key, _, raw_value = override.partition("=")
        if key not in BASELINE:
            raise ValueError(f"--set {key}=... : '{key}' is not a BASELINE parameter ({sorted(BASELINE.keys())}).")
        BASELINE[key] = _coerce_value(raw_value)

    tile_path = _resolve_tile_path(args.base_dir, args.project, args.day, args.batch, args.tile)
    print(f"Tile: {tile_path}")

    tile_stem = os.path.splitext(os.path.basename(tile_path))[0]
    output_dir = args.output_dir or os.path.join(
        args.base_dir, args.project, "processed_data", f"Day {args.day}", args.batch,
        "parameter_fitting", tile_stem,
    )
    print(f"Output dir: {output_dir}")

    if args.diagnose:
        run_diagnose(tile_path, args.pixel_to_micron_x, args.pixel_to_micron_y, output_dir)
        return

    sweep_names = args.sweep.split(",") if args.sweep else None
    run_sweep(
        tile_path, args.pixel_to_micron_x, args.pixel_to_micron_y, output_dir,
        sweep_names=sweep_names, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
