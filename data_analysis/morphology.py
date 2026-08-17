"""
Morphology-metrics extraction for a selected project -- reads whatever
segmentation masks segmentation.py has already produced
(``inference_filtered_npy``/``inference_npy``) and writes the per-object/
per-well CSVs under each day/batch's ``inference_metrics/`` (``mask_metrics
.csv``, ``well_metrics_summary.csv``, and -- for runs with
``track_light_dark_split: true`` -- ``light_dark_metrics.csv`` plus the
``light_dark_overlay_jpg/`` QC images).

Split out of segmentation.py on purpose: segmentation (producing masks) and
morphology extraction (measuring them) are independent stages that don't
need to run back-to-back in the same process, and a metrics-only code
change (e.g. a fixed morphology formula) now only means re-running this
script, not touching segmentation.py at all. Run this after segmentation.py
has produced masks for the days/batches/runs you care about; run plot.py
after this to turn the resulting CSVs into figures/statistics.

Uses experiment_info.py's own project-selection prompt and
``segmentation_metadata.json``-driven ``batches_info``/``get_segmentation_
runs``, exactly like segmentation.py -- the same project selection, the
same per-run config (``days``, ``organoid_mode``, pixel calibration,
``track_light_dark_split`` and its related keys), just without any of
segmentation.py's own model-loading/venv-switching machinery, none of
which morphology extraction needs (it never imports torch/mmsegmentation/
OrgLine, only numpy/pandas/PIL/skimage).

Run: python3 morphology.py [--resume | --force | --check] [--only-methods ...]
"""

import argparse
import os


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Extract morphology metrics for a selected project."
    )
    parser.add_argument(
        "--only-methods",
        default=None,
        help=(
            "Comma-separated segmentation methods (e.g. 'orgline' or 'dl,cl') to "
            "extract metrics for this invocation, skipping every other configured "
            "run -- matches segmentation_metadata.json's own run 'method' values, "
            "purely for filtering which runs this script processes; extraction "
            "itself doesn't care which method produced the masks it reads."
        ),
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--resume",
        dest="mode",
        action="store_const",
        const="resume",
        help=(
            "Default. Extract metrics for every configured day/batch/run whose "
            "masks exist, skipping any (day, batch) combination whose metrics are "
            "already up to date (mtime-compared against its source masks -- see "
            "utils_morphology.py's _metrics_outputs_are_current)."
        ),
    )
    mode_group.add_argument(
        "--force",
        dest="mode",
        action="store_const",
        const="force",
        help=(
            "Ignore staleness checks and recompute every configured day/batch/"
            "run's metrics from scratch, regardless of whether they already look "
            "up to date. Use this after a metrics-only code change (e.g. a fixed "
            "morphology formula) so mask_metrics.csv/well_metrics_summary.csv/"
            "light_dark_metrics.csv get regenerated even though their own mtimes "
            "wouldn't otherwise look stale."
        ),
    )
    mode_group.add_argument(
        "--check",
        dest="mode",
        action="store_const",
        const="check",
        help="Report which day/batch/run metrics are missing/stale/up to date, without extracting anything.",
    )
    parser.set_defaults(mode="resume")
    return parser.parse_args()


ARGS = _parse_args()
ARGS.only_methods = (
    {m.strip() for m in ARGS.only_methods.split(",") if m.strip()} if ARGS.only_methods else None
)

from experiment_info import *
from utils_morphology import (
    _metrics_outputs_are_current,
    extract_light_dark_metrics,
    extract_organoid_morphology_metrics,
    generate_light_dark_overlay_masks,
)


def _run_selected(run):
    return not ARGS.only_methods or run["method"] in ARGS.only_methods


def _masks_dir(batch_name, day):
    processed_dir = os.path.join(base_dir, project_dir, "processed_data", f"Day {day}", batch_name)
    masks_dir = os.path.join(processed_dir, "inference_filtered_npy")
    if not os.path.isdir(masks_dir):
        masks_dir = os.path.join(processed_dir, "inference_npy")
    return masks_dir if os.path.isdir(masks_dir) else None


def _morphology_status(batch_name, day):
    """Report whether mask_metrics.csv/well_metrics_summary.csv are up to
    date for one day/batch, the same way regardless of which segmentation
    method (dl/cl/orgline) produced the masks -- it just looks at whichever
    mask folder (filtered or raw) extraction would read."""
    masks_dir = _masks_dir(batch_name, day)
    if masks_dir is None:
        return False, "no masks generated yet"

    metrics_path = os.path.join(base_dir, project_dir, "processed_data", f"Day {day}", batch_name, "inference_metrics")
    if _metrics_outputs_are_current(metrics_path, masks_dir):
        return True, "morphology metrics up to date"
    return False, "morphology metrics missing or stale"


def _extract_metrics_for_run(batch_name, run, force):
    """Run morphology (and light-dark split, where enabled) metrics
    extraction for one batch/run, against whatever masks already exist for
    ``run["days"]``. Each of the functions called here already skips any
    (day, batch) combination whose own outputs are current unless
    ``force=True``, so this can always be called with the run's full day
    list -- no separate day-by-day pre-filter needed here the way
    segmentation.py's own mask-completeness check has to do for
    segmentation itself.
    """
    days = run["days"]
    pixel_size = get_pixel_to_micron_for_run(run)

    extract_organoid_morphology_metrics(
        base_dir,
        project_dir,
        days,
        [batch_name],
        pixel_size["x"],
        pixel_size["y"],
        organoid_mode=run["organoid_mode"],
        run_name=run["name"],
        images_per_plate=run.get("images_per_plate", run.get("stitched_images_per_batch", 24)),
        force=force,
    )

    if run.get("track_light_dark_split", False):
        extract_light_dark_metrics(
            base_dir,
            project_dir,
            days,
            [batch_name],
            pixel_size["x"],
            pixel_size["y"],
            organoid_mode=run["organoid_mode"],
            foreground_polarity=run.get("foreground_polarity", "dark"),
            min_eta_squared=run.get("light_dark_min_eta_squared", 0.3),
            images_per_plate=run.get("images_per_plate", run.get("stitched_images_per_batch", 24)),
            run_name=run["name"],
            force=force,
        )
        generate_light_dark_overlay_masks(
            base_dir,
            project_dir,
            days,
            [batch_name],
            foreground_polarity=run.get("foreground_polarity", "dark"),
            min_eta_squared=run.get("light_dark_min_eta_squared", 0.3),
            light_color=run.get("light_dark_overlay_color", (255, 0, 255)),
            force=force,
        )


def _print_check_report():
    total = 0
    up_to_date = 0
    for batch in batches_info:
        batch_name = batch["batch"]
        for run in get_segmentation_runs(batch):
            if not run["days"] or not _run_selected(run):
                continue

            print(f"\n{batch_name}/{run['name']} ({run['method']}):")
            for day in run["days"]:
                total += 1
                ok, reason = _morphology_status(batch_name, day)
                up_to_date += int(ok)
                icon = "✅" if ok else "❌"
                print(f"  Day {day}: {icon} ({reason})")

    print(f"\nSummary: morphology metrics up to date for {up_to_date}/{total} day/batch/run combinations.")


if ARGS.mode == "check":
    print("🔍 Check mode: reporting metrics status without extracting anything.")
    _print_check_report()
else:
    if ARGS.mode == "force":
        print("🔁 Force mode enabled: every configured day/batch/run's metrics will be recomputed from scratch.")
    else:
        print("⏯️  Resume mode (default): extracting metrics wherever masks exist and metrics are missing/stale.")

    for batch in batches_info:
        batch_name = batch["batch"]
        for run in get_segmentation_runs(batch):
            if not run["days"] or not _run_selected(run):
                continue

            print(f"\n📐 {batch_name}/{run['name']} ({run['method']}): extracting metrics for {len(run['days'])} day(s).")
            _extract_metrics_for_run(batch_name, run, force=(ARGS.mode == "force"))

    print("Morphology extraction completed for all plates.")
