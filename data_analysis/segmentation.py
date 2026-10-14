import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run organoid segmentation for a selected project."
    )
    parser.add_argument(
        "--only-methods",
        default=None,
        help=(
            "Comma-separated segmentation methods to process this invocation (e.g. "
            "'orgline' or 'dl,cl'), skipping every other configured run -- pass this if "
            "you want to split the work across separate invocations by hand."
        ),
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--resume",
        dest="mode",
        action="store_const",
        const="resume",
        help=(
            "Default. Continue from wherever the analysis stopped: skip any "
            "day/batch/run whose segmentation outputs already exist, for both "
            "deep-learning and classical runs. Run morphology.py afterward to "
            "extract metrics from the masks this produces."
        ),
    )
    mode_group.add_argument(
        "--force",
        dest="mode",
        action="store_const",
        const="force",
        help=(
            "Ignore all existing outputs and recompute segmentation and "
            "filtering from scratch for every configured day/batch/run, "
            "regardless of segmentation method."
        ),
    )
    mode_group.add_argument(
        "--check",
        dest="mode",
        action="store_const",
        const="check",
        help="Report which day/batch/run segmentation outputs already exist, are missing, or are stale, without running anything.",
    )
    mode_group.add_argument(
        "--correct_mask",
        dest="mode",
        action="store_const",
        const="correct_mask",
        help=(
            "Interactively pick a day, then a batch, then a tile whose mask isn't quite "
            "right, open it in napari for manual correction, then re-extract morphology "
            "metrics for that one batch/day once the window is closed. Needs napari plus "
            "a Qt backend and a real display -- see utils_segmentation.py's "
            "open_mask_correction_viewer."
        ),
    )
    parser.set_defaults(mode="resume")
    return parser.parse_args()


ARGS = _parse_args()
ARGS.only_methods = (
    {m.strip() for m in ARGS.only_methods.split(",") if m.strip()} if ARGS.only_methods else None
)

from experiment_info import *
from utils_segmentation import (
    apply_filtering_to_segmentations,
    generate_classical_segmentation_comparison_figures,
    list_correctable_tiles,
    open_mask_correction_viewer,
    prepare_classical_segmentation_images,
    run_classical_brightfield_segmentation,
    run_deep_learning_segmentation,
    run_orgline_segmentation,
    save_corrected_mask,
)

RAW_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".tif", ".tiff"}
CLASSICAL_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
OUTPUT_EXTENSIONS = {
    "inference_npy": {".npy"},
    "inference_filtered_npy": {".npy"},
    "inference_labels_npy": {".npy"},
    "inference_jpg": {".jpg", ".jpeg"},
    "inference_tif": {".tif", ".tiff"},
    "inference_comparison": {".jpg", ".jpeg"},
}
CLASSICAL_SEGMENTATION_OUTPUT_SUBDIRS = [
    "inference_npy",
    "inference_jpg",
    "inference_tif",
    "inference_labels_npy",
]
CLASSICAL_COMPLETE_OUTPUT_SUBDIRS = CLASSICAL_SEGMENTATION_OUTPUT_SUBDIRS + [
    "inference_comparison"
]


def _collect_stems(directory, extensions, prefix=None):
    if not os.path.isdir(directory):
        return set()

    stems = set()
    for filename in os.listdir(directory):
        stem, extension = os.path.splitext(filename)
        if extension.lower() not in extensions:
            continue
        if prefix and not stem.startswith(prefix):
            continue
        stems.add(stem)

    return stems


def _raw_data_dir(day, batch_name):
    return os.path.join(base_dir, project_dir, "raw_data", f"Day {day}", batch_name)


def _processed_data_dir(day, batch_name):
    return os.path.join(base_dir, project_dir, "processed_data", f"Day {day}", batch_name)


def _output_stems(processed_dir, output_subdir):
    return _collect_stems(
        os.path.join(processed_dir, output_subdir),
        OUTPUT_EXTENSIONS[output_subdir],
    )


def _expected_classical_output_count(run):
    images_per_plate = run.get("images_per_plate", run.get("stitched_images_per_batch", 0))
    num_plates = run.get("num_plates", 1)
    try:
        return int(images_per_plate) * int(num_plates)
    except (TypeError, ValueError):
        return 0


def _expected_tile_stems(expected_count):
    return {f"Tile{index:06d}" for index in range(1, expected_count + 1)}


def _required_outputs_complete(processed_dir, expected_stems, expected_count, required_subdirs):
    for output_subdir in required_subdirs:
        output_stems = _output_stems(processed_dir, output_subdir)
        if expected_stems:
            missing = expected_stems - output_stems
            if missing:
                return False, f"missing {len(missing)} files in {output_subdir}"

        if expected_count > 0:
            if len(output_stems) < expected_count:
                return False, f"{output_subdir} has {len(output_stems)}/{expected_count} files"
        elif not expected_stems:
            return False, "could not determine expected output count"

    return True, "all expected output files found"


def _deep_learning_day_complete(batch_name, run, day):
    raw_dir = _raw_data_dir(day, batch_name)
    expected_stems = _collect_stems(raw_dir, RAW_IMAGE_EXTENSIONS)
    if not expected_stems:
        return False, "no raw images found for completion check"

    processed_dir = _processed_data_dir(day, batch_name)
    required_subdirs = ["inference_npy", "inference_jpg"]
    if run.get("filter_masks", True):
        required_subdirs.append("inference_filtered_npy")

    return _required_outputs_complete(
        processed_dir,
        expected_stems,
        len(expected_stems),
        required_subdirs,
    )


def _classical_expected_outputs(batch_name, run, day):
    raw_dir = _raw_data_dir(day, batch_name)
    raw_tile_stems = _collect_stems(raw_dir, CLASSICAL_IMAGE_EXTENSIONS, prefix="Tile")
    expected_count = _expected_classical_output_count(run)

    if run.get("stitch", False) and expected_count > 0:
        expected_stems = _expected_tile_stems(expected_count)
    else:
        expected_stems = raw_tile_stems
        expected_count = len(expected_stems) or expected_count

    processed_dir = _processed_data_dir(day, batch_name)
    return processed_dir, expected_stems, expected_count


def _prepared_classical_images_complete(batch_name, run, day, expected_stems, expected_count):
    if not run.get("stitch", False) or expected_count <= 0:
        return True, "prepared image check not required"

    raw_dir = _raw_data_dir(day, batch_name)
    stitched_dir = os.path.join(raw_dir, "stitched")
    stitched_stems = _collect_stems(stitched_dir, CLASSICAL_IMAGE_EXTENSIONS, prefix="Tile")
    if stitched_stems:
        if expected_stems:
            missing = expected_stems - stitched_stems
            if missing:
                return False, f"stitched folder missing {len(missing)} prepared images"
        if len(stitched_stems) < expected_count:
            return False, f"stitched folder has {len(stitched_stems)}/{expected_count} prepared images"
        return True, "stitched images found"

    raw_tile_stems = _collect_stems(raw_dir, CLASSICAL_IMAGE_EXTENSIONS, prefix="Tile")
    if raw_tile_stems and len(raw_tile_stems) <= expected_count:
        return True, "prepared Tile images found"

    return (
        False,
        f"missing prepared stitched images ({len(raw_tile_stems)} root Tile images found, "
        f"expected {expected_count} final images)",
    )


def _classical_segmentation_day_complete(batch_name, run, day):
    processed_dir, expected_stems, expected_count = _classical_expected_outputs(
        batch_name, run, day
    )

    prepared_complete, prepared_reason = _prepared_classical_images_complete(
        batch_name, run, day, expected_stems, expected_count
    )
    if not prepared_complete:
        return False, prepared_reason

    return _required_outputs_complete(
        processed_dir,
        expected_stems,
        expected_count,
        CLASSICAL_SEGMENTATION_OUTPUT_SUBDIRS,
    )


def _classical_day_complete(
    batch_name, run, day, summary_filename="classical_segmentation_summary.csv", extra_required_subdirs=None
):
    """Shared completeness check for both classical and OrgLine runs -- the
    two methods write to the same directory layout (see
    CLASSICAL_COMPLETE_OUTPUT_SUBDIRS), differing only in which per-method
    summary CSV they produce (``summary_filename``) and, for OrgLine with
    ``filter_masks`` enabled, an extra required output directory
    (``extra_required_subdirs``) -- classical mode has no equivalent
    post-filter step, so it never passes this."""
    processed_dir, expected_stems, expected_count = _classical_expected_outputs(
        batch_name, run, day
    )

    prepared_complete, prepared_reason = _prepared_classical_images_complete(
        batch_name, run, day, expected_stems, expected_count
    )
    if not prepared_complete:
        return False, prepared_reason

    required_subdirs = CLASSICAL_COMPLETE_OUTPUT_SUBDIRS + list(extra_required_subdirs or [])
    complete, reason = _required_outputs_complete(
        processed_dir,
        expected_stems,
        expected_count,
        required_subdirs,
    )
    if not complete:
        return complete, reason

    summary_path = os.path.join(processed_dir, summary_filename)
    if not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0:
        return False, f"missing {summary_filename}"

    return True, reason


def _run_day_complete(batch_name, run, day):
    if run["method"] == "dl":
        return _deep_learning_day_complete(batch_name, run, day)
    if run["method"] == "cl":
        return _classical_day_complete(batch_name, run, day)
    if run["method"] == "orgline":
        extra = ["inference_filtered_npy"] if run.get("filter_masks", True) else None
        return _classical_day_complete(
            batch_name, run, day, summary_filename="orgline_segmentation_summary.csv",
            extra_required_subdirs=extra,
        )
    return False, f"unknown method {run['method']}"


def _days_to_process(batch_name, run):
    if ARGS.mode == "force":
        return run["days"]

    pending_days = []
    for day in run["days"]:
        complete, reason = _run_day_complete(batch_name, run, day)
        if complete:
            print(f"⏭️  Skipping {batch_name}/{run['name']} Day {day}: segmentation complete ({reason}).")
        else:
            print(f"▶️  Processing {batch_name}/{run['name']} Day {day}: {reason}.")
            pending_days.append(day)

    return pending_days


def _run_deep_learning_pipeline(batch_name, run):
    days = run["days"]
    pixel_size = get_pixel_to_micron_for_run(run)

    print(
        f"\n🧠 Processing Plate {batch_name} "
        f"({run['name']}: deep learning, {run['magnification']}, {run['tile_grid'][0]}x{run['tile_grid'][1]})"
    )

    run_deep_learning_segmentation(
        run["checkpoint"],
        base_dir,
        project_dir,
        days,
        [batch_name],
        crop_days=run.get("crop_days", []),
        run_name=run["name"],
        model_config_path=run.get("model_config_path"),
        device=run.get("device", "cpu"),
    )

    if run.get("filter_masks", True):
        apply_filtering_to_segmentations(
            base_dir,
            project_dir,
            days,
            [batch_name],
            pixel_size["x"],
            pixel_size["y"],
            run_name=run["name"],
            min_area=run["min_area"],
            max_area=run["max_area"],
            min_circularity=run["min_circularity"],
            erosion_kernel_size=run.get("erosion_kernel_size", (0, 0)),
            max_objects_before_circularity_pick=run.get("max_objects_before_circularity_pick", 1),
            dilation_initial_kernel_size=run.get("dilation_initial_kernel_size", (0, 0)),
            dilation_iterations=run.get("dilation_iterations", 0),
            dilation_reduction_factor=run.get("dilation_reduction_factor", 0.8),
        )


def _run_classical_pipeline(batch_name, run):
    days = run["days"]
    pixel_size = get_pixel_to_micron_for_run(run)
    tile_grid = run["tile_grid"]

    print(
        f"\n🧠 Processing Plate {batch_name} "
        f"({run['name']}: classical, {run['magnification']}, "
        f"{tile_grid[0]}x{tile_grid[1]}, {run['organoid_mode']} organoid mode)"
    )

    segmentation_days = []
    for day in days:
        if ARGS.mode == "force":
            segmentation_days.append(day)
            continue

        complete, reason = _classical_segmentation_day_complete(batch_name, run, day)
        if complete:
            print(f"Classical masks already present for {batch_name}, Day {day} ({reason}).")
        else:
            print(f"Classical masks missing for {batch_name}, Day {day} ({reason}).")
            segmentation_days.append(day)

    if segmentation_days:
        if run.get("stitch", True):
            prepare_classical_segmentation_images(
                base_dir,
                project_dir,
                segmentation_days,
                [batch_name],
                [run["num_plates"]],
                tile_grid=tile_grid,
                stitched_images_per_batch=run["stitched_images_per_batch"],
                overlap_x=run["overlap_x"],
                overlap_y=run["overlap_y"],
                tile_order=run["tile_order"],
            )

        run_classical_brightfield_segmentation(
            base_dir,
            project_dir,
            segmentation_days,
            [batch_name],
            pixel_size["x"],
            pixel_size["y"],
            organoid_mode=run["organoid_mode"],
            threshold=run["threshold"],
            max_area=run["max_area"],
            min_area=run["min_area"],
            min_circularity=run["min_circularity"],
            images_per_plate=run.get("images_per_plate", run.get("stitched_images_per_batch", 24)),
            foreground_polarity=run.get("foreground_polarity", "dark"),
            center_fraction=run.get("center_fraction"),
            close_kernel_size=run.get("close_kernel_size"),
            run_name=run["name"],
        )

    generate_classical_segmentation_comparison_figures(
        base_dir,
        project_dir,
        days,
        [batch_name],
        overwrite=True,
    )


def _run_orgline_pipeline(batch_name, run):
    days = run["days"]
    pixel_size = get_pixel_to_micron_for_run(run)
    tile_grid = run["tile_grid"]

    print(
        f"\n🧬 Processing Plate {batch_name} "
        f"({run['name']}: OrgLine, {run['magnification']}, "
        f"{tile_grid[0]}x{tile_grid[1]}, {run['organoid_mode']} organoid mode)"
    )

    segmentation_days = []
    for day in days:
        if ARGS.mode == "force":
            segmentation_days.append(day)
            continue

        complete, reason = _classical_day_complete(
            batch_name, run, day, summary_filename="orgline_segmentation_summary.csv"
        )
        if complete:
            print(f"OrgLine masks already present for {batch_name}, Day {day} ({reason}).")
        else:
            print(f"OrgLine masks missing for {batch_name}, Day {day} ({reason}).")
            segmentation_days.append(day)

    if segmentation_days:
        if run.get("stitch", False):
            prepare_classical_segmentation_images(
                base_dir,
                project_dir,
                segmentation_days,
                [batch_name],
                [run["num_plates"]],
                tile_grid=tile_grid,
                stitched_images_per_batch=run["stitched_images_per_batch"],
                overlap_x=run["overlap_x"],
                overlap_y=run["overlap_y"],
                tile_order=run["tile_order"],
            )

        run_orgline_segmentation(
            base_dir,
            project_dir,
            segmentation_days,
            [batch_name],
            orgline_repo=run["orgline_repo"],
            orgdet_checkpoint=run.get("orgdet_checkpoint"),
            sam2_checkpoint=run.get("sam2_checkpoint"),
            conf=run.get("orgline_conf", 0.2),
            iou=run.get("orgline_iou", 0.7),
            imgsz=run.get("orgline_imgsz", 640),
            agnostic_nms=run.get("orgline_agnostic_nms", False),
            organoid_mode=run["organoid_mode"],
            images_per_plate=run.get("images_per_plate", run.get("stitched_images_per_batch", 24)),
            device=run.get("device", "cpu"),
            run_name=run["name"],
        )

    # Optional, same post-inference area/circularity filter DL mode uses:
    # filter_masks defaults to True the same way it does for dl, so set it
    # explicitly false per-run when it doesn't suit that run.
    # max_objects_before_circularity_pick defaults to 1 (keep only the single
    # most circular survivor), which is only correct for
    # organoid_mode "single" -- a "multiple" run relying on the default
    # would silently collapse every image down to one kept structure. Set
    # max_objects_before_circularity_pick explicitly (e.g. a large number) or
    # filter_masks: false for "multiple" orgline runs that want every
    # detection kept.
    if run.get("filter_masks", True):
        apply_filtering_to_segmentations(
            base_dir,
            project_dir,
            days,
            [batch_name],
            pixel_size["x"],
            pixel_size["y"],
            run_name=run["name"],
            min_area=run["min_area"],
            max_area=run["max_area"],
            min_circularity=run["min_circularity"],
            erosion_kernel_size=run.get("erosion_kernel_size", (0, 0)),
            max_objects_before_circularity_pick=run.get("max_objects_before_circularity_pick", 1),
            dilation_initial_kernel_size=run.get("dilation_initial_kernel_size", (0, 0)),
            dilation_iterations=run.get("dilation_iterations", 0),
            dilation_reduction_factor=run.get("dilation_reduction_factor", 0.8),
        )

    generate_classical_segmentation_comparison_figures(
        base_dir,
        project_dir,
        days,
        [batch_name],
        overwrite=True,
    )


def _run_mask_correction():
    """Interactively pick a day, then a batch, then a tile, open its mask in
    napari for manual correction, and re-extract morphology metrics for
    that one batch/day once the window is closed.
    """
    import questionary

    all_days = sorted({
        day
        for batch in batches_info
        for run in get_segmentation_runs(batch)
        for day in run["days"]
    })
    if not all_days:
        print("No days configured for this project.")
        return

    day_str = questionary.select(
        "Select a day to correct a mask for:",
        choices=[str(day) for day in all_days],
    ).ask()
    if day_str is None:
        return
    day = next(day for day in all_days if str(day) == day_str)

    candidate_batches = sorted({
        batch["batch"]
        for batch in batches_info
        for run in get_segmentation_runs(batch)
        if day in run["days"]
    })
    available_batches = [
        batch_name for batch_name in candidate_batches
        if list_correctable_tiles(base_dir, project_dir, day, batch_name)
    ]
    if not available_batches:
        print(f"No segmented masks found for Day {day} yet -- run segmentation.py first.")
        return

    batch_name = questionary.select(
        f"Select a batch (Day {day}):", choices=available_batches,
    ).ask()
    if batch_name is None:
        return

    tiles = list_correctable_tiles(base_dir, project_dir, day, batch_name)
    tile_name = questionary.select(
        f"Select a tile to correct ({batch_name}, Day {day}):", choices=tiles,
    ).ask()
    if tile_name is None:
        return

    edited_labels, masks_dir, tile_name = open_mask_correction_viewer(
        base_dir, project_dir, day, batch_name, tile_name,
    )
    save_corrected_mask(base_dir, project_dir, day, batch_name, tile_name, masks_dir, edited_labels)

    # Deliberate exception to this script's usual separation from morphology
    # extraction -- see this function's own docstring.
    from utils_morphology import (
        extract_light_dark_metrics,
        extract_organoid_morphology_metrics,
        generate_light_dark_overlay_masks,
    )

    matching_runs = [
        run for batch in batches_info if batch["batch"] == batch_name
        for run in get_segmentation_runs(batch) if day in run["days"]
    ]
    if not matching_runs:
        print(f"⚠️  No configured run covers {batch_name}, Day {day} -- skipping metrics re-extraction.")
        return
    run = matching_runs[0]
    pixel_size = get_pixel_to_micron_for_run(run)
    images_per_plate = run.get("images_per_plate", run.get("stitched_images_per_batch", 24))

    print(f"\nRe-extracting morphology metrics for {batch_name}, Day {day} ...")
    extract_organoid_morphology_metrics(
        base_dir, project_dir, [day], [batch_name], pixel_size["x"], pixel_size["y"],
        organoid_mode=run["organoid_mode"], run_name=run["name"],
        images_per_plate=images_per_plate, force=True,
    )
    if run.get("track_light_dark_split", False):
        extract_light_dark_metrics(
            base_dir, project_dir, [day], [batch_name], pixel_size["x"], pixel_size["y"],
            organoid_mode=run["organoid_mode"],
            foreground_polarity=run.get("foreground_polarity", "dark"),
            min_eta_squared=run.get("light_dark_min_eta_squared", 0.3),
            images_per_plate=images_per_plate, run_name=run["name"], force=True,
        )
        generate_light_dark_overlay_masks(
            base_dir, project_dir, [day], [batch_name],
            foreground_polarity=run.get("foreground_polarity", "dark"),
            min_eta_squared=run.get("light_dark_min_eta_squared", 0.3),
            light_color=run.get("light_dark_overlay_color", (255, 0, 255)),
            force=True,
        )
    print("Done -- morphology metrics now reflect the corrected mask.")


def _print_check_report():
    """Report segmentation output status per day/batch/run without running
    anything. Run morphology.py --check for the corresponding metrics
    status."""
    total = 0
    segmentation_complete = 0

    for batch in batches_info:
        batch_name = batch["batch"]
        for run in get_segmentation_runs(batch):
            if not run["days"] or not _run_selected(run):
                continue

            print(f"\n{batch_name}/{run['name']} ({run['method']}):")
            for day in run["days"]:
                total += 1
                seg_ok, seg_reason = _run_day_complete(batch_name, run, day)
                segmentation_complete += int(seg_ok)

                seg_icon = "✅" if seg_ok else "❌"
                print(f"  Day {day}: segmentation {seg_icon} ({seg_reason})")

    print(f"\nSummary: segmentation complete for {segmentation_complete}/{total} day/batch/run combinations.")


def _run_selected(run):
    """Whether ``run`` should be processed by *this* process's own dispatch
    loops -- respects an explicit --only-methods filter."""
    if ARGS.only_methods and run["method"] not in ARGS.only_methods:
        return False
    return True


if ARGS.mode == "check":
    print("🔍 Check mode: reporting output status without processing anything.")
    _print_check_report()
elif ARGS.mode == "correct_mask":
    print("🖌️  Mask-correction mode: pick a day/batch/tile to edit in napari.")
    _run_mask_correction()
elif ARGS.mode == "force":
    print("🔁 Force mode enabled: every configured day/batch/run will be fully recomputed.")
else:
    print("⏯️  Resume mode (default): continuing from wherever the analysis stopped.")

if ARGS.mode not in ("check", "correct_mask"):
    for batch in batches_info:
        batch_name = batch["batch"]

        for run in get_segmentation_runs(batch):
            if not run["days"] or not _run_selected(run):
                continue

            pending_days = _days_to_process(batch_name, run)
            if not pending_days:
                print(f"✅ All outputs already complete for {batch_name}/{run['name']}.")
                continue

            run_to_process = dict(run)
            run_to_process["days"] = pending_days

            if run_to_process["method"] == "dl":
                _run_deep_learning_pipeline(batch_name, run_to_process)
            elif run_to_process["method"] == "cl":
                _run_classical_pipeline(batch_name, run_to_process)
            elif run_to_process["method"] == "orgline":
                _run_orgline_pipeline(batch_name, run_to_process)
            else:
                print(f"Skipping {batch_name}/{run_to_process['name']}: unknown method {run_to_process['method']}")

    print("Processing completed for all plates.")
