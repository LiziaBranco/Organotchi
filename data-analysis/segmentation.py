import argparse
import os


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run organoid segmentation for a selected project."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue an interrupted segmentation run by skipping day/batch/run "
            "combinations whose expected outputs already exist."
        ),
    )
    return parser.parse_args()


ARGS = _parse_args()

from experiment_info import *
from utils_segmentation import (
    apply_filtering_to_segmentations,
    extract_organoid_morphology_metrics,
    generate_classical_segmentation_comparison_figures,
    prepare_classical_segmentation_images,
    run_classical_brightfield_segmentation,
    run_deep_learning_segmentation,
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


def _classical_day_complete(batch_name, run, day):
    processed_dir, expected_stems, expected_count = _classical_expected_outputs(
        batch_name, run, day
    )

    prepared_complete, prepared_reason = _prepared_classical_images_complete(
        batch_name, run, day, expected_stems, expected_count
    )
    if not prepared_complete:
        return False, prepared_reason

    complete, reason = _required_outputs_complete(
        processed_dir,
        expected_stems,
        expected_count,
        CLASSICAL_COMPLETE_OUTPUT_SUBDIRS,
    )
    if not complete:
        return complete, reason

    summary_path = os.path.join(processed_dir, "classical_segmentation_summary.csv")
    if not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0:
        return False, "missing classical_segmentation_summary.csv"

    return True, reason


def _run_day_complete(batch_name, run, day):
    if run["method"] == "dl":
        return _deep_learning_day_complete(batch_name, run, day)
    if run["method"] == "cl":
        return _classical_day_complete(batch_name, run, day)
    return False, f"unknown method {run['method']}"


def _days_to_process(batch_name, run):
    if not ARGS.resume:
        return run["days"]

    pending_days = []
    for day in run["days"]:
        complete, reason = _run_day_complete(batch_name, run, day)
        if complete:
            print(f"⏭️  Resume: skipping {batch_name}/{run['name']} Day {day} ({reason}).")
        else:
            print(f"▶️  Resume: processing {batch_name}/{run['name']} Day {day} ({reason}).")
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
    )

    if run.get("filter_masks", True):
        apply_filtering_to_segmentations(
            base_dir,
            project_dir,
            days,
            [batch_name],
            pixel_size["x"],
            pixel_size["y"],
        )

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
            center_fraction=run["center_fraction"],
            images_per_plate=run.get("images_per_plate", run.get("stitched_images_per_batch", 24)),
            foreground_polarity=run.get("foreground_polarity", "dark"),
            threshold_method=run.get("threshold_method", "fixed"),
            blur_kernel_size=run.get("blur_kernel_size", 0),
        )

    generate_classical_segmentation_comparison_figures(
        base_dir,
        project_dir,
        days,
        [batch_name],
    )

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
    )


if ARGS.resume:
    print("🔁 Resume mode enabled: completed day/batch/run outputs will be skipped.")

for batch in batches_info:
    batch_name = batch["batch"]

    for run in get_segmentation_runs(batch):
        if not run["days"]:
            continue

        pending_days = _days_to_process(batch_name, run)
        if not pending_days:
            print(f"✅ Resume: all outputs already complete for {batch_name}/{run['name']}.")
            continue

        run_to_process = dict(run)
        run_to_process["days"] = pending_days

        if run_to_process["method"] == "dl":
            _run_deep_learning_pipeline(batch_name, run_to_process)
        elif run_to_process["method"] == "cl":
            _run_classical_pipeline(batch_name, run_to_process)
        else:
            print(f"Skipping {batch_name}/{run_to_process['name']}: unknown method {run_to_process['method']}")

print("Processing completed for all plates.")
