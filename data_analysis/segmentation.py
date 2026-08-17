import argparse
import os
import sys
import subprocess

# Set before any torch/numpy/MKL-linked import happens (including
# transitively, via utils_segmentation.py below) -- on a many-core machine,
# these libraries default to spawning one thread per core for every
# individual op, and the resulting thread-creation/scheduling overhead can
# dwarf the actual compute for a single-image forward pass. Measured
# directly for this exact class of problem in the Organotchi-manuscript
# repo's benchmarking/environments/README.md ("CPU thread oversubscription"):
# a single Cellpose call on this machine dropped from 4m40s+ (sys time alone
# >3m28s) to a few seconds once these were set to 4. OrgLine (also
# torch-based) was never given this fix when the "orgline" segmentation
# method was integrated here, and a real production run showed exactly the
# same symptom (~17s/image, most of it in the detector/SAM2 forward passes,
# not the actual image content). setdefault so an operator's own explicit
# setting is never overridden. Inherited automatically by both the
# subprocess- and execv-based venv switches below (os.environ is passed
# through by default in each case), so this covers every OrgLine invocation
# path, not just a direct one.
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
            "'orgline' or 'dl,cl'), skipping every other configured run. Mainly set "
            "automatically when this script re-invokes itself under OrgLine's venv "
            "for a project mixing 'orgline' with 'dl'/'cl' runs (see the top-level "
            "venv-switch logic below) -- pass it yourself only if you want to split "
            "the work across separate invocations by hand."
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
    parser.set_defaults(mode="resume")
    return parser.parse_args()


ARGS = _parse_args()
ARGS.only_methods = (
    {m.strip() for m in ARGS.only_methods.split(",") if m.strip()} if ARGS.only_methods else None
)

# Methods delegated to a child process this invocation (see the venv-switch
# logic near the bottom of this file) -- excluded from this process's own
# dispatch loops so the same work isn't done twice.
DELEGATED_METHODS = set()

from experiment_info import *
from utils_segmentation import (
    apply_filtering_to_segmentations,
    generate_classical_segmentation_comparison_figures,
    prepare_classical_segmentation_images,
    run_classical_brightfield_segmentation,
    run_deep_learning_segmentation,
    run_orgline_segmentation,
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
            organoid_mode=run["organoid_mode"],
            images_per_plate=run.get("images_per_plate", run.get("stitched_images_per_batch", 24)),
            device=run.get("device", "cpu"),
            run_name=run["name"],
        )

    # Optional, same post-inference area/circularity filter DL mode uses (off
    # by default here is NOT the case -- filter_masks defaults to True the
    # same way it does for dl; set it explicitly false per-run if it doesn't
    # suit that run). NOTE: max_objects_before_circularity_pick defaults to 1
    # (keep only the single most circular survivor), which is only correct
    # for organoid_mode "single" -- a "multiple" run relying on the default
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


def _print_check_report():
    """Report segmentation output status per day/batch/run without running
    anything. Run morphology.py --check for the corresponding metrics
    status -- that's a separate stage now, with its own check mode."""
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
    loops -- respects an explicit --only-methods filter, and excludes
    whatever's already been delegated to a child process this invocation
    (see the venv-switch logic below)."""
    if ARGS.only_methods and run["method"] not in ARGS.only_methods:
        return False
    if run["method"] in DELEGATED_METHODS:
        return False
    return True


def _orgline_venv_python(orgline_repo):
    """OrgLine's isolated venv interpreter, resolved as the sibling
    orgline_venv/ next to orgline_repo's own environments/ folder (see
    ../environments/README.md) -- not hardcoded, so a project
    pointing orgline_repo somewhere else still resolves correctly."""
    environments_dir = os.path.dirname(os.path.abspath(orgline_repo))
    return os.path.join(environments_dir, "orgline_venv", "bin", "python")


def _running_under(python_path):
    """Whether the CURRENT process is running under the venv that
    ``python_path`` (a venv's own bin/python) belongs to. Compares
    ``sys.prefix`` (the venv's own root directory, as the interpreter itself
    resolved it at startup) rather than resolving ``python_path``/
    ``sys.executable`` as file paths -- venvs are typically created without
    ``--copies``, so bin/python is a symlink to the same underlying system
    interpreter for every venv; realpath-resolving it collapses every venv
    on the machine to one identical path and can never distinguish them."""
    venv_root = os.path.dirname(os.path.dirname(os.path.abspath(os.path.expanduser(python_path))))
    return os.path.abspath(sys.prefix) == venv_root


def _pending_methods():
    """Which segmentation methods actually have day/batch/run work left this
    invocation, via cheap filesystem-only completeness checks (no model
    loading) -- used to decide whether OrgLine's isolated venv needs to be
    engaged at all, and whether anything else also needs the current/native
    environment."""
    methods = set()
    for batch in batches_info:
        batch_name = batch["batch"]
        for run in get_segmentation_runs(batch):
            if not run["days"] or not _run_selected(run):
                continue
            if ARGS.mode == "force":
                methods.add(run["method"])
                continue
            for day in run["days"]:
                complete, _ = _run_day_complete(batch_name, run, day)
                if not complete:
                    methods.add(run["method"])
                    break
    return methods


def _first_orgline_repo():
    for batch in batches_info:
        for run in get_segmentation_runs(batch):
            if run["method"] == "orgline":
                return run["orgline_repo"]
    return None


def _switch_to_orgline_venv_if_needed():
    """If any 'orgline'-method run still has pending work, make sure it's
    processed under OrgLine's own venv -- never the current one, since
    OrgLine's dependencies (its own torch/torchvision pins, the vendored
    detector, SAM2) are isolated from mmsegmentation's and generally aren't
    importable from wherever this script happened to be launched.

    - If OrgLine is the *only* thing pending, this process re-execs itself
      into OrgLine's venv outright (os.execv fully replaces the process --
      nothing of the previously active environment lingers).
    - If something else (dl/cl, needing the current/native environment) is
      *also* pending, OrgLine's share of the work is delegated to a child
      process running under its venv instead (subprocess.run, with
      --only-methods orgline) -- an isolated child process is itself the
      "deactivation" of that venv the moment it exits, and this process's
      own environment is never touched, so it can go on to handle the
      dl/cl portion right after.
    - If this process is already running under OrgLine's venv but dl/cl
      work is ALSO pending, there's no reliable way to locate "the native
      environment" to switch back to (unlike orgline_venv, nothing in this
      project's config points to it) -- that portion is reported instead of
      silently attempted or skipped.
    """
    if ARGS.mode not in ("resume", "force"):
        return  # --check needs no model loading at all

    pending = _pending_methods()
    if "orgline" not in pending:
        return

    orgline_repo = _first_orgline_repo()
    orgline_python = _orgline_venv_python(orgline_repo)

    if _running_under(orgline_python):
        other_pending = pending - {"orgline"}
        if other_pending:
            print(
                f"⚠️  Running under OrgLine's venv, but {', '.join(sorted(other_pending))} "
                "work is also pending and needs the native environment -- this "
                "invocation only handles the OrgLine portion. Re-run from the native "
                "environment for the rest."
            )
        return

    other_pending = pending - {"orgline"}
    child_argv = [orgline_python, os.path.abspath(__file__)] + sys.argv[1:] + ["--only-methods", "orgline"]

    if not other_pending:
        print(
            f"🔀 Only OrgLine-method work is pending -- switching this process to its venv ({orgline_python}).",
            flush=True,
        )
        os.execv(orgline_python, child_argv)
        return  # unreachable -- execv replaces this process outright

    print(
        f"🔀 OrgLine-method work is pending alongside {', '.join(sorted(other_pending))} work -- "
        f"delegating the OrgLine portion to its own venv ({orgline_python}); this process's own "
        "environment stays active for the rest.",
        flush=True,
    )
    result = subprocess.run(child_argv)
    if result.returncode != 0:
        sys.exit(result.returncode)
    DELEGATED_METHODS.add("orgline")


_switch_to_orgline_venv_if_needed()

if ARGS.mode == "check":
    print("🔍 Check mode: reporting output status without processing anything.")
    _print_check_report()
elif ARGS.mode == "force":
    print("🔁 Force mode enabled: every configured day/batch/run will be fully recomputed.")
else:
    print("⏯️  Resume mode (default): continuing from wherever the analysis stopped.")

if ARGS.mode != "check":
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
