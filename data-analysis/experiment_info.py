import questionary
import sys
import os
import json


def _reject_duplicate_json_keys(pairs):
    """Reject duplicate keys that Python's json parser would otherwise overwrite."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"Duplicate key '{key}' found in experiment_metadata.json. "
                "If this is a batch entry, each batch must be its own object in "
                "the batches_info list."
            )
        result[key] = value

    return result


def _resolve_metadata_path(project_dir):
    metadata_override = os.environ.get("ORGANOID_METADATA_FILE")
    json_filename = metadata_override or "experiment_metadata.json"

    if os.path.isabs(json_filename):
        return json_filename

    project_json_path = os.path.join(base_dir, project_dir, json_filename)
    if os.path.exists(project_json_path) or not metadata_override:
        return project_json_path

    return os.path.join(os.path.dirname(__file__), json_filename)


def run_project_logic(project_dir):
    global batches_info, batches_complete_info

    json_path = _resolve_metadata_path(project_dir)

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No project data found for: {project_dir} at {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f, object_pairs_hook=_reject_duplicate_json_keys)

    batches_info = data.get("batches_info", [])
    batches_complete_info = data.get("batches_complete_info", [])

    print(f"📦 Project data loaded successfully from {json_path}.")
    
# Define the available projects
AVAILABLE_PROJECT_DIRS = [
    '2026_06_09-PCH2_CB',
    '2025_03_05-Organoid-Segmentation',
    '2025_03_05-Theresa-SmallScale-Trial/AB108-AB112',
    '2025_03_05-Theresa-SmallScale-Trial/SB60-SB64',
    '2025_03_05-Theresa-SmallScale-Trial/QB01-QB05',
    '6_WNT_treatment/AB115-AB117',
    'Francesco',
    'Isabel',
    'Zeynep',
    'Kenzie',
    'Dasha',
    '2025_12_24-Francesco',
    '6_WNT_treatment/WNT_ligands_treatment',
    '2026_04_30-Zeynep-CP',
    'AnnaRaic',
    '2026_06_08-Dasha-DMA',
    'Erin',
    '2026_06_17_Dasha',
    '2026_06_25_Dasha',
    '2026_06_25_Isabel_PB02_PB03'
]

# Shared config
checkpoint = 'Training-Results/work_dirs/split-0/iter_144000.pth'
base_dir = '../../../AGMayer/Lizia/Organoid_Segmentation'

pixel_to_micron = {
    #"96_well": {"x": 0.8882, "y": 0.8873}, #considering last calibration october/november 2025
    #"24_well": {"x": 2.8422*1.2, "y": 2.8394*1.2}  #1.97 correction to correspond to measure on day of change of plates
    #"24_well": {"x": 2.8422, "y": 2.8394},  #considering last calibration october/november 2025
    "5x_leica": {"x": 0.154, "y": 0.154},
    "4x": {"x": 0.8882, "y": 0.8873},
    "1.25x": {"x": 2.8422, "y": 2.8394},
    "96_well": {"x": 0.8882, "y": 0.8873},
    "24_well": {"x": 2.8422, "y": 2.8394},
}

features = ['Area_um2', 'Diameter_um', 'Perimeter_um', 'Circularity',
            'Skeleton_Length_um', 'Solidity', 'Eccentricity', 'Roughness_Index']

SEGMENTATION_RUN_KEYS = (
    "segmentation_runs",
    "segmentations",
    "imaging_runs",
    "acquisitions",
)


def normalize_tile_grid(tile_grid, default=(1, 1)):
    """Normalize tile-grid metadata such as '2x2', [2, 2], or {'rows': 2, 'cols': 2}."""
    if tile_grid is None:
        return default

    if isinstance(tile_grid, str):
        grid_text = tile_grid.lower().replace(" ", "").replace("by", "x")
        if "x" in grid_text:
            rows, cols = grid_text.split("x", 1)
            return int(rows), int(cols)

    if isinstance(tile_grid, dict):
        rows = tile_grid.get("rows", tile_grid.get("height", tile_grid.get("y")))
        cols = tile_grid.get("cols", tile_grid.get("columns", tile_grid.get("width", tile_grid.get("x"))))
        if rows is not None and cols is not None:
            return int(rows), int(cols)

    if isinstance(tile_grid, (list, tuple)) and len(tile_grid) == 2:
        return int(tile_grid[0]), int(tile_grid[1])

    if isinstance(tile_grid, int):
        return tile_grid, tile_grid

    return default


def normalize_organoid_mode(value, default="single"):
    """Return 'single' or 'multiple' from flexible organoid-per-image metadata."""
    if value is None:
        return default

    value_text = str(value).lower()
    if value_text in {"multiple", "multi", "many", "true", "yes"}:
        return "multiple"
    if value_text in {"single", "individual", "one", "false", "no"}:
        return "single"

    return default


def _get_explicit_segmentation_runs(batch):
    for key in SEGMENTATION_RUN_KEYS:
        if key in batch:
            runs = batch[key]
            return runs if isinstance(runs, list) else [runs]

    return []


def _normalize_segmentation_run(batch, run_config, default_name=None):
    method = run_config.get(
        "method",
        run_config.get("model", run_config.get("segmentation_model", "cl"))
    )
    method = str(method).lower()

    if method in {"deep_learning", "deeplearning", "neural", "nn"}:
        method = "dl"
    if method in {"classical", "classic", "threshold", "traditional"}:
        method = "cl"

    default_grid = (1, 1) if method == "dl" else (2, 2)
    tile_grid = normalize_tile_grid(
        run_config.get("tile_grid", run_config.get("tiles", run_config.get("grid_shape"))),
        default=default_grid,
    )

    magnification = str(run_config.get(
        "magnification",
        run_config.get("amplification", "4x" if method == "dl" else "1.25x")
    ))
    pixel_key = run_config.get(
        "pixel_to_micron_key",
        run_config.get("pixel_size_key", "96_well" if method == "dl" else "24_well")
    )

    organoid_mode = normalize_organoid_mode(
        run_config.get(
            "organoids_per_image",
            run_config.get("organoid_mode", run_config.get("multiple_organoids"))
        ),
        default="single",
    )
    default_center_fraction = 1.0 if organoid_mode == "multiple" else 0.95
    days = run_config.get("days", [])
    if days is None:
        days = []
    elif not isinstance(days, (list, tuple)):
        days = [days]

    images_per_plate = run_config.get(
        "images_per_plate",
        run_config.get(
            "wells_per_plate",
            run_config.get("stitched_images_per_batch", 24)
        )
    )

    return {
        "name": run_config.get("name", default_name or method),
        "method": method,
        "days": list(days),
        "tile_grid": tile_grid,
        "tiles_per_image": tile_grid[0] * tile_grid[1],
        "magnification": magnification,
        "pixel_to_micron_key": pixel_key,
        "pixel_to_micron": run_config.get("pixel_to_micron"),
        "num_plates": run_config.get("num_plates", batch.get("num_plates", 1)),
        "organoid_mode": organoid_mode,
        "checkpoint": run_config.get("checkpoint", checkpoint),
        "crop_days": run_config.get("crop_days", ["0"] if method == "dl" else []),
        "stitch": run_config.get("stitch", method == "cl" and tile_grid != (1, 1)),
        "tile_order": run_config.get("tile_order", "serpentine"),
        "overlap_x": run_config.get("overlap_x", 32),
        "overlap_y": run_config.get("overlap_y", 31),
        "stitched_images_per_batch": run_config.get("stitched_images_per_batch", images_per_plate),
        "images_per_plate": images_per_plate,
        "threshold": run_config.get("threshold", 128),
        "threshold_method": run_config.get("threshold_method", run_config.get("threshold_mode", "fixed")),
        "foreground_polarity": run_config.get("foreground_polarity", run_config.get("foreground", "dark")),
        "blur_kernel_size": run_config.get("blur_kernel_size", 0),
        "max_area": run_config.get("max_area", 3000000),
        "min_area": run_config.get("min_area", 30000),
        "min_circularity": run_config.get("min_circularity", 0.2),
        "center_fraction": run_config.get("center_fraction", default_center_fraction),
    }


def get_segmentation_runs(batch):
    """
    Return normalized segmentation runs for one batch.

    New metadata can use a ``segmentation_runs`` list. Legacy metadata with
    ``days_96`` and ``days_24`` is still supported.
    """
    explicit_runs = _get_explicit_segmentation_runs(batch)
    if explicit_runs:
        return [
            _normalize_segmentation_run(batch, run_config, default_name=f"run_{idx + 1}")
            for idx, run_config in enumerate(explicit_runs)
        ]

    runs = []
    if batch.get("days_96"):
        runs.append(_normalize_segmentation_run(batch, {
            "name": "days_96",
            "method": "dl",
            "days": batch["days_96"],
            "tile_grid": "1x1",
            "magnification": "4x",
            "pixel_to_micron_key": "96_well",
            "organoids_per_image": "single",
        }))

    if batch.get("days_24"):
        runs.append(_normalize_segmentation_run(batch, {
            "name": "days_24",
            "method": "cl",
            "days": batch["days_24"],
            "tile_grid": "2x2",
            "magnification": "1.25x",
            "pixel_to_micron_key": "24_well",
            "organoids_per_image": "single",
            "num_plates": batch.get("num_plates", 1),
        }))

    for legacy_key in ("multi_1.25x", "multi_125x", "days_multi_125x"):
        if batch.get(legacy_key):
            runs.append(_normalize_segmentation_run(batch, {
                "name": legacy_key,
                "method": "cl",
                "days": batch[legacy_key],
                "tile_grid": batch.get("tile_grid", "5x5"),
                "magnification": "1.25x",
                "pixel_to_micron_key": "24_well",
                "organoids_per_image": "multiple",
                "num_plates": batch.get("num_plates", 1),
                "tile_order": batch.get("tile_order", "serpentine"),
            }))

    return runs


def get_pixel_to_micron_for_run(run):
    """Resolve pixel-to-micron calibration for a normalized segmentation run."""
    if run.get("pixel_to_micron"):
        return run["pixel_to_micron"]

    pixel_key = run.get("pixel_to_micron_key")
    if pixel_key not in pixel_to_micron:
        raise KeyError(f"Unknown pixel_to_micron key: {pixel_key}")

    return pixel_to_micron[pixel_key]

# Prompt menu
print("🧠 Organoid Segmentation - Project Selector\n")

while True:
    action = questionary.select(
        "What would you like to do?",
        choices=[
            "Select a project directory",
            "List all project directories",
            "Exit"
        ]
    ).ask()

    if action == "List all project directories":
        print("\n📂 Available project directories:")
        for proj in AVAILABLE_PROJECT_DIRS:
            print(f"  - {proj}")
        print("")  # Blank line for readability
        # Instead of exiting, loop back to ask again

    elif action == "Select a project directory":
        project_dir = questionary.select(
            "Select a project directory:",
            choices=AVAILABLE_PROJECT_DIRS
        ).ask()

        if not project_dir:
            print("❌ No project selected. Try again.\n")
            continue  # Ask again
        print(f"\n✅ You selected: {project_dir}\n")
        run_project_logic(project_dir)
        break  # Exit the loop after successful selection

    elif action == "Exit":
        print("👋 Goodbye!")
        sys.exit(0)
