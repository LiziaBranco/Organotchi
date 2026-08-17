import questionary
import sys
import os
import json
import glob

# Default local clone location for OrgLine ("orgline" segmentation method) --
# see ../environments/README.md for how this clone + its venv are set up.
# (This used to default into ../benchmarking/environments/OrgLine, back when
# benchmarking/ lived in this same repo -- moved to ../environments/ when
# benchmarking/ and manuscript/ split into their own Organotchi-manuscript
# repo, since OrgLine support is a tool capability, not benchmark-specific.)
# A project doesn't need to repeat this path unless its clone actually lives
# somewhere else.
_DEFAULT_ORGLINE_REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "environments", "OrgLine"
))


def _reject_duplicate_json_keys(json_filename):
    """Return an object_pairs_hook that rejects duplicate keys a plain
    ``json.load`` would otherwise silently overwrite, naming the actual
    file being parsed in the error."""
    def hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    f"Duplicate key '{key}' found in {json_filename}. "
                    "If this is a batch entry, each batch must be its own object in "
                    "the batches_info list."
                )
            result[key] = value
        return result
    return hook


def _resolve_project_json_path(project_dir, default_filename, env_var):
    """Resolve one of the two per-project metadata files (segmentation or
    plotting), honoring an env-var override the same way for both."""
    override = os.environ.get(env_var)
    json_filename = override or default_filename

    if os.path.isabs(json_filename):
        return json_filename

    project_json_path = os.path.join(base_dir, project_dir, json_filename)
    if os.path.exists(project_json_path) or not override:
        return project_json_path

    return os.path.join(os.path.dirname(__file__), "templates", json_filename)


def _resolve_metadata_path(project_dir):
    return _resolve_project_json_path(project_dir, "segmentation_metadata.json", "ORGANOID_SEGMENTATION_METADATA_FILE")


def _resolve_plotting_metadata_path(project_dir):
    return _resolve_project_json_path(project_dir, "plotting_metadata.json", "ORGANOID_PLOTTING_METADATA_FILE")


def _plotting_metadata_suffix(path):
    """'plotting_metadata.json' -> None (default 'plots' output dir);
    'plotting_metadata_e10_vs_e11.json' -> 'e10_vs_e11' (output dir
    'plots_e10_vs_e11'). Falls back to the full stem for any filename that
    doesn't start with the expected prefix (shouldn't happen given the glob
    pattern discover_plotting_metadata_files uses, but avoids a crash)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem == "plotting_metadata":
        return None
    prefix = "plotting_metadata_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def discover_plotting_metadata_files(project_dir):
    """Every plotting_metadata*.json file for ``project_dir``, as a sorted
    list of (suffix, path) pairs -- one entry per file a project keeps for
    a distinct comparison (e.g. plotting_metadata_mtesr_vs_ctrl.json /
    plotting_metadata_e10_vs_e11.json for 2026_04_30-Zeynep-CP), run one
    after another by plot.py's driver loop, each into its own
    plots_<suffix>/ directory (see configure_plots_dirname in
    utils_plot.py) rather than overwriting a single shared plots/.

    Files with "copy" anywhere in the name (case-insensitive) are skipped
    -- these are ad hoc backup copies (e.g. "plotting_metadata - Copy.json"),
    not real alternate configs, mirroring the same convention already seen
    on segmentation_metadata backups in project directories.

    For a project with only the plain plotting_metadata.json (the common
    case), this returns exactly that one file with suffix=None -> output
    dir stays "plots", identical to the pre-existing single-file behavior.

    Honors ORGANOID_PLOTTING_METADATA_FILE the same way
    _resolve_plotting_metadata_path does: if set, only that one file is
    considered (an explicit single-file override takes precedence over
    multi-variant discovery).
    """
    override = os.environ.get("ORGANOID_PLOTTING_METADATA_FILE")
    if override:
        path = override if os.path.isabs(override) else os.path.join(base_dir, project_dir, override)
        return [(_plotting_metadata_suffix(path), path)]

    project_path = os.path.join(base_dir, project_dir)
    candidates = sorted(
        path for path in glob.glob(os.path.join(project_path, "plotting_metadata*.json"))
        if "copy" not in os.path.basename(path).lower()
    )
    if not candidates:
        # No file at all yet -- return the default path anyway so the
        # caller's "file not found" handling (a clear warning, not a crash)
        # still runs exactly as it did before this function existed.
        return [(None, os.path.join(project_path, "plotting_metadata.json"))]

    return [(_plotting_metadata_suffix(path), path) for path in candidates]


def load_plotting_metadata_file(path):
    """Load + normalize one plotting_metadata*.json file in isolation --
    for plot.py's multi-variant driver loop, which needs each
    discovered file's own config without going through
    run_project_logic's single-file prompt-time load. Does not touch the
    plotting_metadata/batches_complete_info globals set by
    run_project_logic."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No plotting metadata found at: {path}")

    with open(path, 'r') as f:
        data = json.load(f, object_pairs_hook=_reject_duplicate_json_keys(os.path.basename(path)))

    return _normalize_plotting_metadata(data)


# Defaults used for any key a project's plotting_metadata.json omits --
# mirrors _normalize_segmentation_run's per-key defaulting for the
# segmentation side, so a minimal plotting_metadata.json is still runnable.
DEFAULT_PLOTTING_METADATA = {
    "batches_complete_info": [],
    "features": [
        'Area_um2', 'Diameter_um', 'Perimeter_um',
        'Feret_Diameter_Max_um', 'Aspect_Ratio', 'Circularity',
        'Roundness', 'Solidity', 'Eccentricity', 'Skeleton_Length_um',
    ],
    "feature_labels": None,  # None -> derived from `features` verbatim
    "groups": ["batch"],
    "days_for_group_comparison": [],
    # Days to run plot_pca_grid_by_group and plot_umap_clusters on -- kept
    # separate from days_for_group_comparison (used for the boxplot/grid-
    # scatter group comparisons instead) since the two analyses serve
    # different purposes and a project may well want a different day
    # selection for each. Empty by default (no silent fallback to
    # days_for_group_comparison): set this explicitly per project, or PCA/
    # UMAP won't run.
    "pca_umap_days": [],
    "plot_aggregation_levels": ["object"],
    "outlier_detection": {
        "method": "iqr",
        "iqr_multiplier": 1.5,
        "std_multiplier": 3.0,
        "manual_std_thresholds": None,
        "aggregation_level": "object",
    },
    # Minimum number of independent wells required in EACH group before a
    # pairwise statistical comparison is attempted (Welch's t-test/Hedges'
    # g, everywhere in utils_plot.py that compares groups on
    # well-level means). Below this, the comparison is skipped and
    # reported as such (a "Skipped_Insufficient_N" row) rather than either
    # silently vanishing or running on a handful of wells that can't
    # support a meaningful test. 3 is a practical floor, not a statistical
    # guarantee of power -- raise it for projects that can afford to be
    # stricter, or lower it (a single well per group still lets Welch's
    # t-test run, since _hedges_g/ttest_ind only require n>=2) if a
    # project's design genuinely has no more replicates to give.
    "min_wells_for_stats": 3,
    # {"<day>": "<label>"} -- days where the imaging/plating regime itself
    # changed (e.g. 96-well/single-organoid -> 6-well/multiple-organoids-
    # per-well), annotated as a vertical marker on every longitudinal
    # line plot (plot_feature_over_time, plot_variability_and_outliers's
    # CV/outlier-percentage plots, plot_well_organoid_counts_over_time,
    # plot_well_metric_over_time, plot_feature_over_time_by_window,
    # analyze_growth_trajectories's trajectory plot) so a reader isn't left
    # to guess why a metric jumps/shifts at that day. Empty by default (no
    # markers) -- day keys are read as given (str or int; compared against
    # the Day column as int) and don't need to be contiguous or ordered.
    "transition_days": {},
    "well_metric_plots": [],
    # Error bar shown on every mean+/-error line plot in utils_plot.py
    # (plot_feature_over_time, plot_well_metric_over_time,
    # plot_feature_over_time_by_window, analyze_expansion_rate_by_window's
    # window-rate plot, plot_well_organoid_counts_over_time) -- see
    # _resolve_error_bar_values. One of "sem" (default, standard error of
    # the mean), "sd" (standard deviation), or "ci" (parametric 95%
    # confidence interval via the t-distribution).
    "line_plot_error_bar": "sem",
    # Minimum |Spearman r| above which plot_pca_grid_by_group's
    # _select_non_redundant_features drops a feature in favor of an
    # earlier, correlated one already selected -- see that function's
    # docstring in utils_plot.py.
    "pca_correlation_threshold": 0.9,
    # analyze_growth_trajectories runs once per entry here (only
    # Organoid_Mode == "single" rows -- i.e. one physical well imaged over
    # the whole timecourse -- contribute; see that function's docstring).
    # Area_um2 is a sensible default for every project; add more (e.g. a
    # light-dark ratio column) per project as needed.
    "growth_rate_features": [
        {"feature": "Area_um2", "metric_label": "Area (um^2)"},
    ],
    # List of developmental "windows" (each a list of days), e.g.
    # [[4,7,8,10], [12,14,16,17,18], [21,...,50]] -- used by
    # plot_feature_over_time_by_window/plot_feature_boxplots_by_window/
    # analyze_expansion_rate_by_window. Empty by default (opt-in): there's
    # no generic default that would make sense across projects with very
    # different developmental timelines.
    "developmental_windows": [],
    # Per-day/per-object metrics to run the window line+box plots on --
    # e.g. Light_Fraction/Dark_Fraction from light_dark_metrics.csv (requires
    # track_light_dark_split: true in segmentation_metadata.json), but any
    # mask_metrics.csv feature works too.
    "window_instant_metrics": [],
    # Features to run analyze_expansion_rate_by_window on (single-organoid
    # mode only -- see that function's docstring).
    "window_expansion_rate_features": [],
    # HDBSCAN's minimum cluster size for plot_umap_clusters -- see that
    # function's docstring for the full UMAP+clustering design (per-day
    # embedding on the same non-redundant feature set as the PCA grid,
    # cluster-vs-rest Hedges' g/Welch's-t to find distinguishing features).
    "umap_min_cluster_size": 5,
    # "eom" (HDBSCAN's own default) -- adopted as this pipeline's default
    # per explicit instruction. "leaf" (over-splitting into many small,
    # tighter clusters rather than eom's fewer, more stable ones) was tried
    # first and is still available per-project if "eom" ever collapses a
    # loosely-connected population too aggressively; see plot_umap_clusters'
    # docstring.
    "umap_cluster_selection_method": "eom",
    "umap_n_neighbors": 15,
    "umap_min_dist": 0.1,
    "umap_random_state": 42,
}


def _normalize_plotting_metadata(data):
    merged = {**DEFAULT_PLOTTING_METADATA, **data}
    merged["outlier_detection"] = {
        **DEFAULT_PLOTTING_METADATA["outlier_detection"],
        **data.get("outlier_detection", {}),
    }
    if not merged.get("feature_labels"):
        merged["feature_labels"] = list(merged["features"])
    # JSON object keys are always strings ("14"); normalize to int here
    # once so every plotting function can compare directly against the
    # (already-int) Day column instead of re-coercing this per call site.
    merged["transition_days"] = {
        int(day): label for day, label in (merged.get("transition_days") or {}).items()
    }
    return merged


def run_project_logic(project_dir):
    global batches_info, batches_complete_info, plotting_metadata

    json_path = _resolve_metadata_path(project_dir)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No project data found for: {project_dir} at {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f, object_pairs_hook=_reject_duplicate_json_keys(os.path.basename(json_path)))

    batches_info = data.get("batches_info", [])
    print(f"📦 Segmentation metadata loaded successfully from {json_path}.")

    plotting_json_path = _resolve_plotting_metadata_path(project_dir)
    if os.path.exists(plotting_json_path):
        with open(plotting_json_path, 'r') as f:
            plotting_data = json.load(f, object_pairs_hook=_reject_duplicate_json_keys(os.path.basename(plotting_json_path)))
        print(f"📦 Plotting metadata loaded successfully from {plotting_json_path}.")
    else:
        plotting_data = {}
        print(
            f"⚠️  No plotting_metadata.json found at {plotting_json_path}; "
            "plot.py will have nothing to group/plot until one is added "
            "(see plotting_metadata.json in data_analysis/ for a template)."
        )

    plotting_metadata = _normalize_plotting_metadata(plotting_data)
    batches_complete_info = plotting_metadata["batches_complete_info"]

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
    '2026_06_25_Isabel_PB02_PB03',
    '2026-07-21_Dasha',
    '2026_07_13_Isabel_QB24_QB25',
    '2026_07_13_Isabel_QB26',
    'Training-Dataset-Lancaster',
    '2026_04_30-Zeynep-CP-comparison'
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
    if method in {"org_line", "orgdet", "orgline_tiny"}:
        method = "orgline"

    default_grid = (1, 1) if method in {"dl", "orgline"} else (2, 2)
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

    # DL's min_area/min_circularity defaults were chosen by grid search against
    # split-0's held-out validation set (see the Organotchi-manuscript repo's
    # train_and_test/tune_classical_filter.py) -- the previous shared defaults (30000 um^2 /
    # 0.3) made aggregate filtered Dice/IoU on the test set *worse* than no
    # filtering at all, driven mostly by real, correctly-shaped early-
    # timepoint organoids whose area fell under the old area floor. The
    # classical pipeline's own contour filter keeps its previous defaults
    # unchanged -- this tuning was scoped to the DL post-filter only.
    default_min_area = 10000 if method == "dl" else 30000
    default_min_circularity = 0.1 if method == "dl" else 0.2
    # DL's post-filter upper area bound is off by default (organoids legitimately
    # grow larger over the developmental timeline; an unconditional upper bound
    # risked discarding genuine late-timepoint organoids) -- the classical
    # pipeline's own contour filter keeps its previous default unchanged.
    default_max_area = None if method == "dl" else 3000000

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
        "model_config_path": run_config.get("model_config_path"),
        "device": run_config.get("device", "cpu"),
        # DL only: whether to run the post-inference area/circularity filter
        # pass (apply_filtering_to_segmentations) at all.
        "filter_masks": bool(run_config.get("filter_masks", True)),
        "crop_days": run_config.get("crop_days", ["0"] if method == "dl" else []),
        "stitch": run_config.get("stitch", method in {"cl", "orgline"} and tile_grid != (1, 1)),
        "tile_order": run_config.get("tile_order", "serpentine"),
        "overlap_x": run_config.get("overlap_x", 32),
        "overlap_y": run_config.get("overlap_y", 31),
        "stitched_images_per_batch": run_config.get("stitched_images_per_batch", images_per_plate),
        "images_per_plate": images_per_plate,
        "threshold": run_config.get("threshold", 128),
        "foreground_polarity": run_config.get("foreground_polarity", run_config.get("foreground", "dark")),
        "max_area": run_config.get("max_area", default_max_area),
        "min_area": run_config.get("min_area", default_min_area),
        "min_circularity": run_config.get("min_circularity", default_min_circularity),
        # Classical only: excludes everything outside a circle concentric
        # with the image, of this fraction of its width -- width, not the
        # shorter dimension, since in these stitched multi-well images the
        # well is wider than it is tall and its physical diameter in pixels
        # is close to the image's width. Approximates the physical well
        # boundary well enough to drop a well-edge ring before it ever
        # reaches contour extraction, without detecting the well's actual
        # edge per image. None (default) disables this; set e.g. 0.9-0.98
        # for runs where the well boundary is visible in frame. See
        # utils_segmentation.py's
        # _apply_center_fraction_mask.
        "center_fraction": run_config.get("center_fraction"),
        # Classical only: bridges small gaps in an otherwise near-complete
        # object boundary (e.g. a cystic organoid rim that didn't fully
        # threshold all the way around) via morphological closing, before
        # min_area/max_area filtering -- so area/circularity filtering
        # judges the object's real, whole shape rather than an open
        # fragment of it. An int is a square kernel size in pixels (e.g.
        # 40); None/0 (default) disables this. Too large a kernel risks
        # bridging the gap *between* two separate, adjacent organoids --
        # keep it just big enough for the gaps actually seen (see
        # fit_experiment_data.py --sweep close_kernel_size). See
        # utils_segmentation.py's _apply_morphological_closing.
        "close_kernel_size": run_config.get("close_kernel_size"),
        # Enables per-organoid light-vs-dark area estimation
        # (extract_light_dark_metrics), from each mask's own grayscale
        # histogram -- see utils_segmentation.py's
        # estimate_light_dark_split_from_mask. Off by default; set true for
        # batches/runs where this split is of interest -- originally built
        # for cystic (translucent) vs solid tissue area, then generalized
        # once to "core"/"rim", but the same split is useful more generally
        # for tracking how an organoid's own dense (dark) region is
        # expanding/shrinking over a developmental timeline (with consistent
        # imaging settings), regardless of whether the lighter region
        # reflects true cyst biology for a given project -- see that
        # function's docstring. Accepts the old ``track_core_rim_split``/
        # ``has_cystic_organoids``/``cystic_organoids`` keys too, for
        # projects not yet migrated to the new name.
        "track_light_dark_split": bool(run_config.get(
            "track_light_dark_split", run_config.get(
                "track_core_rim_split", run_config.get(
                    "has_cystic_organoids", run_config.get("cystic_organoids", False)
                )
            )
        )),
        # Minimum eta-squared (fraction of pixel-intensity variance
        # explained by the Otsu light/dark split) for
        # estimate_light_dark_split_from_mask to consider that split real
        # rather than an artifact of forcing a two-class split onto a
        # still-unimodal histogram -- see that function's docstring. Only
        # used when track_light_dark_split is enabled. Accepts the old
        # ``core_rim_min_eta_squared``/``cystic_min_eta_squared`` keys too.
        "light_dark_min_eta_squared": run_config.get(
            "light_dark_min_eta_squared", run_config.get(
                "core_rim_min_eta_squared", run_config.get("cystic_min_eta_squared", 0.3)
            )
        ),
        # RGB color for the light class in generate_light_dark_overlay_masks's
        # rendered classification images (background is always black, dark
        # is always white). Magenta by default -- distinct from both.
        "light_dark_overlay_color": tuple(run_config.get(
            "light_dark_overlay_color", run_config.get(
                "core_rim_overlay_color", run_config.get("cystic_overlay_color", (255, 0, 255))
            )
        )),
        # DL post-filter morphology cleanup (erosion/dilation) parameters, previously
        # hardcoded inside utils_segmentation.py with no way to tune per run.
        # Disabled by default -- the same grid search found erosion/dilation
        # made the tuned area/circularity thresholds' results slightly worse,
        # not better (see tune_classical_filter.py's "disabled" profile
        # winning); kept overridable per-run for images that do need it.
        "erosion_kernel_size": tuple(run_config.get("erosion_kernel_size", (0, 0))),
        # If more than this many structures survive the size/circularity gates,
        # keep only the single most circular one; otherwise every survivor is
        # kept (e.g. just 1). See keep_most_circular_if_crowded.
        "max_objects_before_circularity_pick": run_config.get("max_objects_before_circularity_pick", 1),
        "dilation_initial_kernel_size": tuple(run_config.get("dilation_initial_kernel_size", (0, 0))),
        "dilation_iterations": run_config.get("dilation_iterations", 0),
        "dilation_reduction_factor": run_config.get("dilation_reduction_factor", 0.8),
        # OrgLine only: local clone of https://github.com/ucas-dx/OrgLine and
        # its two "tiny"/low-VRAM checkpoints (orgdet_tiny.pt + sam2_hiera_tiny.pt,
        # both under <orgline_repo>/checkpoints/ by convention -- see
        # ../environments/README.md for how they're obtained) --
        # the exact configuration benchmarked in the Organotchi-manuscript
        # repo's benchmarking/, where OrgLine was the strongest tool across
        # every tissue tried. Must be run with
        # the orgline_venv interpreter (see run_orgline_segmentation's
        # docstring) -- OrgLine's dependencies are isolated from mmsegmentation's.
        "orgline_repo": run_config.get("orgline_repo", _DEFAULT_ORGLINE_REPO),
        "orgdet_checkpoint": run_config.get("orgdet_checkpoint"),
        "sam2_checkpoint": run_config.get("sam2_checkpoint"),
        "orgline_conf": run_config.get("orgline_conf", 0.2),
        "orgline_iou": run_config.get("orgline_iou", 0.7),
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
