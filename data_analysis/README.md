# data_analysis

Organotchi's pipeline: segmentation of brightfield organoid images,
extraction of per-organoid/per-well morphology metrics from the resulting
masks, and plotting of those metrics across days, batches, and experimental
conditions. These three independent stages are treated independently in 
one script each (`segmentation.py`, `morphology.py`,`plot.py`) and are configured
per project through two JSON files (check templates/).

## How it works

1. **Configure.** `experiment_info.py` loads lab-wide settings
   (from `local_config.json`, incluidng available project directories,
   data root, default SegFormer checkpoint) and the pixel-to-micron calibration
   table. Each project folder then gets its own `segmentation_metadata.json`
   (what to segment and how) and `plotting_metadata.json` (what to plot and how to
   compare it) -- copy the fully-commented templates in `templates/` and
   edit them there. Every accepted key and value is documented in the
   templates themselves and in `experiment_info.py`'s normalization
   functions.
2. **Organize raw data** as `raw_data/Day <day>/<batch>/Tile*.jpg` under
   each project folder (see `templates/raw_data_folder_organization.txt`).
   Day and batch folder names must match `segmentation_metadata.json`
   exactly.
3. **Segment**: `python3 segmentation.py`. Reads the selected project's
   `segmentation_metadata.json`, runs the configured method(s) (`dl`
   SegFormer-based, `cl` classical threshold/contour, or `orgline`
   detector-guided SAM2), and writes masks under `processed_data/`.
4. **Extract morphology**: `python3 morphology.py`. Reads the masks
   `segmentation.py` produced and writes per-object/per-well metrics CSVs
   under each day/batch's `inference_metrics/`.
5. **Plot**: `python3 plot.py`. Reads those CSVs and `plotting_metadata.json`
   and writes figures/statistics under `plots/`.

All three scripts share an interactive project-selection prompt
(`experiment_info.py`) and the same three run modes:

```bash
python3 segmentation.py --resume   # default: continue from wherever the analysis stopped
python3 segmentation.py --force    # ignore existing outputs, recompute everything
python3 segmentation.py --check    # report what's done/missing/stale; runs nothing
```

`morphology.py` supports the same three modes. `segmentation.py` also has
`--correct_mask`, for interactive manual mask correction in
[napari](https://napari.org) -- see its own `--help`.

## Files in this folder

| File | Purpose |
|---|---|
| `experiment_info.py` | Loads `local_config.json`, the pixel-to-micron table, and normalizes each project's `segmentation_metadata.json`/`plotting_metadata.json` (defaults, aliases, accepted values). Every other script imports from this one. |
| `local_config.json` / `local_config.example.json` | Lab-specific settings (available project directories, data root, default checkpoint). `local_config.json` is gitignored; copy `local_config.example.json` to create your own. |
| `segmentation.py` | Entry point for segmentation. Selects a project, runs the configured `dl`/`cl`/`orgline` runs, writes masks. |
| `utils_segmentation.py` | Segmentation implementation: classical thresholding/contour filtering, SegFormer-based deep-learning inference, OrgLine integration, stitching, post-filtering. |
| `morphology.py` | Entry point for morphology extraction. Reads masks, writes metrics CSVs. |
| `utils_morphology.py` | Morphology-metrics implementation: per-object/per-well shape metrics, light/dark split estimation. |
| `plot.py` | Entry point for plotting. Reads metrics CSVs and `plotting_metadata.json`, generates figures and statistics. |
| `utils_plot.py` | Plotting/statistics implementation: all individual plot functions, outlier detection, group comparisons. |
| `orgline_bridge.py` | Shared model-loading code for the `orgline` method, working around upstream OrgLine bugs (see its module docstring). |
| `eval_metrics.py` | Pixel-level (Dice, IoU, accuracy) and instance-matching (AP50/AP75/F1) metrics for comparing a predicted mask against ground truth. Not used in this pipeline and is instead a library module for external users validating segmentation output against their own ground truth. |
| `rename_images_for_segmentation.py` | Standalone CLI to rename/normalize raw image filenames into the `Tile######` convention `segmentation.py` expects. |
| `__init__.py` | Makes `data_analysis` importable as a package while keeping every module's internal imports bare, so scripts still run directly (see its own docstring). |
| `templates/` | Fully-commented `segmentation_metadata.json`/`plotting_metadata.json` templates and the raw-data folder-naming convention. |

## Outputs

For each processed day and batch, under `processed_data/Day <day>/<batch>/`:

- `inference_jpg/`, `inference_npy/`, `inference_tif/`: binary masks
- `inference_labels_npy/`: labeled object masks
- `inference_comparison/`: raw-image/mask comparison previews
- `inference_metrics/`: `mask_metrics.csv`, `well_metrics_summary.csv`, and
  (for runs with `track_light_dark_split: true`) `light_dark_metrics.csv`
  plus `light_dark_overlay_jpg/`
- `segmentation_run_manifest.json`: resolved run parameters, for provenance

Project-level plots and quality-control outputs are written under `plots/`.

## Troubleshooting

- No project in the menu -> add it to `local_config.json`'s
  `available_project_dirs`.
- Metadata not found -> confirm `segmentation_metadata.json`/
  `plotting_metadata.json` are inside the selected project folder (or set
  `ORGANOID_SEGMENTATION_METADATA_FILE`/`ORGANOID_PLOTTING_METADATA_FILE`).
- Images not found -> confirm folder names are exactly
  `raw_data/Day <day>/<batch>/`.
- Plots look outdated -> `python3 morphology.py --check` to find stale
  metrics, then `--force` to regenerate before rerunning `plot.py`.
