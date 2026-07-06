# Organoid Segmentation and Morphology Analysis

This folder contains the scripts used to run organoid brightfield segmentation
and downstream morphology analysis. The pipeline is configured per experiment:
raw images live in the experiment folder, while project paths and segmentation
settings are defined in `experiment_info.py` and `experiment_metadata.json`.

## 1. Configure Experiment Locations

Edit `experiment_info.py` before running a new project.

The most important settings are:

```python
base_dir = '../../../AGMayer/Lizia/Organoid_Segmentation'

AVAILABLE_PROJECT_DIRS = [
    '2026_04_30-Zeynep-CP',
    ...
]
```

`base_dir` should point to the parent folder where all experiment folders are
stored. Each entry in `AVAILABLE_PROJECT_DIRS` should be a project folder inside
`base_dir`. For example, with the settings above the pipeline expects:

```text
../../../AGMayer/Lizia/Organoid_Segmentation/2026_04_30-Zeynep-CP/
```

Also check the `pixel_to_micron` dictionary in `experiment_info.py`. Each
segmentation run in the metadata refers to one of these calibration keys, such
as `4x`, `1.25x`, `96_well`, or `24_well`.

## 2. Organize Raw Data

Each experiment folder should contain a `raw_data` directory with this structure:

```text
Project_name/
    experiment_metadata.json
    raw_data/
        Day <day>/
            <batch_or_condition_name>/
                Tile000001.jpg
                Tile000002.jpg
                Tile000003.jpg
                ...
```

Example:

```text
2026_04_30-Zeynep-CP/
    raw_data/
        Day 10/
            CP03_1/
                Tile000001.jpg
                Tile000002.jpg
            CP03_2/
                Tile000001.jpg
                Tile000002.jpg
```

The day folder names must match the day values in `experiment_metadata.json`,
for example `Day 0`, `Day 10`, or `Day 16.5`. Batch or condition folder names
must match the `batch` values in the metadata exactly.

Accepted image formats include `.jpg`, `.jpeg`, `.tif`, `.tiff`, and `.png`.
See `raw_data_folder_organization.txt` for the full raw-data naming guide.

## 3. Define Experiment Metadata

Each project should contain its own:

```text
Project_name/experiment_metadata.json
```

Use the `experiment_metadata.json` file in this repository as a template. The
metadata defines:

- batches or conditions to process
- days/time points
- segmentation method: `dl` for deep learning or `cl` for classical
- tile grid, for example `1x1`, `2x2`, or `5x5`
- magnification and pixel calibration
- whether images contain `single` or `multiple` organoids
- classical segmentation parameters such as threshold, area, circularity, and
  foreground polarity
- grouping information for morphology plots, such as condition, cell line, or
  replicate

Minimal classical run example:

```json
{
  "batch": "CP03_1",
  "num_plates": 1,
  "segmentation_runs": [
    {
      "name": "late_6well_classical_multiple",
      "method": "cl",
      "days": [10, 13, 15],
      "tile_grid": "5x5",
      "magnification": "1.25x",
      "images_per_plate": 6,
      "stitched_images_per_batch": 6,
      "organoids_per_image": "multiple",
      "pixel_to_micron_key": "1.25x",
      "threshold_method": "fixed",
      "threshold": 128,
      "foreground_polarity": "dark",
      "min_area": 30000,
      "max_area": 3000000,
      "min_circularity": 0.2
    }
  ]
}
```

## 4. Run Segmentation

From the `data-analysis` folder:

```bash
python3 segmentation.py
```

Select the project from the interactive menu. The script reads that project's
`experiment_metadata.json`, runs the configured segmentation methods, writes
masks into `processed_data`, and extracts morphology metrics.

Useful option:

```bash
python3 segmentation.py --resume
```

Use `--resume` to continue an interrupted segmentation run. It checks existing
outputs and processes only missing day/batch/run combinations.

If masks already exist but morphology metrics need to be regenerated, force
morphology recomputation:

```bash
ORGANOID_FORCE_MORPHOLOGY=1 python3 segmentation.py
```

## 5. Run Morphology Analysis and Plots

After segmentation and morphology extraction are complete, run:

```bash
python3 morphology.py
```

This script reads the generated metrics and creates morphology plots, outlier
reports, detection summaries, and organoid-count quality-control plots.

To change what is plotted, edit `morphology.py`. Common settings include:

- `features`
- `features_names`
- `groups`
- `plot_aggregation_levels`
- `outlier_method`
- outlier thresholds and variability settings

## 6. Expected Outputs

The pipeline writes results under:

```text
Project_name/
    processed_data/
        Day <day>/
            <batch>/
                inference_jpg/
                inference_npy/
                inference_tif/
                inference_labels_npy/
                inference_comparison/
                inference_metrics/
```

Important files include:

- `inference_jpg/`: binary mask images
- `inference_npy/`: binary masks as NumPy arrays
- `inference_tif/`: binary masks as TIFF images
- `inference_labels_npy/`: labeled object masks
- `inference_comparison/`: raw image and mask comparison previews
- `inference_metrics/mask_metrics.csv`: per-organoid morphology metrics
- `inference_metrics/well_metrics_summary.csv`: per-image or per-well summaries
- `plots/`: morphology plots and quality-control outputs

## Troubleshooting

- If no project appears in the menu, add it to `AVAILABLE_PROJECT_DIRS` in
  `experiment_info.py`.
- If metadata is not found, check that `experiment_metadata.json` is inside the
  selected project folder.
- If images are not found, check that the folder names are exactly
  `raw_data/Day <day>/<batch>/`.
- If masks are generated but plots look outdated, rerun segmentation with
  `ORGANOID_FORCE_MORPHOLOGY=1` before rerunning `morphology.py`.
