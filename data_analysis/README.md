# Organoid Segmentation and Morphology Analysis

This folder contains the scripts used to run organoid brightfield segmentation
and downstream morphology analysis. The pipeline is configured per experiment,
across three files:

- `experiment_info.py`: shared, repo-wide settings (available projects, base
  directory, pixel-to-micron calibration table) -- edited directly, not JSON.
- `segmentation_metadata.json` (one copy per project folder): **what to
  segment and how** -- batches, days, method (classical/deep-learning),
  thresholds, area/circularity filters.
- `plotting_metadata.json` (one copy per project folder): **what to plot and
  how to compare it** -- which features/metrics, which grouping variables,
  which days to run statistics on, outlier-detection settings.

Splitting configuration this way means re-running `plot.py` with a
different grouping or a different day range never requires touching
segmentation settings (or vice versa). Template copies of both JSON files
live in this folder's `templates/` subfolder (`data_analysis/templates/segmentation_metadata.json`,
`data_analysis/templates/plotting_metadata.json`) -- copy either one into a project
folder and edit it there.

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

Also check the `pixel_to_micron` dictionary in `experiment_info.py` -- every
`pixel_to_micron_key` referenced from `segmentation_metadata.json` must be one
of its keys:

```python
pixel_to_micron = {
    "5x_leica": {"x": 0.154, "y": 0.154},
    "4x": {"x": 0.8882, "y": 0.8873},
    "1.25x": {"x": 2.8422, "y": 2.8394},
    "96_well": {"x": 0.8882, "y": 0.8873},
    "24_well": {"x": 2.8422, "y": 2.8394},
}
```

Add a new entry here (µm per pixel, X and Y separately for anisotropic
pixels) for any magnification/objective not already listed.

## 2. Organize Raw Data

Each experiment folder should contain a `raw_data` directory with this structure:

```text
Project_name/
    segmentation_metadata.json
    plotting_metadata.json
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

The day folder names must match the day values in `segmentation_metadata.json`,
for example `Day 0`, `Day 10`, or `Day 16.5`. Batch or condition folder names
must match the `batch` values in the metadata exactly.

Accepted image formats include `.jpg`, `.jpeg`, `.tif`, `.tiff`, and `.png`.
See `templates/raw_data_folder_organization.txt` for the full raw-data naming guide.

## 3. `segmentation_metadata.json` Reference

Use `data_analysis/templates/segmentation_metadata.json` as a starting template. Its
top-level shape:

```json
{
  "project_info": { "...": "purely informational, see below" },
  "batches_info": [
    {
      "batch": "BATCH_NAME",
      "num_plates": 2,
      "segmentation_runs": [ { "...": "one or more run configs, see below" } ]
    }
  ]
}
```

### Top-level keys

| Key | Required | Meaning |
|---|---|---|
| `project_info` | No | Free-form, **not read by any script** -- `project_name`/`description`/`raw_data_layout`/`processed_data_layout` are for humans reading the file, not configuration. Safe to omit or extend. |
| `batches_info` | Yes | List of batch/condition entries. Each `batch` value must exactly match a folder name under `raw_data/Day <day>/`. |

### Each `batches_info[]` entry

| Key | Default | Meaning |
|---|---|---|
| `batch` | — (required) | Folder name under `raw_data/Day <day>/<batch>/`. |
| `num_plates` | `1` | Default `num_plates` for any run below that doesn't set its own. |
| `segmentation_runs` | — (required) | List of run configs (see below). Aliases: `segmentations`, `imaging_runs`, `acquisitions`. |

A batch can define more than one run (e.g. an early deep-learning run and a
later classical run for the same batch, as in the template) -- each becomes
an independent entry in `Run`/manifest/output naming.

**Legacy shorthand** (still supported, but `segmentation_runs` is
recommended for new projects since every parameter is explicit): a batch
with `days_96` is expanded into a single-organoid 96-well DL run;
`days_24` into a single-organoid 24-well classical run; `multi_1.25x` /
`multi_125x` / `days_multi_125x` into a multiple-organoid 1.25x classical
run.

### Each `segmentation_runs[]` entry ("run config")

Applies-to: **both** = used by every method. **cl**/**dl**/**orgline** = that method only.

| Key | Applies to | Default | Meaning |
|---|---|---|---|
| `name` | both | `"run_<N>"` (auto) | Identifies this run in filenames, manifests, and the `Run` column of CSVs. |
| `method` | both | `"cl"` | `"dl"` (deep learning), `"cl"` (classical), or `"orgline"` (OrgLine -- detector-guided SAM2 prompting; see below). Aliases: `model`, `segmentation_model`. Values `deep_learning`/`neural`/`nn` normalize to `dl`; `classical`/`threshold`/`traditional` normalize to `cl`; `org_line`/`orgdet`/`orgline_tiny` normalize to `orgline`. |
| `days` | both | `[]` | Day numbers this run processes; must match `raw_data/Day <day>/` folders (floats allowed, e.g. `16.5`). |
| `organoids_per_image` | both | `"single"` | `"single"` or `"multiple"`. Aliases: `organoid_mode`, `multiple_organoids`. Accepted values: `multiple`/`multi`/`many`/`true`/`yes` -> multiple; `single`/`individual`/`one`/`false`/`no` -> single. For orgline, `"single"` keeps only the single highest-confidence detection per image; `"multiple"` keeps every detection surviving `orgline_conf`/`orgline_iou`. |
| `tile_grid` | both | `1x1` (dl, orgline) / `2x2` (cl) | Stitching grid, e.g. `"5x5"`, `[5, 5]`, `{"rows": 5, "cols": 5}`, or a single int for a square grid. Aliases: `tiles`, `grid_shape`. |
| `magnification` | both | `"4x"` (dl) / `"1.25x"` (cl) | Informational, and picks the default `pixel_to_micron_key`. Alias: `amplification`. |
| `pixel_to_micron_key` | both | `"96_well"` (dl) / `"24_well"` (cl) | Key into `experiment_info.py`'s `pixel_to_micron` dict. Alias: `pixel_size_key`. |
| `pixel_to_micron` | both | `null` | Explicit `{"x": ..., "y": ...}` override; takes precedence over `pixel_to_micron_key` when given. |
| `num_plates` | both | the batch's own `num_plates`, else `1` | Number of physical plates this run covers. |
| `images_per_plate` | both | `24` | Wells/tiles per plate; drives `Plate_Index`/`Well_Index`/`Well_ID` derivation. Aliases: `wells_per_plate`, falls back to `stitched_images_per_batch`. |
| `stitched_images_per_batch` | cl, orgline | = `images_per_plate` | Expected number of stitched images per batch. |
| `stitch` | cl, orgline | `true` if `tile_grid` != `1x1` (cl, orgline) | Whether to stitch raw tiles before segmentation. orgline reuses the same `prepare_classical_segmentation_images` stitching step as cl. |
| `tile_order` | cl, orgline | `"serpentine"` | Raster order tiles were acquired in (also accepts e.g. `"column_snake"`). |
| `overlap_x` / `overlap_y` | cl, orgline | `32` / `31` | Pixel overlap between adjacent tiles when stitching. |
| `threshold` | cl only | `128` | Global grayscale cutoff separating organoid from background (0-255). Set to the literal string `"otsu"` instead of a number to resolve it per image (`skimage.filters.threshold_otsu`) rather than use a fixed value -- see the note below. |
| `foreground_polarity` | both | `"dark"` | `"dark"` (organoid darker than background) or `"bright"`. Alias: `foreground`. |
| `min_area` | both | `30000` (cl) / `10000` (dl) | Minimum object area, in µm². For dl, applied *before* erosion (see below), so a legitimately-sized object is judged on its actual size rather than its post-erosion size. The dl default was chosen by grid search against split-0's validation set (see the `Organotchi-manuscript` repo's `train_and_test/tune_classical_filter.py`) -- `30000` made real, correctly-shaped early-timepoint (Day 2) organoids fail this gate outright. |
| `max_area` | both | `3000000` (cl) / disabled (dl) | Maximum object area, in µm². For dl this is **optional and off by default** -- organoids legitimately grow larger over the developmental timeline, so an unconditional upper bound risked discarding genuine late-timepoint organoids; set it explicitly to re-enable a ceiling. Unchanged for cl. |
| `min_circularity` | both | `0.2` (cl) / `0.1` (dl) | Minimum `4·pi·area / perimeter²` (1.0 = perfect circle). The dl default (also grid-searched) is deliberately permissive -- real organoids in this dataset already score 0.85-0.95, so `0.1` is really only there to reject clearly non-organoid shapes (e.g. thin well-edge artifacts, circularity ~0.1-0.2), not to police roundness among genuine organoids. |
| `center_fraction` | cl only | disabled (`None`) | Excludes everything outside a circle concentric with the image, of this fraction of its **width** (e.g. `0.9` keeps the central 90% of the width) -- applied right after thresholding, before `min_area`/`max_area`. Width, not the shorter dimension, since in these stitched multi-well images the well is wider than it is tall and its physical diameter in pixels is close to the image's width. Approximates the physical well boundary well enough to drop a well-edge ring before it ever reaches contour extraction, without detecting the well's actual edge per image. See the note below. |
| `close_kernel_size` | cl only | disabled (`None`) | Morphological closing (dilate then erode) applied right after `center_fraction`, before `min_area`/`max_area`. An int is a square kernel size in pixels (e.g. `40`). Bridges small gaps in an otherwise near-complete object boundary -- e.g. a cystic organoid rim that didn't fully threshold all the way around -- into one closed loop, before it's judged by area/circularity or drawn into the final mask. See the note below. |
| `max_objects_before_circularity_pick` | dl, orgline | `1` | If more than this many structures survive the size/circularity gates, keep only the single most circular one; otherwise every survivor is kept. Set to `1` (always narrow to a single winner once there's more than one candidate) after finding that allowing exactly 2 survivors could keep a real organoid *and* a spurious artifact together, each individually passing the gates. **For an orgline run with `organoids_per_image: "multiple"`, the default of `1` will silently collapse every image down to one kept structure once `filter_masks` runs** -- set this explicitly higher (or `filter_masks: false`) for genuinely multi-organoid orgline runs. |
| `track_light_dark_split` | both | `false` | Enables `extract_light_dark_metrics`, writing `inference_metrics/light_dark_metrics.csv`, plus `generate_light_dark_overlay_masks`'s QC overlay images (`light_dark_overlay_jpg/`) (see below). |
| `foreground_polarity` (light/dark split) | both | `"dark"` | Reused from the segmentation-side `foreground_polarity` above to decide which side of each organoid's own internal split counts as "light" vs "dark". |
| `light_dark_min_eta_squared` | both | `0.3` | Minimum eta-squared (bimodal-split effect size) below which an organoid's light/dark split is flagged as not yet separable (`Has_Bimodal_Split: false`) rather than a real result close to zero. |
| `light_dark_overlay_color` | both | `(255, 0, 255)` (magenta) | RGB color used for the "light" region in `light_dark_overlay_jpg/` QC images. |
| `checkpoint` | dl only | module-level `checkpoint` in `experiment_info.py` | Path to the `.pth` checkpoint file. |
| `model_config_path` | dl only | auto (the one `.py` file beside the checkpoint) | Explicit mmseg config path override. |
| `device` | dl, orgline | `"cpu"` | e.g. `"cuda:0"` if a GPU is available. |
| `crop_days` | dl only | `["0"]` | Days whose images get center-cropped before inference. |
| `filter_masks` | dl, orgline | `true` | Whether to run the post-inference area/circularity filter pass (`apply_filtering_to_segmentations`) at all. For orgline, this is the same post-filter dl uses, applied to OrgLine's own output -- see the `max_objects_before_circularity_pick` caveat above before enabling it for a "multiple" orgline run. |
| `erosion_kernel_size` | dl, orgline | `[0, 0]` (disabled) | Post-filter mask cleanup (erosion kernel), shared by dl and orgline. Disabled by default -- the same grid search found erosion/dilation made the tuned area/circularity thresholds' results slightly worse, not better; set e.g. `[10, 10]` to re-enable if a specific run needs it. |
| `dilation_initial_kernel_size` | dl, orgline | `[0, 0]` (disabled) | Post-filter mask cleanup (initial dilation kernel), shared by dl and orgline. |
| `dilation_iterations` | dl, orgline | `0` (disabled) | Post-filter mask cleanup (number of dilation passes), shared by dl and orgline. |
| `dilation_reduction_factor` | dl, orgline | `0.8` | Post-filter mask cleanup (kernel shrink factor per pass), shared by dl and orgline. Only takes effect if `dilation_iterations` > 0. |
| `orgline_repo` | orgline only | local clone under `../environments/OrgLine` | Path to a local clone of https://github.com/ucas-dx/OrgLine. Only set this if your clone lives somewhere else. |
| `orgdet_checkpoint` | orgline only | `<orgline_repo>/checkpoints/orgdet_tiny.pt` | Detector checkpoint path. |
| `sam2_checkpoint` | orgline only | `<orgline_repo>/checkpoints/sam2_hiera_tiny.pt` | SAM2 mask-predictor checkpoint path. |
| `orgline_conf` | orgline only | `0.2` | Detection confidence threshold, passed straight to OrgLine's detector -- the same value used in the `Organotchi-manuscript` benchmark. |
| `orgline_iou` | orgline only | `0.7` | Detection NMS IoU threshold, passed straight to OrgLine's detector -- the same value used in the `Organotchi-manuscript` benchmark. |

**DL post-filter order** (`apply_filtering_to_segmentations`, all in
`utils_segmentation.py`): remove objects failing the `min_area`/`max_area`
gate (`remove_small_objects_by_area`, run *before* erosion) -> erosion
(`apply_erosion`) -> remove objects below `min_circularity`
(`remove_low_circularity_objects`) -> if more than
`max_objects_before_circularity_pick` structures still remain, keep only the
most circular one, otherwise keep every survivor
(`keep_most_circular_if_crowded`) -> dilation (`apply_dilation`) ->
hole-filling (`fill_black_holes`). Each intermediate step keeps every
qualifying structure, not just one -- only the last step ever narrows down
to a single winner, and only when crowded.

The five threshold-like parameters above (`min_area`, `min_circularity`,
`erosion_kernel_size`/`dilation_*` as four paired strength profiles) were
last chosen by a grid search against split-0's held-out validation set --
see the `Organotchi-manuscript` repo's `train_and_test/tune_classical_filter.py`,
which also reports the chosen combination's honest performance on the
(untouched-during-search) test set. Re-run it if the checkpoint or dataset
changes enough that these defaults might no longer be optimal.

**Classical segmentation is intentionally minimal for the time being.** The
only filters applied are `threshold`, `min_area`, `max_area`, and
`min_circularity` (plus `foreground_polarity`, which isn't a filter but says
which side of `threshold` the organoid is on). Pipeline order:
threshold -> `center_fraction` mask -> `close_kernel_size` closing ->
`min_area` filter -> `max_area` filter -> contour extraction -> circularity
filter -> area filter -> final accepted objects (single-organoid mode
additionally keeps only the largest survivor at that last step -- this is
the classical pipeline's own separate keep-largest logic, unrelated to the
dl pipeline's `keep_most_circular_if_crowded` above). Halo growth
(cystic-organoid halo recovery), touching-object splitting, and the
`"adaptive"`/`"combined"` threshold methods that existed in earlier
iterations of this pipeline are still removed for now, with three exceptions:

- `threshold` also accepts the literal string `"otsu"`, resolved fresh per
  image (`_resolve_threshold_value` in `utils_segmentation.py`) rather than
  a single fixed cutoff shared across an entire batch/timecourse. Brought
  back because a fixed threshold is a real liability whenever illumination
  genuinely drifts day to day -- confirmed on AnnaRaic, where a fixed
  threshold detected organoids on Day 0 but missed them entirely on Day 1.
  `fit_experiment_data.py --diagnose` prints the resolved per-image value
  when this is set, and `--sweep threshold` includes it as one of the
  candidates alongside fixed values, so it's easy to compare against a
  fixed cutoff before committing to it for a full run.
- `center_fraction` (`_apply_center_fraction_mask`) excludes everything
  outside a circle concentric with the image, of this fraction of its
  **width** -- width, not the shorter dimension, since in these stitched
  multi-well images the well is wider than it is tall and its physical
  diameter in pixels is close to the image's width; using width means
  values close to 1.0 track the well's actual edge instead of leaving a
  wide unaccounted margin on the long axis. Brought back because a visible
  well boundary is a real liability for multi-organoid classical runs --
  its own contour traces along the image edge (however large its enclosed
  area), so without this it's left to survive contour extraction and get
  judged (and sometimes wrongly accepted) by circularity/area alone, same
  as any other candidate object. Confirmed valuable on Zeynep-CP's
  late-timepoint multi-organoid images: a fixed circular crop dropped a
  well-ring contour that had otherwise been sitting among the real organoid
  detections. `fit_experiment_data.py --sweep center_fraction` compares a
  few crop strengths (including `None`, disabled) side by side.
- `close_kernel_size` (`_apply_morphological_closing`) runs a dilate-then-
  erode morphological closing pass right after `center_fraction`, before
  `min_area`/`max_area`, to bridge small gaps in an otherwise near-complete
  object boundary. Motivating case: a cystic organoid's rim sometimes
  doesn't fully threshold all the way around (uneven staining/focus across
  part of it), leaving an open C-shape rather than a closed loop -- an open
  contour's filled draw only fills the thin sliver it traces, not the disk
  it would enclose if closed, so the object is either dropped outright (an
  open C-shape scores very low on circularity) or saved as a fragment of its
  true extent. No separate hole-filling step is needed on top of this:
  closing the gap into one loop means the existing filled-contour draw at
  final selection already recovers the whole object once area/circularity
  filtering judges its real, closed shape rather than a fragment of it.
  Confirmed directly against a synthetic gapped ring: closing it, then
  running the result through contour extraction and a filled draw, fully
  recovers the disk (center pixel included), while skipping closing leaves
  either nothing (fails circularity) or just the thin crescent. Keep the
  kernel just big enough for the gaps actually seen: a kernel that bridges a
  gap in one organoid's own rim can just as easily bridge the *spacing*
  between two separate, adjacent organoids, merging them into one.
  `fit_experiment_data.py --sweep close_kernel_size` compares a few kernel
  sizes (including `None`, disabled) side by side.

Use `python3 fit_experiment_data.py --diagnose` to inspect
each remaining stage step by step if organoids are being lost unexpectedly
(see that script's own docstring for usage).

`min_area`/`min_circularity` are honored by both the classical and
deep-learning pipelines (the DL post-filter used to only threshold on
circularity and a hardcoded 150 µm diameter floor). `max_area` is honored by
both too, but is off by default for dl specifically (see above).

### `track_light_dark_split` in more detail

Estimates a lighter ("light") vs darker ("dark") area split *within* each
individual organoid's own mask, from that organoid's own grayscale pixel
histogram -- this is **not** a segmentation parameter, it doesn't affect the
generated masks at all, it just adds `inference_metrics/light_dark_metrics.csv`
(one row per organoid) plus `light_dark_overlay_jpg/` QC images. See
`utils_morphology.py`'s `estimate_light_dark_split_from_mask`: Otsu's
threshold, restricted to just that organoid's own masked pixels, splits them
into a lighter ("light") and darker ("dark") class -- an optical brightness
split between an organoid's periphery and its denser center, not necessarily
true cyst/lumen formation (this metric was originally named "cystic", then
"core"/"rim", before being generalized to "light"/"dark" -- see that
function's own docstring for the full history). Before a visible light
region forms, this is expected to be a near-arbitrary split of an
effectively unimodal histogram (`Has_Bimodal_Split: false`, per
`light_dark_min_eta_squared`); as it develops, a real second, lighter peak
should pull `Light_Fraction`/`Light_Dark_Ratio` up over the developmental
timeline. Projects with genuine cystic/translucent structures can still read
these same numbers as cystic-vs-solid area.

### Minimal classical run example

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
      "threshold": 128,
      "foreground_polarity": "dark",
      "min_area": 30000,
      "max_area": 3000000,
      "min_circularity": 0.2
    }
  ]
}
```

### Minimal orgline run example

The same tiny-checkpoint OrgLine configuration (`orgdet_tiny.pt` +
`sam2_hiera_tiny.pt`, confidence 0.20, IoU 0.70) benchmarked in the
`Organotchi-manuscript` repo's `benchmarking/`, where it was the strongest
tool across every tissue tried:

```json
{
  "batch": "CP03_1",
  "num_plates": 1,
  "segmentation_runs": [
    {
      "name": "early_96well_orgline",
      "method": "orgline",
      "days": [1, 2, 4, 7, 10, 14],
      "tile_grid": "1x1",
      "magnification": "4x",
      "images_per_plate": 96,
      "organoids_per_image": "single",
      "pixel_to_micron_key": "4x"
    }
  ]
}
```

`orgline_repo`/`orgdet_checkpoint`/`sam2_checkpoint` default to this tool's
own local clone and checkpoints under `../environments/OrgLine` (see the
table above) -- add them explicitly only if your clone lives elsewhere. No need to
invoke OrgLine's own venv interpreter yourself -- `segmentation.py`
auto-detects and switches to it when needed (see "Run Segmentation" below).

## 4. Run Segmentation

From the `data_analysis` folder:

```bash
python3 segmentation.py
```

Select the project from the interactive menu. The script reads that project's
`segmentation_metadata.json`, runs the configured segmentation methods, and
writes masks into `processed_data`. Run `morphology.py` afterward to extract
metrics from those masks (see the next section).

**`orgline`-method runs need OrgLine's own environment, and this is handled
automatically.** `dl` and `cl` runs both work from the main environment
(`mmsegmentation`/`mmengine` for `dl`, plain OpenCV/NumPy for `cl`);
`orgline` runs need their own isolated torch/torchvision pins plus a
vendored detector and SAM2 (see `../environments/README.md`).
You never need to invoke `orgline_venv`'s interpreter yourself -- launch
`segmentation.py` normally (`python3 segmentation.py`, `--resume`/`--force`
only; `--check` never touches any model, so this never applies to it), and
before any actual segmentation runs, it checks whether OrgLine-method work is
pending and, if so:

- **OrgLine-only pending work**: the process re-executes itself into
  `orgline_venv` outright (`os.execv` -- this fully replaces the process;
  nothing of the environment it started under lingers) and continues
  normally from there.
- **OrgLine pending alongside `dl`/`cl` work**: the OrgLine portion is
  delegated to a child process running under `orgline_venv`
  (`--only-methods orgline`, an internal flag); once that child exits (its
  venv "deactivated" simply by the process ending), this process continues
  handling the `dl`/`cl` portion itself, under whichever environment it was
  originally launched from.
- **Already running under `orgline_venv`, but `dl`/`cl` work is *also*
  pending**: reported rather than attempted -- unlike `orgline_venv`
  (locatable from each run's own `orgline_repo` config), nothing in this
  project's config points back to "the native environment", so re-run from
  there for that remaining portion.

`--only-methods dl,cl` (or any comma-separated method list) is also a
regular flag if you want to split the work across invocations by hand
instead of relying on the automatic switch.

`segmentation.py` runs in one of three mutually exclusive modes, chosen with a
flag (they work identically across every segmentation method):

```bash
python3 segmentation.py             # --resume is the default
python3 segmentation.py --resume    # continue from wherever the analysis stopped
python3 segmentation.py --force     # ignore all existing outputs, recompute everything
python3 segmentation.py --check     # report status only; runs nothing
```

- `--resume` (default): skips any day/batch/run whose segmentation outputs
  already exist. Safe to re-run after an interrupted or partial run.
- `--force`: ignores every existing output and recomputes segmentation and
  filtering from scratch for every configured day/batch/run. Use this after
  changing thresholds/calibration/parameters that wouldn't otherwise be
  detected as "stale".
- `--check`: prints, per day/batch/run, whether segmentation outputs are
  complete (with the reason), plus a summary count. Does not run anything —
  use it to see what's missing before committing to a full run.

## 5. Run Morphology Extraction

Once masks exist for the days/batches/runs you care about, extract
per-object/per-well morphology metrics from them:

```bash
python3 morphology.py
```

Select the project from the same interactive menu `segmentation.py` uses.
The script reads whichever masks `segmentation.py` has already produced
(`inference_filtered_npy`, falling back to `inference_npy`) and writes the
metrics CSVs into each day/batch's `inference_metrics/` folder
(`mask_metrics.csv`, `well_metrics_summary.csv`, and -- for runs with
`track_light_dark_split: true` -- `light_dark_metrics.csv` plus
`light_dark_overlay_jpg/` QC images). It never loads a segmentation model
(no `torch`/`mmsegmentation`/OrgLine), so it doesn't need OrgLine's own
environment even for masks OrgLine produced.

Split out of `segmentation.py` on purpose: segmentation (producing masks)
and morphology extraction (measuring them) are independent stages that
don't need to run back-to-back in the same process, and a metrics-only code
change (e.g. a fixed morphology formula) now only means re-running this
script, not touching `segmentation.py` at all.

`morphology.py` supports the same three mutually exclusive modes as
`segmentation.py`:

```bash
python3 morphology.py             # --resume is the default
python3 morphology.py --resume    # extract metrics wherever masks exist and metrics are missing/stale
python3 morphology.py --force     # ignore staleness checks, recompute every configured run's metrics from scratch
python3 morphology.py --check     # report status only; extracts nothing
```

- `--resume` (default): skips any (day, batch) combination whose metrics are
  already up to date (mtime-compared against its source masks).
- `--force`: recomputes every configured day/batch/run's metrics from
  scratch, regardless of whether they already look up to date. Use this
  after a metrics-only code change so the CSVs get regenerated even though
  their own mtimes wouldn't otherwise look stale.
- `--check`: prints, per day/batch/run, whether morphology metrics are up to
  date (with the reason), plus a summary count. Does not extract anything.

`--only-methods dl,cl` (or any comma-separated method list, matching
`segmentation_metadata.json`'s own run `method` values) restricts extraction
to a subset of configured runs for this invocation.

## 6. `plotting_metadata.json` Reference

Use `data_analysis/templates/plotting_metadata.json` as a starting template. Unlike
`segmentation_metadata.json`, every key here has a working default (see
`experiment_info.py`'s `DEFAULT_PLOTTING_METADATA`) — a project can start
with just `batches_complete_info` filled in and everything else omitted.
If a project has no `plotting_metadata.json` at all, `plot.py` still
runs, but with nothing to group/plot (`batches_complete_info` defaults to
empty).

| Key | Default | Meaning |
|---|---|---|
| `batches_complete_info` | `[]` | One entry per batch (see below) — grouping/labeling info for plots. Independent of `segmentation_metadata.json`'s `batches_info`: this is about how to *compare* results, not how to *produce* them. |
| `features` | `["Area_um2", "Diameter_um", "Perimeter_um", "Feret_Diameter_Max_um", "Aspect_Ratio", "Circularity", "Roundness", "Solidity", "Eccentricity", "Skeleton_Length_um"]` | Which `mask_metrics.csv` columns to plot with `plot_feature_over_time`/`plot_feature_grid_scatters`/`plot_variability_and_outliers`. |
| `feature_labels` | = `features` verbatim | Human-readable axis labels, same order and length as `features`. |
| `groups` | `["batch"]` | Which `batches_complete_info` field(s) to group/compare by. A full set of plots is generated once per entry in this list (e.g. once grouped by `"condition"`, once by `"cell-line"`). |
| `days_for_group_comparison` | `[]` | Explicit day subset for the per-day panel plots (`plot_feature_grid_scatters`, `plot_well_metric_boxplots_by_group`) — pairwise statistical testing across groups runs on these days specifically, not every day in `batches_complete_info`. |
| `plot_aggregation_levels` | `["object"]` | Passed to `plot_feature_over_time`: `"object"` (one point per organoid) and/or `"well"` (one point per image, averaging its objects). |
| `outlier_detection` | see below | Settings for `plot_variability_and_outliers`. |
| `well_metric_plots` | `[]` | List of per-well metrics to plot (see below). |

### Each `batches_complete_info[]` entry

| Key | Required | Meaning |
|---|---|---|
| `batch` | Yes | Must match a `batch` value in `segmentation_metadata.json`'s `batches_info`. |
| `days` | Yes | Days this batch has data for — drives the x-axis of line plots. |
| *(anything else)* | No | Arbitrary grouping fields referenced by `groups`, e.g. `condition`, `cell-line`, `treatment`, `replicate`, `batch_no_plate`. Add whatever fields your `groups` list needs. |

### `outlier_detection`

| Key | Default | Meaning |
|---|---|---|
| `method` | `"iqr"` | `"iqr"` (Tukey fences) or `"mean_sd"` (mean ± N standard deviations). `"percentile"` is also accepted by the underlying function (see below). |
| `iqr_multiplier` | `1.5` | Whisker multiplier when `method` is `"iqr"`. |
| `std_multiplier` | `3.0` | Default N when `method` is `"mean_sd"`. |
| `manual_std_thresholds` | `null` | Optional override for `std_multiplier`: either a single number (applies everywhere), or a dict keyed by feature name, optionally nested one level deeper by day (as a string) with a `"default"` fallback, e.g. `{"Area_um2": {"10": 2.5, "default": 3.0}}`. See `_resolve_std_multiplier` in `utils_plot.py` — its lookup also supports tuple keys like `(feature, group, day)`, but those aren't representable in JSON, so only the feature/day-keyed forms are reachable from `plotting_metadata.json`. |
| `aggregation_level` | `"object"` | `"object"` or `"well"` — aggregation level outlier detection itself runs at (independent of `plot_aggregation_levels` above). |

`plot_variability_and_outliers` (in `utils_plot.py`) also accepts
`percentile_bounds` (default `(0.05, 0.95)`, used when `method` is
`"percentile"`/`"quantile"`) and `mad_threshold` (median-absolute-deviation
cutoff, used when `method` is `"mad"`) — not currently exposed as
`plotting_metadata.json` keys (they weren't configurable outside the
function's own defaults before this file existed either); pass them
directly in `plot.py` if you need them.

### Each `well_metric_plots[]` entry

Each entry drives one call to both `plot_well_metric_over_time` (line plot
over time) and `plot_well_metric_boxplots_by_group` (per-day box plots),
for every value in `groups` — both with pairwise Welch's t-tests and
Benjamini-Hochberg FDR correction across groups.

| Key | Required | Meaning |
|---|---|---|
| `metric_column` | Yes | Column name to read from `csv_filename`. |
| `metric_label` | Yes | Human-readable axis label. |
| `csv_filename` | Yes | CSV filename under each day/batch's `inference_metrics/` folder. Currently meaningful values: `well_metrics_summary.csv`'s `Total_Segmented_Area_um2` (or any other `well_metrics_summary.csv` column, e.g. `Area_um2_Mean`), and — for batches with `track_light_dark_split: true` — `light_dark_metrics.csv`'s `Light_Dark_Ratio`, `Light_Fraction`, or `Dark_Fraction`. |

The template ships with both the total-segmented-area and light-dark-ratio
entries pre-configured; delete either one if it doesn't apply to your
project (e.g. no batch has `track_light_dark_split: true`).

## 7. Run Plotting

After segmentation and morphology extraction are complete, run:

```bash
python3 plot.py
```

This script reads `plotting_metadata.json` (via `experiment_info.py`'s
project-selection prompt) and the metrics CSVs `morphology.py` already
extracted, then creates morphology plots, outlier reports, detection
summaries, and organoid-count/well-metric quality-control plots for every
grouping in `groups`.

To change what is plotted, edit the project's `plotting_metadata.json` —
`plot.py` itself no longer hardcodes features, groupings, days, or
outlier settings (see the reference above). `plot.py` is still the
place to edit if you want to add/remove *which plotting functions run at
all*, e.g. uncommenting the `plot_pca_grid_by_group`/`create_detection_analysis`
calls already present but commented out in the script.

## 8. Expected Outputs

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
    plots/
```

Important files include:

- `inference_jpg/`: binary mask images
- `inference_npy/`: binary masks as NumPy arrays
- `inference_tif/`: binary masks as TIFF images
- `inference_labels_npy/`: labeled object masks
- `inference_comparison/`: raw image and mask comparison previews
- `inference_metrics/mask_metrics.csv`: per-organoid morphology metrics
- `inference_metrics/well_metrics_summary.csv`: per-image or per-well
  summaries, including `Total_Segmented_Area_um2` (sum, not mean, of that
  well's `Area_um2` -- the well's total organoid coverage regardless of
  object count, so it stays meaningful even when touching/merged organoids
  are traced as one connected component and under-counted)
- `inference_metrics/light_dark_metrics.csv`: one row per organoid with
  `Dark_Area_um2`, `Light_Area_um2`, `Light_Dark_Ratio`, `Light_Fraction`,
  `Dark_Fraction`, `Split_Threshold`, `Otsu_Eta_Squared`,
  `Has_Bimodal_Split`, and per-light-region stats -- only written when
  `track_light_dark_split: true` for that run
- `inference_metrics/empty_or_failed_masks.csv`: images with zero detected
  organoids or a measurement failure, so these are distinguishable from images
  that were never processed
- `segmentation_run_manifest.json`: resolved parameters (calibration,
  thresholds, checkpoint/model config, device) used to produce that
  batch/day's outputs, for provenance
- `classical_segmentation_summary.csv` / `orgline_segmentation_summary.csv`:
  one row per image with that method's own per-image detection metadata
  (organoid count, mask/label paths, thresholds used) -- written by `cl`/
  `orgline` runs respectively, not `dl`
- `segmentation_errors.csv`: images that failed to load/segment/filter, with
  the stage and error message (a bad image no longer halts the whole run)
- `plots/`: morphology plots and quality-control outputs, plus
  `plots/plot_data/`: the underlying per-record/per-summary/per-test CSVs
  behind each plot

## 9. Implementation Notes

A few non-obvious fixes worth knowing about if segmentation results look
wrong in a way that doesn't match any `segmentation_metadata.json` setting:

- **Well-boundary rings swallowing organoids**: classical contour extraction
  used to use `cv2.RETR_EXTERNAL`, which discards any contour enclosed
  inside another foreground region — so a well-boundary ring that thresholds
  as foreground silently swallowed every organoid sitting inside it, whether
  or not they physically touched the ring. Contour extraction now uses
  `cv2.RETR_CCOMP` and keeps only parentless (non-hole) contours, so
  organoids inside the ring are correctly recovered while the ring itself is
  still rejected by the area/circularity filters as before. Automatic, no
  metadata changes needed.
- **16-bit TIFF tiles going solid white after stitching**: `Image.convert("RGB")`
  does not rescale a 16-bit image's dynamic range — it treats pixel values as
  if already 8-bit, clipping a real sensor's ~0-65535 range to 255 for
  nearly every pixel. Tile loading now rescales any high-bit-depth tile
  (16-bit or 32-bit TIFF) to 8-bit using one min/max shared across the whole
  tile group before stitching (not a separate min/max per tile, which would
  reintroduce tile-to-tile brightness inconsistency). Already-8-bit tiles
  are unaffected. This only fixes *newly stitched* images — existing
  `raw_data/.../stitched/Tile*.jpg` files from before this fix should be
  deleted so they regenerate (`--force`, or `--resume` after removing the
  `stitched/` folder).
- **Stale `.tif` → `.jpg` conversion cache**: `.tif` → `.jpg` conversion for
  segmentation writes its output `.jpg` into the *same* `raw_data/` folder
  as the source `.tif`. Tile resolution used to always prefer a `.jpg` over
  a same-stem `.tif` unconditionally, so any `.jpg` ever generated by a past
  run permanently shadowed the `.tif` forever, regardless of the `.tif`
  being replaced or fixed later — and `--force` does **not** clear this (it
  only affects segmentation-stage skip logic, not this raw-data-adjacent
  cache). Tile resolution now compares `.tif` vs `.jpg` mtimes and prefers
  the `.tif` whenever it is not older than the cached `.jpg`. If you suspect
  this happened to your data, either delete the stray `.jpg` siblings under
  `raw_data/` or `touch` the corresponding `.tif` files, then re-run.

## Troubleshooting

- If no project appears in the menu, add it to `AVAILABLE_PROJECT_DIRS` in
  `experiment_info.py`.
- If segmentation metadata is not found, check that `segmentation_metadata.json`
  is inside the selected project folder (or set the
  `ORGANOID_SEGMENTATION_METADATA_FILE` environment variable to point at it
  directly).
- If `plot.py` produces no plots, check that `plotting_metadata.json`
  is inside the selected project folder and its `batches_complete_info` is
  non-empty — a missing file prints a warning at project-selection time and
  runs with an empty default. (`ORGANOID_PLOTTING_METADATA_FILE` overrides
  its path the same way.)
- If images are not found, check that the folder names are exactly
  `raw_data/Day <day>/<batch>/`.
- If masks are generated but plots look outdated, run
  `python3 morphology.py --check` to see which morphology metrics are
  stale, or `python3 morphology.py --force` to regenerate them before
  rerunning `plot.py`.
- If `well_metric_plots` entries produce "No data found for ... " messages,
  confirm the referenced `csv_filename` actually exists for that
  batch/day (`light_dark_metrics.csv` only exists for runs with
  `track_light_dark_split: true`) and that `metric_column` is spelled exactly
  as it appears in that CSV.
