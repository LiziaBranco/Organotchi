# Organotchi

Organotchi is a Python workflow for brightfield brain organoid image segmentation and downstream morphology analysis. It supports both deep-learning segmentation with MMSegmentation/SegFormer and configurable classical segmentation for threshold-based brightfield workflows.

The repository is organized around four tasks:

- train and evaluate segmentation models for single-organoid brightfield images
- segment new experiments using either deep-learning or classical pipelines
- extract organoid morphology metrics from the resulting masks
- plot those metrics across days, batches, and experimental conditions

## Main Capabilities

- Deep-learning segmentation using MMSegmentation and SegFormer checkpoints.
- Classical brightfield segmentation for different magnification values (1.25x, 4x, etc)
- OrgLine (detector-guided SAM2 prompting) as an alternative segmentation method (`orgline`), feeding the same downstream morphology pipeline as `dl`/`cl`.
- Stitching support for tiled microscope acquisitions such as `2x2` or `5x5` tile grids.
- Project-level metadata configuration through `segmentation_metadata.json` and `plotting_metadata.json`.
- Support for one organoid per image and multiple organoids per well/image.
- Export of binary masks, labeled masks, raw-vs-mask comparison images, and segmentation summaries.
- Morphology extraction at object and well/image level, including area, diameter, perimeter, circularity, solidity, eccentricity, Feret diameter, aspect ratio, roundness, and skeleton length.
- Longitudinal plotting, group comparisons, outlier reports, detection summaries, and quality-control plots.
- Resume mode for interrupted segmentation runs.

## Repository Structure

This is the **tool** repository -- `data_analysis` is what you `pip
install`: the segmentation/morphology/plotting pipeline itself, run
against already-trained checkpoints. Everything else -- the manuscript
(figures, tables), the tool-comparison benchmark, and the SegFormer
train/test pipeline (`train_and_test/`, `mmsegmentation/`) that produced
this tool's own checkpoints -- lives in a separate repository,
`Organotchi-manuscript`, which depends on this one as a package rather
than living alongside it. See "History" at the bottom for why.

```text
Organotchi/
    data_analysis/
        segmentation.py
        morphology.py
        plot.py
        experiment_info.py
        templates/
            segmentation_metadata.json
            plotting_metadata.json
            raw_data_folder_organization.txt
        utils_segmentation.py
        utils_morphology.py
        utils_plot.py
        __init__.py
        README.md
    pyproject.toml
    requirements.txt
```

The detailed instructions for preparing experiment folders and running segmentation/morphology are in `data_analysis/README.md`.

## Available Segmentation Checkpoints

All trained/plannned through the same dataset-agnostic `train_and_test/`
pipeline -- now in the separate `Organotchi-manuscript` repository (see
"Repository Structure" above) -- each pointed at its own
`--dataset-dir`/`--output-dir`:

| Tissue | Data source | Status |
|---|---|---|
| Brain | in-house curated dataset | trained, in production use |
| Brain, 1.25x magnification | classical-segmentation pseudo-labels, bootstrapped from in-house 1.25x images (`harvest_from_classical_outputs.py`) | training plan documented, not yet trained/validated -- see `Organotchi-manuscript/train_and_test/README.md`'s caveats |
| Intestine, Colon, PDAC | [OrgLine Zenodo benchmark](https://zenodo.org/records/16355179) (manually annotated, aggregated from published sources; `download_zenodo_benchmark.py`) | pipeline + data ready, not yet trained (no GPU on the dev machine used to set this up) |
| Lung | -- | not attempted: the OrgLine benchmark has no Lung segmentation masks, only bounding boxes |

Each dataset is independent (its own `--dataset-dir`/`--output-dir`, its own
`dataset_config.json`) and none read from or write to another's.

## Benchmarking Against Other Tools

Benchmarking this tool's checkpoints against Cellpose, StarDist, MOrgAna,
OrgLine, and others (per Deng et al. 2026, "OrgLine: A versatile pipeline
for organoid morphometry using detector-guided prompts", Cell Reports
Methods) lives in the separate `Organotchi-manuscript` repository's
`benchmarking/`, not here -- see that repo's `benchmarking/README.md` for
the full protocol, every tool's setup caveats, and the results table.
OrgLine's own detector+SAM2 loading code (the bug-workarounds it needed)
lives here instead, in `data_analysis/orgline_bridge.py`, since it also
backs this pipeline's own `orgline` segmentation method (see
`data_analysis/README.md`'s "Minimal orgline run example") -- the
benchmark repo imports it from here rather than the other way around.

## Installation

Two ways to install, depending on what you're doing:

**As a dependency of another project** (e.g. the manuscript repo, or your
own analysis scripts) -- install as a package:

```bash
pip install -e /path/to/Organotchi          # editable, for local development
pip install "organotchi[dl] @ git+https://github.com/<user>/Organotchi.git"  # from GitHub, with the DL extras
```

This installs `data_analysis` as a top-level importable package
(`from data_analysis.eval_metrics import evaluate_image`, etc.). The
deep-learning stack (`torch`, `mmengine`, `mmcv`, `mmdet` -- everything
SegFormer inference and the `orgline` method need) is the optional `dl`
extra (`pip install -e ".[dl]"`), since classical-segmentation and
morphology-analysis workflows don't need it -- every module that does soft-
imports it. This repo does not include the SegFormer *training* pipeline
(`train_and_test/`, `mmsegmentation/`) -- that, along with its own
project-specific mmsegmentation fork with a custom dataset class, moved to
`Organotchi-manuscript` (see "History" below); `data_analysis` only needs
mmsegmentation/mmengine at inference time, against checkpoints already
trained there.

**To work on this repo directly** (run `segmentation.py`/`morphology.py`/
`plot.py` etc. from within `data_analysis/`, the way this README's examples
below do) -- a plain venv with the full pinned `requirements.txt` is
simpler, since it doesn't require `cd`-ing anywhere relative to a package
install:

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r Organotchi/requirements.txt
```

The current `requirements.txt` was synchronized against the local `myenv` environment used for this project. It includes the OpenMMLab stack (`mmengine`, `mmcv`, `mmdet`, `mmsegmentation`) plus scientific Python and plotting dependencies.

If you only need classical segmentation and morphology analysis, the deep-learning dependencies are still listed because this repository also contains DL-checkpoint *inference* code (`segmentation.py`'s `dl` method) -- not training, which lives in `Organotchi-manuscript/train_and_test/`.

## Data Layout

Raw experiment data are expected outside the repository, under the base directory configured in `data_analysis/experiment_info.py`.

A typical experiment folder looks like this:

```text
Project_name/
    segmentation_metadata.json
    plotting_metadata.json
    raw_data/
        Day <day>/
            <batch_or_condition>/
                Tile000001.jpg
                Tile000002.jpg
                ...
    processed_data/
        ... generated by the pipeline ...
```

Image names, day folders, and batch names must match the project metadata. See `data_analysis/templates/raw_data_folder_organization.txt` and `data_analysis/README.md` for the full convention.

## Running an Analysis

The pipeline is three independent stages -- segmentation (masks), morphology
extraction (metrics CSVs), and plotting (figures/statistics) -- each its own
script. Run them in order from `data_analysis`:

```bash
cd Organotchi/data_analysis
python3 segmentation.py
```

Select the project from the interactive menu. The script reads the selected project's `segmentation_metadata.json` and generates masks.

`segmentation.py` supports three mutually exclusive modes (identical behavior for deep-learning and classical runs):

```bash
python3 segmentation.py --resume   # default: continue from wherever the analysis stopped
python3 segmentation.py --force    # ignore existing outputs, recompute everything from scratch
python3 segmentation.py --check    # report what's done/missing/stale; runs nothing
```

See `data_analysis/README.md` for details on what each mode checks.

Once masks exist, extract morphology metrics:

```bash
python3 morphology.py
```

`morphology.py` supports the same `--resume`/`--force`/`--check` modes, reading whichever masks `segmentation.py` has already produced and writing the per-object/per-well CSVs under each day/batch's `inference_metrics/`.

Finally, generate plots and reports from those CSVs:

```bash
python3 plot.py
```

Plot features, grouping variables, aggregation levels, and outlier settings are configured directly in `plot.py`.

## Configuration Model

Each experiment should have a project-level `segmentation_metadata.json`, defining:

- batches or conditions
- days/time points
- segmentation method: `dl`, `cl`, or `orgline`
- magnification and pixel-to-micron calibration key
- tile grid and stitching parameters
- number of images/wells per plate
- whether each image contains single or multiple organoids
- classical segmentation thresholds and filters

and a project-level `plotting_metadata.json`, defining:

- grouping variables used by morphology plots
- plot features, aggregation levels, and outlier-detection settings

The templates in `data_analysis/templates/segmentation_metadata.json` and `data_analysis/templates/plotting_metadata.json` can be copied into a project folder and edited for a new experiment. See `data_analysis/README.md` for the full key-by-key reference.

## Outputs

For each processed day and batch, the pipeline writes outputs under:

```text
processed_data/Day <day>/<batch>/
```

Common output folders include:

- `inference_jpg/`: binary mask images
- `inference_npy/`: binary masks as NumPy arrays
- `inference_tif/`: binary masks as TIFF images
- `inference_labels_npy/`: labeled object masks
- `inference_comparison/`: raw image and mask comparison previews
- `inference_metrics/`: per-object and per-well morphology CSV files

Project-level plots and quality-control outputs are written under `plots/`.

## Training and Testing

Scripts for dataset splitting, model training, checkpoint testing, and
checkpoint comparison (`train_and_test/`), plus the MMSegmentation
configuration files/custom dataset class they use (`mmsegmentation/`), live
in the separate `Organotchi-manuscript` repository -- not here, since
training a checkpoint is a one-time, paper-adjacent activity, while this
repository's own job is running already-trained checkpoints against new
experiments. See that repo's `train_and_test/README.md`.

## Notes

- Large raw and processed image data are not intended to be stored in this repository.
- Classical segmentation performance depends strongly on threshold, foreground polarity, magnification, tile stitching, and whether organoids are cystic, dense, single, or multiple per image.
- For new imaging protocols, validate masks visually using `inference_comparison/` before interpreting morphology plots.

## History

This repository used to also contain `benchmarking/` (tool-comparison
infrastructure) and `manuscript/` (figures/tables for the paper describing
this tool) directly, as `Organoid-Segmentation` -- a single repo mixing the
tool itself with paper-specific, non-reusable content. Those two moved out
into a separate `Organotchi-manuscript` repository, which now depends on
this one as an installed package (see "Installation" above) instead of
reaching into a sibling checkout by a hardcoded relative path. `data-analysis/`
and `train-and-test/` were also renamed to `data_analysis/`/`train_and_test/`
at the same time, since Python package names can't contain hyphens --
that's what makes `pip install`-ing this repo and getting clean
`from data_analysis.x import y`-style imports possible at all.

In a follow-up pass, `train_and_test/` and `mmsegmentation/` also moved to
`Organotchi-manuscript` -- training a SegFormer checkpoint is, like the
benchmark, a one-time/paper-adjacent activity, not something this tool's
own end users (who just run inference against an already-trained
checkpoint) need. `data_analysis/compare_zeynep_cp_segmentation_methods.py`
and `data_analysis/organize_training_dataset.py` moved to
`Organotchi-manuscript/analysis/` for the same reason: both are one-off
manuscript-specific analyses, not part of the reusable pipeline. The three
per-project config templates (`segmentation_metadata.json`,
`plotting_metadata.json`, `raw_data_folder_organization.txt`) that used to
sit directly in `data_analysis/` moved into `data_analysis/templates/`, to
separate them from the pipeline's own code.
