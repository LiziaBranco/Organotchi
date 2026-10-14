# OrgLine environment

This folder is only needed if you want to use Organotchi's `orgline`
segmentation method (detector-guided SAM2 prompting, via
`data_analysis/utils_segmentation.py`'s `run_orgline_segmentation` and the
shared bridge module `data_analysis/orgline_bridge.py`). It is **not**
needed for the `dl` (SegFormer) or `cl` (classical) methods.

## Setup

All steps below are run from the repository root (`Organotchi/`).

1. Clone [OrgLine](https://github.com/ucas-dx/OrgLine) into this folder:

   ```bash
   git clone https://github.com/ucas-dx/OrgLine environments/OrgLine
   ```

2. Install OrgLine's own dependencies into the *same* environment you
   already use for Organotchi (the `orgline` extra -- see the repository
   root `pyproject.toml`, or `requirements.txt` if you installed that way
   instead):

   ```bash
   pip install -e ".[orgline]"
   cd environments/OrgLine && pip install --no-deps -e . && cd ../..
   ```

3. Download the checkpoints if not available (the "tiny"/low-VRAM variant -- the "Standard"
   variant is ~15x the parameter count and was judged infeasible for
   CPU-only use):

   ```bash
   # SAM2 tiny -- public Meta URL, no auth needed
   curl -L -o environments/OrgLine/checkpoints/sam2_hiera_tiny.pt \
       https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
   cp environments/OrgLine/checkpoints/sam2_hiera_tiny.pt environments/OrgLine/  # see bug #4 below

   # OrgDet tiny -- Google Drive, but a real (non-dead) link
   python -m gdown 169OIs_wCGidXcbe5g6u3kWcji3PKhlmW \
       -O environments/OrgLine/checkpoints/orgdet_tiny.pt
   ```

That's it -- no separate virtual environment, and no further configuration.
With the clone at exactly `Organotchi/environments/OrgLine` as set up above,
`segmentation.py` finds it automatically (`experiment_info.py`'s
`_DEFAULT_ORGLINE_REPO`); set a run's `orgline_repo`/`orgdet_checkpoint`/
`sam2_checkpoint` keys in `segmentation_metadata.json` only if your clone
lives elsewhere.

## Four real upstream bugs, worked around

All undocumented in OrgLine's own README (as cloned 2026-07-28). Full
detail, with the exact evidence for each, is in `../data_analysis/orgline_bridge.py`'s
module docstring -- the loader code that works around bugs #1-#3 lives
there, shared by every OrgLine caller in this repo:

1. **`orgdet_tiny.pt` expects `nc=18`**, not `nc=1` as every shipped config
   (`orgline_tiny*.yaml`, `data/*.yaml`) declares. Loading it into an nc=1
   model gives a clean 0-missing/0-unexpected *name* match but a hard shape
   mismatch on the final `model.24.cv3.*.weight` (18 vs 1 output channels).
   `orgline_bridge.py` writes a local nc=18 copy of the config on first
   run rather than hand-editing the shipped one.
2. **It also expects the PANet head** (`orgline_tiny_pan.yaml`), not the
   plainer `orgline_tiny.yaml` the README's own "Low-VRAM" bullet names --
   same evidence (exact parameter-name match) confirms it.
3. **Calling `.eval()` on the top-level `ORGDET` object crashes** with
   `KeyError('Distillation')`. `ORGDET` mimics the Ultralytics `YOLO` API,
   whose `Model.train()` is a full *dataset*-training entrypoint (expects
   `overrides['Distillation']` etc.) -- but `torch.nn.Module.eval()` is
   implemented as `self.train(False)`, so it calls OrgLine's *training*
   method with `False` instead of toggling inference mode. Fixed by never
   calling `.eval()` on the wrapper, only on the underlying `model.model`.
   For the same reason, `weights_merge.merge_and_load_model_weights()`
   (which calls the wrapper's `.eval()` internally) isn't used either --
   `orgline_bridge.py` calls the lower-level `ModelWeightMerger` methods
   directly instead.
4. **The SAM2 checkpoint path is hard-coded, with no parameter to override
   it**: `samhiera3_tiny.__init__` (the actual backbone used inside
   `ORGDET`) does `checkpoint = "sam2_hiera_tiny.pt"` and passes that bare
   relative filename straight to `build_sam2()`. That means the file must
   exist as a literal `./sam2_hiera_tiny.pt` relative to whatever the
   process's cwd is at model-construction time -- not configurable via any
   of `ORGDET`'s or `weights_merge`'s own arguments. Every caller handles
   this with `os.chdir()` into the OrgLine clone before building the
   model; you need the extra copy in the repo root (not just under
   `checkpoints/`) for the same reason if calling the lower-level
   functions yourself outside those callers.
