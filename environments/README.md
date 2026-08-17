# OrgLine environment

The `orgline` segmentation method (`data_analysis/utils_segmentation.py`'s
`run_orgline_segmentation`, and `data_analysis-orgline`'s bridge module
`orgline_bridge.py`) needs a local clone of
[OrgLine](https://github.com/ucas-dx/OrgLine) (detector-guided SAM2
prompting) plus its own isolated Python environment -- its dependencies
(a specific `torch`/`torchvision` pin, a vendored Ultralytics-style
detector, SAM2) are independent of this project's main environment.

This is the tool's own copy, separate from (but architecturally identical
to) whatever copy the `Organotchi-manuscript` repo's benchmark sets up for
itself under its own `benchmarking/environments/` -- see that repo's
`environments/README.md` for the fuller multi-tool environment story if
you're working on the benchmark specifically. `../data_analysis/orgline_bridge.py`
is the shared bug-fix loading code both repos' OrgLine usage goes through.

## Setup

```bash
python3 -m venv orgline_venv
orgline_venv/bin/pip install torch torchvision
orgline_venv/bin/pip install opencv-python-headless pillow imageio scikit-image numpy scipy pandas \
    einops timm albumentations scikit-learn seaborn matplotlib roifile \
    hydra-core omegaconf tqdm rich psutil h5py pycocotools tifffile gdown py-cpuinfo
git clone https://github.com/ucas-dx/OrgLine OrgLine
cd OrgLine && ../orgline_venv/bin/pip install --no-deps -e .
```

Checkpoints (the "tiny"/low-VRAM variant -- the "Standard" variant is ~15x
the parameter count and was judged infeasible for CPU-only use):

```bash
# SAM2 tiny -- public Meta URL, no auth needed
curl -L -o OrgLine/checkpoints/sam2_hiera_tiny.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
cp OrgLine/checkpoints/sam2_hiera_tiny.pt OrgLine/  # see bug #4 below

# OrgDet tiny -- Google Drive, but a real (non-dead) link
orgline_venv/bin/python -m gdown 169OIs_wCGidXcbe5g6u3kWcji3PKhlmW \
    -O OrgLine/checkpoints/orgdet_tiny.pt
```

Once set up, point a project's `segmentation_metadata.json` at this
environment (`orgline_repo`/`orgdet_checkpoint`/`sam2_checkpoint` keys --
see `../data_analysis/README.md`'s "orgline" reference) -- or, if this
clone lives at exactly `Organotchi/environments/OrgLine` as set up above,
no configuration is needed at all: that's `experiment_info.py`'s own
`_DEFAULT_ORGLINE_REPO`.

To run this environment's Python directly (e.g. for the `orgline` method,
which needs `orgline_venv`'s interpreter, not the main one):

```bash
Organotchi/environments/orgline_venv/bin/python Organotchi/data_analysis/segmentation.py
```

## Four real upstream bugs, worked around

All undocumented in OrgLine's own README (as cloned 2026-07-28). Full
detail, with the exact evidence for each, is in `../data_analysis/orgline_bridge.py`'s
module docstring -- the loader code that works around bugs #1-#3 lives
there, shared by every OrgLine caller in this repo (and by
`Organotchi-manuscript/benchmarking/adapters/predict_orgline.py`, which
imports it from here as an installed package):

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
