"""
Shared OrgLine (detector-guided SAM2 prompting) model-loading bridge.

Centralizes three bugs in the upstream OrgLine repo
(https://github.com/ucas-dx/OrgLine, as cloned 2026-07-28), all left
undocumented in OrgLine's own README, that every caller of OrgLine in this
codebase needs worked around identically:

1. `orgdet_tiny.pt`'s detection head was trained with nc=18 (18 output
   classes), not nc=1 as every shipped config file (`orgline_tiny*.yaml`,
   `data/*.yaml`) declares. Test: loading the checkpoint into
   an nc=1 model produced 0 missing/0 unexpected *names* but a hard shape
   mismatch on every `model.24.cv3.*.weight` (18 vs 1 output channels);
   setting nc=18 in a local config copy loads with an exact 787/787 match.
   We don't know what the 18 classes correspond to (probably a fine-grained
   scheme from OrgLine's own multi-tissue training set), but every detection
   is treated as "an organoid" here regardless of class id.
2. `orgdet_tiny.pt` was also trained with the PANet-style head in
   `orgline_tiny_pan.yaml`, not the plainer `orgline_tiny.yaml` the README's
   "Low-VRAM" bullet names -- same evidence (shape/name match) confirms it.
3. Calling `.eval()` on the top-level `ORGDET` object crashes with
   `KeyError('Distillation')`: `ORGDET` mimics the Ultralytics `YOLO` API,
   whose `Model.train()` is a full *dataset* training entrypoint (needs
   `overrides['Distillation']` etc.) -- but `torch.nn.Module.eval()` is
   implemented as `self.train(False)`, so it calls OrgLine's *training*
   method with `False` instead of toggling inference mode. Worked around by
   never calling `.eval()` on the wrapper -- only on the underlying
   `model.model` (the real `nn.Module`), which is all inference actually needs.

Weights aren't merged through `weights_merge.merge_and_load_model_weights()`
for the same reason (it calls the wrapper's `.eval()` internally and would
hit the same crash) -- this module calls the lower-level
`ModelWeightMerger.load_and_merge_weights()` /
`.safe_load_weights_to_model()` directly instead.

Used by every OrgLine consumer of this tool:
- `run_orgline_segmentation` in `utils_segmentation.py` (this folder), the
  `"orgline"` method available directly from `segmentation.py`.
- `Organotchi-manuscript/benchmarking/adapters/predict_orgline.py`, the
  tool-comparison benchmark.
"""

import os
import sys
from pathlib import Path

NC18_CONFIG_NAME = "orgline_tiny_pan_nc18.yaml"


def _ensure_nc18_config(orgline_repo):
    """Write a local nc=18 copy of orgline_tiny_pan.yaml next to the original
    if it doesn't already exist (see module docstring, bugs #1/#2). Returns
    the new config's path relative to `orgline_repo` (what `ORGDET(...)`
    expects)."""
    cfg_dir = Path(orgline_repo) / "orgline" / "cfg" / "models" / "our_model"
    src = cfg_dir / "orgline_tiny_pan.yaml"
    dest = cfg_dir / NC18_CONFIG_NAME
    if not dest.exists():
        text = src.read_text()
        text = text.replace("nc: 1  # number of classes", "nc: 18  # number of classes", 1)
        dest.write_text(text)
    return str(dest.relative_to(orgline_repo))


def build_detector(orgline_repo, orgdet_checkpoint, sam2_checkpoint, device="cpu"):
    """Build OrgLine's ORGDET detector with the nc=18/PANet-head/`.eval()`
    workarounds applied (see module docstring). Must be called with
    `orgline_repo` already the current working directory -- OrgLine's own
    architecture code hard-codes the SAM2 checkpoint path as relative to
    cwd, with no parameter to override it."""
    sys.path.insert(0, orgline_repo)
    from orgline import ORGDET
    from orgline.weights_merge import ModelWeightMerger

    config_rel_path = _ensure_nc18_config(orgline_repo)
    model = ORGDET(config_rel_path)

    merger = ModelWeightMerger(verbose=False)
    _checkpoint, complete_state = merger.load_and_merge_weights(orgdet_checkpoint, sam2_checkpoint)
    temp_model = merger.safe_load_weights_to_model(model, complete_state)
    model.model = temp_model
    model.model.eval()  # NOT model.eval() -- see module docstring, bug #3
    model.model.to(device)
    return model


def build_sam2_predictor(orgline_repo, sam2_config_name, checkpoint_dir, device="cpu"):
    """Build the SAM2 mask predictor OrgLine's detections are prompted
    through. `orgline_repo` must already be importable (see `build_detector`)."""
    from orgline.build_model import load_sam2_pretrained

    model_name = {"sam2_hiera_t.yaml": "hiera_tiny", "sam2_hiera_l.yaml": "hiera_large"}[sam2_config_name]
    return load_sam2_pretrained(
        model_name=model_name,
        config_dir=os.path.join(orgline_repo, "orgline", "sam2_configs"),
        device=device,
        download_dir=checkpoint_dir,
    )
