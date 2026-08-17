"""
Shared metrics for comparing instance-labeled segmentation masks (0 =
background, each object a distinct positive integer) against ground truth.

Two families of metric are reported, deliberately, because not every tool
this repository segments with is instance-aware:

1. Instance-matching metrics (AP50 / AP75 / mAP, F1, precision, recall) --
   fair between tools (e.g. Cellpose, StarDist, OrgLine) that natively
   separate touching/overlapping organoids into distinct instances.
2. Pixel-level metrics (Dice, IoU, accuracy on the binarized organoid/
   background mask) -- fair for our own SegFormer model too, which is
   semantic-only; its "instances" for metric #1 are just connected
   components of the binary mask and will under-count touching organoids
   as a structural disadvantage, not a measure of mask quality.

The "AP" here follows the convention used in the Cellpose (Stringer et al.
2021) and StarDist (Schmidt et al. 2018) papers and the 2018 Kaggle Data
Science Bowl nucleus-segmentation challenge: AP_t = TP / (TP + FP + FN) at a
one-to-one greedy IoU match with threshold t. This is NOT the COCO
precision-recall-curve AP (that needs per-instance confidence scores, which
several tools/output formats don't expose uniformly) -- it's the simpler,
score-free matching accuracy that's the de facto standard metric in exactly
this sub-field. Naming it "AP" here is for consistency with that literature,
not COCO.

Used, as an installed package (`from data_analysis.eval_metrics import ...`),
by the separate `Organotchi-manuscript` repo's `train_and_test/checkpoint_predictions.py`'s
`evaluate_image`, `train_and_test/tune_classical_filter.py`'s
`pixel_metrics`, and `benchmarking/eval_metrics.py` (which adds
`aggregate_image_results` -- the benchmark-reporting summary stat (mean/
median/IQR across a whole tissue) that has no reason to live on the tool
side, since nothing outside that repo's `benchmarking/` needs it).
"""

import numpy as np


def _instance_ids(mask):
    ids = np.unique(mask)
    return ids[ids != 0]


def compute_iou_matrix(pred_mask, gt_mask):
    """IoU matrix, shape (n_pred, n_gt), between every pred instance and every
    GT instance (both instance-labeled arrays, 0=background).

    Vectorized via a joint (pred_id, gt_id) pixel histogram rather than a
    per-pair mask-and/mask-or loop -- the naive O(n_pred * n_gt) loop, each
    iteration doing a full-image boolean op, is fine for Brain (~1
    instance/image) but becomes minutes-to-hung on PDAC (500+ instances/image
    -> 250,000+ full-image passes per image). This does one O(H*W) pass
    regardless of instance count."""
    # Some ground truth (e.g. Brain, one-organoid-per-image, saved straight from
    # Zenodo's own .npy) is a plain bool array rather than an int label map.
    # Cast up front so `bool` labels don't get treated as a boolean *index*
    # later (numpy's fancy indexing overloads bool arrays as masks, not values).
    if pred_mask.dtype == bool:
        pred_mask = pred_mask.astype(np.int64)
    if gt_mask.dtype == bool:
        gt_mask = gt_mask.astype(np.int64)

    pred_ids = _instance_ids(pred_mask)
    gt_ids = _instance_ids(gt_mask)
    n_pred, n_gt = len(pred_ids), len(gt_ids)

    if n_pred == 0 or n_gt == 0:
        return np.zeros((n_pred, n_gt), dtype=np.float64), pred_ids, gt_ids

    # Dense 1..n remap (arbitrary/sparse source IDs -> compact indices) so the
    # histogram below is sized by instance COUNT, not by the largest ID value.
    pred_remap = np.zeros(int(pred_mask.max()) + 1, dtype=np.int64)
    pred_remap[pred_ids] = np.arange(1, n_pred + 1)
    gt_remap = np.zeros(int(gt_mask.max()) + 1, dtype=np.int64)
    gt_remap[gt_ids] = np.arange(1, n_gt + 1)

    pred_dense = pred_remap[pred_mask]
    gt_dense = gt_remap[gt_mask]

    combined = pred_dense.astype(np.int64) * (n_gt + 1) + gt_dense.astype(np.int64)
    hist = np.bincount(combined.ravel(), minlength=(n_pred + 1) * (n_gt + 1))
    hist = hist.reshape(n_pred + 1, n_gt + 1)
    intersection = hist[1:, 1:].astype(np.float64)  # drop the background row/col

    pred_areas = np.bincount(pred_dense.ravel(), minlength=n_pred + 1)[1:].astype(np.float64)
    gt_areas = np.bincount(gt_dense.ravel(), minlength=n_gt + 1)[1:].astype(np.float64)

    union = pred_areas[:, None] + gt_areas[None, :] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)

    return iou, pred_ids, gt_ids


def match_at_threshold(iou_matrix, threshold):
    """Greedy one-to-one matching, highest-IoU-first, at a fixed IoU
    threshold. Returns (tp, fp, fn) counts."""
    n_pred, n_gt = iou_matrix.shape
    if n_pred == 0 and n_gt == 0:
        return 0, 0, 0
    if n_pred == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, n_pred, 0

    candidates = np.argwhere(iou_matrix >= threshold)
    if len(candidates) == 0:
        return 0, n_pred, n_gt

    order = np.argsort(-iou_matrix[candidates[:, 0], candidates[:, 1]])
    candidates = candidates[order]

    matched_pred, matched_gt = set(), set()
    for pi, gi in candidates:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)

    tp = len(matched_pred)
    fp = n_pred - tp
    fn = n_gt - tp
    return tp, fp, fn


DEFAULT_THRESHOLDS = tuple(np.round(np.arange(0.5, 1.0, 0.05), 2))


def instance_metrics(pred_mask, gt_mask, thresholds=DEFAULT_THRESHOLDS):
    iou_matrix, pred_ids, gt_ids = compute_iou_matrix(pred_mask, gt_mask)

    per_threshold = {}
    for t in thresholds:
        tp, fp, fn = match_at_threshold(iou_matrix, t)
        denom = tp + fp + fn
        ap = tp / denom if denom else 1.0  # both empty at this threshold -> trivially perfect
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
        per_threshold[t] = dict(tp=tp, fp=fp, fn=fn, ap=ap, precision=precision, recall=recall, f1=f1)

    ap50 = per_threshold[0.5]["ap"] if 0.5 in per_threshold else None
    ap75 = per_threshold[0.75]["ap"] if 0.75 in per_threshold else None
    mean_ap = float(np.mean([v["ap"] for v in per_threshold.values()]))
    f1_50 = per_threshold[0.5]["f1"] if 0.5 in per_threshold else None

    return dict(
        n_pred=len(pred_ids),
        n_gt=len(gt_ids),
        ap50=ap50,
        ap75=ap75,
        mean_ap=mean_ap,
        f1_at_50=f1_50,
        per_threshold=per_threshold,
    )


def pixel_metrics(pred_mask, gt_mask):
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    intersection = np.count_nonzero(pred_bin & gt_bin)
    union = np.count_nonzero(pred_bin | gt_bin)
    pred_sum = pred_bin.sum()
    gt_sum = gt_bin.sum()

    iou = intersection / union if union else 1.0
    dice = (2 * intersection) / (pred_sum + gt_sum) if (pred_sum + gt_sum) else 1.0
    accuracy = np.mean(pred_bin == gt_bin)

    return dict(iou=float(iou), dice=float(dice), accuracy=float(accuracy))


def evaluate_image(pred_mask, gt_mask, thresholds=DEFAULT_THRESHOLDS):
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"Shape mismatch: pred {pred_mask.shape} vs gt {gt_mask.shape}")
    result = pixel_metrics(pred_mask, gt_mask)
    result.update(instance_metrics(pred_mask, gt_mask, thresholds=thresholds))
    return result
