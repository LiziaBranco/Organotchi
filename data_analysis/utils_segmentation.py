import numpy as np
from PIL import Image, ImageDraw
from os.path import join as opj
import os
import re
import sys
import json
import tempfile
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from skimage import measure
from skimage.filters import threshold_otsu
from scipy.ndimage import distance_transform_edt
import cv2
import glob
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="mmengine.runner.checkpoint")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="mmengine.optim.optimizer.zero_optimizer")

try:
    from mmengine.config import Config
    from mmseg.apis import inference_model, init_model
except ModuleNotFoundError:
    Config = None
    inference_model = None
    init_model = None

# OrgLine ("orgline" method, run_orgline_segmentation) needs its own isolated
# torch/torchvision pins plus a vendored Ultralytics-style detector and SAM2,
# just like mmsegmentation above -- soft-imported the same way so importing
# this module doesn't hard-fail outside OrgLine's own venv.
try:
    import torch
except ModuleNotFoundError:
    torch = None

# Connectivity convention shared by every labeling/filtering step in this module.
# (Picked 8-connectivity since the final saved labels and morphology metrics
# already used it; the DL/classical filtering helpers were the outliers.)
LABEL_CONNECTIVITY = 2


def _pixel_area_to_micron(pixel_area, pixel_to_micron_x, pixel_to_micron_y):
    """Convert a pixel-count area to um^2 using true anisotropic (x, y) spacing."""
    return pixel_area * pixel_to_micron_x * pixel_to_micron_y


def _pixel_length_to_micron(pixel_length, pixel_to_micron_x, pixel_to_micron_y):
    """Convert a 1D pixel length (perimeter, Feret diameter, skeleton) to um.

    Length measures mix x/y pixel steps, so there is no exact anisotropic
    conversion; every call site in this module uses the same average-spacing
    approximation so circularity/perimeter are comparable across the
    classical and deep-learning pipelines.
    """
    avg_spacing = (pixel_to_micron_x + pixel_to_micron_y) / 2
    return pixel_length * avg_spacing


def _write_run_manifest(processed_data_dir, section, params):
    """Merge resolved run parameters into a per-batch/day JSON manifest.

    This gives every output CSV a traceable record of exactly which
    thresholds/calibration/model were used to produce it, since the CSVs
    themselves only carry a subset of that (e.g. classical_segmentation_summary.csv).
    """
    os.makedirs(processed_data_dir, exist_ok=True)
    manifest_path = opj(processed_data_dir, "segmentation_run_manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError):
            manifest = {}

    manifest[section] = {
        **{key: value for key, value in params.items()},
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)


def _resolve_model_config_path(checkpoint_path, model_config_path=None):
    """Resolve the mmseg model config that matches a given checkpoint.

    Previously the config path was a single string hardcoded in
    ``run_deep_learning_segmentation`` and shared by every run regardless of
    which checkpoint was selected. Defaulting to "the .py file next to the
    checkpoint" keeps each checkpoint paired with its own config, while still
    allowing an explicit override.
    """
    if model_config_path:
        return model_config_path

    checkpoint_dir = os.path.dirname(checkpoint_path)
    candidates = sorted(glob.glob(opj(checkpoint_dir, "*.py")))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a model config (.py) file next to checkpoint '{checkpoint_path}'. "
            "Pass model_config_path explicitly to run_deep_learning_segmentation."
        )

    for candidate in candidates:
        if os.path.splitext(os.path.basename(candidate))[0] == os.path.basename(checkpoint_dir):
            return candidate

    return candidates[0]


def clean_folder_names(directory, batch):
    """
    Renames folders by keeping only the base name and any 'XinY'-style tags (like '1in3'),
    removing everything after the first space or underscore. If no 'XinY' is present, only base is kept.
    """
    folder_path = opj(directory, batch)
    if os.path.exists(folder_path):
        print(f"Skipping rename: {batch} already exists.")
    else:
        for folder in os.listdir(directory):
            folder_path = opj(directory, folder)
            if os.path.exists(folder_path):
                # Extract base name: everything before the first space or underscore
                if batch in folder:
                    base = batch
                    xinY_match = re.search(r"(\d+)in(\d+)", folder)
                    xinY_part = f"_{xinY_match.group()}" if xinY_match else ""

                    # Combine base and optional 'XinY' part
                    cleaned_name = base + xinY_part
                    cleaned_path = opj(directory, cleaned_name)

                    # Rename if necessary
                    if cleaned_name != folder:
                        if not os.path.exists(cleaned_path):
                            os.rename(folder_path, cleaned_path)
                            print(f"Renamed: {folder} → {cleaned_name}")
                        else:
                            print(f"Skipping rename: {cleaned_name} already exists.")
                else:
                    continue

def remove_small_objects_by_area(
    mask,
    pixel_to_micron_x,
    pixel_to_micron_y,
    min_area=30000,
    max_area=None,
):
    """
    Removes connected components whose area (in um^2) is <= min_area, or (if
    max_area is given) >= max_area. max_area is optional and None by default
    -- unlike earlier versions of this pipeline, an upper size bound is not
    applied unless explicitly requested, since real organoids legitimately
    grow larger over the developmental timeline and an unconditional upper
    bound risked discarding genuine late-timepoint organoids.

    Unlike the single-structure selection this used to be bundled with (see
    keep_most_circular_if_crowded), every component that passes the area gate
    is kept here -- this step only removes objects that are clearly too small
    (or, if requested, too large) to be a real structure, run BEFORE erosion
    so a legitimately-sized object is judged on its actual size, not its
    post-erosion size.

    Area is computed with the same unit-conversion helper used by the
    classical pipeline (``_pixel_area_to_micron``), so DL and classical size
    filtering stay directly comparable.
    """
    labeled_mask, num_features = measure.label(mask, connectivity=LABEL_CONNECTIVITY, return_num=True)
    properties = measure.regionprops(labeled_mask)

    filtered_mask = np.zeros(mask.shape, dtype=np.uint8)
    for prop in properties:
        area = _pixel_area_to_micron(prop.area, pixel_to_micron_x, pixel_to_micron_y)
        if area <= min_area:
            continue
        if max_area is not None and area >= max_area:
            continue
        filtered_mask[labeled_mask == prop.label] = 255

    return filtered_mask


def remove_low_circularity_objects(
    mask,
    pixel_to_micron_x,
    pixel_to_micron_y,
    circularity_threshold=0.3,
):
    """
    Removes connected components whose circularity is <= circularity_threshold.
    Every component that passes is kept (not just the single largest) -- see
    keep_most_circular_if_crowded for the step that narrows down to one
    structure when more than a couple of candidates remain.

    Circularity is computed with the same unit-conversion helpers used by the
    classical pipeline (``_pixel_area_to_micron``/``_pixel_length_to_micron``).
    """
    labeled_mask, num_features = measure.label(mask, connectivity=LABEL_CONNECTIVITY, return_num=True)
    properties = measure.regionprops(labeled_mask)

    filtered_mask = np.zeros(mask.shape, dtype=np.uint8)
    for prop in properties:
        area = _pixel_area_to_micron(prop.area, pixel_to_micron_x, pixel_to_micron_y)
        perimeter = _pixel_length_to_micron(prop.perimeter, pixel_to_micron_x, pixel_to_micron_y)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        if circularity > circularity_threshold:
            filtered_mask[labeled_mask == prop.label] = 255

    return filtered_mask


def keep_most_circular_if_crowded(
    mask,
    pixel_to_micron_x,
    pixel_to_micron_y,
    max_objects=2,
):
    """
    If more than max_objects connected components remain (default: more than
    2), keeps only the single one with the highest circularity and discards
    the rest. If max_objects or fewer components remain, the mask is
    returned unchanged (every surviving component kept) -- this step only
    forces a single winner once there are enough leftover candidates that
    the result would otherwise be ambiguous.

    Circularity, for choosing among crowded candidates, is computed with the
    same unit-conversion helpers used elsewhere in this pipeline
    (``_pixel_area_to_micron``/``_pixel_length_to_micron``).

    Returns (filtered_mask, structure_count).
    """
    labeled_mask, num_features = measure.label(mask, connectivity=LABEL_CONNECTIVITY, return_num=True)
    properties = measure.regionprops(labeled_mask)

    if len(properties) <= max_objects:
        return mask.astype(np.uint8), len(properties)

    best_circularity = -1
    best_label = -1
    for prop in properties:
        area = _pixel_area_to_micron(prop.area, pixel_to_micron_x, pixel_to_micron_y)
        perimeter = _pixel_length_to_micron(prop.perimeter, pixel_to_micron_x, pixel_to_micron_y)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        if circularity > best_circularity:
            best_circularity = circularity
            best_label = prop.label

    filtered_mask = np.zeros(mask.shape, dtype=np.uint8)
    structure_count = 0
    if best_label != -1:
        filtered_mask[labeled_mask == best_label] = 255
        structure_count = 1

    return filtered_mask, structure_count

def _get_center_crop_box(width, height, keep_fraction=0.85):
    """Return a centered crop box that preserves the original aspect ratio."""
    crop_width = max(1, int(round(width * keep_fraction)))
    crop_height = max(1, int(round(height * keep_fraction)))

    left = max((width - crop_width) // 2, 0)
    top = max((height - crop_height) // 2, 0)
    right = min(left + crop_width, width)
    bottom = min(top + crop_height, height)

    return left, top, right, bottom

def _resize_image_to_original_size(image, original_size):
    """Resize the cropped image back to the original size before inference."""
    resampling_attr = getattr(Image, "Resampling", Image)
    return image.resize(original_size, resampling_attr.BILINEAR)

def _restore_resized_crop_mask_to_original_size(mask, original_size, crop_box):
    """
    Map the prediction from the expanded crop back into the original crop window.

    The mask returned by inference corresponds to the resized crop, so it is reduced to
    the crop dimensions and then placed onto a black canvas with the original size.
    """
    original_width, original_height = original_size
    left, top, right, bottom = crop_box
    crop_width = right - left
    crop_height = bottom - top

    original_dtype = mask.dtype
    if mask.shape != (original_height, original_width):
        mask = cv2.resize(
            mask.astype(np.uint8),
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST
        ).astype(original_dtype)

    restored_crop_mask = cv2.resize(
        mask.astype(np.uint8),
        (crop_width, crop_height),
        interpolation=cv2.INTER_NEAREST
    ).astype(original_dtype)

    padded_mask = np.zeros((original_height, original_width), dtype=original_dtype)
    padded_mask[top:bottom, left:right] = restored_crop_mask

    return padded_mask

def _convert_tif_to_jpg_array(img_np):
    """Shared 8-bit min-max normalization used for every TIFF->JPG conversion in this module."""
    img_np = img_np.astype(float)
    img_min = img_np.min()
    img_max = img_np.max()

    if img_max > img_min:
        img_scaled = (img_np - img_min) / (img_max - img_min) * 255
    else:
        img_scaled = np.zeros_like(img_np)  # Flat/empty image: avoid silently passing raw values through.

    return Image.fromarray(img_scaled.astype(np.uint8))


def run_deep_learning_segmentation(
    checkpoint,
    base_dir,
    project_dir,
    days,
    batches,
    crop_days=None,
    run_name=None,
    model_config_path=None,
    device="cpu",
):
    """
    Run segmentation inference and optionally crop the center of selected days before inference.

    When a day is included in ``crop_days``, the centered crop of the original image is
    first resized back to the original size and sent to ``inference_model``. The predicted
    mask is then resized back into the crop window and padded with black pixels so that the
    saved output matches the original image size.

    ``model_config_path`` defaults to whichever single .py file sits next to the
    checkpoint, so each checkpoint stays paired with its own config instead of
    every run sharing one hardcoded config path. TIFF->JPG conversions are written
    to ``processed_data`` rather than mutating the raw_data folder in place.
    """
    if Config is None or inference_model is None or init_model is None:
        raise ImportError(
            "run_deep_learning_segmentation requires mmengine and mmsegmentation. "
            "Install those packages or use the classical segmentation functions."
        )

    resolved_config_path = _resolve_model_config_path(checkpoint, model_config_path)

    # Load the model once per run rather than once per (day, batch) pair.
    cfg = Config.fromfile(resolved_config_path)
    cfg.load_from = checkpoint
    cfg.model.decode_head.norm_cfg = dict(type='BN', requires_grad=True)
    model = init_model(cfg, checkpoint=checkpoint, device=device)

    crop_days = {str(crop_day) for crop_day in (crop_days or [])}

    print(f"\n🤖 Automated Segmentation Started.")
    for day in days:
        print(f"- Processing Day {day}")
        crop_current_day = str(day) in crop_days
        for batch in batches:
            raw_data_dir = opj(base_dir, project_dir, 'raw_data', f'Day {day}')

            clean_folder_names(raw_data_dir, batch)

            raw_data_dir = opj(base_dir, project_dir, 'raw_data', f'Day {day}', batch)
            processed_data_dir = opj(base_dir, project_dir, 'processed_data', f'Day {day}', batch)

            if not os.path.isdir(raw_data_dir):
                print(f"⚠️  Skipping {batch}, Day {day}: raw data folder not found at {raw_data_dir}")
                continue

            outdirs = {
                'inference_npy': opj(processed_data_dir, 'inference_npy'),
                'inference_jpg': opj(processed_data_dir, 'inference_jpg'),
                'converted_jpg': opj(processed_data_dir, 'converted_jpg'),
            }
            for directory in outdirs.values():
                os.makedirs(directory, exist_ok=True)

            failed_images = []

            # Convert TIFF images to JPG if needed. Written to processed_data instead of
            # raw_data so the collaborator-provided raw folder is never mutated.
            for img_loc in sorted(os.listdir(raw_data_dir)):
                if not img_loc.lower().endswith(('.tif', '.tiff')):
                    continue
                img_path = opj(raw_data_dir, img_loc)
                try:
                    with Image.open(img_path) as img:
                        img_8bit = _convert_tif_to_jpg_array(np.array(img))
                    jpg_path = opj(outdirs['converted_jpg'], os.path.splitext(img_loc)[0] + '.jpg')
                    img_8bit.convert("RGB").save(jpg_path, "JPEG")
                except Exception as exc:
                    print(f"⚠️  Failed to convert {img_path}: {exc}")
                    failed_images.append({"Image": img_loc, "Stage": "tif_to_jpg", "Error": str(exc)})

            # Build the set of JPGs to run inference on: native JPGs in raw_data take
            # priority over a converted copy of the same stem.
            jpg_candidates = {}
            for img_loc in sorted(os.listdir(raw_data_dir)):
                if img_loc.lower().endswith(('.jpg', '.jpeg')):
                    jpg_candidates[os.path.splitext(img_loc)[0]] = opj(raw_data_dir, img_loc)
            if os.path.isdir(outdirs['converted_jpg']):
                for img_loc in sorted(os.listdir(outdirs['converted_jpg'])):
                    stem = os.path.splitext(img_loc)[0]
                    jpg_candidates.setdefault(stem, opj(outdirs['converted_jpg'], img_loc))

            # Perform inference
            for img_id, img in sorted(jpg_candidates.items()):
                try:
                    if crop_current_day:
                        with Image.open(img) as original_img:
                            original_img = original_img.convert("RGB")
                            original_size = original_img.size
                            crop_box = _get_center_crop_box(*original_size, keep_fraction=0.90)
                            cropped_img = original_img.crop(crop_box)
                            resized_cropped_img = _resize_image_to_original_size(cropped_img, original_size)

                        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_img_file:
                            temp_img_path = temp_img_file.name

                        try:
                            resized_cropped_img.save(temp_img_path, "PNG")
                            result = inference_model(model, temp_img_path)
                        finally:
                            if os.path.exists(temp_img_path):
                                os.remove(temp_img_path)

                        pred = result.pred_sem_seg.data[0].cpu().numpy()
                        pred = _restore_resized_crop_mask_to_original_size(pred, original_size, crop_box)
                    else:
                        result = inference_model(model, img)
                        pred = result.pred_sem_seg.data[0].cpu().numpy()

                    # Save masks
                    np.save(opj(outdirs['inference_npy'], f'{img_id}.npy'), pred)
                    Image.fromarray((pred * 255).astype(np.uint8)).save(opj(outdirs['inference_jpg'], f'{img_id}.jpg'))
                except Exception as exc:
                    print(f"⚠️  Failed to run inference on {img}: {exc}")
                    failed_images.append({"Image": img_id, "Stage": "inference", "Error": str(exc)})

            if failed_images:
                pd.DataFrame(failed_images).to_csv(
                    opj(processed_data_dir, "segmentation_errors.csv"), index=False
                )

            _write_run_manifest(processed_data_dir, "deep_learning_inference", {
                "run_name": run_name,
                "checkpoint": checkpoint,
                "model_config_path": resolved_config_path,
                "device": device,
                "crop_applied": crop_current_day,
                "crop_keep_fraction": 0.90 if crop_current_day else None,
                "images_attempted": len(jpg_candidates),
                "images_failed": len(failed_images),
            })

def apply_erosion(mask, kernel_size=(10, 10), iterations=1):
    """
    Apply subtle erosion to the binary mask to refine structures slightly.
    The kernel_size determines the size of the structuring element, and iterations
    control the number of times erosion is applied.

    A degenerate kernel_size (any dimension <= 0) or iterations <= 0 is treated
    as "erosion disabled" and returns the mask unchanged (cast to uint8), matching
    apply_dilation's existing no-op behavior at iterations=0. Without this guard,
    cv2.erode silently falls back to its default 3x3 structuring element when
    given an empty kernel (e.g. kernel_size=(0, 0)), which still shrinks the mask
    instead of skipping erosion as intended.
    """
    if iterations <= 0 or min(kernel_size) <= 0:
        return mask.astype(np.uint8)
    kernel = np.ones(kernel_size, np.uint8)  # Create a square kernel
    eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=iterations)
    return eroded_mask

def apply_dilation(mask, initial_kernel_size=(10, 10), iterations=10, reduction_factor=0.8):
    """
    Apply progressive dilation to the binary mask, reducing the kernel size on each iteration.
    
    - initial_kernel_size: Starting kernel size (width, height).
    - iterations: Number of dilation steps.
    - reduction_factor: Multiplier to decrease kernel size per iteration (e.g., 0.7 reduces it by 30% per step).
    """
    dilated_mask = mask.copy()

    kernel_size = np.array(initial_kernel_size)

    for i in range(iterations):
        # Ensure kernel size does not go below 1x1
        kernel_size = np.maximum((kernel_size * reduction_factor).astype(int), (1, 1))
        kernel = np.ones(tuple(kernel_size), np.uint8)

        # Apply dilation with the reduced kernel
        dilated_mask = cv2.dilate(dilated_mask.astype(np.uint8), kernel, iterations=1)

    return dilated_mask

def fill_black_holes(mask):
    """
    Fill black holes (internal gaps) within white structures in a binary mask.
    This helps restore continuous structures after erosion and dilation.

    Uses ``_find_object_contours`` (RETR_CCOMP, top-level contours only)
    rather than raw ``RETR_EXTERNAL``: if the mask contains any enclosing
    foreground shape (e.g. a well-boundary ring), ``RETR_EXTERNAL`` returns
    only that ring's outer boundary as a single contour, and filling it solid
    would paint over the entire enclosed region -- including genuine
    background gaps between the ring and real objects inside it -- rather
    than just the intended small internal gaps.
    """
    mask_filled = mask.copy()

    contours = _find_object_contours(mask.astype(np.uint8))

    # Fill each contour to remove black holes
    cv2.drawContours(mask_filled, contours, -1, (255), thickness=cv2.FILLED)

    return mask_filled

def apply_filtering_to_segmentations(
    base_dir,
    project_dir,
    days,
    batches,
    pixel_to_micron_x,
    pixel_to_micron_y,
    run_name=None,
    min_area=10000,
    max_area=None,
    min_circularity=0.1,
    erosion_kernel_size=(0, 0),
    max_objects_before_circularity_pick=1,
    dilation_initial_kernel_size=(0, 0),
    dilation_iterations=0,
    dilation_reduction_factor=0.8,
):
    """
    Applies post-inference filtering to the segmentation masks in `inference_jpg/`
    and saves the filtered masks in:
    - `inference_filtered/` (JPG images)
    - `inference_filtered_npy/` (Numpy array masks)
    - `inference_comparison/` (Comparison images)

    Defaults (min_area=10000, min_circularity=0.1, erosion/dilation disabled,
    max_objects_before_circularity_pick=1) are chosen by grid search against
    split-0's held-out VALIDATION set (see the Organotchi-manuscript repo's
    train_and_test/tune_classical_filter.py), not hand-picked -- the previous defaults
    (min_area=30000, min_circularity=0.3, erosion/dilation kernel (10,10))
    were found to make aggregate Dice/IoU on the TEST set *worse* than no
    filtering at all (mean Dice 0.963 raw vs. 0.888 filtered), driven mostly
    by real, correctly-shaped early-timepoint (Day 2) organoids whose area
    fell under the old 30,000 um^2 floor and were discarded outright. The
    tuned defaults instead improve on the (untouched-during-tuning) test set:
    mean Dice 0.963 raw vs. 0.967 filtered, mean IoU 0.933 vs. 0.941, and
    only 1/286 images end up with no surviving structure (down from 21).
    Re-run tune_classical_filter.py if the checkpoint or dataset changes
    enough that these defaults might no longer be optimal.

    Filter order: (1) remove objects too small to be a real structure (by
    area, BEFORE erosion, so a legitimately-sized object is judged on its
    actual size rather than its post-erosion size -- an upper area bound is
    optional, off by default, since real organoids legitimately grow larger
    over the developmental timeline); (2) erosion; (3) remove structures with
    low circularity; (4) if more than max_objects_before_circularity_pick
    structures still remain, keep only the single most circular one
    (otherwise every surviving structure is kept, e.g. 1 or 2); (5) dilation;
    (6) hole-filling. See remove_small_objects_by_area,
    remove_low_circularity_objects, and keep_most_circular_if_crowded for
    each step's own docstring.
    """
    print(f"\n🪄 Filtering to segmentations initiated.")
    for day in days:
        print(f"- Processing Day {day}")
        for batch in batches:
            processed_data_dir = opj(base_dir, project_dir, 'processed_data', f'Day {day}', batch)
            inference_jpg_dir = opj(processed_data_dir, 'inference_jpg')
            filtered_jpg_dir = opj(processed_data_dir, 'inference_filtered_jpg')
            filtered_npy_dir = opj(processed_data_dir, 'inference_filtered_npy')
            comparison_dir = opj(processed_data_dir, 'inference_comparison')

            if not os.path.isdir(inference_jpg_dir):
                print(f"⚠️  Skipping {batch}, Day {day}: {inference_jpg_dir} not found.")
                continue

            os.makedirs(filtered_jpg_dir, exist_ok=True)
            os.makedirs(filtered_npy_dir, exist_ok=True)
            os.makedirs(comparison_dir, exist_ok=True)

            failed_images = []

            for img_loc in sorted(os.listdir(inference_jpg_dir)):
                if not img_loc.lower().endswith('.jpg'):
                    continue  # Skip non-JPG files

                img_path = opj(inference_jpg_dir, img_loc)
                img_id = os.path.splitext(img_loc)[0]

                try:
                    # Load the segmentation mask
                    mask = np.array(Image.open(img_path).convert('L'))  # Convert to grayscale
                    mask = mask > 128  # Binarize (thresholding)

                    # (1) Remove objects too small (or, optionally, too large) to be real,
                    # judged on their actual size before erosion shrinks them.
                    size_filtered_mask = remove_small_objects_by_area(
                        mask, pixel_to_micron_x, pixel_to_micron_y,
                        min_area=min_area, max_area=max_area,
                    )

                    # (2) Apply subtle erosion to refine the surviving structures.
                    eroded_mask = apply_erosion(size_filtered_mask, kernel_size=erosion_kernel_size)

                    # (3) Remove structures with low circularity.
                    circularity_filtered_mask = remove_low_circularity_objects(
                        eroded_mask, pixel_to_micron_x, pixel_to_micron_y,
                        circularity_threshold=min_circularity,
                    )

                    # (4) If more than max_objects_before_circularity_pick structures remain,
                    # narrow down to the single most circular one.
                    filtered_mask, structure_count = keep_most_circular_if_crowded(
                        circularity_filtered_mask, pixel_to_micron_x, pixel_to_micron_y,
                        max_objects=max_objects_before_circularity_pick,
                    )

                    # (5) Dilate back out, then (6) fill internal holes.
                    filtered_mask = apply_dilation(
                        filtered_mask,
                        initial_kernel_size=dilation_initial_kernel_size,
                        iterations=dilation_iterations,
                        reduction_factor=dilation_reduction_factor,
                    )
                    filtered_mask = fill_black_holes(filtered_mask)

                    # Save the filtered mask as JPG
                    Image.fromarray(filtered_mask).save(opj(filtered_jpg_dir, f'{img_id}.jpg'))

                    # Save the filtered mask as NPY
                    np.save(opj(filtered_npy_dir, f'{img_id}.npy'), filtered_mask)

                    # Load original image for comparison
                    orig_img_path = opj(base_dir, project_dir, 'raw_data', f'Day {day}', batch, f"{img_id}.jpg")
                    orig_img = Image.open(orig_img_path) if os.path.exists(orig_img_path) else None

                    # Create a comparison plot
                    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

                    if orig_img:
                        ax[0].imshow(orig_img)
                        ax[0].set_title('Original Image')
                    else:
                        ax[0].imshow(mask, cmap='gray')
                        ax[0].set_title('Original Mask')

                    ax[0].axis('off')
                    ax[1].imshow(mask, cmap='gray')
                    ax[1].set_title('Unfiltered Mask')
                    ax[1].axis('off')
                    ax[2].imshow(filtered_mask, cmap='gray')
                    ax[2].set_title('Filtered Mask')
                    ax[2].axis('off')

                    # Save the comparison figure
                    plt.savefig(opj(comparison_dir, f'{img_id}.jpg'), bbox_inches='tight')
                    plt.close()
                except Exception as exc:
                    plt.close('all')
                    print(f"⚠️  Failed to filter {img_path}: {exc}")
                    failed_images.append({"Image": img_id, "Stage": "dl_filtering", "Error": str(exc)})

            if failed_images:
                errors_path = opj(processed_data_dir, "segmentation_errors.csv")
                if os.path.exists(errors_path):
                    pd.concat(
                        [pd.read_csv(errors_path), pd.DataFrame(failed_images)],
                        ignore_index=True,
                    ).to_csv(errors_path, index=False)
                else:
                    pd.DataFrame(failed_images).to_csv(errors_path, index=False)

            _write_run_manifest(processed_data_dir, "deep_learning_filtering", {
                "run_name": run_name,
                "pixel_to_micron_x": pixel_to_micron_x,
                "pixel_to_micron_y": pixel_to_micron_y,
                "min_area": min_area,
                "max_area": max_area,
                "min_circularity": min_circularity,
                "erosion_kernel_size": erosion_kernel_size,
                "dilation_initial_kernel_size": dilation_initial_kernel_size,
                "dilation_iterations": dilation_iterations,
                "dilation_reduction_factor": dilation_reduction_factor,
                "images_failed": len(failed_images),
            })

def _get_grid_position(index, grid_shape, tile_order="snake"):
    """Return row/column placement for microscope tiles in a stitched grid."""
    rows, cols = grid_shape

    if tile_order in {"snake", "serpentine", "d_order"}:
        row = index // cols
        col_in_row = index % cols
        col = col_in_row if row % 2 == 0 else cols - 1 - col_in_row
    elif tile_order in {"row", "row_major"}:
        row = index // cols
        col = index % cols
    elif tile_order in {"column", "column_major"}:
        col = index // rows
        row = index % rows
    elif tile_order in {"column_snake", "vertical_snake"}:
        col = index // rows
        row_in_col = index % rows
        row = row_in_col if col % 2 == 0 else rows - 1 - row_in_col
    else:
        raise ValueError(
            "tile_order must be one of: snake, row_major, column_major, column_snake"
        )

    return row, col


def _natural_image_sort_key(image_path):
    """Sort image names by numeric chunks when microscope files are not zero-padded."""
    filename = os.path.basename(image_path)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", filename)
    ]


def _collect_stitch_input_images(folder_path):
    """Collect common image formats for tile stitching."""
    image_paths = []
    for extension in ("jpg", "jpeg", "png", "tif", "tiff"):
        image_paths.extend(glob.glob(opj(folder_path, f"*.{extension}")))
        image_paths.extend(glob.glob(opj(folder_path, f"*.{extension.upper()}")))

    return sorted(set(image_paths), key=_natural_image_sort_key)


def _normalize_tile_grid_shape(tile_grid, default=(2, 2)):
    """Normalize tile-grid values such as '2x2', [2, 2], or {'rows': 2, 'cols': 2}."""
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


def _find_tiled_image_source_folder(raw_data_dir, project, batch, num_plates, tiles_per_image):
    """Find the raw microscope folder that contains tiled brightfield acquisitions."""
    candidate_names = [
        f"{project}_{batch}in{num_plates}",
        f"{project} {batch}in{num_plates}",
    ]

    if num_plates == 1:
        candidate_names.extend([
            project,
            f"{project}_N0000000",
            f"{project} N0000000",
        ])
    else:
        candidate_names.extend([
            f"{project}_{batch}_N0000000",
            f"{project} {batch} N0000000",
        ])

    candidate_paths = [
        opj(raw_data_dir, candidate_name)
        for candidate_name in candidate_names
    ]

    if os.path.exists(raw_data_dir):
        folder_prefixes = [f"{project}_{batch}", f"{project} {batch}"]
        if num_plates == 1:
            folder_prefixes.extend([project])

        for folder_name in os.listdir(raw_data_dir):
            folder_path = opj(raw_data_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            if any(folder_name.startswith(prefix) for prefix in folder_prefixes):
                candidate_paths.append(folder_path)

    for folder_path in sorted(set(candidate_paths), key=_natural_image_sort_key):
        if not os.path.isdir(folder_path):
            continue

        if len(_collect_stitch_input_images(folder_path)) >= tiles_per_image:
            return folder_path

    return None


# PIL modes used for higher-bit-depth-than-8 pixel data (16-bit and 32-bit
# TIFFs commonly load as one of these rather than "L"/"RGB").
_HIGH_BIT_DEPTH_MODES = {"I", "I;16", "I;16B", "I;16L", "I;16N", "F"}


def _load_tile_images_for_stitching(image_paths):
    """Open every tile in one stitch group, normalizing high-bit-depth (e.g.
    16-bit TIFF) tiles the same way the rest of the pipeline does.

    PIL's plain ``Image.convert("RGB")`` does *not* rescale a 16-bit image's
    dynamic range -- it treats pixel values as if they were already 8-bit,
    which silently clips a real sensor's ~0-65535 range to 255 for nearly
    every pixel. A tile whose actual contrast lives between, say, 20000 and
    50000 becomes a single solid white rectangle once naively downcast,
    destroying every organoid in it before contour extraction ever sees the
    stitched image -- confirmed directly: a synthetic 16-bit tile with a
    clearly darker organoid core became uniformly 255 end to end.

    The rescale uses ONE min/max shared across every tile in the group, not
    a separate min/max per tile: rescaling each tile independently would
    reintroduce exactly the kind of tile-to-tile brightness inconsistency
    that trips up a single downstream threshold.

    Non-high-bit-depth tiles (already 8-bit, or any other PIL mode such as
    palette) are loaded exactly as before -- ``img.convert("RGB")`` on the
    freshly opened image, untouched by the rescale path.
    """
    raw_arrays_by_path = {}
    modes_by_path = {}
    for img_path in image_paths:
        with Image.open(img_path) as img:
            modes_by_path[img_path] = img.mode
            if img.mode in _HIGH_BIT_DEPTH_MODES:
                raw_arrays_by_path[img_path] = np.array(img)

    group_min = group_max = None
    if raw_arrays_by_path:
        pooled = np.concatenate([arr.ravel().astype(float) for arr in raw_arrays_by_path.values()])
        group_min, group_max = pooled.min(), pooled.max()

    images = []
    for img_path in image_paths:
        if img_path in raw_arrays_by_path:
            arr_f = raw_arrays_by_path[img_path].astype(float)
            if group_max > group_min:
                scaled = (arr_f - group_min) / (group_max - group_min) * 255
            else:
                scaled = np.zeros_like(arr_f)
            images.append(Image.fromarray(scaled.astype(np.uint8)).convert("RGB"))
        else:
            with Image.open(img_path) as img:
                images.append(img.convert("RGB"))

    return images


def stitch_microscope_tile_grid(
    image_paths,
    output_path,
    tile_grid=(2, 2),
    overlap_x=32,
    overlap_y=31,
    tile_order="serpentine",
):
    """Stitch a tiled microscope acquisition into a single brightfield image."""
    grid_shape = _normalize_tile_grid_shape(tile_grid)
    expected_tiles = grid_shape[0] * grid_shape[1]

    if len(image_paths) != expected_tiles:
        print(f"Skipping set, expected {expected_tiles} images but got {len(image_paths)}")
        return False

    images = _load_tile_images_for_stitching(image_paths)

    widths, heights = zip(*(img.size for img in images))
    if len(set(widths)) > 1 or len(set(heights)) > 1:
        print("Skipping set, images have different sizes.")
        return False

    tile_width, tile_height = images[0].size
    if overlap_x >= tile_width or overlap_y >= tile_height:
        print("Skipping set, overlap must be smaller than the tile dimensions.")
        return False

    rows, cols = grid_shape
    step_x = tile_width - overlap_x
    step_y = tile_height - overlap_y
    stitched_width = tile_width + step_x * (cols - 1)
    stitched_height = tile_height + step_y * (rows - 1)

    stitched_image = Image.new("RGB", (stitched_width, stitched_height))

    for index, image in enumerate(images):
        row, col = _get_grid_position(index, grid_shape, tile_order=tile_order)
        stitched_image.paste(image, (col * step_x, row * step_y))

    stitched_image.save(output_path)
    print(f"Saved: {output_path}")
    return True


def prepare_classical_segmentation_images(
    base_dir,
    project_dir,
    days,
    batch_names,
    project_batches,
    tile_grid=(2, 2),
    stitched_images_per_batch=24,
    overlap_x=32,
    overlap_y=31,
    tile_order="serpentine",
):
    """
    Prepare tiled brightfield images for classical segmentation.

    The stitched outputs are written to ``raw_data/Day <day>/<batch>/Tile*.jpg``.
    If the raw microscope tiles are already stored directly in that batch folder,
    stitched outputs are written to ``raw_data/Day <day>/<batch>/stitched/Tile*.jpg``
    so the source tiles are not overwritten.
    ``tile_grid`` controls how many raw microscope tiles are grouped per image.
    """
    grid_shape = _normalize_tile_grid_shape(tile_grid)
    tiles_per_stitched_image = grid_shape[0] * grid_shape[1]

    for day in days:
        raw_data_dir = opj(base_dir, project_dir, 'raw_data', f'Day {day}')
        os.makedirs(opj(base_dir, project_dir, 'processed_data', f'Day {day}'), exist_ok=True)

        for project, num_plates in zip(batch_names, project_batches):
            project_output_dir = opj(raw_data_dir, project)
            os.makedirs(project_output_dir, exist_ok=True)

            if tiles_per_stitched_image == 1:
                print(f"Skipping stitching for {project}, Day {day}: tile_grid is 1x1.")
                continue

            for batch in range(1, num_plates + 1):
                folder_path = _find_tiled_image_source_folder(
                    raw_data_dir,
                    project,
                    batch,
                    num_plates,
                    tiles_per_stitched_image,
                )

                if not folder_path:
                    print(f"Warning: No tiled source folder found for {project}, batch {batch}.")
                    continue

                output_dir = project_output_dir
                if os.path.abspath(folder_path) == os.path.abspath(project_output_dir):
                    output_dir = opj(project_output_dir, "stitched")
                    os.makedirs(output_dir, exist_ok=True)

                image_list = _collect_stitch_input_images(folder_path)
                if not image_list:
                    print(f"Warning: No images found in {folder_path}")
                    continue

                if len(image_list) % tiles_per_stitched_image != 0:
                    print(
                        f"Warning: {folder_path} contains {len(image_list)} images, "
                        f"which is not a multiple of {tiles_per_stitched_image}."
                    )

                image_groups = [
                    image_list[i:i + tiles_per_stitched_image]
                    for i in range(0, len(image_list), tiles_per_stitched_image)
                ]
                start_tile = (batch - 1) * stitched_images_per_batch + 1

                for local_idx, group in enumerate(image_groups):
                    global_tile_num = start_tile + local_idx
                    output_filename = opj(
                        output_dir,
                        f"Tile{global_tile_num:06d}.jpg"
                    )

                    stitch_microscope_tile_grid(
                        group,
                        output_filename,
                        tile_grid=grid_shape,
                        overlap_x=overlap_x,
                        overlap_y=overlap_y,
                        tile_order=tile_order,
                    )

def _find_object_contours(mask):
    """Find candidate object contours, excluding anything enclosed by another
    foreground region (e.g. a well-boundary ring).

    ``cv2.RETR_EXTERNAL`` only returns hierarchy-level-0 contours: if the
    image has any enclosing foreground shape -- most relevantly, a well
    boundary ring that thresholds as foreground -- every organoid contour
    sitting inside that ring's hole is a *deeper* hierarchy level and gets
    silently discarded, regardless of the fact that the organoids are not
    physically touching the ring at all. This was confirmed as the dominant
    cause of severe under-detection in a real crowded/ringed well image:
    ``RETR_EXTERNAL`` returned a single contour (the ring), while
    ``RETR_CCOMP`` correctly recovered every organoid inside it.

    ``RETR_CCOMP`` returns a 2-level hierarchy (outer boundaries and their
    holes); keeping only contours with no parent (``hierarchy[3] == -1``)
    gives every real foreground blob's outer boundary -- including ones
    nested inside another blob's hole -- while excluding hole boundaries
    (e.g. a cystic organoid's translucent center, or the ring's own hole)
    that would otherwise be miscounted as separate objects.
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    return [contour for contour, h in zip(contours, hierarchy[0]) if h[3] == -1]




def _filter_single_organoid_contours(mask, pixel_to_micron_x, pixel_to_micron_y, max_area=3000000, min_area=30000, min_circularity=0.2, debug_capture=None):
    """
    Filters contours based on circularity and area.

    If multiple objects pass the filter, only the largest one is retained.

    Parameters:
    - mask: Binary image mask.
    - pixel_to_micron_x: Conversion factor for pixels to microns (X-axis).
    - pixel_to_micron_y: Conversion factor for pixels to microns (Y-axis).
    - max_area: Maximum object area in square microns.
    - min_area: Minimum object area in square microns.
    - min_circularity: Minimum circularity threshold.
    - debug_capture: optional dict, populated with ``"raw_contours"`` (all
      contours found), then ``"after_circularity_filter"``, ``"after_area_filter"``
      (the same two sequential checks ``_filter_multi_organoid_contours``
      uses, applied here in the same order, before the largest-only pick),
      and ``"accepted_contours"`` (the final single kept contour, if any) --
      for step-by-step pipeline inspection.

    Returns:
    - filtered_mask: Binary mask with only valid objects.
    """

    contours = _find_object_contours(mask)

    if debug_capture is not None:
        debug_capture["raw_contours"] = list(contours)

    # Apply the two checks sequentially (circularity -> area), so each
    # one's own contribution to the final count is separately inspectable
    # (see `fit_experiment_data.py`'s `--diagnose` mode).
    after_circularity = [c for c in contours if _contour_passes_circularity_filter(c, pixel_to_micron_x, pixel_to_micron_y, min_circularity)]
    if debug_capture is not None:
        debug_capture["after_circularity_filter"] = list(after_circularity)

    after_area = [c for c in after_circularity if _contour_passes_area_filter(c, pixel_to_micron_x, pixel_to_micron_y, min_area, max_area)]
    if debug_capture is not None:
        debug_capture["after_area_filter"] = list(after_area)

    valid_contours = [(c, _contour_shape_metrics(c, pixel_to_micron_x, pixel_to_micron_y)[0]) for c in after_area]

    # In single-organoid mode, keep exactly one candidate: the largest valid contour.
    if valid_contours:
        valid_contours = [max(valid_contours, key=lambda item: item[1])]

    if debug_capture is not None:
        debug_capture["accepted_contours"] = [contour for contour, _ in valid_contours]

    # Draw final filtered contours
    filtered_mask = np.zeros_like(mask)
    for contour, _ in valid_contours:
        cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return _keep_largest_connected_component(filtered_mask)


def _keep_largest_connected_component(mask):
    """Remove tiny disconnected artifacts from a single-organoid mask."""
    labeled_mask = measure.label(mask > 0, connectivity=LABEL_CONNECTIVITY)
    if labeled_mask.max() == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    regions = measure.regionprops(labeled_mask)
    largest_region = max(regions, key=lambda region: region.area)
    cleaned_mask = np.zeros_like(mask, dtype=np.uint8)
    cleaned_mask[labeled_mask == largest_region.label] = 255

    return cleaned_mask

def _normalize_foreground_polarity(foreground_polarity):
    foreground_polarity = str(foreground_polarity).lower()
    if foreground_polarity in {"bright", "light", "white", "bright_on_dark"}:
        return "bright"
    if foreground_polarity in {"dark", "black", "dark_on_light"}:
        return "dark"
    raise ValueError("foreground_polarity must be 'dark' or 'bright'.")


def _resolve_threshold_value(img_uint8, threshold):
    """Resolve ``threshold`` to a numeric cutoff: passed through as-is unless
    it's the literal string ``"otsu"``, in which case it's computed fresh
    from this image's own grayscale histogram (``skimage.filters.threshold_otsu``).
    A fixed numeric threshold assumes stable, uniform illumination across the
    whole timecourse; Otsu instead adapts per image, which matters for
    projects where brightness genuinely drifts day to day (confirmed on
    AnnaRaic -- a fixed threshold detected organoids on Day 0 but missed them
    entirely on Day 1)."""
    if isinstance(threshold, str) and threshold.strip().lower() == "otsu":
        return float(threshold_otsu(img_uint8))
    return float(threshold)


def _global_threshold_mask(img_uint8, threshold, foreground_polarity):
    threshold_value = _resolve_threshold_value(img_uint8, threshold)
    if foreground_polarity == "bright":
        return np.where(img_uint8 > threshold_value, 255, 0).astype(np.uint8)
    return np.where(img_uint8 < threshold_value, 255, 0).astype(np.uint8)


def _apply_center_fraction_mask(mask, center_fraction):
    """Zero out every foreground pixel outside a circle concentric with the
    image, of diameter ``center_fraction`` times the image's **width**.
    ``center_fraction=None`` (or >= 1.0) disables this and returns ``mask``
    unchanged.

    Width, not the shorter dimension, is the right reference here: in these
    stitched multi-well images the well itself is wider than it is tall, and
    its physical diameter in pixels is close to the image's width -- using
    width means ``center_fraction`` close to 1.0 tracks the well's actual
    edge, rather than leaving a wide unaccounted margin on the image's long
    axis the way ``min(height, width)`` would (confirmed on Zeynep-CP).

    Approximates the physical well boundary as a circle rather than
    detecting the well's actual edge per image -- cheaper and more robust,
    and correct as long as a batch's magnification/tile grid keeps the
    well's apparent size in-frame consistent across its images (true within
    any one batch here). Pixels are dropped outright, right after
    thresholding, rather than scored per detected object later: a
    well-boundary ring's own contour traces along the image edge however
    large its enclosed area is, so removing everything outside the circle
    before contour extraction ever runs means the ring never survives to be
    misjudged as an organoid downstream, instead of relying on
    circularity/area filtering to reject it after the fact.
    """
    if center_fraction is None or center_fraction >= 1.0:
        return mask
    height, width = mask.shape[:2]
    center_y, center_x = height / 2.0, width / 2.0
    radius = center_fraction * width / 2.0
    yy, xx = np.ogrid[:height, :width]
    distance_from_center = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    return np.where(distance_from_center <= radius, mask, 0).astype(mask.dtype)


def _apply_morphological_closing(mask, close_kernel_size):
    """Bridge small gaps in an otherwise near-complete foreground boundary
    via a dilate-then-erode morphological closing pass (``cv2.MORPH_CLOSE``),
    before contour extraction ever runs. ``close_kernel_size`` of falsy
    (``None``, ``0``, or ``(0, 0)``) disables this and returns ``mask``
    unchanged; an int is treated as a square kernel of that size, matching
    the single-number convention this project's earlier, pre-uniformization
    pipeline used for the same parameter.

    Motivating case: a cystic organoid's rim sometimes doesn't fully
    threshold all the way around (uneven staining/focus across part of the
    boundary), leaving an open, gapped ring rather than a closed loop. An
    *open* contour's ``cv2.drawContours(..., thickness=cv2.FILLED)`` only
    fills the thin sliver the contour's own points trace out, not the disk
    it would enclose if closed -- so the organoid is either dropped outright
    (an open C-shape scores very low on circularity) or, if it survives,
    saved as a thin crescent rather than its true extent. Closing the gap
    here, before contour extraction, converts it back into one closed loop;
    everything downstream (area/circularity filtering, and the existing
    filled-contour draw at final selection) then treats it as the solid
    object it actually is -- no separate hole-filling step is needed, since
    filling a topologically closed contour is exactly what that final draw
    already does. Confirmed directly: closing a synthetic gapped ring, then
    running it through contour extraction and a filled draw, fully recovers
    the disk (center pixel included), while skipping closing leaves either
    nothing (fails circularity) or just the thin crescent.

    A kernel large enough to bridge the gaps actually seen is also large
    enough to bridge the *spacing* between two genuinely separate, adjacent
    organoids -- keep it just big enough for the former, checked visually
    (``fit_experiment_data.py --sweep close_kernel_size``), not larger.
    """
    if not close_kernel_size:
        return mask
    kernel_size = tuple(close_kernel_size) if isinstance(close_kernel_size, (tuple, list)) else (int(close_kernel_size), int(close_kernel_size))
    if kernel_size == (0, 0):
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def _filter_regions_by_min_area(mask, pixel_to_micron_x, pixel_to_micron_y, min_area):
    """Drop connected regions below ``min_area`` (in um^2) right after
    thresholding, before contour extraction/circularity filtering ever sees
    them.

    In a heavily speckled image, raw thresholding can produce many small
    noise regions nowhere close to a real organoid's size -- applying the
    real, calibrated ``min_area`` cutoff immediately after thresholding
    removes that noise before it has any chance to masquerade as a real
    organoid.

    ``cv2.contourArea`` (the enclosed/would-be-filled area) is used to decide
    which contours survive -- not the region's raw foreground pixel count --
    so a real but currently thin/hollow shape (a cystic organoid's bare
    core, or a well-boundary ring) is judged by how big it would be once
    whole, not by how little of that footprint is actually thresholded yet.
    But surviving contours must **not** be redrawn filled: a contour here can
    be a well-boundary ring, whose "filled interior" is the rest of the well
    -- background gaps *and every organoid inside it* -- not a hole
    belonging to one object. Force-filling it here would silently merge
    everything geometrically inside the ring into one solid blob before
    contour extraction ever gets a chance to run on the real, unmodified
    thresholded pixels. Instead, only the pixels already thresholded as
    foreground within a surviving contour's footprint are kept, exactly as
    thresholded -- rings stay rings, hollow cores stay hollow, only the
    decision of which contours survive uses the would-be-filled area.
    """
    contours = _find_object_contours(mask)
    return _keep_contours_matching(
        mask, contours,
        lambda contour: _pixel_area_to_micron(cv2.contourArea(contour), pixel_to_micron_x, pixel_to_micron_y) > min_area,
    )


def _filter_regions_by_max_area(mask, pixel_to_micron_x, pixel_to_micron_y, max_area):
    """Drop connected regions at/above ``max_area`` (in um^2), right after
    ``_filter_regions_by_min_area`` -- same early stage, same rationale.

    Note this also rejects a genuine cluster of several touching/merged
    organoids traced as one oversized contour -- there is no touching-object
    split step to recover them into individual organoids, so an oversized
    contour here is simply dropped.
    """
    contours = _find_object_contours(mask)
    return _keep_contours_matching(
        mask, contours,
        lambda contour: _pixel_area_to_micron(cv2.contourArea(contour), pixel_to_micron_x, pixel_to_micron_y) < max_area,
    )


def _keep_contours_matching(mask, contours, predicate):
    """Shared reconstruction step for ``_filter_regions_by_min_area`` /
    ``_filter_regions_by_max_area``: keep only the original foreground
    pixels of contours where ``predicate(contour)`` is true, never
    force-filling a surviving contour's interior (see
    ``_filter_regions_by_min_area``'s docstring for why that matters)."""
    filtered = np.zeros_like(mask)
    for contour in contours:
        if not predicate(contour):
            continue
        contour_footprint = np.zeros_like(mask)
        cv2.drawContours(contour_footprint, [contour], -1, 255, thickness=cv2.FILLED)
        keep = (contour_footprint > 0) & (mask > 0)
        filtered[keep] = 255
    return filtered


def _create_classical_binary_mask(
    img_array,
    threshold=128,
    foreground_polarity="dark",
    pixel_to_micron_x=None,
    pixel_to_micron_y=None,
    min_area=None,
    max_area=None,
    center_fraction=None,
    close_kernel_size=None,
    debug_capture=None,
):
    """Create a binary foreground mask for dark-on-light or bright-on-dark images.

    A single global cutoff (``threshold``) separates foreground from
    background; ``foreground_polarity`` ("dark" or "bright") says which side
    of that cutoff the organoid is on.

    ``center_fraction`` (see ``_apply_center_fraction_mask``), if given,
    is applied immediately after thresholding -- before ``min_area``/
    ``max_area`` -- to drop a well-boundary ring (or any other large
    edge-of-frame artifact) before it can reach contour extraction at all.

    ``close_kernel_size`` (see ``_apply_morphological_closing``) runs next,
    still before ``min_area``/``max_area``: it bridges small gaps in an
    otherwise near-complete object boundary (e.g. an incompletely-thresholded
    cystic organoid rim) into one closed loop, so area filtering judges the
    object's real, whole shape rather than a fragment of it.

    ``min_area`` filtering (see ``_filter_regions_by_min_area``) happens
    next, followed by ``max_area`` filtering (see
    ``_filter_regions_by_max_area``) -- pruning obvious noise/oversized
    artifacts before contour extraction. Each of ``center_fraction``,
    ``close_kernel_size``, ``min_area``, ``max_area`` is only applied when
    given (``min_area``/``max_area`` additionally require
    ``pixel_to_micron_x``/``_y``); a direct caller that omits all four gets
    the unfiltered threshold mask, unchanged.

    ``debug_capture``, if given a dict, gets populated with the intermediate
    mask after each sub-stage (``"threshold"``, ``"after_center_fraction_mask"``,
    ``"after_closing"``, ``"after_min_area_filter"``, ``"after_max_area_filter"``)
    for step-by-step pipeline inspection (see ``fit_experiment_data.py``'s
    ``--diagnose`` mode), plus ``"threshold_value"``, the actual numeric
    cutoff used -- always worth checking when ``threshold="otsu"``, since
    it's resolved fresh per image rather than being the fixed number a
    caller passed in. Has no effect on the returned mask.
    """
    foreground_polarity = _normalize_foreground_polarity(foreground_polarity)
    img_uint8 = img_array.astype(np.uint8)
    if debug_capture is not None:
        debug_capture["threshold_value"] = _resolve_threshold_value(img_uint8, threshold)
    binary_mask = _global_threshold_mask(img_uint8, threshold, foreground_polarity)

    if debug_capture is not None:
        debug_capture["threshold"] = binary_mask.copy()

    if center_fraction is not None:
        binary_mask = _apply_center_fraction_mask(binary_mask, center_fraction)
        if debug_capture is not None:
            debug_capture["after_center_fraction_mask"] = binary_mask.copy()

    if close_kernel_size:
        binary_mask = _apply_morphological_closing(binary_mask, close_kernel_size)
        if debug_capture is not None:
            debug_capture["after_closing"] = binary_mask.copy()

    if min_area is not None and pixel_to_micron_x is not None and pixel_to_micron_y is not None:
        binary_mask = _filter_regions_by_min_area(binary_mask, pixel_to_micron_x, pixel_to_micron_y, min_area)
        if debug_capture is not None:
            debug_capture["after_min_area_filter"] = binary_mask.copy()

    if max_area is not None and pixel_to_micron_x is not None and pixel_to_micron_y is not None:
        binary_mask = _filter_regions_by_max_area(binary_mask, pixel_to_micron_x, pixel_to_micron_y, max_area)
        if debug_capture is not None:
            debug_capture["after_max_area_filter"] = binary_mask.copy()

    return binary_mask.astype(np.uint8)




def _convert_tif_to_jpg_for_segmentation(image_path):
    """Convert TIFF images to an 8-bit JPG sibling before segmentation.

    Shares ``_convert_tif_to_jpg_array`` with the deep-learning pipeline so
    16-bit TIFF normalization behaves identically in both pipelines.
    """
    extension = os.path.splitext(image_path)[1].lower()
    if extension not in {".tif", ".tiff"}:
        return image_path

    jpg_path = os.path.splitext(image_path)[0] + ".jpg"
    with Image.open(image_path) as img:
        img_8bit = _convert_tif_to_jpg_array(np.array(img))

    img_8bit.convert("RGB").save(jpg_path, "JPEG")

    return jpg_path


def _segment_single_organoid_brightfield_image(
    image_path,
    pixel_to_micron_x,
    pixel_to_micron_y,
    threshold=128,
    max_area=3000000,
    min_area=30000,
    min_circularity=0.2,
    foreground_polarity="dark",
    center_fraction=None,
    close_kernel_size=None,
    debug_capture=None,
):
    """
    Segment an image based on grayscale intensity and apply contour filtering.

    Parameters:
    - image_path: Path to the input image.
    - pixel_to_micron_x: Conversion factor for pixels to microns (X-axis).
    - pixel_to_micron_y: Conversion factor for pixels to microns (Y-axis).
    - threshold: Grayscale threshold for binarization.
    - max_area: Maximum object area in square microns.
    - min_area: Minimum object area in square microns.
    - min_circularity: Minimum circularity threshold.
    - center_fraction: if given, excludes everything outside a concentric
      circle of this fraction of the image's width (see
      ``_apply_center_fraction_mask``) -- e.g. to drop a well-boundary ring.
    - close_kernel_size: if given, bridges small gaps in an object's
      boundary via morphological closing (see
      ``_apply_morphological_closing``), before ``min_area``/``max_area``.
    - debug_capture: optional dict, populated with intermediate masks/contour
      lists from every pipeline sub-stage (see ``_create_classical_binary_mask``,
      ``_filter_single_organoid_contours``) for step-by-step pipeline
      inspection. Has no effect on the result.

    Returns:
    - filtered_mask: Binary mask with valid objects.
    """

    image_path = _convert_tif_to_jpg_for_segmentation(image_path)
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img_array = np.array(img)

    binary_mask = _create_classical_binary_mask(
        img_array,
        threshold=threshold,
        foreground_polarity=foreground_polarity,
        pixel_to_micron_x=pixel_to_micron_x,
        pixel_to_micron_y=pixel_to_micron_y,
        min_area=min_area,
        max_area=max_area,
        center_fraction=center_fraction,
        close_kernel_size=close_kernel_size,
        debug_capture=debug_capture,
    )

    filtered_mask = _filter_single_organoid_contours(
        binary_mask, pixel_to_micron_x, pixel_to_micron_y, max_area, min_area, min_circularity,
        debug_capture=debug_capture,
    )

    return filtered_mask


def _contour_shape_metrics(contour, pixel_to_micron_x, pixel_to_micron_y):
    area = _pixel_area_to_micron(cv2.contourArea(contour), pixel_to_micron_x, pixel_to_micron_y)
    perimeter = _pixel_length_to_micron(cv2.arcLength(contour, True), pixel_to_micron_x, pixel_to_micron_y)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    return area, circularity


def _contour_passes_area_filter(contour, pixel_to_micron_x, pixel_to_micron_y, min_area, max_area):
    area, _ = _contour_shape_metrics(contour, pixel_to_micron_x, pixel_to_micron_y)
    lower_ok = area > min_area if min_area is not None else True
    upper_ok = area < max_area if max_area is not None else True
    return lower_ok and upper_ok


def _contour_passes_circularity_filter(contour, pixel_to_micron_x, pixel_to_micron_y, min_circularity):
    _, circularity = _contour_shape_metrics(contour, pixel_to_micron_x, pixel_to_micron_y)
    return circularity >= min_circularity


def _filter_multi_organoid_contours(
    mask,
    pixel_to_micron_x,
    pixel_to_micron_y,
    max_area=3000000,
    min_area=30000,
    min_circularity=0.2,
    debug_capture=None,
):
    """
    Filter a binary mask while preserving all valid organoid-like contours.

    This function keeps every contour that passes the circularity and area
    filters. It is intended for fields where multiple organoids can be
    present in the same image. Touching/merged organoids are not split --
    a contour spanning several organoids is judged (and accepted or
    rejected) as one shape.

    ``debug_capture``, if given a dict, gets populated with the contour list
    at each stage -- ``"raw_contours"`` (straight out of
    ``_find_object_contours``), then ``"after_circularity_filter"``, then
    ``"after_area_filter"`` (same as ``"accepted_contours"``, the final
    result) -- for step-by-step pipeline inspection. Has no effect on the
    returned mask/count.
    """
    contours = _find_object_contours(mask)

    if debug_capture is not None:
        debug_capture["raw_contours"] = list(contours)

    # Apply the two checks sequentially (circularity -> area), so each
    # one's own contribution to the final count is separately inspectable
    # (see `fit_experiment_data.py`'s `--diagnose` mode).
    after_circularity = [c for c in contours if _contour_passes_circularity_filter(c, pixel_to_micron_x, pixel_to_micron_y, min_circularity)]
    if debug_capture is not None:
        debug_capture["after_circularity_filter"] = list(after_circularity)

    valid_contours = [c for c in after_circularity if _contour_passes_area_filter(c, pixel_to_micron_x, pixel_to_micron_y, min_area, max_area)]
    if debug_capture is not None:
        debug_capture["after_area_filter"] = list(valid_contours)

    filtered_mask = np.zeros_like(mask)
    for contour in valid_contours:
        cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

    if debug_capture is not None:
        debug_capture["accepted_contours"] = valid_contours

    return filtered_mask, len(valid_contours)


def _segment_multi_organoid_brightfield_image(
    image_path,
    pixel_to_micron_x,
    pixel_to_micron_y,
    threshold=128,
    max_area=3000000,
    min_area=30000,
    min_circularity=0.2,
    foreground_polarity="dark",
    center_fraction=None,
    close_kernel_size=None,
    debug_capture=None,
):
    """
    Segment all organoid-like objects in one stitched brightfield image.

    The binarization logic mirrors the single-organoid classical segmentation:
    foreground structures become binary mask objects, then contour filtering
    removes small, very large, or non-circular objects without reducing the
    mask to a single organoid.

    ``center_fraction``: see ``_segment_single_organoid_brightfield_image``
    -- especially relevant here, since a well-boundary ring visible in a
    multi-organoid field would otherwise sit among the real organoids as
    just another (very large, low-circularity) candidate contour.

    ``close_kernel_size``: see ``_segment_single_organoid_brightfield_image``
    -- particularly relevant for cystic organoids, whose rim sometimes
    doesn't fully threshold all the way around, leaving an open gap rather
    than a closed loop.

    ``debug_capture``, if given a dict, gets populated with intermediate
    masks/contour lists from every pipeline sub-stage (see
    ``_create_classical_binary_mask``, ``_filter_multi_organoid_contours``)
    for step-by-step pipeline inspection. Has no effect on the result.
    """
    image_path = _convert_tif_to_jpg_for_segmentation(image_path)
    with Image.open(image_path) as img:
        img_array = np.array(img.convert("L"))

    binary_mask = _create_classical_binary_mask(
        img_array,
        threshold=threshold,
        foreground_polarity=foreground_polarity,
        pixel_to_micron_x=pixel_to_micron_x,
        pixel_to_micron_y=pixel_to_micron_y,
        min_area=min_area,
        max_area=max_area,
        center_fraction=center_fraction,
        close_kernel_size=close_kernel_size,
        debug_capture=debug_capture,
    )

    filtered_mask, object_count = _filter_multi_organoid_contours(
        binary_mask,
        pixel_to_micron_x,
        pixel_to_micron_y,
        max_area=max_area,
        min_area=min_area,
        min_circularity=min_circularity,
        debug_capture=debug_capture,
    )

    return filtered_mask, object_count


def segment_classical_brightfield_image(
    image_path,
    pixel_to_micron_x,
    pixel_to_micron_y,
    organoid_mode="single",
    threshold=128,
    max_area=3000000,
    min_area=30000,
    min_circularity=0.2,
    foreground_polarity="dark",
    center_fraction=None,
    close_kernel_size=None,
    debug_capture=None,
):
    """Segment one brightfield image using the classical threshold/contour workflow.

    ``center_fraction`` / ``close_kernel_size``: see
    ``_segment_single_organoid_brightfield_image``.

    ``debug_capture``, if given a dict, gets populated with intermediate
    masks/contour lists from every pipeline sub-stage, regardless of
    ``organoid_mode`` -- see ``_segment_single_organoid_brightfield_image`` /
    ``_segment_multi_organoid_brightfield_image`` for the exact keys. Used by
    ``fit_experiment_data.py``'s ``--diagnose`` mode for step-by-step
    pipeline inspection; has no effect on the result.
    """
    organoid_mode = str(organoid_mode).lower()
    if organoid_mode in {"multiple", "multi", "many"}:
        return _segment_multi_organoid_brightfield_image(
            image_path,
            pixel_to_micron_x,
            pixel_to_micron_y,
            threshold=threshold,
            max_area=max_area,
            min_area=min_area,
            min_circularity=min_circularity,
            foreground_polarity=foreground_polarity,
            center_fraction=center_fraction,
            close_kernel_size=close_kernel_size,
            debug_capture=debug_capture,
        )

    mask = _segment_single_organoid_brightfield_image(
        image_path,
        pixel_to_micron_x,
        pixel_to_micron_y,
        threshold=threshold,
        max_area=max_area,
        min_area=min_area,
        min_circularity=min_circularity,
        foreground_polarity=foreground_polarity,
        center_fraction=center_fraction,
        close_kernel_size=close_kernel_size,
        debug_capture=debug_capture,
    )
    object_count = measure.label(mask > 0, connectivity=LABEL_CONNECTIVITY).max()

    return mask, object_count


def _parse_trailing_number(value):
    match = re.search(r"(\d+)$", str(value))
    return int(match.group(1)) if match else None


def _get_image_measurement_context(image_id, images_per_plate=24):
    """Return image/well identifiers derived from sequential Tile IDs when possible."""
    tile_number = _parse_trailing_number(image_id)
    if tile_number is None or not images_per_plate:
        return {
            "Image_ID": image_id,
            "Plate_Index": np.nan,
            "Well_Index": np.nan,
            "Well_ID": image_id,
        }

    plate_index = ((tile_number - 1) // images_per_plate) + 1
    well_index = ((tile_number - 1) % images_per_plate) + 1

    return {
        "Image_ID": image_id,
        "Plate_Index": plate_index,
        "Well_Index": well_index,
        "Well_ID": f"Plate{plate_index:02d}_Well{well_index:02d}",
    }


def _folder_has_tile_images(directory):
    if not os.path.isdir(directory):
        return False

    for extension in ("jpg", "jpeg", "tif", "tiff", "png"):
        if glob.glob(opj(directory, f"Tile*.{extension}")):
            return True
        if glob.glob(opj(directory, f"Tile*.{extension.upper()}")):
            return True

    return False


def _resolve_prepared_tile_image_dir(raw_data_dir):
    stitched_dir = opj(raw_data_dir, "stitched")
    if _folder_has_tile_images(stitched_dir):
        return stitched_dir
    return raw_data_dir


_TIF_EXTENSIONS = {".tif", ".tiff"}
_JPG_EXTENSIONS = {".jpg", ".jpeg"}


def _register_tile_candidate(tile_images_by_stem, tile_path, extension_priority):
    stem = os.path.splitext(os.path.basename(tile_path))[0]
    new_extension = os.path.splitext(tile_path)[1].lower()
    current_path = tile_images_by_stem.get(stem)

    if current_path is None:
        tile_images_by_stem[stem] = tile_path
        return

    current_extension = os.path.splitext(current_path)[1].lower()

    if current_extension in _JPG_EXTENSIONS and new_extension in _TIF_EXTENSIONS:
        # A .jpg conversion sibling is normally preferred over its source
        # .tif (already 8-bit, no need to re-normalize) -- *unless* the .tif
        # is newer, which means the .jpg is a stale cache: left over from
        # before the source .tif was replaced, or from an earlier (since
        # fixed) version of the TIFF->JPG normalization logic. Nothing else
        # in the pipeline invalidates this cache -- `--force` only affects
        # whether *segmentation* re-runs for a day, not whether this
        # raw-data-adjacent conversion artifact gets regenerated. Without
        # this check, a stale .jpg silently shadows its .tif forever, and no
        # amount of downstream threshold/parameter tuning can fix pixel data
        # that never actually gets re-read.
        if os.path.getmtime(tile_path) > os.path.getmtime(current_path):
            tile_images_by_stem[stem] = tile_path
        return

    if extension_priority[new_extension] < extension_priority[current_extension]:
        tile_images_by_stem[stem] = tile_path


def _collect_tile_images(raw_data_dir):
    """Collect Tile images, preferring JPG siblings created from TIFF inputs
    (unless the TIFF is newer -- see ``_register_tile_candidate``)."""
    tile_image_dir = _resolve_prepared_tile_image_dir(raw_data_dir)
    extension_priority = {
        ".jpg": 0,
        ".jpeg": 1,
        ".png": 2,
        ".tif": 3,
        ".tiff": 4,
    }
    tile_images_by_stem = {}

    for extension in ("jpg", "jpeg", "tif", "tiff", "png"):
        for tile_path in glob.glob(opj(tile_image_dir, f"Tile*.{extension}")):
            _register_tile_candidate(tile_images_by_stem, tile_path, extension_priority)
        for tile_path in glob.glob(opj(tile_image_dir, f"Tile*.{extension.upper()}")):
            _register_tile_candidate(tile_images_by_stem, tile_path, extension_priority)

    return sorted(tile_images_by_stem.values())


def _comparison_resample_filter(binary=False):
    resampling = getattr(Image, "Resampling", Image)
    return resampling.NEAREST if binary else resampling.BILINEAR


def _load_image_preview_for_comparison(image_path, max_side=1200, binary=False):
    """Load a display-sized image preview without rendering the full image as a figure."""
    original_pixel_limit = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(image_path) as img:
                img.draft("L", (max_side, max_side))
                img = img.convert("L")
                img.thumbnail(
                    (max_side, max_side),
                    resample=_comparison_resample_filter(binary=binary),
                )
                preview = img.copy()
    finally:
        Image.MAX_IMAGE_PIXELS = original_pixel_limit

    if binary:
        preview = preview.point(lambda value: 255 if value > 0 else 0)

    return preview.convert("RGB")


def _load_npy_mask_preview_for_comparison(mask_path, max_side=1200):
    """Load a downsampled binary preview from a potentially large NPY mask."""
    mask = np.load(mask_path, mmap_mode="r")
    height, width = mask.shape[:2]
    stride = max(1, int(np.ceil(max(height, width) / max_side)))
    preview = np.asarray(mask[::stride, ::stride] > 0, dtype=np.uint8) * 255
    preview_img = Image.fromarray(preview)
    preview_img.thumbnail(
        (max_side, max_side),
        resample=_comparison_resample_filter(binary=True),
    )
    return preview_img.convert("RGB")


def _load_mask_preview_for_comparison(mask_path, max_side=1200):
    """Load a saved segmentation mask as a display-sized binary preview."""
    extension = os.path.splitext(mask_path)[1].lower()
    if extension == ".npy":
        return _load_npy_mask_preview_for_comparison(mask_path, max_side=max_side)

    return _load_image_preview_for_comparison(mask_path, max_side=max_side, binary=True)


def _compose_classical_comparison_preview(raw_img, mask_img):
    """Paste raw and mask previews into one lightweight QC image."""
    title_height = 34
    padding = 14
    gap = 16
    panel_height = max(raw_img.height, mask_img.height)
    canvas_width = raw_img.width + mask_img.width + gap + padding * 2
    canvas_height = panel_height + title_height + padding * 2
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    panels = [
        (raw_img, padding, "Raw stitched image"),
        (mask_img, padding + raw_img.width + gap, "Classical segmentation mask"),
    ]
    for panel_img, left, title in panels:
        text_box = draw.textbbox((0, 0), title)
        text_width = text_box[2] - text_box[0]
        text_x = left + max((panel_img.width - text_width) // 2, 0)
        draw.text((text_x, padding), title, fill="black")
        top = padding + title_height + max((panel_height - panel_img.height) // 2, 0)
        canvas.paste(panel_img, (left, top))

    return canvas


def _find_saved_classical_mask(processed_data_dir, tile_name):
    """Return the saved classical mask path for a Tile ID, preferring JPG output."""
    candidate_paths = [
        opj(processed_data_dir, "inference_jpg", f"{tile_name}.jpg"),
        opj(processed_data_dir, "inference_tif", f"{tile_name}.tif"),
        opj(processed_data_dir, "inference_npy", f"{tile_name}.npy"),
    ]

    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path

    return None


def generate_classical_segmentation_comparison_figures(
    base_dir,
    project_dir,
    days,
    project_names,
    overwrite=False,
    max_preview_side=1200,
):
    """Create lightweight side-by-side raw image and binary-mask QC previews."""
    for day in days:
        for project in project_names:
            raw_data_dir = opj(base_dir, project_dir, 'raw_data', f'Day {day}', project)
            processed_data_dir = opj(base_dir, project_dir, 'processed_data', f'Day {day}', project)
            comparison_dir = opj(processed_data_dir, "inference_comparison")

            tile_images = _collect_tile_images(raw_data_dir)
            if not tile_images:
                print(f"No raw Tile images found for {project}, Day {day}; skipping comparison figures.")
                continue

            os.makedirs(comparison_dir, exist_ok=True)

            generated_count = 0
            skipped_count = 0
            missing_mask_count = 0

            for tile_path in tile_images:
                tile_name = os.path.splitext(os.path.basename(tile_path))[0]
                comparison_path = opj(comparison_dir, f"{tile_name}.jpg")
                if os.path.exists(comparison_path) and not overwrite:
                    skipped_count += 1
                    continue

                mask_path = _find_saved_classical_mask(processed_data_dir, tile_name)
                if mask_path is None:
                    missing_mask_count += 1
                    continue

                raw_img = _load_image_preview_for_comparison(
                    tile_path,
                    max_side=max_preview_side,
                    binary=False,
                )
                mask_img = _load_mask_preview_for_comparison(
                    mask_path,
                    max_side=max_preview_side,
                )
                comparison_img = _compose_classical_comparison_preview(raw_img, mask_img)
                comparison_img.save(comparison_path, "JPEG", quality=88)
                generated_count += 1

            print(
                f"Classical comparison figures for {project}, Day {day}: "
                f"{generated_count} generated, {skipped_count} already present"
                + (
                    f", {missing_mask_count} missing masks"
                    if missing_mask_count
                    else ""
                )
                + "."
            )


def run_classical_brightfield_segmentation(
    base_dir,
    project_dir,
    days,
    project_names,
    pixel_to_micron_x,
    pixel_to_micron_y,
    organoid_mode="single",
    threshold=128,
    max_area=3000000,
    min_area=30000,
    min_circularity=0.2,
    images_per_plate=24,
    foreground_polarity="dark",
    center_fraction=None,
    close_kernel_size=None,
    run_name=None,
):
    """Segment prepared brightfield images and save masks, labels, and a summary CSV."""
    for day in days:
        for project in project_names:
            raw_data_dir = opj(base_dir, project_dir, 'raw_data', f'Day {day}', project)
            project_dir_path = opj(base_dir, project_dir, 'processed_data', f'Day {day}', project)
            inference_jpg_dir = opj(project_dir_path, "inference_jpg")
            inference_npy_dir = opj(project_dir_path, "inference_npy")
            inference_tif_dir = opj(project_dir_path, "inference_tif")
            inference_labels_dir = opj(project_dir_path, "inference_labels_npy")

            os.makedirs(inference_jpg_dir, exist_ok=True)
            os.makedirs(inference_npy_dir, exist_ok=True)
            os.makedirs(inference_tif_dir, exist_ok=True)
            os.makedirs(inference_labels_dir, exist_ok=True)

            tile_images = _collect_tile_images(raw_data_dir)
            segmentation_summary = []
            failed_images = []

            for tile_path in tile_images:
                tile_name = os.path.splitext(os.path.basename(tile_path))[0]
                image_context = _get_image_measurement_context(tile_name, images_per_plate)
                try:
                    mask, object_count = segment_classical_brightfield_image(
                        tile_path,
                        pixel_to_micron_x,
                        pixel_to_micron_y,
                        organoid_mode=organoid_mode,
                        threshold=threshold,
                        max_area=max_area,
                        min_area=min_area,
                        min_circularity=min_circularity,
                        foreground_polarity=foreground_polarity,
                        center_fraction=center_fraction,
                        close_kernel_size=close_kernel_size,
                    )
                except Exception as exc:
                    print(f"⚠️  Failed to segment {tile_path}: {exc}")
                    failed_images.append({"Image": tile_name, "Stage": "classical_segmentation", "Error": str(exc)})
                    continue

                mask_img = Image.fromarray(mask)
                mask_img.save(opj(inference_jpg_dir, f"{tile_name}.jpg"))
                np.save(opj(inference_npy_dir, f"{tile_name}.npy"), mask)
                mask_img.save(opj(inference_tif_dir, f"{tile_name}.tif"))

                labeled_mask = measure.label(mask > 0, connectivity=LABEL_CONNECTIVITY).astype(np.uint16)
                np.save(opj(inference_labels_dir, f"{tile_name}.npy"), labeled_mask)

                segmentation_summary.append({
                    "Tile_ID": tile_name,
                    **image_context,
                    "Organoid_Count": object_count,
                    "Mask_Path": opj(inference_npy_dir, f"{tile_name}.npy"),
                    "Label_Path": opj(inference_labels_dir, f"{tile_name}.npy"),
                    "Organoid_Mode": organoid_mode,
                    "Foreground_Polarity": foreground_polarity,
                    "Threshold": threshold,
                    "Min_Area_um2": min_area,
                    "Max_Area_um2": max_area,
                    "Min_Circularity": min_circularity,
                    "Center_Fraction": center_fraction,
                    "Close_Kernel_Size": close_kernel_size,
                })

                print(
                    f"Saved masks for {tile_name} in {project_dir_path} "
                    f"({object_count} organoids)"
                )

            if segmentation_summary:
                summary_df = pd.DataFrame(segmentation_summary)
                summary_df.to_csv(
                    opj(project_dir_path, "classical_segmentation_summary.csv"),
                    index=False,
                )

            if failed_images:
                pd.DataFrame(failed_images).to_csv(
                    opj(project_dir_path, "segmentation_errors.csv"), index=False
                )

            _write_run_manifest(project_dir_path, "classical_segmentation", {
                "run_name": run_name,
                "organoid_mode": organoid_mode,
                "pixel_to_micron_x": pixel_to_micron_x,
                "pixel_to_micron_y": pixel_to_micron_y,
                "threshold": threshold,
                "foreground_polarity": foreground_polarity,
                "min_area": min_area,
                "max_area": max_area,
                "min_circularity": min_circularity,
                "center_fraction": center_fraction,
                "close_kernel_size": close_kernel_size,
                "images_processed": len(segmentation_summary),
                "images_failed": len(failed_images),
            })

            generate_classical_segmentation_comparison_figures(
                base_dir,
                project_dir,
                [day],
                [project],
                overwrite=True,
            )


# Keyed by (orgline_repo, orgdet_checkpoint, sam2_checkpoint, device) so
# run_orgline_segmentation only builds the detector/SAM2 predictor once per
# distinct config per process, not once per (batch, run) call -- see that
# function's own comment at the cache lookup site.
_ORGLINE_MODEL_CACHE = {}

# The nc=18/PANet-head/`.eval()` bug-fix loader is shared with
# Organotchi-manuscript/benchmarking/adapters/predict_orgline.py (a
# separate repo, imported from there as `data_analysis.orgline_bridge`) --
# see orgline_bridge.py's module docstring for the three upstream bugs it
# works around.
from orgline_bridge import build_detector as _build_orgline_detector  # noqa: E402
from orgline_bridge import build_sam2_predictor as _build_orgline_sam2_predictor  # noqa: E402


def run_orgline_segmentation(
    base_dir,
    project_dir,
    days,
    project_names,
    orgline_repo,
    orgdet_checkpoint=None,
    sam2_checkpoint=None,
    conf=0.2,
    iou=0.7,
    organoid_mode="single",
    images_per_plate=24,
    device="cpu",
    run_name=None,
):
    """Segment prepared brightfield images with OrgLine (a YOLO-style
    detector whose backbone is a SAM2 Hiera image encoder, generating
    bounding-box prompts for a SAM2 mask predictor) and save masks, labels,
    and a summary CSV -- the same "tiny"/low-VRAM checkpoint configuration
    (orgdet_tiny.pt + sam2_hiera_tiny.pt, confidence 0.20, IoU 0.70) used in
    Organotchi-manuscript/benchmarking/adapters/predict_orgline.py (a
    separate repo), where OrgLine was the strongest tool across every
    tissue tried (see that repo's benchmarking/README.md).

    Must be run with the OrgLine venv's interpreter (see
    ../environments/README.md) -- OrgLine's dependencies (its
    own torch/torchvision pins, the vendored Ultralytics-style detector,
    SAM2) are isolated from mmsegmentation's ("dl" method) and are not
    expected to be importable from the main environment.

    Reads the same pre-stitched "Tile*" images, via the same
    ``_collect_tile_images``, as ``run_classical_brightfield_segmentation``
    -- OrgLine mode expects the same per-well image layout (stitched or not)
    as classical mode, and shares its ``prepare_classical_segmentation_images``
    stitching step where a run configures ``stitch: true``.

    ``organoid_mode="single"`` keeps only OrgLine's single highest-confidence
    detection per image (matching the classical/DL pipelines' single-organoid
    convention, e.g. Lancaster brain); ``"multiple"`` keeps every detection
    that survives ``conf``/``iou``, unrestricted, for genuinely multi-organoid
    wells.
    """
    if torch is None:
        raise ImportError(
            "run_orgline_segmentation requires OrgLine's own environment (torch, "
            "the vendored Ultralytics-style detector, SAM2), which is not importable "
            "from the current interpreter. Run this with "
            "Organotchi/environments/orgline_venv/bin/python "
            "-- see ../environments/README.md for how to build it."
        )

    organoid_mode = str(organoid_mode).lower()
    organoid_mode = "multiple" if organoid_mode in {"multiple", "multi", "many"} else "single"

    orgline_repo = os.path.abspath(orgline_repo)
    orgdet_checkpoint = os.path.abspath(orgdet_checkpoint or opj(orgline_repo, "checkpoints", "orgdet_tiny.pt"))
    sam2_checkpoint = os.path.abspath(sam2_checkpoint or opj(orgline_repo, "checkpoints", "sam2_hiera_tiny.pt"))
    checkpoints_dir = os.path.dirname(sam2_checkpoint)

    # Resolve every one of our own paths to absolute BEFORE changing cwd --
    # OrgLine's architecture code hard-codes the SAM2 checkpoint path
    # ("sam2_hiera_tiny.pt", no directory) as relative to cwd with no
    # override parameter, so building the model requires cwd == orgline_repo.
    base_dir = os.path.abspath(base_dir)

    cache_key = (orgline_repo, orgdet_checkpoint, sam2_checkpoint, device)
    cached = _ORGLINE_MODEL_CACHE.get(cache_key)
    if cached is not None:
        detector, sam2_predictor = cached
    else:
        # ~30s of pure model-build overhead (measured: ~28s detector + ~3s
        # SAM2), unrelated to how many images actually get processed --
        # segmentation.py calls this once per (batch, run) pair, so a
        # project with several batches configured for "orgline" would
        # otherwise pay this cost again for every single batch despite
        # every one of them using the exact same checkpoint. Cached at
        # module level (keyed by the exact config) so it's paid once per
        # process, not once per call.
        original_cwd = os.getcwd()
        os.chdir(orgline_repo)
        try:
            print("Building ORGDET detector (tiny) ...")
            detector = _build_orgline_detector(orgline_repo, orgdet_checkpoint, sam2_checkpoint, device=device)
            print("Building SAM2 predictor (hiera_tiny) ...")
            sam2_predictor = _build_orgline_sam2_predictor(orgline_repo, "sam2_hiera_t.yaml", checkpoints_dir, device=device)
        finally:
            os.chdir(original_cwd)
        _ORGLINE_MODEL_CACHE[cache_key] = (detector, sam2_predictor)

    for day in days:
        for project in project_names:
            raw_data_dir = opj(base_dir, project_dir, 'raw_data', f'Day {day}', project)
            project_dir_path = opj(base_dir, project_dir, 'processed_data', f'Day {day}', project)
            inference_jpg_dir = opj(project_dir_path, "inference_jpg")
            inference_npy_dir = opj(project_dir_path, "inference_npy")
            inference_tif_dir = opj(project_dir_path, "inference_tif")
            inference_labels_dir = opj(project_dir_path, "inference_labels_npy")

            os.makedirs(inference_jpg_dir, exist_ok=True)
            os.makedirs(inference_npy_dir, exist_ok=True)
            os.makedirs(inference_tif_dir, exist_ok=True)
            os.makedirs(inference_labels_dir, exist_ok=True)

            tile_images = _collect_tile_images(raw_data_dir)
            segmentation_summary = []
            failed_images = []

            for tile_path in tile_images:
                tile_name = os.path.splitext(os.path.basename(tile_path))[0]
                image_context = _get_image_measurement_context(tile_name, images_per_plate)
                try:
                    results = detector.predict(source=tile_path, conf=conf, iou=iou, verbose=False)
                    r = results[0]
                    h, w = r.orig_shape
                    instance_mask = np.zeros((h, w), dtype=np.int32)
                    n_boxes = 0 if r.boxes is None else len(r.boxes)

                    if n_boxes > 0:
                        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                        if organoid_mode == "single" and n_boxes > 1:
                            best_idx = int(r.boxes.conf.cpu().numpy().argmax())
                            boxes_xyxy = boxes_xyxy[[best_idx]]
                            n_boxes = 1

                        img_bgr = cv2.imread(tile_path)
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        sam2_predictor.set_image(img_rgb)
                        with torch.no_grad():
                            masks, _scores, _logits = sam2_predictor.predict(
                                point_coords=None, point_labels=None, box=boxes_xyxy, multimask_output=False,
                            )
                        if masks.ndim == 3:  # single box -> (1, H, W) instead of (1, 1, H, W)
                            masks = masks[:, None, :, :]
                        for i in range(masks.shape[0]):
                            instance_mask[masks[i, 0].astype(bool)] = i + 1
                except Exception as exc:
                    print(f"⚠️  Failed to segment {tile_path}: {exc}")
                    failed_images.append({"Image": tile_name, "Stage": "orgline_segmentation", "Error": str(exc)})
                    continue

                object_count = int(instance_mask.max())
                binary_mask = ((instance_mask > 0).astype(np.uint8)) * 255

                mask_img = Image.fromarray(binary_mask)
                mask_img.save(opj(inference_jpg_dir, f"{tile_name}.jpg"))
                np.save(opj(inference_npy_dir, f"{tile_name}.npy"), binary_mask)
                mask_img.save(opj(inference_tif_dir, f"{tile_name}.tif"))
                np.save(opj(inference_labels_dir, f"{tile_name}.npy"), instance_mask.astype(np.uint16))

                segmentation_summary.append({
                    "Tile_ID": tile_name,
                    **image_context,
                    "Organoid_Count": object_count,
                    "Mask_Path": opj(inference_npy_dir, f"{tile_name}.npy"),
                    "Label_Path": opj(inference_labels_dir, f"{tile_name}.npy"),
                    "Organoid_Mode": organoid_mode,
                    "Confidence_Threshold": conf,
                    "IoU_Threshold": iou,
                })

                print(
                    f"Saved masks for {tile_name} in {project_dir_path} "
                    f"({object_count} organoids)"
                )

            if segmentation_summary:
                summary_df = pd.DataFrame(segmentation_summary)
                summary_df.to_csv(
                    opj(project_dir_path, "orgline_segmentation_summary.csv"),
                    index=False,
                )

            if failed_images:
                pd.DataFrame(failed_images).to_csv(
                    opj(project_dir_path, "segmentation_errors.csv"), index=False
                )

            _write_run_manifest(project_dir_path, "orgline_segmentation", {
                "run_name": run_name,
                "orgline_repo": orgline_repo,
                "orgdet_checkpoint": orgdet_checkpoint,
                "sam2_checkpoint": sam2_checkpoint,
                "checkpoint_variant": "tiny",
                "organoid_mode": organoid_mode,
                "conf_threshold": conf,
                "iou_threshold": iou,
                "device": device,
                "images_processed": len(segmentation_summary),
                "images_failed": len(failed_images),
            })

            generate_classical_segmentation_comparison_figures(
                base_dir,
                project_dir,
                [day],
                [project],
                overwrite=True,
            )

