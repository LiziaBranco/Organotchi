import numpy as np
from PIL import Image, ImageDraw
from os.path import join as opj
import os
import re
import tempfile
import matplotlib.pyplot as plt
import pandas as pd
from skimage import measure, morphology
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

def filter_and_blackout_low_circularity(mask, pixel_to_micron_x, pixel_to_micron_y, circularity_threshold=0.3):
    """
    Filters the mask to keep only the structure with the highest area among those with circularity above the threshold.
    Structures with circularity lower than the threshold are blacked out.
    """
    labeled_mask, num_features = measure.label(mask, connectivity=1, return_num=True)  # Identify white structures
    properties = measure.regionprops(labeled_mask)  # Get region properties

    filtered_mask = np.zeros_like(mask)  # Start with a black mask
    max_area = 0  # To track the largest structure's area
    largest_structure_label = -1  # Label of the largest structure
    structure_count = 0  # Counter for valid structures

    for prop in properties:
        area = prop.area * pixel_to_micron_x * pixel_to_micron_y
        perimeter = prop.perimeter * pixel_to_micron_x
        diameter = np.sqrt(4 * area / np.pi)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

        if circularity > circularity_threshold and diameter > 150:
            # If this structure is larger than the previously largest one, update
            if area > max_area:
                max_area = area
                largest_structure_label = prop.label

    # Retain the largest valid structure
    if largest_structure_label != -1:
        filtered_mask[labeled_mask == largest_structure_label] = 255
        structure_count = 1  # We are keeping only one structure

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

def run_deep_learning_segmentation(checkpoint, base_dir, project_dir, days, batches, crop_days=None):
    """
    Run segmentation inference and optionally crop the center of selected days before inference.

    When a day is included in ``crop_days``, the centered crop of the original image is
    first resized back to the original size and sent to ``inference_model``. The predicted
    mask is then resized back into the crop window and padded with black pixels so that the
    saved output matches the original image size.
    """
    if Config is None or inference_model is None or init_model is None:
        raise ImportError(
            "run_deep_learning_segmentation requires mmengine and mmsegmentation. "
            "Install those packages or use the classical segmentation functions."
        )

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
            
            outdirs = {
                'inference_npy': opj(processed_data_dir, 'inference_npy'),
                'inference_jpg': opj(processed_data_dir, 'inference_jpg')
            }            
            for directory in outdirs.values():
                os.makedirs(directory, exist_ok=True)
            
            # Convert TIF images to JPG if needed
            for img_loc in sorted(os.listdir(raw_data_dir)):
                img_path = opj(raw_data_dir, img_loc)
                # Convert TIF images to JPG if needed
                if img_loc.lower().endswith('.tif'):
                    img = Image.open(img_path)
                    img_np = np.array(img).astype(float)

                    # 1. Min-Max Normalization (Scales actual signal to 0-255)
                    img_min = img_np.min()
                    img_max = img_np.max()
                    
                    if img_max > img_min:
                        img_scaled = (img_np - img_min) / (img_max - img_min) * 255
                    else:
                        img_scaled = img_np # Handle flat/empty images
                        
                    # 2. Convert to 8-bit uint8
                    img_8bit = Image.fromarray(img_scaled.astype(np.uint8))

                    # 3. Convert grayscale to RGB and save
                    img_rgb = img_8bit.convert("RGB")
                    jpg_path = opj(raw_data_dir, os.path.splitext(img_loc)[0] + '.jpg')
                    img_rgb.save(jpg_path, "JPEG")
                    
            # Load model config and checkpoint
            cfg = Config.fromfile('Training-Results/work_dirs/split-0/split-0.py')
            cfg.load_from = checkpoint
            cfg.model.decode_head.norm_cfg = dict(type='BN', requires_grad=True)
            
            # Initialize model
            model = init_model(cfg, checkpoint=checkpoint, device='cpu')

            # Perform inference
            for img_loc in sorted(os.listdir(raw_data_dir)):
                if not img_loc.lower().endswith('.jpg'):
                    continue  # Skip non-JPG images
                
                img = opj(raw_data_dir, img_loc)
                img_id = os.path.splitext(img_loc)[0]

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

def apply_erosion(mask, kernel_size=(10, 10), iterations=1):
    """
    Apply subtle erosion to the binary mask to refine structures slightly.
    The kernel_size determines the size of the structuring element, and iterations 
    control the number of times erosion is applied.
    """
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
    """
    mask_filled = mask.copy()
    
    # Find contours of the white regions
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Fill each contour to remove black holes
    cv2.drawContours(mask_filled, contours, -1, (255), thickness=cv2.FILLED)
    
    return mask_filled

def apply_filtering_to_segmentations(base_dir, project_dir, days, batches, pixel_to_micron_x, pixel_to_micron_y):
    """
    Applies high-circularity filtering to the segmentation masks in `inference_jpg/`
    and saves the filtered masks in:
    - `inference_filtered/` (JPG images)
    - `inference_filtered_npy/` (Numpy array masks)
    - `inference_comparison/` (Comparison images)
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

            os.makedirs(filtered_jpg_dir, exist_ok=True)
            os.makedirs(filtered_npy_dir, exist_ok=True)
            os.makedirs(comparison_dir, exist_ok=True)

            for img_loc in sorted(os.listdir(inference_jpg_dir)):
                if not img_loc.lower().endswith('.jpg'):
                    continue  # Skip non-JPG files
                
                img_path = opj(inference_jpg_dir, img_loc)
                img_id = os.path.splitext(img_loc)[0]

                # Load the segmentation mask
                mask = np.array(Image.open(img_path).convert('L'))  # Convert to grayscale
                mask = mask > 128  # Binarize (thresholding)

                # Apply subtle erosion to refine the mask
                eroded_mask = apply_erosion(mask)

                # Apply filtering: remove low-circularity structures
                filtered_mask, structure_count = filter_and_blackout_low_circularity(eroded_mask, pixel_to_micron_x, pixel_to_micron_y)
                
                filtered_mask = apply_dilation(filtered_mask)
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
            if not os.path.isdir(folder_path) or folder_name == project:
                continue

            if any(folder_name.startswith(prefix) for prefix in folder_prefixes):
                candidate_paths.append(folder_path)

    for folder_path in sorted(set(candidate_paths), key=_natural_image_sort_key):
        if not os.path.isdir(folder_path):
            continue

        if len(_collect_stitch_input_images(folder_path)) >= tiles_per_image:
            return folder_path

    return None


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

    images = []
    for img_path in image_paths:
        with Image.open(img_path) as img:
            images.append(img.convert("RGB"))

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
                        project_output_dir,
                        f"Tile0000{global_tile_num:02d}.jpg"
                    )

                    stitch_microscope_tile_grid(
                        group,
                        output_filename,
                        tile_grid=grid_shape,
                        overlap_x=overlap_x,
                        overlap_y=overlap_y,
                        tile_order=tile_order,
                    )

def _filter_single_organoid_contours(mask, pixel_to_micron_x, pixel_to_micron_y, max_area=3000000, min_area=30000, min_circularity=0.2, center_fraction=0.95):
    """
    Filters contours based on area, circularity, and position inside a central region.
    Only keeps objects within the central rectangle of the image.

    If multiple objects pass the filter, only the largest one is retained.

    Parameters:
    - mask: Binary image mask.
    - pixel_to_micron_x: Conversion factor for pixels to microns (X-axis).
    - pixel_to_micron_y: Conversion factor for pixels to microns (Y-axis).
    - max_area: Maximum object area in square microns.
    - min_circularity: Minimum circularity threshold.
    - center_fraction: Fraction of the image to be considered as the "central area" (0.0 - 1.0).

    Returns:
    - filtered_mask: Binary mask with only valid objects.
    """

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []

    # Define central region
    height, width = mask.shape
    center_x1 = int(width * (1 - center_fraction) / 2)
    center_x2 = int(width * (1 + center_fraction) / 2)
    center_y1 = int(height * (1 - center_fraction) / 2)
    center_y2 = int(height * (1 + center_fraction) / 2)

    # First pass: apply main filters
    for contour in contours:
        area = cv2.contourArea(contour) * pixel_to_micron_x * pixel_to_micron_y

        if min_area < area < max_area:
            perimeter = cv2.arcLength(contour, True) * pixel_to_micron_x
            circularity = 4 * np.pi * (area / (perimeter ** 2)) if perimeter > 0 else 0

            if circularity >= min_circularity:
                x, y, w, h = cv2.boundingRect(contour)
                if (x >= center_x1 and x + w <= center_x2 and 
                    y >= center_y1 and y + h <= center_y2):
                    valid_contours.append((contour, area))

    # In single-organoid mode, keep exactly one candidate: the largest valid contour.
    if valid_contours:
        valid_contours = [max(valid_contours, key=lambda item: item[1])]


    # Draw final filtered contours
    filtered_mask = np.zeros_like(mask)
    for contour, _ in valid_contours:
        cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return _keep_largest_connected_component(filtered_mask)


def _keep_largest_connected_component(mask):
    """Remove tiny disconnected artifacts from a single-organoid mask."""
    labeled_mask = measure.label(mask > 0, connectivity=1)
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


def _create_classical_binary_mask(
    img_array,
    threshold=128,
    threshold_method="fixed",
    foreground_polarity="dark",
    blur_kernel_size=0,
):
    """Create a binary foreground mask for dark-on-light or bright-on-dark images."""
    foreground_polarity = _normalize_foreground_polarity(foreground_polarity)
    threshold_method = str(threshold_method).lower()
    if isinstance(threshold, str) and threshold.lower() in {"otsu", "auto"}:
        threshold_method = "otsu"

    img_uint8 = img_array.astype(np.uint8)
    blur_kernel_size = int(blur_kernel_size or 0)
    if blur_kernel_size > 1:
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        img_uint8 = cv2.GaussianBlur(img_uint8, (blur_kernel_size, blur_kernel_size), 0)

    if threshold_method in {"otsu", "auto"}:
        threshold_type = cv2.THRESH_BINARY if foreground_polarity == "bright" else cv2.THRESH_BINARY_INV
        _, binary_mask = cv2.threshold(
            img_uint8,
            0,
            255,
            threshold_type + cv2.THRESH_OTSU,
        )
        return binary_mask.astype(np.uint8)

    if threshold_method not in {"fixed", "manual"}:
        raise ValueError("threshold_method must be 'fixed' or 'otsu'.")

    threshold_value = float(threshold)
    if foreground_polarity == "bright":
        return np.where(img_uint8 > threshold_value, 255, 0).astype(np.uint8)

    return np.where(img_uint8 < threshold_value, 255, 0).astype(np.uint8)




def _convert_tif_to_jpg_for_segmentation(image_path):
    """Convert TIFF images to an 8-bit JPG sibling before segmentation."""
    extension = os.path.splitext(image_path)[1].lower()
    if extension not in {".tif", ".tiff"}:
        return image_path

    jpg_path = os.path.splitext(image_path)[0] + ".jpg"
    with Image.open(image_path) as img:
        img_np = np.array(img).astype(float)

    img_min = img_np.min()
    img_max = img_np.max()
    if img_max > img_min:
        img_scaled = (img_np - img_min) / (img_max - img_min) * 255
    else:
        img_scaled = img_np

    img_8bit = Image.fromarray(img_scaled.astype(np.uint8))
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
    center_fraction=0.95,
    foreground_polarity="dark",
    threshold_method="fixed",
    blur_kernel_size=0,
):
    """
    Segment an image based on grayscale intensity and apply contour filtering,
    keeping only objects inside a defined central region.

    Parameters:
    - image_path: Path to the input image.
    - pixel_to_micron_x: Conversion factor for pixels to microns (X-axis).
    - pixel_to_micron_y: Conversion factor for pixels to microns (Y-axis).
    - threshold: Grayscale threshold for binarization.
    - max_area: Maximum object area in square microns.
    - min_circularity: Minimum circularity threshold.
    - center_fraction: Fraction of the image to be considered as the "central area" (0.0 - 1.0).

    Returns:
    - filtered_mask: Binary mask with valid objects.
    """

    image_path = _convert_tif_to_jpg_for_segmentation(image_path)
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img_array = np.array(img)

    binary_mask = _create_classical_binary_mask(
        img_array,
        threshold=threshold,
        threshold_method=threshold_method,
        foreground_polarity=foreground_polarity,
        blur_kernel_size=blur_kernel_size,
    )

    # Apply filtering to remove unwanted structures
    filtered_mask = _filter_single_organoid_contours(binary_mask, pixel_to_micron_x, pixel_to_micron_y, max_area, min_area, min_circularity, center_fraction)

    return filtered_mask


def _filter_multi_organoid_contours(
    mask,
    pixel_to_micron_x,
    pixel_to_micron_y,
    max_area=3000000,
    min_area=30000,
    min_circularity=0.2,
    center_fraction=1.0,
):
    """
    Filter a binary mask while preserving all valid organoid-like contours.

    This function keeps every contour that passes the area/circularity filters.
    It is intended for fields where multiple organoids can be present in the
    same image.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    height, width = mask.shape
    avg_spacing = (pixel_to_micron_x + pixel_to_micron_y) / 2

    if center_fraction is None or center_fraction >= 1:
        center_x1, center_y1 = 0, 0
        center_x2, center_y2 = width, height
    else:
        center_x1 = int(width * (1 - center_fraction) / 2)
        center_x2 = int(width * (1 + center_fraction) / 2)
        center_y1 = int(height * (1 - center_fraction) / 2)
        center_y2 = int(height * (1 + center_fraction) / 2)

    for contour in contours:
        area = cv2.contourArea(contour) * pixel_to_micron_x * pixel_to_micron_y
        if not min_area < area < max_area:
            continue

        perimeter = cv2.arcLength(contour, True) * avg_spacing
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        if circularity < min_circularity:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if (
            x >= center_x1 and x + w <= center_x2 and
            y >= center_y1 and y + h <= center_y2
        ):
            valid_contours.append(contour)

    filtered_mask = np.zeros_like(mask)
    for contour in valid_contours:
        cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return filtered_mask, len(valid_contours)


def _segment_multi_organoid_brightfield_image(
    image_path,
    pixel_to_micron_x,
    pixel_to_micron_y,
    threshold=128,
    max_area=3000000,
    min_area=30000,
    min_circularity=0.2,
    center_fraction=1.0,
    foreground_polarity="dark",
    threshold_method="fixed",
    blur_kernel_size=0,
):
    """
    Segment all organoid-like objects in one stitched brightfield image.

    The binarization logic mirrors the single-organoid classical segmentation:
    foreground structures become binary mask objects, then contour filtering removes small,
    very large, or irregular objects without reducing the mask to a single
    organoid.
    """
    with Image.open(image_path) as img:
        img_array = np.array(img.convert("L"))

    binary_mask = _create_classical_binary_mask(
        img_array,
        threshold=threshold,
        threshold_method=threshold_method,
        foreground_polarity=foreground_polarity,
        blur_kernel_size=blur_kernel_size,
    )
    filtered_mask, object_count = _filter_multi_organoid_contours(
        binary_mask,
        pixel_to_micron_x,
        pixel_to_micron_y,
        max_area=max_area,
        min_area=min_area,
        min_circularity=min_circularity,
        center_fraction=center_fraction,
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
    center_fraction=0.95,
    foreground_polarity="dark",
    threshold_method="fixed",
    blur_kernel_size=0,
):
    """Segment one brightfield image using the classical threshold/contour workflow."""
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
            center_fraction=center_fraction,
            foreground_polarity=foreground_polarity,
            threshold_method=threshold_method,
            blur_kernel_size=blur_kernel_size,
        )

    mask = _segment_single_organoid_brightfield_image(
        image_path,
        pixel_to_micron_x,
        pixel_to_micron_y,
        threshold=threshold,
        max_area=max_area,
        min_area=min_area,
        min_circularity=min_circularity,
        center_fraction=center_fraction,
        foreground_polarity=foreground_polarity,
        threshold_method=threshold_method,
        blur_kernel_size=blur_kernel_size,
    )
    object_count = measure.label(mask > 0, connectivity=1).max()

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


def _collect_tile_images(raw_data_dir):
    """Collect Tile images, preferring JPG siblings created from TIFF inputs."""
    extension_priority = {
        ".jpg": 0,
        ".jpeg": 1,
        ".png": 2,
        ".tif": 3,
        ".tiff": 4,
    }
    tile_images_by_stem = {}

    for extension in ("jpg", "jpeg", "tif", "tiff", "png"):
        for tile_path in glob.glob(opj(raw_data_dir, f"Tile*.{extension}")):
            stem = os.path.splitext(os.path.basename(tile_path))[0]
            current_path = tile_images_by_stem.get(stem)
            if current_path is None:
                tile_images_by_stem[stem] = tile_path
                continue

            current_extension = os.path.splitext(current_path)[1].lower()
            new_extension = os.path.splitext(tile_path)[1].lower()
            if extension_priority[new_extension] < extension_priority[current_extension]:
                tile_images_by_stem[stem] = tile_path

        for tile_path in glob.glob(opj(raw_data_dir, f"Tile*.{extension.upper()}")):
            stem = os.path.splitext(os.path.basename(tile_path))[0]
            current_path = tile_images_by_stem.get(stem)
            if current_path is None:
                tile_images_by_stem[stem] = tile_path
                continue

            current_extension = os.path.splitext(current_path)[1].lower()
            new_extension = os.path.splitext(tile_path)[1].lower()
            if extension_priority[new_extension] < extension_priority[current_extension]:
                tile_images_by_stem[stem] = tile_path

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


def _metrics_outputs_are_current(metrics_path, inference_npy_path):
    """Return True when saved morphology CSVs are newer than all source masks."""
    metrics_csv = os.path.join(metrics_path, 'mask_metrics.csv')
    well_summary_csv = os.path.join(metrics_path, 'well_metrics_summary.csv')
    output_paths = [metrics_csv, well_summary_csv]

    if any(not os.path.exists(path) or os.path.getsize(path) == 0 for path in output_paths):
        return False

    mask_paths = [
        os.path.join(inference_npy_path, filename)
        for filename in os.listdir(inference_npy_path)
        if filename.endswith('.npy')
    ]
    if not mask_paths:
        return False

    oldest_output_mtime = min(os.path.getmtime(path) for path in output_paths)
    newest_mask_mtime = max(os.path.getmtime(path) for path in mask_paths)

    return oldest_output_mtime >= newest_mask_mtime


def _summarize_metrics_by_image(metrics_df, feature_columns):
    """Create one row per image/well with object counts and feature summaries."""
    if metrics_df.empty:
        return pd.DataFrame()

    group_cols = [
        "Batch", "Day", "Run", "Image_ID", "Well_ID",
        "Plate_Index", "Well_Index", "Organoid_Mode",
    ]
    group_cols = [column for column in group_cols if column in metrics_df.columns]
    summary = (
        metrics_df
        .groupby(group_cols, dropna=False)
        .agg(Detected_Organoid_Count=("Organoid_ID", "count"))
        .reset_index()
    )

    for feature in feature_columns:
        feature_summary = (
            metrics_df
            .groupby(group_cols, dropna=False)[feature]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )
        feature_summary.rename(
            columns={
                "mean": f"{feature}_Mean",
                "std": f"{feature}_Std",
                "min": f"{feature}_Min",
                "max": f"{feature}_Max",
            },
            inplace=True,
        )
        summary = summary.merge(feature_summary, on=group_cols, how="left")
        mean_col = f"{feature}_Mean"
        std_col = f"{feature}_Std"
        cv_col = f"{feature}_CV"
        summary[cv_col] = np.where(
            summary[mean_col].abs() > 0,
            summary[std_col] / summary[mean_col],
            np.nan,
        )

    return summary


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
    center_fraction=0.95,
    images_per_plate=24,
    foreground_polarity="dark",
    threshold_method="fixed",
    blur_kernel_size=0,
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

            for tile_path in tile_images:
                tile_name = os.path.splitext(os.path.basename(tile_path))[0]
                image_context = _get_image_measurement_context(tile_name, images_per_plate)
                mask, object_count = segment_classical_brightfield_image(
                    tile_path,
                    pixel_to_micron_x,
                    pixel_to_micron_y,
                    organoid_mode=organoid_mode,
                    threshold=threshold,
                    max_area=max_area,
                    min_area=min_area,
                    min_circularity=min_circularity,
                    center_fraction=center_fraction,
                    foreground_polarity=foreground_polarity,
                    threshold_method=threshold_method,
                    blur_kernel_size=blur_kernel_size,
                )

                mask_img = Image.fromarray(mask)
                mask_img.save(opj(inference_jpg_dir, f"{tile_name}.jpg"))
                np.save(opj(inference_npy_dir, f"{tile_name}.npy"), mask)
                mask_img.save(opj(inference_tif_dir, f"{tile_name}.tif"))

                labeled_mask = measure.label(mask > 0, connectivity=2).astype(np.uint16)
                np.save(opj(inference_labels_dir, f"{tile_name}.npy"), labeled_mask)

                segmentation_summary.append({
                    "Tile_ID": tile_name,
                    **image_context,
                    "Organoid_Count": object_count,
                    "Mask_Path": opj(inference_npy_dir, f"{tile_name}.npy"),
                    "Label_Path": opj(inference_labels_dir, f"{tile_name}.npy"),
                    "Organoid_Mode": organoid_mode,
                    "Foreground_Polarity": foreground_polarity,
                    "Threshold_Method": threshold_method,
                    "Threshold": threshold,
                    "Blur_Kernel_Size": blur_kernel_size,
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

            generate_classical_segmentation_comparison_figures(
                base_dir,
                project_dir,
                [day],
                [project],
            )


def extract_organoid_morphology_metrics(
    base_dir,
    project_dir,
    days,
    batches,
    pixel_to_micron_x,
    pixel_to_micron_y,
    organoid_mode="single",
    run_name=None,
    images_per_plate=24,
    force=False,
):
    """Extracts morphological features handling anisotropic pixels (X != Y)."""
    organoid_mode = str(organoid_mode).lower()
    organoid_mode = "multiple" if organoid_mode in {"multiple", "multi", "many"} else "single"
    print(f"\n📐 Extracting Organoid Morphology Metrics")
    
    # Calculate average spacing for 1D scaling (Perimeter, Feret, Skeleton)
    avg_spacing = (pixel_to_micron_x + pixel_to_micron_y) / 2
    
    for day in days:
        print(f"- Processing Day {day}")
        batch_processed_path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}')
        for batch in batches:
            inference_npy_path = os.path.join(batch_processed_path, batch, 'inference_filtered_npy')
            if not os.path.exists(inference_npy_path):
                inference_npy_path = os.path.join(batch_processed_path, batch, 'inference_npy')

            if not os.path.exists(inference_npy_path):
                continue

            metrics_path = os.path.join(batch_processed_path, batch, 'inference_metrics')
            os.makedirs(metrics_path, exist_ok=True)
            force_current = force or os.environ.get("ORGANOID_FORCE_MORPHOLOGY", "").lower() in {"1", "true", "yes"}
            if not force_current and _metrics_outputs_are_current(metrics_path, inference_npy_path):
                print(f"  Skipping {batch}, Day {day}: morphology metrics are up to date.")
                continue

            metrics_data = []
            feature_columns = [
                'Area_um2', 'Diameter_um', 'Perimeter_um',
                'Feret_Diameter_Max_um', 'Aspect_Ratio', 'Circularity',
                'Roundness', 'Solidity', 'Eccentricity', 'Skeleton_Length_um'
            ]

            for filename in sorted(os.listdir(inference_npy_path)):
                if filename.endswith('.npy'):
                    mask = np.load(os.path.join(inference_npy_path, filename), mmap_mode='r')
                    labeled_mask = measure.label(mask > 0, connectivity=2)
                    object_count = int(labeled_mask.max())
                    if object_count == 0:
                        continue

                    image_id = filename.replace('.npy', '')
                    image_context = _get_image_measurement_context(image_id, images_per_plate)

                    # 1. Get properties WITH spacing (for Area and Axis lengths which are supported)
                    # Note: major/minor axis lengths ARE supported with spacing
                    props_um = measure.regionprops(labeled_mask, spacing=(pixel_to_micron_y, pixel_to_micron_x), cache=True)
                    
                    # 2. Get properties WITHOUT spacing (pixels) to avoid the NotImplementedError
                    props_px = measure.regionprops(labeled_mask, cache=True)

                    for p_um, p_px in zip(props_um, props_px):
                        organoid_id = image_id
                        if organoid_mode == "multiple" or object_count > 1:
                            organoid_id = f"{image_id}_object{p_px.label:03d}"

                        # --- Basic Metrics (Supported by spacing) ---
                        area_um2 = p_um.area
                        major_axis_um = p_um.axis_major_length
                        minor_axis_um = p_um.axis_minor_length
                        solidity = p_um.solidity
                        eccentricity = p_um.eccentricity
                        
                        # --- 1D Metrics (Scaled manually to avoid spacing error) ---
                        perimeter_um = p_px.perimeter * avg_spacing
                        
                        try:
                            feret_max_um = p_px.feret_diameter_max * avg_spacing
                        except AttributeError:
                            feret_max_um = np.nan

                        # --- Derived Metrics ---
                        diameter_um = np.sqrt(4 * area_um2 / np.pi)
                        circularity = (4 * np.pi * area_um2) / (perimeter_um ** 2) if perimeter_um > 0 else np.nan
                        aspect_ratio = major_axis_um / minor_axis_um if minor_axis_um > 0 else np.nan
                        roundness = (4 * area_um2) / (np.pi * (major_axis_um ** 2)) if major_axis_um > 0 else np.nan
                        
                        # Skeletonization is the slowest metric for multi-object images.
                        # Use the object's bounding-box crop instead of the full stitched mask.
                        skeleton = morphology.skeletonize(p_px.image)
                        skeleton_length_um = np.sum(skeleton) * avg_spacing
                        
                        metrics_data.append([
                            batch,
                            day,
                            run_name,
                            organoid_mode,
                            image_context["Image_ID"],
                            image_context["Well_ID"],
                            image_context["Plate_Index"],
                            image_context["Well_Index"],
                            organoid_id,
                            p_px.label,
                            object_count,
                            area_um2, diameter_um, perimeter_um, 
                            feret_max_um, aspect_ratio, circularity, 
                            roundness, solidity, eccentricity, skeleton_length_um
                        ])

            if metrics_data:
                columns = [
                    'Batch', 'Day', 'Run', 'Organoid_Mode', 'Image_ID', 'Well_ID',
                    'Plate_Index', 'Well_Index', 'Organoid_ID', 'Object_Label',
                    'Object_Count_In_Image',
                    'Area_um2', 'Diameter_um', 'Perimeter_um',
                    'Feret_Diameter_Max_um', 'Aspect_Ratio', 'Circularity',
                    'Roundness', 'Solidity', 'Eccentricity', 'Skeleton_Length_um'
                ]
                df = pd.DataFrame(metrics_data, columns=columns)
                df.to_csv(os.path.join(metrics_path, 'mask_metrics.csv'), index=False)
                image_summary = _summarize_metrics_by_image(df, feature_columns)
                image_summary.to_csv(os.path.join(metrics_path, 'well_metrics_summary.csv'), index=False)
                                         
def detect_outliers_morphology(
    base_dir,
    project_dir,
    days,
    batches,
    manual_thresholds=None,
    features = [
        'Area_um2', 'Diameter_um', 
        'Skeleton_Length_um', 'Solidity', 'Eccentricity'
    ],
    mad_multiplier=3,
    save_global_summary=True,
    global_summary_filename='outliers_all_batches_summary.csv'
):
    """
    Detects outlier organoids using either manual thresholds or MAD-based filtering.
    Saves per-batch/day CSVs and optionally a global summary CSV with mean and std info.

    Parameters:
    - base_dir: Root directory for data.
    - project_dir: Project directory name.
    - days: List of days to analyze.
    - batches: List of batch names.
    - manual_thresholds: Optional dictionary {day: threshold} for manual outlier detection.
    - feature: List of morphological features to analyze (e.g., 'Area_um2').
    - mad_multiplier: Multiplier for MAD thresholding.
    - save_global_summary: If True, saves combined CSV for all batches/days.
    - global_summary_filename: Output filename for global summary.
    """
    for feature in features:
        print(f"\n📊 Detecting Outliers for Feature: {feature}")
        global_summary_data = []

        for day in days:
            for batch in batches:
                metrics_path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_metrics')
                csv_path = os.path.join(metrics_path, 'mask_metrics.csv')
                if not os.path.exists(csv_path):
                    print(f"⚠️  Metrics file not found for {batch} on Day {day}. Skipping...")
                    continue

                df = pd.read_csv(csv_path)

                if feature not in df.columns:
                    print(f"⚠️  Feature '{feature}' not found in metrics for {batch} on Day {day}. Skipping...")
                    continue

                values = df[feature]
                mean = values.mean()
                std = values.std()

                if manual_thresholds and day in manual_thresholds:
                    lower_threshold, upper_threshold = manual_thresholds[day]
                    method = "Manual"
                else:
                    median = np.median(values)
                    mad = np.median(np.abs(values - median))
                    lower_threshold = median - mad_multiplier * mad
                    upper_threshold = median + mad_multiplier * mad
                    method = f"MAD ±{mad_multiplier}"

                # Detect outliers on both sides
                outliers = df[(df[feature] < lower_threshold) | (df[feature] > upper_threshold)]

                outlier_ids = outliers['Organoid_ID'].tolist()
                total = len(df)
                num_outliers = len(outliers)
                if num_outliers == 0:
                    continue
                else:
                    percent = (num_outliers / total) * 100 if total > 0 else 0

                    summary_row = {
                        'Day': day,
                        'batch': batch,
                        'Feature': feature,
                        'Detection_Method': method,
                        'Lower_Threshold': lower_threshold,
                        'Upper_Threshold': upper_threshold,
                        'Mean': mean,
                        'Std_Dev': std,
                        'Total_Objects': total,
                        'Outlier_Count': num_outliers,
                        'Outlier_Percent': percent,
                        'Outlier_IDs': ', '.join(outlier_ids)
                    }

                    global_summary_data.append(summary_row)

                    # Save individual summary
                    individual_csv = os.path.join(metrics_path, f'{feature}_outliers_summary.csv')
                    if os.path.exists(individual_csv):
                        existing = pd.read_csv(individual_csv)
                        combined = pd.concat([existing, pd.DataFrame([summary_row])], ignore_index=True)
                    else:
                        combined = pd.DataFrame([summary_row])
                    combined.to_csv(individual_csv, index=False)

        # Save global summary if enabled
        if save_global_summary and global_summary_data:
            global_df = pd.DataFrame(global_summary_data)
            global_path = os.path.join(base_dir, project_dir, global_summary_filename)
            if os.path.exists(global_path):
                existing = pd.read_csv(global_path)
                combined = pd.concat([existing, global_df], ignore_index=True)
            else:
                combined = global_df
            combined.to_csv(global_path, index=False)

            print(f"\n📁 Global outlier summary saved to: {global_path}")
