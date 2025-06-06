import numpy as np
from torch import nn
from PIL import Image
from os.path import join as opj
from mmengine.config import Config
from mmseg.apis import inference_model, init_model
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
from skimage import measure, morphology
import cv2
import glob
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="mmengine.runner.checkpoint")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="mmengine.optim.optimizer.zero_optimizer")

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
                base_match = re.split(r"[ _]", folder)[0]
                base = base_match.strip()
                if base == batch:
                    # Find 'XinY' pattern (like 1in2) anywhere in the name
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

                    
def all_eval_except_dropout(m):
    """Ensure dropout layers stay in evaluation mode."""
    if not isinstance(m, nn.Dropout) and not isinstance(m, nn.Dropout2d):
        m.eval()
    else:
        if isinstance(m, nn.Dropout2d):
            m.p = 1.0
            m.inplace = True
            m.train()

def save_as_tif(npy_array, save_path):
    """Save the npy array as a TIF image."""
    tif_image = Image.fromarray((npy_array * 255).astype(np.uint8))  # Convert binary mask to uint8
    tif_image.save(save_path)
    
def filter_and_blackout_low_circularity(mask, pixel_to_micron_x, pixel_to_micron_y, circularity_threshold=0.2):
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


def organoid_segmentation_inference(checkpoint, base_dir, project_dir, days, batches):
    print(f"\n🤖 Automated Segmentation Started.")
    for day in days:
        print(f"- Processing Day {day}")
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
                if img_loc.lower().endswith('.tif'):
                    img = Image.open(img_path)
                    jpg_path = opj(raw_data_dir, os.path.splitext(img_loc)[0] + '.jpg')
                    img_8bit = Image.fromarray((np.array(img) / 256).astype(np.uint8))

                    # Convert grayscale to RGB
                    img_rgb = img_8bit.convert("RGB")

                    # Save as JPG
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
             
def stitch_24wellplate_images(image_paths, output_path):
    """Stitch four images into a 2x2 grid and save the result."""
    if len(image_paths) != 4:
        print(f"Skipping set, expected 4 images but got {len(image_paths)}")
        return
    
    images = [Image.open(img) for img in image_paths]
    
    # Ensure all images have the same size
    widths, heights = zip(*(img.size for img in images))
    if len(set(widths)) > 1 or len(set(heights)) > 1:
        print("Skipping set, images have different sizes.")
        return

    tile_width, tile_height = images[0].size
    stitched_image = Image.new("RGB", (tile_width * 2, tile_height * 2))

    # Place images in the correct positions
    stitched_image.paste(images[0], (0, 0))  # Top left
    stitched_image.paste(images[1], (tile_width, 0))  # Top right
    stitched_image.paste(images[2], (tile_width, tile_height))  # Bottom left
    stitched_image.paste(images[3], (0, tile_height))  # Bottom right

    stitched_image.save(output_path)
    print(f"Saved: {output_path}")

def process_24wellplate_images(base_dir, project_dir, days, batch_names, project_batches):
    """Process images from raw_data and save stitched images in processed_data."""
    
    for day in days:
        raw_data_dir = opj(base_dir, project_dir, 'raw_data', f'Day {day}')
        
        processed_data_dir = opj(base_dir, project_dir, 'processed_data', f'Day {day}')
        os.makedirs(processed_data_dir, exist_ok=True)
        
        for project, num_plates in zip(batch_names, project_batches):
            clean_folder_names(raw_data_dir, project)
            project_output_dir = opj(raw_data_dir, project)
            os.makedirs(project_output_dir, exist_ok=True)

            # Gather images from all subfolders for this project
            all_images = []
            for batch in range(1, num_plates + 1):
                folder_name = f"{project}_{batch}in{num_plates}"
                folder_path = opj(raw_data_dir, folder_name)
                
                if os.path.exists(folder_path):
                    image_files = sorted(glob.glob(opj(folder_path, "*.jpg")))
                    all_images.extend(image_files)
                else:
                    folder_name = f"{project} {batch}in{num_plates}"
                    folder_path = opj(raw_data_dir, folder_name)
                    
                    if os.path.exists(folder_path):
                        image_files = sorted(glob.glob(opj(folder_path, "*.jpg")))
                        all_images.extend(image_files)                
                    else:
                        print(f"Warning: {folder_path} does not exist.")

            # Group images into sets of four
            image_groups = [all_images[i:i+4] for i in range(0, len(all_images), 4)]

            for idx, group in enumerate(image_groups, start=1):
                output_filename = opj(project_output_dir, f"Tile0000{idx:02d}.jpg")
                stitch_24wellplate_images(group, output_filename)

def filter_contours(mask, pixel_to_micron_x, pixel_to_micron_y, max_area=3000000, min_area=30000, min_circularity=0.2, center_fraction=0.95):
    """
    Filters contours based on area, circularity, and position inside a central region.
    Only keeps objects within the central rectangle of the image.

    Additionally, if multiple objects pass the filter, only retains the largest one and 
    any others that are at least half its size.

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

    # If more than one valid contour, keep only those >= 50% of max area
    if len(valid_contours) > 1:
        max_area_found = max(area for _, area in valid_contours)
        valid_contours = [c for c in valid_contours if c[1] >= 0.7 * max_area_found]

    # Draw final filtered contours
    filtered_mask = np.zeros_like(mask)
    for contour, _ in valid_contours:
        cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return filtered_mask

def segment_image(image_path, pixel_to_micron_x, pixel_to_micron_y, threshold=128, max_area=3000000, min_area=30000, min_circularity=0.2, center_fraction=0.95):
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

    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img_array = np.array(img)

    # Initial binary mask (inverted: dark -> white, light -> black)
    binary_mask = np.where(img_array < threshold, 255, 0).astype(np.uint8)

    # Apply filtering to remove unwanted structures
    filtered_mask = filter_contours(binary_mask, pixel_to_micron_x, pixel_to_micron_y, max_area, min_area, min_circularity, center_fraction)

    return filtered_mask


def process_24wellplate_segmentation(base_dir, project_dir, days, project_names, pixel_to_micron_x, pixel_to_micron_y):
    """
    Process all images in stitched directories and save segmentation results.
    """
    for day in days:
        for project in project_names:
            raw_data_dir = opj(base_dir, project_dir, 'raw_data', f'Day {day}', project)
            project_dir_path = opj(base_dir, project_dir, 'processed_data', f'Day {day}', project)
            inference_jpg_dir = opj(project_dir_path, "inference_jpg")
            inference_npy_dir = opj(project_dir_path, "inference_npy")
            inference_tif_dir = opj(project_dir_path, "inference_tif")

            os.makedirs(inference_jpg_dir, exist_ok=True)
            os.makedirs(inference_npy_dir, exist_ok=True)
            os.makedirs(inference_tif_dir, exist_ok=True)

            # Find all tiles in the project directory
            tile_images = sorted(glob.glob(opj(raw_data_dir, "Tile*.jpg")))

            for tile_path in tile_images:
                tile_name = os.path.basename(tile_path).replace(".jpg", "")

                # Perform segmentation
                mask = segment_image(tile_path, pixel_to_micron_x, pixel_to_micron_y)

                # Save as JPEG
                mask_img = Image.fromarray(mask)
                mask_img.save(opj(inference_jpg_dir, f"{tile_name}.jpg"))

                # Save as NumPy array
                np.save(opj(inference_npy_dir, f"{tile_name}.npy"), mask)

                # Save as TIFF
                mask_img.save(opj(inference_tif_dir, f"{tile_name}.tif"))

                print(f"Saved masks for {tile_name} in {project_dir_path}")
                

def analyze_organoid_morphology(base_dir, project_dir, days, batches, pixel_to_micron_x, pixel_to_micron_y):
    """Extracts selected morphological features from organoid masks and saves them to CSV."""
    print(f"\n📐 Extracting Organoid Morphology Metrics")
    for day in days:
        print(f"- Processing Day {day}")
        batch_processed_path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}')
        for batch in batches:
            # Try 'inference_filtered_npy' first, otherwise use 'inference_npy'
            inference_npy_path = os.path.join(batch_processed_path, batch, 'inference_filtered_npy')
            if not os.path.exists(inference_npy_path):
                inference_npy_path = os.path.join(batch_processed_path, batch, 'inference_npy')

            # Ensure the fallback path exists before proceeding
            if not os.path.exists(inference_npy_path):
                print(f"Warning: No valid inference directory found for {batch} on Day {day}. Skipping...")
                continue

            metrics_path = os.path.join(batch_processed_path, batch, 'inference_metrics')
            os.makedirs(metrics_path, exist_ok=True)

            metrics_data = []

            for filename in os.listdir(inference_npy_path):
                if filename.endswith('.npy'):
                    mask = np.load(os.path.join(inference_npy_path, filename))
                    mask = mask > 0  # Ensure binary

                    properties = measure.regionprops(mask.astype(int))

                    for prop in properties:
                        area_pixels = prop.area
                        area = area_pixels * pixel_to_micron_x * pixel_to_micron_y
                        diameter = np.sqrt(4 * area / np.pi)
                        perimeter = prop.perimeter * pixel_to_micron_x
                        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else np.nan
                        skeleton = morphology.skeletonize(mask)
                        skeleton_length = np.sum(skeleton) * pixel_to_micron_x

                        # Convex hull for roughness index
                        convex_hull = morphology.convex_hull_image(mask)
                        convex_perimeter = measure.perimeter(convex_hull) * pixel_to_micron_x
                        roughness = perimeter / convex_perimeter if convex_perimeter > 0 else np.nan

                        # Additional useful descriptors
                        solidity = prop.solidity
                        eccentricity = prop.eccentricity

                        metrics_data.append([
                            filename.replace('.npy', ''),
                            area_pixels, area, diameter, perimeter, circularity,
                            skeleton_length, solidity, eccentricity, roughness
                        ])

            if metrics_data:
                df = pd.DataFrame(metrics_data, columns=[
                    'Organoid_ID', 'Area_px2', 'Area_um2', 'Diameter_um', 'Perimeter_um',
                    'Circularity', 'Skeleton_Length_um', 'Solidity', 'Eccentricity', 'Roughness_Index'
                ])
                csv_path = os.path.join(metrics_path, 'mask_metrics.csv')
                df.to_csv(csv_path, index=False)
            else:
                print(f"No valid masks found for {batch} on Day {day}. Skipping CSV generation.")
                
def detect_outliers_morphology(
    base_dir,
    project_dir,
    days,
    batches,
    manual_thresholds=None,
    features = [
        'Area_um2', 'Diameter_um', 'Perimeter_um', 'Circularity',
        'Skeleton_Length_um', 'Solidity', 'Eccentricity', 'Roughness_Index'
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