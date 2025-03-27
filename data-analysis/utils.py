import numpy as np
from torch import nn
from PIL import Image
from os.path import join as opj
from mmengine.config import Config
from mmseg.apis import inference_model, init_model
import os
import matplotlib.pyplot as plt
import pandas as pd
from skimage import measure, morphology
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import markers
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import cv2

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

        if circularity > circularity_threshold and diameter > 170:
            # If this structure is larger than the previously largest one, update
            if area > max_area:
                max_area = area
                largest_structure_label = prop.label

    # Retain the largest valid structure
    if largest_structure_label != -1:
        filtered_mask[labeled_mask == largest_structure_label] = 255
        structure_count = 1  # We are keeping only one structure

    return filtered_mask, structure_count


def organoid_segmentation_inference(checkpoint, base_dir, project_dir, days, plates):
    for day in days:
        for plate in plates:
            raw_data_dir = opj(base_dir, project_dir, 'raw_data', f'Day {day}', plate)
            processed_data_dir = opj(base_dir, project_dir, 'processed_data', f'Day {day}', plate)
            
            outdirs = {
                'inference_npy': opj(processed_data_dir, 'inference_npy'),
                'inference_tif': opj(processed_data_dir, 'inference_tif'),
                'inference_jpg': opj(processed_data_dir, 'inference_jpg'),
                'inference_comparison': opj(processed_data_dir, 'inference_comparison'),
                'inference_metrics': opj(processed_data_dir, 'inference_metrics')
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

def apply_filtering_to_segmentations(base_dir, project_dir, days, plates, pixel_to_micron_x, pixel_to_micron_y):
    """
    Applies high-circularity filtering to the segmentation masks in `inference_jpg/`
    and saves the filtered masks in:
    - `inference_filtered/` (JPG images)
    - `inference_filtered_npy/` (Numpy array masks)
    - `inference_comparison/` (Comparison images)
    """
    for day in days:
        print(day)
        for plate in plates:
            processed_data_dir = opj(base_dir, project_dir, 'processed_data', f'Day {day}', plate)
            inference_jpg_dir = opj(processed_data_dir, 'inference_jpg')
            filtered_jpg_dir = opj(processed_data_dir, 'inference_filtered')
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
                orig_img_path = opj(base_dir, project_dir, 'raw_data', f'Day {day}', plate, f"{img_id}.jpg")
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

     
def analyze_organoid_morphology(base_dir, project_dir, days, plates, pixel_to_micron_x, pixel_to_micron_y):
    """Extracts morphological features from organoid masks and saves them to CSV."""
    for day in days:
        batch_processed_path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}')
        for plate in plates:
            inference_npy_path = os.path.join(batch_processed_path, plate, 'inference_filtered_npy')
            metrics_path = os.path.join(batch_processed_path, plate, 'inference_metrics')
            os.makedirs(metrics_path, exist_ok=True)
            
            metrics_data = []
            
            for filename in os.listdir(inference_npy_path):
                if filename.endswith('.npy'):
                    mask = np.load(os.path.join(inference_npy_path, filename))
                    mask = mask > 0  # Ensure binary
                    
                    properties = measure.regionprops(mask.astype(int))
                    
                    for prop in properties:
                        area_pixels = prop.area
                        area = prop.area * pixel_to_micron_x * pixel_to_micron_y
                        diameter = np.sqrt(4 * area / np.pi)
                        perimeter = prop.perimeter * pixel_to_micron_x
                        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else np.nan
                        skeleton = morphology.skeletonize(mask)
                        curvature = np.sum(skeleton) * pixel_to_micron_x  # Approximate curvature
                        
                        metrics_data.append([filename.replace('.npy', ''), area_pixels, area, diameter, perimeter, circularity, curvature])
            
            df = pd.DataFrame(metrics_data, columns=['Organoid_ID', 'Area_px2', 'Area_um2', 'Diameter_um', 'Perimeter_um', 'Circularity', 'Curvature_um'])
            csv_path = os.path.join(metrics_path, 'mask_metrics.csv')
            df.to_csv(csv_path, index=False)
            print(f"Metrics saved: {csv_path}")
            
def plot_pca_for_all_plates(base_dir, project_dir, days, plates):
    """Performs PCA on organoid morphology data from all plates and visualizes the results."""
    all_metrics = []
    
    for day in days:
        batch_processed_path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}')
        if not os.path.exists(batch_processed_path):
            continue
        
        for plate in plates:
            metrics_csv = os.path.join(batch_processed_path, plate, 'inference_metrics', 'mask_metrics.csv')
            if os.path.exists(metrics_csv):
                df = pd.read_csv(metrics_csv)
                df = df[df['Diameter_um'] > 170]  # Keep only values greater than 170
                df = df[df['Diameter_um'] < 1000]
                df = df[df['Circularity'] > 0.40]  # Keep only values greater than 170   
                
                df['Plate'] = plate
                df['Day'] = day
                all_metrics.append(df)
    
    if not all_metrics:
        print("No data found for PCA.")
        return
    
    df_all = pd.concat(all_metrics, ignore_index=True)
    features = ['Area_um2', 'Perimeter_um', 'Circularity', 'Curvature_um']
    df_all.dropna(subset=features, inplace=True)
    
    X = df_all[features].values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    
    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    df_pca['Plate'] = df_all['Plate'].values
    df_pca['Day'] = df_all['Day'].values

    # Define colors for days and markers for plates
    unique_days = sorted(df_pca['Day'].unique())
    unique_plates = sorted(df_pca['Plate'].unique())
    plate_markers = list(markers.MarkerStyle.markers.keys())  # Different shapes for plates
    plate_markers = ["o", "v", "s", "*"]
    print(plate_markers)

    # Normalize day values for color mapping
    day_norm = mcolors.Normalize(vmin=min(unique_days), vmax=max(unique_days))
    day_cmap = cm.get_cmap('gnuplot2')  # Use the gnuplot2 colormap

    fig, ax = plt.subplots(figsize=(10, 6))

    # Dictionaries to store handles for legends
    plate_handles = {}
    scatter_plots = []

    for i, plate in enumerate(unique_plates):
        for j, day in enumerate(unique_days):
            subset = df_pca[(df_pca['Plate'] == plate) & (df_pca['Day'] == day)]
            if not subset.empty:
                color = day_cmap(day_norm(day))  # Get color from colormap based on day value

                scatter = ax.scatter(
                    subset['PC1'], subset['PC2'],
                    color=color, marker=plate_markers[i], 
                    alpha=0.7
                )
                scatter_plots.append(scatter)
                
                # Store one handle per plate
                if plate not in plate_handles:
                    plate_handles[plate] = mlines.Line2D([], [], color='black', 
                                                        marker=plate_markers[i % len(plate_markers)], 
                                                        linestyle='None', markersize=8, label=f"Plate {plate}")

    # Set labels and title
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title('PCA of Organoid Morphology')
    ax.grid(True)

    # Create separate legend for plates (shapes)
    legend1 = ax.legend(handles=plate_handles.values(), title='Plates', loc='lower right', fontsize='small')

    # Add colorbar for day gradient
    sm = cm.ScalarMappable(cmap=day_cmap, norm=day_norm)
    sm.set_array([])  # Required for colorbar to work
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Days')

    # Add legend for shapes
    ax.add_artist(legend1)

    # Save and show plot
    plot_dir = os.path.join(base_dir, project_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    filename = os.path.join(plot_dir, 'PCA_organoid_morphology.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"PCA plot saved: {filename}")
