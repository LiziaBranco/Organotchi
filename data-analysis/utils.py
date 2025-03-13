import numpy as np
from torch import nn
from PIL import Image
from os.path import join as opj
from mmengine.config import Config
from mmseg.apis import inference_model, init_model
import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import tifffile
from skimage.measure import regionprops, label
from skimage.morphology import skeletonize
from skimage import measure, morphology, io
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import markers
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

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

                # Generate comparison image
                orig_img = Image.open(img)
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(orig_img)
                ax[0].set_title('Original Image')
                ax[0].axis('off')
                ax[1].imshow(pred, cmap='gray')
                ax[1].set_title('Predicted Mask')
                ax[1].axis('off')
                
                # Save comparison figure
                plt.savefig(opj(outdirs['inference_comparison'], f'{img_id}.jpg'), bbox_inches='tight')
                plt.close()
            
def analyze_organoid_morphology(base_dir, project_dir, days, plates, pixel_to_micron_x, pixel_to_micron_y):
    """Extracts morphological features from organoid masks and saves them to CSV."""
    for day in days:
        batch_processed_path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}')
        for plate in plates:
            inference_npy_path = os.path.join(batch_processed_path, plate, 'inference_npy')
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
                df = df[df['Circularity'] > 0.70]  # Keep only values greater than 170   
                
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