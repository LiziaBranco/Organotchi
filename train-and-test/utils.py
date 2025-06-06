import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from PIL import Image
from sklearn.metrics import jaccard_score, accuracy_score

def dice_coefficient(pred, target):
    """Calculate Dice Coefficient (F1 Score)."""
    intersection = np.sum(pred * target)
    dice = (2.0 * intersection) / (np.sum(pred) + np.sum(target) + 1e-6)  # Avoid division by zero
    return dice

def iou_score(pred, target):
    """Calculate Intersection over Union (IoU)."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    iou = jaccard_score(target_flat, pred_flat, average='weighted')
    return iou

def accuracy(pred, target):
    """Calculate accuracy score."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    return accuracy_score(target_flat, pred_flat)

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

def save_metrics_to_csv(metrics_data, csv_path):
    """Save metrics data to a CSV file."""
    metrics_df = pd.DataFrame(metrics_data, columns=['Image_ID', 'Dice', 'IoU', 'Accuracy'])
    metrics_df.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")

def create_directories(directories):
    """Create directories if they don't exist."""
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def aggregate_metrics(metrics_data):
    """Aggregate metrics data to compute mean, std, min, max for each metric."""
    metrics_df = pd.DataFrame(metrics_data, columns=['Image_ID', 'Dice', 'IoU', 'Accuracy'])
    
    # Compute the statistics for each metric
    stats = {
        'Dice_Min': metrics_df['Dice'].min(),
        'Dice_Max': metrics_df['Dice'].max(),
        'Dice_Mean': metrics_df['Dice'].mean(),
        'Dice_Std': metrics_df['Dice'].std(),
        
        'IoU_Min': metrics_df['IoU'].min(),
        'IoU_Max': metrics_df['IoU'].max(),
        'IoU_Mean': metrics_df['IoU'].mean(),
        'IoU_Std': metrics_df['IoU'].std(),
        
        'Acc_Min': metrics_df['Accuracy'].min(),
        'Acc_Max': metrics_df['Accuracy'].max(),
        'Acc_Mean': metrics_df['Accuracy'].mean(),
        'Acc_Std': metrics_df['Accuracy'].std(),
    }

    return stats

def load_image_as_array(image_path):
    """Load an image as a numpy array."""
    try:
        img = Image.open(image_path)
        return np.array(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def plot_metrics_from_csv(project_directory, csv_file_path, split):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Ensure checkpoints are sorted before plotting
    df = df.sort_values(by='Iteration')

    # Extract the data for plotting
    checkpoints = df['Iteration']
    dice_means = df['Dice_Mean']
    iou_means = df['IoU_Mean']
    acc_means = df['Acc_Mean']
    
    dice_stds = df['Dice_Std']
    iou_stds = df['IoU_Std']
    acc_stds = df['Acc_Std']

    # Plot Means as connected lines with markers
    plt.figure(figsize=(10, 6))
    plt.plot(checkpoints, dice_means, marker='o', label='Dice Mean', color='b')
    plt.plot(checkpoints, iou_means, marker='x', label='IoU Mean', color='g')
    plt.plot(checkpoints, acc_means, marker='^', label='Accuracy Mean', color='r')
    
    plt.title(f'Mean Metrics Across Checkpoints - Split {split}')
    plt.xlabel('Checkpoints (Iterations)')
    plt.ylabel('Mean Values')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{project_directory}/Training-Results/plots/mean_metrics_split_{split}.png')  # Save with split info
    plt.show()

    # Plot Stds as connected lines with markers
    plt.figure(figsize=(10, 6))
    plt.plot(checkpoints, dice_stds, marker='o', label='Dice Std', color='b')
    plt.plot(checkpoints, iou_stds, marker='x', label='IoU Std', color='g')
    plt.plot(checkpoints, acc_stds, marker='^', label='Accuracy Std', color='r')
    
    plt.title(f'Standard Deviations of Metrics Across Checkpoints - Split {split}')
    plt.xlabel('Checkpoints (Iterations)')
    plt.ylabel('Standard Deviation Values')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{project_directory}/Training-Results/plots/std_metrics_split_{split}.png')  # Save with split info
    plt.show()

def load_and_convert_npy_image(npy_path):
    """Load .npy file, convert 0s and 1s to 255 for visualization."""
    array = np.load(npy_path)
    img = Image.fromarray((array * 255).astype(np.uint8))  # Convert binary mask to an image
    return img

def load_jpg_image(jpg_path):
    """Load a jpg image."""
    return Image.open(jpg_path)

def save_concatenated_image(ground_truth_path, inference_path, original_images, output_folder, image_name):
    # Define paths
    ground_truth_path = os.path.join(ground_truth_path, f"{image_name}.npy")
    inference_path = os.path.join(inference_path, f"{image_name}.npy")
    jpg_path = os.path.join(original_images, f"{image_name}.jpg")

    # Check if all the required files exist
    if not (os.path.exists(ground_truth_path) and os.path.exists(inference_path) and os.path.exists(jpg_path)):
        print(f"Skipping {image_name} due to missing files.")
        return

    # Load images
    img1 = load_and_convert_npy_image(ground_truth_path)
    img2 = load_and_convert_npy_image(inference_path)
    img3 = load_jpg_image(jpg_path)

    # Resize images to the same size (optional, adjust as necessary)
    img2 = img2.resize(img1.size)
    img3 = img3.resize(img1.size)

    # Concatenate the images horizontally
    total_width = img1.width + img2.width + img3.width
    max_height = max(img1.height, img2.height, img3.height)
    
    # Create a new blank image to hold the concatenated result
    concatenated_image = Image.new('RGB', (total_width, max_height))

    # Paste the images side by side
    concatenated_image.paste(img1.convert('RGB'), (0, 0))  # Convert to RGB for consistency
    concatenated_image.paste(img2.convert('RGB'), (img1.width, 0))
    concatenated_image.paste(img3.convert('RGB'), (img1.width + img2.width, 0))

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save the concatenated image
    output_path = os.path.join(output_folder, f"{image_name}_concatenated.jpg")
    concatenated_image.save(output_path)
    print(f"Concatenated image saved at {output_path}")

def process_all_npy_images(ground_truth_path, inference_path, original_images, output_folder):
    """Process all npy files in npy_folder2 and save concatenated images in output_folder."""
    for npy_file in os.listdir(inference_path):
        if npy_file.endswith('.npy'):
            # Extract the base image name (without the .npy extension)
            image_name = os.path.splitext(npy_file)[0]

            # Call the function to process and save the concatenated image
            save_concatenated_image(ground_truth_path, inference_path, original_images, output_folder, image_name)
            
def plot_means_stds_across_checkpoints(data_dir, plot_dir, splits=5):
    """
    Generate plots comparing means and standard deviations across checkpoints for each split,
    ensuring the checkpoints are ordered.
    
    Args:
        data_dir (str): Directory containing the summary CSV files (split{split}_summary_metrics.csv).
        plot_dir (str): Directory to save the generated plots.
        splits (int): Number of splits (default is 5).
    """
    # Ensure the plot directory exists
    os.makedirs(plot_dir, exist_ok=True)
    
    # Initialize figures with three subplots for means and stds
    fig_means, ax_means = plt.subplots(1, 3, figsize=(18, 5))
    fig_stds, ax_stds = plt.subplots(1, 3, figsize=(18, 5))
    metric_names = ['Dice', 'IoU', 'Acc']
    
    for split in range(splits):
        # Load the summary metrics CSV for each split
        summary_csv_path = os.path.join(data_dir, f'split-{split}/split{split}_summary_metrics.csv')
        if not os.path.exists(summary_csv_path):
            print(f"Summary file for split {split} not found at {summary_csv_path}")
            continue
        
        # Read the CSV data into a DataFrame and sort by Iteration
        summary_df = pd.read_csv(summary_csv_path)
        summary_df = summary_df.sort_values(by='Iteration')  # Ensure iterations are ordered
        
        # Extract ordered iteration numbers for x-axis
        iterations = summary_df['Iteration']
        
        # Plot each metric's mean and std
        for idx, metric in enumerate(metric_names):
            mean_col = f'{metric}_Mean'
            std_col = f'{metric}_Std'
            
            # Plot mean
            ax_means[idx].plot(iterations, summary_df[mean_col], marker='o', label=f'Split {split}')
            ax_means[idx].set_title(f'{metric} Mean across Checkpoints')
            ax_means[idx].set_xlabel('Iteration')
            ax_means[idx].set_ylabel(f'{metric} Mean')
            ax_means[idx].legend(loc='best')
            
            # Plot standard deviation
            ax_stds[idx].plot(iterations, summary_df[std_col], marker='o', label=f'Split {split}')
            ax_stds[idx].set_title(f'{metric} Standard Deviation across Checkpoints')
            ax_stds[idx].set_xlabel('Iteration')
            ax_stds[idx].set_ylabel(f'{metric} Std')
            ax_stds[idx].legend(loc='best')

    # Adjust layout for clarity and save figures
    fig_means.tight_layout()
    fig_stds.tight_layout()
    fig_means.savefig(os.path.join(plot_dir, 'means_across_checkpoints.png'))
    fig_stds.savefig(os.path.join(plot_dir, 'stds_across_checkpoints.png'))
    plt.show()