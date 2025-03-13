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
    iou = jaccard_score(target_flat, pred_flat, average='binary')
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

def plot_metrics_from_csv(csv_file_path, split):
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
    plt.savefig(f'Training_Results/plots/mean_metrics_split_{split}.png')  # Save with split info
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
    plt.savefig(f'Training_Results/plots/std_metrics_split_{split}.png')  # Save with split info
    plt.show()

