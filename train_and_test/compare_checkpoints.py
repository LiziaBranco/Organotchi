import os
import pandas as pd
import matplotlib.pyplot as plt

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

# Specify the directory where summary CSV files and plots are stored
data_directory = 'Training_Results/work_dirs'
plot_directory = 'Training_Results/plots'
plot_means_stds_across_checkpoints(data_directory, plot_directory)
