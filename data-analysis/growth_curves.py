import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_pca_for_all_plates

checkpoint = 'Training-Results/work_dirs/split-0/iter_144000.pth'
base_dir = '../../../AGMayer/Lizia/Organoid_Segmentation'

#project_dir = '2025_03_05-Dasha-DMA'
#project_dir = '2025_03_05-Theresa-SmallScale-Trial'
project_dir = '2025_03_05-Quality-Prediction'

days = [1, 2, 3, 4, 6, 7, 8, 9, 10]
plates = ['AB108', 'QB01_1', 'QB01_2', "SB60"]

# Create directory for saving plots
plot_dir = os.path.join(base_dir, project_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

def save_plot(fig, filename):
    """ Saves the figure to the designated plots directory. """
    filepath = os.path.join(plot_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {filepath}")

def plot_plate_average_growth(base_dir, project_dir, days, plates):
    """
    Plots the average growth of each plate over time and saves the figure.
    
    Parameters:
    - plate (str): The specific plate to analyze. If None, processes all plates.
    """
    plate_growth = {}  # Dictionary to store mean diameters per plate
    plate_std = {}     # Dictionary to store standard deviations per plate

    for day in days:
        batch_processed_path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}')

        for plate in plates:
            metrics_csv_path = os.path.join(batch_processed_path, plate, 'inference_metrics', 'mask_metrics.csv')

            if os.path.exists(metrics_csv_path):
                df = pd.read_csv(metrics_csv_path)

                if 'Diameter_um' in df.columns:
                    df = df[df['Diameter_um'] > 170]  # Keep only values greater than 170
                    df = df[df['Diameter_um'] < 1000]
                    df_filtered = df[df['Circularity'] > 0.70]  # Keep only values greater than 170
                    #df_filtered = df_filtered[df_filtered['Diameter_um'] < 500]
                    
                    if not df_filtered.empty:
                        mean_diameter = np.mean(df_filtered['Diameter_um'])
                        std_diameter = np.std(df_filtered['Diameter_um'])
                    else:
                        mean_diameter = np.nan
                        std_diameter = np.nan
                else:
                    print(f"Warning: 'Diameter_um' column missing in {metrics_csv_path}")
                    mean_diameter = np.nan
                    std_diameter = np.nan
            else:
                print(f"Warning: CSV file not found for {plate} on Day {day}")
                mean_diameter = np.nan
                std_diameter = np.nan

            # Store growth data per plate
            if plate not in plate_growth:
                plate_growth[plate] = []
                plate_std[plate] = []

            plate_growth[plate].append(mean_diameter)
            plate_std[plate].append(std_diameter)

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))

    for plate, growth_values in plate_growth.items():
        std_values = plate_std[plate]
        ax.errorbar(days, growth_values, yerr=std_values, label=plate,
                    marker='o', capsize=5, linestyle='-')

    ax.set_xlabel("Days")
    ax.set_ylabel("Average Diameter (microns)")
    ax.set_title(f"Average Organoid Growth per Plate Over Time")
    ax.legend(title="Plates", loc="best")
    ax.grid(True)

    # Save and show plot
    filename = f"plate_average_growth.png"
    save_plot(fig, filename)
    plt.show()

def plot_organoid_growth(base_dir, project_dir, days, plate, organoid_indices):
    """
    Plots the growth of specific organoids over time and saves the figure.
    
    Parameters:
    - plate (str): Name of the plate to analyze.
    - organoid_indices (list of int): List of organoid row indices to track.
    """
    organoid_data = {idx: [] for idx in organoid_indices}  # Store data per organoid

    for day in days:
        batch_processed_path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}')
        metrics_csv_path = os.path.join(batch_processed_path, plate, 'inference_metrics', 'mask_metrics.csv')

        if os.path.exists(metrics_csv_path):
            df = pd.read_csv(metrics_csv_path)

            if 'Diameter_um' in df.columns:
                for idx in organoid_indices:
                    if idx < len(df):  # Ensure index is within bounds
                        organoid_data[idx].append(df.loc[idx, 'Diameter_um'])
                    else:
                        organoid_data[idx].append(np.nan)  # Fill missing data with NaN
            else:
                print(f"Warning: 'Diameter_um' column missing in {metrics_csv_path}")
                for idx in organoid_indices:
                    organoid_data[idx].append(np.nan)
        else:
            print(f"Warning: CSV file not found for {plate}")
            for idx in organoid_indices:
                organoid_data[idx].append(np.nan)

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, values in organoid_data.items():
        ax.plot(days, values, marker='o', linestyle='-', label=f"Organoid {idx+1}")

    ax.set_xlabel("Days")
    ax.set_ylabel("Diameter (microns)")
    ax.set_title(f"Growth of selected organoids in {plate}")
    ax.legend()
    ax.grid(True)

    # Save and show plot
    filename = f"organoid_growth_{plate}.png"
    save_plot(fig, filename)
    plt.show()

# Example usage: track specific plates and organoids
plot_plate_average_growth(base_dir, project_dir, days, plates)
plot_organoid_growth(base_dir, project_dir, days, plate="AB108", organoid_indices=[8, 12, 16, 56])
plot_organoid_growth(base_dir, project_dir, days, plate="QB01_1", organoid_indices=[8, 12, 16, 56])
plot_organoid_growth(base_dir, project_dir, days, plate="QB01_2", organoid_indices=[8, 12, 16, 56])
plot_organoid_growth(base_dir, project_dir, days, plate="SB60", organoid_indices=[8, 12, 16, 56])
plot_pca_for_all_plates(base_dir, project_dir, days, plates)