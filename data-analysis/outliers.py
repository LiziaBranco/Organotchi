import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt

from experiment_info import *

plot_dir = os.path.join(base_dir, project_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

def save_plot(fig, filename):
    path = os.path.join(plot_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {path}")

def load_outlier_summary(csv_path):
    """Load the global summary CSV into a DataFrame."""
    return pd.read_csv(csv_path)

def merge_plate_metadata(summary_df, plates_info):
    """Attach condition, cell-line, and batch info to the summary DataFrame."""
    metadata_df = pd.DataFrame(plates_info)
    metadata_df = metadata_df.rename(columns={'batch': 'batch'})
    return summary_df.merge(metadata_df, on='batch', how='left')

def plot_outlier_rate_over_days(df, group_by='condition', feature='Area_um2'):
    """Plot outlier % over days for each group (e.g., condition, cell-line, batch)."""
    filtered_df = df[df['Feature'] == feature].copy()

    # Define consistent color palette
    unique_groups = sorted(filtered_df[group_by].dropna().unique())
    palette = sns.color_palette("husl", len(unique_groups))
    group_color_map = dict(zip(unique_groups, palette))

    fig = plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=filtered_df,
        x='Day',
        y='Outlier_Percent',
        hue=group_by,
        marker='o',
        palette=group_color_map
    )
    plt.title(f'Outlier Rate Over Time by {group_by.capitalize()} ({feature})')
    plt.ylabel('Outlier Percentage')
    plt.xlabel('Day')
    plt.grid(True)
    plt.legend(title=group_by)
    plt.tight_layout()
    save_plot(fig, f"{feature}_outlier_rate_over_time_{group_by}.png")
    plt.show()


def plot_outlier_distribution(df, group_by='condition', feature='Area_um2'):
    """Boxplot of outlier percentages grouped by condition/cell-line/batch."""
    filtered_df = df[df['Feature'] == feature].copy()

    # Define consistent color palette
    unique_groups = sorted(filtered_df[group_by].dropna().unique())
    palette = sns.color_palette("husl", len(unique_groups))
    group_color_map = dict(zip(unique_groups, palette))

    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=filtered_df,
        x=group_by,
        y='Outlier_Percent',
        palette=group_color_map
    )
    sns.stripplot(
        data=filtered_df,
        x=group_by,
        y='Outlier_Percent',
        color='black',
        alpha=0.5,
        jitter=True
    )
    plt.title(f'Outlier Rate Distribution by {group_by.capitalize()} ({feature})')
    plt.ylabel('Outlier Percentage')
    plt.xlabel(group_by.capitalize())
    plt.grid(True)
    plt.tight_layout()
    save_plot(fig, f"{feature}_outlier_distribution_{group_by}.png")
    plt.show()

summary_df = load_outlier_summary(f'{base_dir}/{project_dir}/outliers_all_batches_summary.csv')
merged_df = merge_plate_metadata(summary_df, batches_complete_info)

plot_outlier_rate_over_days(merged_df, group_by='condition')
plot_outlier_distribution(merged_df, group_by='cell-line')