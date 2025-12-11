import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from itertools import cycle
from statannotations.Annotator import Annotator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.lines as mlines

from experiment_info import *

plot_dir = os.path.join(base_dir, project_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

# === Utilities ===

def save_plot(fig, filename):
    path = os.path.join(plot_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {path}")

def filter_by_diameter(df, day):
    if 'Diameter_um' not in df.columns:
        return pd.DataFrame()
    df = df[df['Circularity'] > 0.2]
    if day < 14:
        return df[(df['Diameter_um'] > 150) & (df['Diameter_um'] < 1000)]
    elif day >= 14 and day < 30:
        return df[(df['Diameter_um'] > 450) & (df['Diameter_um'] < 2000)]
    else:
        return df[(df['Diameter_um'] > 700) & (df['Diameter_um'] < 2000)] #return df[(df['Diameter_um'] > 300) & (df['Diameter_um'] < 1500)]

# === General Plotting Functions ===

def plot_boxplots(df, x, y, title, filename):
    unique_categories = df[x].nunique()
    palette = sns.color_palette("husl", unique_categories)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x=x, y=y, ax=ax, palette=palette)
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    save_plot(fig, filename)
    plt.show()
    plt.close()

def plot_lineplot(df, x, y, hue, yerr=None, title='', filename=''):
    unique_groups = df[hue].unique()
    palette = sns.color_palette("husl", len(unique_groups))
    color_map = dict(zip(unique_groups, palette))

    fig, ax = plt.subplots(figsize=(10, 6))
    for key, group in df.groupby(hue):
        ax.errorbar(group[x], group[y], yerr=group[yerr] if yerr else None,
                    label=key, marker='o', capsize=5, color=color_map[key])
    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()
    ax.grid(True)
    save_plot(fig, filename)
    plt.show()
    plt.close()


# === Plotting Wrappers ===

def plot_feature_boxplots(base_dir, project_dir, batches_complete_info, features, group_by="batch"):
    all_data = []
    for batch, info in batches_complete_info.items():
        for day in info["days"]:
            path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_metrics', 'mask_metrics.csv')
            if os.path.exists(path):
                df = pd.read_csv(path)
                df_filtered = filter_by_diameter(df, day)
                if not df_filtered.empty:
                    df_filtered['Group'] = info[group_by] if group_by in info else batch
                    all_data.append(df_filtered)

    if not all_data:
        print("No data available for boxplot comparison.")
        return

    df_all = pd.concat(all_data, ignore_index=True)

    for feature in features:
        if feature in df_all.columns:
            plot_boxplots(df_all, x='Group', y=feature,
                          title=f'{feature} Distribution by {group_by.capitalize()}',
                          filename=f'boxplot_{feature}_by_{group_by}.png')


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_feature_over_time(base_dir, project_dir, batches_complete_info, features, metric_names, group_by="batch"):
    """
    Plots the mean and standard deviation, the standard error of the mean (SEM), 
    and the 95% confidence interval of morphological features over time, grouped by a metadata field.

    Parameters:
    - base_dir: Base directory path.
    - project_dir: Project folder name.
    - batches_complete_info: List of dicts, each with 'batch', 'days', and metadata like 'condition' or 'cell-line'.
    - features: List of feature column names (or a single string).
    - metric_names: List of human-readable names corresponding to features.
    - group_by: Metadata field to group by ('batch', 'condition', or 'cell-line').
    """
    if isinstance(features, str):
        features = [features]

    for i, feature in enumerate(features):
        all_records = []

        for entry in batches_complete_info:
            batch = entry['batch']
            label = entry.get(group_by)

            for day in entry['days']:
                path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_metrics', 'mask_metrics.csv')
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df_filtered = filter_by_diameter(df, day)

                    if feature in df_filtered.columns and not df_filtered.empty:
                        df_filtered['Group'] = label
                        df_filtered['Day'] = day
                        all_records.append(df_filtered[['Group', 'Day', feature]])
                else:
                    print(f"⚠️  File not found: {path}")

        if all_records:
            df_all = pd.concat(all_records, ignore_index=True)

            # Prepare color palette
            unique_groups = sorted(df_all['Group'].unique())
            palette = sns.color_palette("husl", len(unique_groups))
            group_color_map = dict(zip(unique_groups, palette))

            # Compute summary statistics
            summary = df_all.groupby(['Group', 'Day'])[feature].agg(['mean', 'std', 'count']).reset_index()
            summary.rename(columns={'mean': f'Mean {metric_names[i]}', 'std': 'Std'}, inplace=True)
            summary['SEM'] = summary['Std'] / summary['count']**0.5

            # 1. Plot with SD error bars
            fig, ax = plt.subplots(figsize=(10, 6))
            for group in unique_groups:
                group_data = summary[summary['Group'] == group]
                ax.errorbar(
                    group_data['Day'],
                    group_data[f'Mean {metric_names[i]}'],
                    yerr=group_data['Std'],
                    label=group,
                    marker='o',
                    capsize=5,
                    color=group_color_map[group]
                )
            ax.set_title(f'Mean {metric_names[i]} Over Time by {group_by.capitalize()} (±SD)')
            ax.set_xlabel('Day')
            ax.set_ylabel(metric_names[i])
            ax.grid(True)
            ax.legend()
            plt.tight_layout()

            # Save SD error bar plot
            plot_dir = os.path.join(base_dir, project_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            filepath_sd = os.path.join(plot_dir, f"{feature}_over_time_by_{group_by}_sd.png")
            plt.savefig(filepath_sd, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"SD error bar plot saved: {filepath_sd}")

            # 2. Plot with SEM error bars
            fig, ax = plt.subplots(figsize=(10, 6))
            for group in unique_groups:
                group_data = summary[summary['Group'] == group]
                ax.errorbar(
                    group_data['Day'],
                    group_data[f'Mean {metric_names[i]}'],
                    yerr=group_data['SEM'],
                    label=group,
                    marker='o',
                    capsize=5,
                    color=group_color_map[group]
                )
            ax.set_title(f'Mean {metric_names[i]} Over Time by {group_by.capitalize()} (±SEM)')
            ax.set_xlabel('Day')
            ax.set_ylabel(metric_names[i])
            ax.grid(True)
            ax.legend()
            plt.tight_layout()

            filepath_sem = os.path.join(plot_dir, f"{feature}_over_time_by_{group_by}_sem.svg")
            plt.savefig(filepath_sem, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"SEM error bar plot saved: {filepath_sem}")

            # 3. Plot with 95% CI shaded regions using seaborn directly
            fig = plt.figure(figsize=(10, 6))
            sns.lineplot(
                data=df_all,
                x='Day',
                y=feature,
                hue='Group',
                errorbar=('ci', 95),
                marker='o',
                palette=group_color_map
            )
            plt.title(f'{metric_names[i]} Over Time by {group_by.capitalize()} (95% CI)')
            plt.ylabel(metric_names[i])
            plt.xlabel('Day')
            plt.grid(True)
            plt.tight_layout()

            filepath_ci = os.path.join(plot_dir, f"{feature}_over_time_by_{group_by}_ci.png")
            plt.savefig(filepath_ci, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"CI plot saved: {filepath_ci}")

def plot_pca_for_all_plates(base_dir, project_dir, days, batches_complete_info, group_by='batch'):
    """
    Performs PCA on organoid morphology metrics and visualizes results grouped by batch, condition, or cell-line.
    
    Parameters:
    - group_by: 'batch', 'condition', or 'cell-line'
    """
    #assert group_by in ['batch', 'condition', 'cell-line'], "group_by must be 'batch', 'condition', or 'cell-line'"

    all_metrics = []

    # Map batch name to its metadata group
    plate_group_map = {batch['batch']: batch[group_by] for batch in batches_complete_info}
    valid_plates = plate_group_map.keys()

    for day in days:
        batch_processed_path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}')
        if not os.path.exists(batch_processed_path):
            continue

        for batch in valid_plates:
            metrics_csv = os.path.join(batch_processed_path, batch, 'inference_metrics', 'mask_metrics.csv')
            if os.path.exists(metrics_csv):
                df = pd.read_csv(metrics_csv)

                # Basic filters
                if day < 14:
                    df = df[(df['Diameter_um'] > 150) & (df['Diameter_um'] < 1000)]
                else:
                    df = df[(df['Diameter_um'] > 300) & (df['Diameter_um'] < 2000)]
                df = df[df['Circularity'] > 0.2]  

                df['batch'] = batch
                df['Day'] = day
                df['Group'] = plate_group_map[batch]  # Add group info
                all_metrics.append(df)

    if not all_metrics:
        print("No data found for PCA.")
        return

    df_all = pd.concat(all_metrics, ignore_index=True)

    features = [
        'Area_um2', 'Diameter_um', 'Perimeter_um', 'Circularity',
        'Skeleton_Length_um', 'Solidity', 'Eccentricity', 'Roughness_Index'
    ]
    df_all.dropna(subset=features, inplace=True)

    # PCA
    X = df_all[features].values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    df_pca['Group'] = df_all['Group'].values
    df_pca['Day'] = df_all['Day'].values

    # Plotting
    unique_groups = sorted(df_pca['Group'].unique())
    unique_days = sorted(df_pca['Day'].unique())
    markers = ["o", "v", "s", "*", "D", "P", "X", "^"]
    color_map = cm.get_cmap('gnuplot2')
    day_norm = mcolors.Normalize(vmin=min(unique_days), vmax=max(unique_days))

    fig, ax = plt.subplots(figsize=(10, 6))
    group_handles = {}

    for i, group in enumerate(unique_groups):
        for day in unique_days:
            subset = df_pca[(df_pca['Group'] == group) & (df_pca['Day'] == day)]
            if not subset.empty:
                color = color_map(day_norm(day))
                ax.scatter(
                    subset['PC1'], subset['PC2'],
                    color=color,
                    marker=markers[i % len(markers)],
                    alpha=0.7
                )
        # Legend marker for the group
        group_handles[group] = mlines.Line2D(
            [], [], color='black',
            marker=markers[i % len(markers)],
            linestyle='None', markersize=8,
            label=f"{group}"
        )

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title(f'PCA of Organoid Morphology Grouped by {group_by.capitalize()}')
    ax.grid(True)

    legend1 = ax.legend(handles=group_handles.values(), title=group_by.capitalize(), loc='lower right', fontsize='small')

    sm = cm.ScalarMappable(cmap=color_map, norm=day_norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Days')
    ax.add_artist(legend1)

    # Save plot
    plot_dir = os.path.join(base_dir, project_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    filename = os.path.join(plot_dir, f'PCA_organoid_morphology_by_{group_by}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"PCA plot saved: {filename}")

def plot_pca_grid_by_group(base_dir, project_dir, days, batches_complete_info, group_by='batch'):
    """
    Creates a grid of PCA plots (one per day), grouped by batch, condition, or cell-line.

    Parameters:
    - group_by: 'batch', 'condition', or 'cell-line'
    """
    #assert group_by in ['batch', 'condition', 'cell-line'], "group_by must be 'batch', 'condition', or 'cell-line'"

    features = [
        'Area_um2', 'Diameter_um', 'Perimeter_um', 'Circularity',
        'Skeleton_Length_um', 'Solidity', 'Eccentricity', 'Roughness_Index'
    ]
    
    plate_group_map = {batch['batch']: batch[group_by] for batch in batches_complete_info}
    valid_plates = plate_group_map.keys()

    markers = ["o", "v", "s", "*", "D", "P", "X", "^"]
    color_map = plt.colormaps['tab10']

    n_cols = 3
    n_rows = int(np.ceil(len(days) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for idx, day in enumerate(days):
        all_metrics = []
        for batch in valid_plates:
            metrics_csv = os.path.join(
                base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_metrics', 'mask_metrics.csv'
            )
            if os.path.exists(metrics_csv):
                df = pd.read_csv(metrics_csv)

                # Filtering based on day
                if day < 14:
                    df = df[(df['Diameter_um'] > 150) & (df['Diameter_um'] < 1000)]
                else:
                    df = df[(df['Diameter_um'] > 300) & (df['Diameter_um'] < 2000)]
                df = df[df['Circularity'] > 0.2]

                df['Group'] = plate_group_map[batch]
                all_metrics.append(df)

        if not all_metrics:
            axes[idx].axis('off')
            axes[idx].set_title(f'Day {day}\nNo data')
            continue

        df_all = pd.concat(all_metrics, ignore_index=True)
        df_all.dropna(subset=features, inplace=True)

        X = df_all[features].values
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X_scaled)

        df_pca = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
        df_pca['Group'] = df_all['Group'].values

        unique_groups = sorted(df_pca['Group'].unique())
        ax = axes[idx]
        group_handles = {}

        for i, group in enumerate(unique_groups):
            subset = df_pca[df_pca['Group'] == group]
            ax.scatter(
                subset['PC1'], subset['PC2'],
                color=color_map(i % 10),
                marker=markers[i % len(markers)],
                label=group,
                alpha=0.7
            )
            group_handles[group] = mlines.Line2D(
                [], [], color=color_map(i % 10),
                marker=markers[i % len(markers)],
                linestyle='None', markersize=8, label=group
            )

        ax.set_title(f'Day {day}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True)

    # Remove unused subplots
    for j in range(len(days), len(axes)):
        fig.delaxes(axes[j])

    # Global legend
    handles = list(group_handles.values())
    fig.legend(handles=handles, loc='upper center', ncol=len(handles), title=group_by.capitalize())

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save plot
    plot_dir = os.path.join(base_dir, project_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    filename = os.path.join(plot_dir, f'PCA_grid_by_{group_by}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"PCA grid plot saved: {filename}")

import seaborn as sns

def plot_violin_grid_by_group(base_dir, project_dir, days, batches_complete_info, metric, metric_name,
                               group_by='condition', title=None, filename=None):
    """
    Creates a grid of violin plots (one per day), grouped by batch, condition, or cell-line.

    Parameters:
    - metric: the metric/feature column to plot (e.g., 'Diameter_um', 'Area_um2', etc.)
    - group_by: 'batch', 'condition', or 'cell-line'
    """
    #assert group_by in ['batch', 'condition', 'cell-line'], "group_by must be 'batch', 'condition', or 'cell-line'"
    
    # Map batch names to group
    plate_group_map = {batch['batch']: batch[group_by] for batch in batches_complete_info}
    valid_plates = plate_group_map.keys()

    # Create subplot grid
    n_cols = 3
    n_rows = int(np.ceil(len(days) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

    palette_colors = sns.color_palette("husl", len(set(plate_group_map.values())))

    for i, day in enumerate(days):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]

        day_data = []
        for batch in valid_plates:
            path = os.path.join(
                base_dir, project_dir, 'processed_data',
                f'Day {day}', batch, 'inference_metrics', 'mask_metrics.csv'
            )
            if os.path.exists(path):
                df = pd.read_csv(path)

                # Basic filtering
                if day < 14:
                    df = df[(df['Diameter_um'] > 150) & (df['Diameter_um'] < 1000)]
                else:
                    df = df[(df['Diameter_um'] > 300) & (df['Diameter_um'] < 2000)]
                df = df[df['Circularity'] > 0.2]

                if metric in df.columns:
                    df['Group'] = plate_group_map[batch]
                    df['batch'] = batch
                    day_data.append(df[['Group', metric]])

        if not day_data:
            ax.axis('off')
            ax.set_title(f'Day {day}\nNo data')
            continue

        df_day = pd.concat(day_data, ignore_index=True)
        sns.violinplot(data=df_day,x='Group',y=metric,hue='Group',legend=False,ax=ax,palette=palette_colors)
        ax.set_title(f'Day {day}')
        ax.set_xlabel(group_by.capitalize())
        ax.set_ylabel(metric_name)

    # Turn off unused subplots
    for j in range(len(days), n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row][col].axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save plot
    plot_dir = os.path.join(base_dir, project_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    filepath = os.path.join(plot_dir, filename or f"violin_grid_{metric}_by_{group_by}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Violin grid plot saved: {filepath}")

def plot_barplot_grid_by_group(base_dir, project_dir, days, batches_complete_info, metric, metric_name,
                                group_by="condition", title=None, filename=None):
    """
    Creates a grid of barplots (one per day), showing mean + SD of a given metric, grouped by a metadata field.
    
    Parameters:
    - metric: Feature column to compare (e.g., 'Area_um2', 'Diameter_um')
    - metric_name: Label to display on the y-axis
    - group_by: One of 'batch', 'condition', or 'cell-line'
    """
    #assert group_by in ['batch', 'condition', 'cell-line'], "group_by must be 'batch', 'condition', or 'cell-line'"
    
    plate_group_map = {batch['batch']: batch[group_by] for batch in batches_complete_info}
    valid_plates = plate_group_map.keys()
    
    n_cols = 3
    n_rows = int(np.ceil(len(days) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

    for i, day in enumerate(days):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]
        day_data = []

        for batch in valid_plates:
            path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_metrics', 'mask_metrics.csv')
            if os.path.exists(path):
                df = pd.read_csv(path)
                df_filtered = filter_by_diameter(df, day)
                if not df_filtered.empty and metric in df_filtered.columns:
                    group_label = plate_group_map[batch]
                    if metric == "Area_um2":
                        df_filtered[metric] = df_filtered[metric] / 1e6  # µm² to mm²
                    df_filtered['Group'] = group_label
                    day_data.append(df_filtered[['Group', metric]])
            else:
                print(f"Data not found for {batch} on Day {day}")

        if not day_data:
            ax.axis('off')
            ax.set_title(f'Day {day}\nNo data')
            continue

        df_day = pd.concat(day_data, ignore_index=True)

        # Unique groups for color palette
        unique_groups = sorted(df_day['Group'].unique())
        palette = sns.color_palette("husl", n_colors=len(unique_groups))
        group_color_map = dict(zip(unique_groups, palette))

        sns.barplot(
            data=df_day, x='Group', y=metric, hue='Group', legend=False,
            errorbar='sd', ax=ax, capsize=0.1, palette=group_color_map
        )

        ax.set_title(f'Day {day}')
        ax.set_ylabel(metric_name)
        ax.set_xlabel(group_by.capitalize())
        ax.tick_params(axis='x')

    # Turn off unused subplots
    for j in range(len(days), n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row][col].axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save figure
    plot_dir = os.path.join(base_dir, project_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    filepath = os.path.join(plot_dir, filename or f"barplot_grid_{metric}_by_{group_by}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Barplot grid saved: {filepath}")

# Modified plotting function that can accept an existing ax
def plot_feature_on_ax(ax, base_dir, project_dir, metric, metric_name, plate_day_pairs, batches_complete_info,
                       group_by="batch", title='Comparison'):
    area_data = []

    for batch, day in plate_day_pairs:
        path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_metrics', 'mask_metrics.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            df_filtered = filter_by_diameter(df, day)
            if not df_filtered.empty and metric in df_filtered.columns:
                label = batches_complete_info[batch].get(group_by, batch) + f"_D{day}"
                df_filtered['Group'] = label
                area_data.append(df_filtered[[metric, 'Group']])
        else:
            print(f"Data not found for {batch} on Day {day}")

    if not area_data:
        print(f"No data available for day {plate_day_pairs[0][1]}")
        return

    df_all = pd.concat(area_data, ignore_index=True)
    unique_groups = sorted(df_all['Group'].unique())
    n_colors = len(unique_groups)
    palette = sns.color_palette("husl", n_colors=n_colors)
    group_color_map = dict(zip(unique_groups, palette))

    sns.barplot(data=df_all, x='Group', y=metric, errorbar='sd', ax=ax, capsize=0.1, palette=group_color_map)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(metric_name)
    ax.set_xlabel(group_by)

    pairs = [(g1, g2) for i, g1 in enumerate(unique_groups) for g2 in unique_groups[i + 1:]]
    annotator = Annotator(ax, pairs, data=df_all, x='Group', y=metric)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    annotator.apply_and_annotate()

def load_outlier_summary(csv_path):
    """Load the global summary CSV into a DataFrame."""
    return pd.read_csv(csv_path)

def merge_plate_metadata(summary_df, plates_info):
    """Attach condition, cell-line, and batch info to the summary DataFrame."""
    metadata_df = pd.DataFrame(plates_info)
    metadata_df = metadata_df.rename(columns={'batch': 'batch'})
    return summary_df.merge(metadata_df, on='batch', how='left')

def plot_feature_scatter_with_stats(
    base_dir,
    project_dir,
    metric,
    metric_name,
    i,
    plate_day_pairs,
    batches_complete_info,
    group_by="batch",
    title_prefix="Neocortical organoids",
    filename="feature_comparison_scatter.png"
):
    area_data = []

    # Collect data
    for batch, day in plate_day_pairs:
        path = os.path.join(
            base_dir, project_dir, "processed_data", f"Day {day}", batch, "inference_metrics", "mask_metrics.csv"
        )
        if os.path.exists(path):
            df = pd.read_csv(path)
            df_filtered = filter_by_diameter(df, day)
            if not df_filtered.empty and metric in df_filtered.columns:
                label = batches_complete_info[batch].get(group_by, batch)
                df_filtered['Group'] = label
                area_data.append(df_filtered[[metric, 'Group']])
        else:
            print(f"Data not found for {batch} on Day {day}")

    if not area_data:
        print("No data available for comparison.")
        return

    df_all = pd.concat(area_data, ignore_index=True)

    # Plot aesthetics
    sns.set(style="whitegrid", context="talk")

    fig, ax = plt.subplots(figsize=(10, 5))
    
    conditions = df_all["Group"].unique()
    palette = sns.color_palette("husl", len(conditions))
    conditions_color_map = dict(zip(conditions, palette))
    
    # Jittered scatter plot with means
    sns.stripplot(
        data=df_all,
        x="Group",
        y=metric,
        hue="Group",
        dodge=False,
        jitter=True,
        palette=conditions_color_map,
        ax=ax,
        s=5
    )

    # Plot mean line
    means = df_all.groupby("Group")[metric].mean()
    for x_pos, (cond, mean_val) in enumerate(means.items()):
        ax.hlines(y=mean_val, xmin=x_pos - 0.2, xmax=x_pos + 0.2, color="black", lw=1)

    unique_groups = sorted(df_all['Group'].unique())
    # Statistical annotation
    pairs = [(g1, g2) for i, g1 in enumerate(unique_groups) for g2 in unique_groups[i + 1:]]
    if pairs:
        annotator = Annotator(ax, pairs, data=df_all, x="Group", y=metric)
        annotator.configure(test="Mann-Whitney", text_format="star", loc="outside")
        annotator.apply_and_annotate()

    # Labels & title
    #ax.set_title(f"{title_prefix} {day}", fontsize=16, weight="bold")
    ax.set_xlabel("")
    ax.set_ylim(0)
    ax.set_ylabel(f"{metric_name}" if i == 0 else "")
    #ax.legend_.remove()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plot_dir = os.path.join(base_dir, project_dir, 'plots')
    filepath= os.path.join(plot_dir, filename)
    
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight")
    plt.show()
    plt.close()
    
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
    plt.show()
    filepath = os.path.join(plot_dir, f"{feature}_outlier_rate_over_time_{group_by}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Outlier % over days plot saved: {filepath}")


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
        hue = group_by,
        y='Outlier_Percent',
        palette=group_color_map,
        legend = False
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
    filepath = os.path.join(plot_dir, f"{feature}_outlier_distribution_{group_by}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Boxplot of outlier percentages saved: {filepath}")
    plt.show()
    
summary_df = load_outlier_summary(f'{base_dir}/{project_dir}/outliers_all_batches_summary.csv')
merged_df = merge_plate_metadata(summary_df, batches_complete_info)