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

plt.rcParams.update({
    "axes.titlesize": 12,           # Title font size
    "axes.labelsize": 12,           # X and Y label font size
    "xtick.labelsize": 10,          # X-axis tick labels font size
    "ytick.labelsize": 10,          # Y-axis tick labels font size
    "legend.fontsize": 10,          # Legend font size
    "axes.spines.top": False,       # Remove top spine
    "axes.spines.right": False,      # Remove right spine
    "axes.spines.bottom": False,       # Remove top spine
    "axes.spines.left": False      # Remove right spine
})

plot_dir = os.path.join(base_dir, project_dir, 'plots')
os.makedirs(plot_dir, exist_ok=True)

# === Utilities ===

def save_plot(fig, filename):
    path = os.path.join(plot_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {path}")

PROJECTS_WITHOUT_DIAMETER_FILTER = {
    "2026_06_08-Dasha-DMA",
}


def _uses_project_specific_unfiltered_morphology():
    """Return True for projects where the legacy diameter gate is not appropriate."""
    return os.path.basename(str(project_dir).rstrip(os.sep)) in PROJECTS_WITHOUT_DIAMETER_FILTER


def filter_by_diameter(df, day):
    if 'Diameter_um' not in df.columns:
        return pd.DataFrame()
    if _uses_project_specific_unfiltered_morphology():
        return df.copy()
    #df = df[df['Circularity'] > 0.2]
    if day < 14:
        return df[(df['Diameter_um'] > 150) & (df['Diameter_um'] < 1000)]
    elif day >= 14 and day < 30:
        return df[(df['Diameter_um'] > 450) & (df['Diameter_um'] < 2000)]
    else:
        return df[(df['Diameter_um'] > 700) & (df['Diameter_um'] < 2000)] #return df[(df['Diameter_um'] > 300) & (df['Diameter_um'] < 1500)]


def _image_id_from_metric_id(organoid_id):
    return str(organoid_id).split("_object", 1)[0]


def _ensure_metric_identity_columns(df):
    """Keep legacy metrics readable while exposing image/well/object identities."""
    df = df.copy()

    if 'Organoid_ID' not in df.columns:
        df['Organoid_ID'] = [f"object{idx:04d}" for idx in range(1, len(df) + 1)]

    if 'Image_ID' not in df.columns:
        df['Image_ID'] = df['Organoid_ID'].map(_image_id_from_metric_id)

    if 'Well_ID' not in df.columns:
        df['Well_ID'] = df['Image_ID']

    if 'Object_Label' not in df.columns:
        object_label = df['Organoid_ID'].astype(str).str.extract(r"_object(\d+)$")[0]
        df['Object_Label'] = pd.to_numeric(object_label, errors='coerce').fillna(1).astype(int)

    if 'Object_Count_In_Image' not in df.columns:
        df['Object_Count_In_Image'] = df.groupby('Image_ID')['Organoid_ID'].transform('count')

    return df


def prepare_morphology_records(df, day, features, aggregation_level="object"):
    """
    Return object-level rows or one row per image/well.

    Object level preserves individual organoids. Well/image level converts each
    image into a per-well mean for each requested feature and keeps the number of
    detected organoids in ``Object_Count_In_Image``.
    """
    features = [features] if isinstance(features, str) else list(features)
    df = _ensure_metric_identity_columns(df)
    df = filter_by_diameter(df, day).copy()
    if df.empty:
        return df

    aggregation_level = str(aggregation_level).lower()
    if aggregation_level in {"object", "organoid", "individual"}:
        df['Aggregation_Level'] = "object"
        return df

    if aggregation_level not in {"well", "image", "well_mean", "image_mean"}:
        raise ValueError("aggregation_level must be 'object' or 'well'.")

    group_cols = [
        column for column in [
            'Batch', 'Day', 'Run', 'Image_ID', 'Well_ID',
            'Plate_Index', 'Well_Index', 'Organoid_Mode'
        ]
        if column in df.columns
    ]
    if not group_cols:
        group_cols = ['Image_ID']

    available_features = [feature for feature in features if feature in df.columns]
    if not available_features:
        return pd.DataFrame()

    well_df = df.groupby(group_cols, dropna=False)[available_features].mean().reset_index()
    counts = (
        df
        .groupby(group_cols, dropna=False)
        .agg(Object_Count_In_Image=('Organoid_ID', 'count'))
        .reset_index()
    )
    well_df = well_df.merge(counts, on=group_cols, how='left')
    well_df['Organoid_ID'] = well_df['Image_ID'] if 'Image_ID' in well_df.columns else well_df.index.astype(str)
    well_df['Aggregation_Level'] = "well"

    return well_df


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

def plot_feature_over_time(base_dir, project_dir, batches_complete_info, features, metric_names, group_by="batch", aggregation_level="object"):
    """
    Plots morphological features over time.
    Performs Welch's T-test with FDR (Benjamini-Hochberg) correction across tested days.
    """
    if isinstance(features, str):
        features = [features]

    aggregation_level = str(aggregation_level).lower()
    if aggregation_level in {"organoid", "individual"}:
        aggregation_level = "object"
    if aggregation_level in {"image", "well_mean", "image_mean"}:
        aggregation_level = "well"

    plot_dir = os.path.join(base_dir, project_dir, 'plots')
    plot_data_dir = os.path.join(plot_dir, 'plot_data')
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(plot_data_dir, exist_ok=True)

    for i, feature in enumerate(features):
        all_records = []
        metric_label = metric_names[i] if i < len(metric_names) else feature

        # --- 1. DATA COLLECTION ---
        for entry in batches_complete_info:
            batch, label = entry['batch'], entry.get(group_by)
            for day in entry['days']:
                path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_metrics', 'mask_metrics.csv')
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df_filtered = prepare_morphology_records(
                        df,
                        day,
                        [feature],
                        aggregation_level=aggregation_level,
                    )
                    
                    if feature in df_filtered.columns and not df_filtered.empty:
                        df_filtered['Batch'] = batch
                        df_filtered['Group'] = label
                        df_filtered['Day'] = day
                        if aggregation_level == "object":
                            df_filtered['Analysis_Unit_ID'] = df_filtered['Organoid_ID'].astype(str)
                        else:
                            df_filtered['Analysis_Unit_ID'] = df_filtered['Image_ID'].astype(str)

                        keep_columns = [
                            'Batch', 'Group', 'Day', 'Image_ID', 'Well_ID',
                            'Organoid_ID', 'Analysis_Unit_ID',
                            'Object_Count_In_Image', 'Aggregation_Level', feature
                        ]
                        keep_columns = [column for column in keep_columns if column in df_filtered.columns]
                        all_records.append(df_filtered[keep_columns])

        if not all_records:
            continue

        df_all = pd.concat(all_records, ignore_index=True)
        unique_groups = sorted(df_all['Group'].unique())
        df_all.to_csv(
            os.path.join(plot_data_dir, f"{feature}_{aggregation_level}_{group_by}_over_time_records.csv"),
            index=False,
        )
        
        # --- 2. SUMMARY STATISTICS ---
        summary = df_all.groupby(['Group', 'Day'])[feature].agg(['mean', 'std', 'count']).reset_index()
        summary['sem'] = summary['std'] / np.sqrt(summary['count'])
        summary['Aggregation_Level'] = aggregation_level
        count_column = 'N_Organoids' if aggregation_level == "object" else 'N_Wells'
        summary.rename(columns={'count': count_column}, inplace=True)
        summary.to_csv(
            os.path.join(plot_data_dir, f"{feature}_{aggregation_level}_{group_by}_over_time_summary.csv"),
            index=False,
        )
        print(
            f"{feature} ({aggregation_level}): plotted {len(df_all)} records; "
            f"summary saved in {plot_data_dir}"
        )
        
        # --- 3. STATISTICAL TESTING WITH FDR CORRECTION ---
        stats_data = []
        if len(unique_groups) == 2:
            g1, g2 = unique_groups
            all_days = sorted(df_all['Day'].unique())
            # Optionally test all days or just the last 3. Testing all days is better for FDR context.
            days_to_test = all_days[-3:] 
            
            raw_pvals = []
            valid_days = []

            for day in days_to_test:
                vals_g1 = df_all[(df_all['Day'] == day) & (df_all['Group'] == g1)][feature].dropna()
                vals_g2 = df_all[(df_all['Day'] == day) & (df_all['Group'] == g2)][feature].dropna()
                
                if len(vals_g1) > 1 and len(vals_g2) > 1:
                    _, p_val = ttest_ind(vals_g1, vals_g2, equal_var=False)
                    raw_pvals.append(p_val)
                    valid_days.append(day)

            if raw_pvals:
                # Apply Benjamini-Hochberg Correction
                reject, p_corrected, _, _ = multipletests(raw_pvals, method='fdr_bh')
                
                print(f"\n--- FDR Corrected Statistics for {metric_label} ({g1} vs {g2}) ---")
                for d_idx, day in enumerate(valid_days):
                    p_adj = p_corrected[d_idx]
                    sig = "***" if p_adj < 0.001 else "**" if p_adj < 0.01 else "*" if p_adj < 0.05 else "ns"
                    
                    # Position logic
                    day_sum = summary[summary['Day'] == day]
                    y_pos = (day_sum['mean'] + day_sum['sem']).max()
                    
                    stats_data.append({'Day': day, 'label': sig, 'y_pos': y_pos})
                    print(f"Day {day}: p_raw = {raw_pvals[d_idx]:.5f}, p_adj = {p_adj:.5f} ({sig})")

        # --- 4. PLOTTING ---
        palette = dict(zip(unique_groups, sns.color_palette("husl", len(unique_groups))))

        def scatter_measurement_points(ax):
            rng = np.random.default_rng(7)
            group_offsets = dict(zip(unique_groups, np.linspace(-0.18, 0.18, max(len(unique_groups), 1))))
            point_size = 8 if aggregation_level == "object" else 22
            alpha = 0.12 if aggregation_level == "object" else 0.40
            jitter_width = 0.035 if aggregation_level == "object" else 0.015

            for group in unique_groups:
                group_data = df_all[df_all['Group'] == group]
                for day, day_data in group_data.groupby('Day'):
                    y_values = pd.to_numeric(day_data[feature], errors='coerce').dropna()
                    if y_values.empty:
                        continue
                    x_values = (
                        np.full(len(y_values), float(day))
                        + group_offsets[group]
                        + rng.normal(0, jitter_width, len(y_values))
                    )
                    ax.scatter(
                        x_values,
                        y_values,
                        s=point_size,
                        alpha=alpha,
                        color=palette[group],
                        linewidths=0,
                        zorder=1,
                    )

        def add_stats_and_save(ax, filename):
            for res in stats_data:
                #if res['label'] != "ns": # Only plot if significant after correction
                ax.text(res['Day'], res['y_pos'] * 1.05, res['label'], 
                            ha='center', va='bottom', color='black', fontsize=12, fontweight='bold')
            ax.set_xlabel('Day')
            ax.set_ylabel(f"{metric_label}")
            ax.grid(True, alpha=0.3)
            ax.legend(frameon=False, loc='best')
            plt.tight_layout()
            save_path = os.path.join(plot_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

        # Plot SD and SEM
        for error_type in ['std', 'sem']:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.set_style("whitegrid")
            scatter_measurement_points(ax)
            for group in unique_groups:
                data = summary[summary['Group'] == group]
                ax.errorbar(data['Day'], data['mean'], yerr=data[error_type], 
                            label=group, marker='o', capsize=5, color=palette[group], linewidth=2, zorder=3)
            
            suffix = 'sd' if error_type == 'std' else 'sem'
            ax.set_ylim(bottom=(data['mean']).min()*0.50, top=(data['mean'] + data['sem']).max()*1.50)
            add_stats_and_save(ax, f"{feature}_{aggregation_level}_{group_by}_over_time_{suffix}.png")

        # Plot 95% CI
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")
    
        sns.lineplot(data=df_all, x='Day', y=feature, hue='Group', errorbar=('ci', 95), 
                     marker='o', palette=palette, ax=ax, linewidth=2)
        ax.set_ylim(bottom=(df_all[feature]).min()*0.75, top=(df_all[feature]).max()*1.25)
        add_stats_and_save(ax,  f"{feature}_{aggregation_level}_{group_by}_over_time_ci.png")
            

def plot_pca_grid_by_group(base_dir, project_dir, days, batches_complete_info, group_by='batch'):
    plot_dir = os.path.join(base_dir, project_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Identify all unique groups across the whole dataset for consistent coloring
    all_possible_groups = list(set(batch[group_by] for batch in batches_complete_info))
    
    # 2. Generate the HUSL palette using Seaborn
    # This creates a list of RGB tuples
    husl_palette = sns.color_palette("husl", len(all_possible_groups))
    color_map = dict(zip(all_possible_groups, husl_palette))

    features = ['Area_um2', 'Diameter_um', 'Perimeter_um',
    'Feret_Diameter_Max_um', 'Aspect_Ratio', 'Circularity',
    'Roundness', 'Solidity', 'Eccentricity', 'Skeleton_Length_um']
    
    plate_group_map = {batch['batch']: batch[group_by] for batch in batches_complete_info}
    valid_plates = plate_group_map.keys()
    markers = ["o", "v", "s", "*", "D", "P", "X", "^"]

    n_cols = 3
    n_rows = int(np.ceil(len(days) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    sns.set_style("whitegrid")
    
    axes = axes.flatten()

    group_handles = {} # To store legend info

    for idx, day in enumerate(days):
        all_metrics = []
        for batch in valid_plates:
            metrics_csv = os.path.join(
                base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_metrics', 'mask_metrics.csv'
            )
            if os.path.exists(metrics_csv):
                df = pd.read_csv(metrics_csv)
                # Filtering
                if day < 14:
                    df = df[(df['Diameter_um'] > 150) & (df['Diameter_um'] < 1000)]
                else:
                    df = df[(df['Diameter_um'] > 300) & (df['Diameter_um'] < 2000)]
                
                df['Group'] = plate_group_map[batch]
                all_metrics.append(df)

        if not all_metrics:
            axes[idx].axis('off')
            continue

        df_all = pd.concat(all_metrics, ignore_index=True)
        df_all.dropna(subset=features, inplace=True)

        X_scaled = StandardScaler().fit_transform(df_all[features].values)
        pcs = PCA(n_components=2).fit_transform(X_scaled)

        df_pca = pd.DataFrame(pcs, columns=['PC1', 'PC2'])
        df_pca['Group'] = df_all['Group'].values

        unique_groups_this_day = sorted(df_pca['Group'].unique())
        ax = axes[idx]

        for i, group in enumerate(all_possible_groups):
            if group not in unique_groups_this_day:
                continue
                
            subset = df_pca[df_pca['Group'] == group]
            
            # Use the pre-calculated HUSL color
            group_color = color_map[group]
            group_marker = markers[i % len(markers)]

            ax.scatter(
                subset['PC1'], subset['PC2'],
                color=group_color,
                marker=group_marker,
                label=group,
                alpha=0.6,
                s=20 # Reduced size for better visibility in density
            )
            
            # Store handle for global legend if not already there
            if group not in group_handles:
                group_handles[group] = mlines.Line2D(
                    [], [], color=group_color,
                    marker=group_marker,
                    linestyle='None', markersize=8, label=group
                )

        ax.set_title(f'Day {day}')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(True, alpha=0.3)
        
    # Remove unused subplots
    for j in range(len(days), len(axes)):
        fig.delaxes(axes[j])

    # Global legend using the handles we collected
    # Sorting handles to match the group order
    sorted_labels = sorted(group_handles.keys())
    handles = [group_handles[lbl] for lbl in sorted_labels]
    
    fig.legend(
        handles=handles, 
        loc='lower center', 
        bbox_to_anchor=(0.5, -0.05), 
        ncol=min(len(handles), 5), 
        title=None,
        frameon=False
    )

    # 3. Use rect to make room at the bottom for the legend (left, bottom, right, top)
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    #fig.tight_layout()

    # Save plot
    filename = os.path.join(plot_dir, f'PCA_grid_by_{group_by}.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

def plot_feature_grid_scatters(base_dir, project_dir, days, batches_complete_info, features, metric_names, group_by='batch'):
    """
    Creates a grid of scatter plots with Welch's T-test and FDR correction.
    Vertical axis labels and ticks are hidden for internal panels.
    """
    plot_dir = os.path.join(base_dir, project_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
        
    if isinstance(features, str):
        features = [features]
        metric_names = [metric_names]

    # 1. Setup metadata mapping
    plate_group_map = {
        batch_id: info.get(group_by, batch_id) 
        for batch_id, info in batches_complete_info.items()
    }
    all_possible_groups = sorted(list(set(plate_group_map.values())))
    
    # 2. Setup Plotting Aesthetics
    husl_palette = sns.color_palette("husl", len(all_possible_groups))
    color_dict = dict(zip(all_possible_groups, husl_palette))

    for f_idx, feature in enumerate(features):
        metric_label = metric_names[f_idx]
        
        # --- PHASE A: DATA GATHERING & STATS CALCULATION ---
        # We need to calculate ALL p-values first to apply FDR correction across the whole feature
        stats_list = []
        feature_data_by_day = {}

        for day in days:
            day_data = []
            for batch_id in plate_group_map.keys():
                path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}', batch_id, 'inference_metrics', 'mask_metrics.csv')
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df_filtered = filter_by_diameter(df, day).copy()
                    if not df_filtered.empty and feature in df_filtered.columns:
                        df_filtered['Group'] = plate_group_map[batch_id]
                        day_data.append(df_filtered[['Group', feature]])
            
            if day_data:
                df_day = pd.concat(day_data, ignore_index=True)
                feature_data_by_day[day] = df_day
                
                # Perform Welch's T-test for all pairs present this day
                u_groups = sorted(df_day['Group'].unique())
                if len(u_groups) > 1:
                    for i in range(len(u_groups)):
                        for j in range(i + 1, len(u_groups)):
                            g1, g2 = u_groups[i], u_groups[j]
                            vals1 = df_day[df_day['Group'] == g1][feature].dropna()
                            vals2 = df_day[df_day['Group'] == g2][feature].dropna()
                            
                            if len(vals1) > 1 and len(vals2) > 1:
                                _, p_val = ttest_ind(vals1, vals2, equal_var=False)
                                stats_list.append({
                                    'day': day, 'pair': (g1, g2), 'p_raw': p_val
                                })

        # Apply FDR correction (Benjamini-Hochberg) across all tests for this feature
        if stats_list:
            raw_pvals = [s['p_raw'] for s in stats_list]
            _, corrected_pvals, _, _ = multipletests(raw_pvals, method='fdr_bh')
            for i, s in enumerate(stats_list):
                s['p_corr'] = corrected_pvals[i]
                # Convert p-value to stars
                p = s['p_corr']
                s['stars'] = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        # --- PHASE B: PLOTTING ---
        n_cols = len(days)
        fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 6), sharey=True)
        sns.set_style("whitegrid")
    
        if n_cols == 1: axes = [axes]
        
        group_handles = {}

        for d_idx, day in enumerate(days):
            ax = axes[d_idx]
            if day not in feature_data_by_day:
                ax.axis('off')
                continue
                
            df_day = feature_data_by_day[day]
            u_groups = sorted(df_day['Group'].unique())

            # 1. Scatter/Stripplot
            sns.stripplot(
                data=df_day, x='Group', y=feature, order=u_groups,
                palette=color_dict, hue='Group', jitter=True, alpha=0.8, 
                s=4, ax=ax, legend=False
            )

            # 2. Mean Lines
            means = df_day.groupby('Group')[feature].mean().reindex(u_groups)
            for x_pos, mean_val in enumerate(means):
                ax.hlines(y=mean_val, xmin=x_pos - 0.25, xmax=x_pos + 0.25, color="black", lw=2.5, zorder=5)

            # 3. Add statistical stars
            day_stats = [s for s in stats_list if s['day'] == day]
            y_max = df_day[feature].max()
            y_offset = y_max * 0.3
            
            for s_idx, s in enumerate(day_stats):
                #if s['stars'] == "ns": continue
                g1, g2 = s['pair']
                x1, x2 = u_groups.index(g1), u_groups.index(g2)
                
                # Draw brackets/stars manually
                h = y_max + (s_idx * y_offset)
                ax.plot([x1, x1, x2, x2], [h, h + y_offset*0.2, h + y_offset*0.2, h], lw=1, color='black')
                ax.text((x1 + x2) * 0.5, h + y_offset*0.2, s['stars'], ha='center', va='bottom', color='black', fontsize=12)

            # 4. Vertical Axis Cleanup
            if d_idx > 0:
                ax.set_ylabel("") # Remove axis title
                #ax.tick_params(labelleft=False) # Remove numerical labels
                ax.spines['left'].set_visible(False) # Optional: remove left spine for "floating" look
            else:
                ax.set_ylabel(metric_label)
            
            ax.tick_params(labelbottom=False)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("")
            ax.set_title(f"Day {day}", fontsize=14)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

        # 5. Global Legend
        for i, grp in enumerate(all_possible_groups):
            group_handles[grp] = mlines.Line2D(
                [], [], color=color_dict[grp], marker='o', 
                linestyle='None', markersize=10, label=grp
            )
        
        handles = [group_handles[k] for k in sorted(group_handles.keys())]
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
                   ncol=len(handles), frameon=False, title=None)

        #plt.suptitle(f"{metric_label} Growth Trajectory", fontsize=16, fontweight='bold', y=1.02)
        # 6. Save
        save_path = os.path.join(plot_dir, f"{feature}_grid_scatterplot.png")
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def filter_organoids(df, day):
    """Consistent filtering logic based on your provided thresholds."""
    if 'Diameter_um' not in df.columns or df.empty:
        return pd.DataFrame()
    if _uses_project_specific_unfiltered_morphology():
        return df.copy()
    if day < 14:
        return df[(df['Diameter_um'] > 150) & (df['Diameter_um'] < 1000)]
    elif 14 <= day < 30:
        return df[(df['Diameter_um'] > 450) & (df['Diameter_um'] < 2000)]
    else:
        return df[(df['Diameter_um'] > 700) & (df['Diameter_um'] < 2000)]

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker

def _legacy_plot_variability_and_outliers(base_dir, project_dir, batches_complete_info, features, metric_names, group_by="batch", mad_threshold=3.0):
    """
    1. Plots SD for each group over time (Scientific Notation).
    2. Plots Percentage of Outliers (MAD-based) over time (Scientific Notation).
    """
    plot_dir = os.path.join(base_dir, project_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    for i, feature in enumerate(features):
        all_records = []
        metric_label = metric_names[i]

        for entry in batches_complete_info:
            batch, label = entry['batch'], entry.get(group_by)
            for day in entry['days']:
                path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_metrics', 'mask_metrics.csv')
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df_f = filter_organoids(df, day) # Using your specific filtering function
                    if not df_f.empty and feature in df_f.columns:
                        df_f = df_f.copy()
                        df_f['Group'], df_f['Day'] = label, day
                        all_records.append(df_f[['Group', 'Day', feature]])

        if not all_records: 
            continue
            
        df_all = pd.concat(all_records, ignore_index=True)
        unique_groups = sorted(df_all['Group'].unique())
        palette = dict(zip(unique_groups, sns.color_palette("husl", len(unique_groups))))

        # --- Task 1: Plot Standard Deviation ---
        sd_summary = df_all.groupby(['Group', 'Day'])[feature].std().reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")
        
        sns.lineplot(data=sd_summary, x='Day', y=feature, hue='Group', marker='o', palette=palette, ax=ax)
        
        #ax.set_title(f'Variability (SD) of {metric_label} Over Time')
        ax.set_ylabel(f'Standard Deviation - {metric_label}')
        
        # Apply Scientific Notation
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(plot_dir, f"{feature}_variability_sd.png"), bbox_inches='tight')
        plt.close()

        # --- Task 2: Calculate and Plot Normalized Outliers (Percentage) ---
        def calc_outlier_percent(group):
            vals = group[feature]
            median = vals.median()
            mad = np.median(np.abs(vals - median))
            total_count = len(vals)
            
            if mad == 0 or total_count == 0: 
                return 0.0
            
            outliers = np.abs(vals - median) > (mad_threshold * mad)
            return (outliers.sum() / total_count) * 100

        outlier_summary = df_all.groupby(['Group', 'Day']).apply(calc_outlier_percent).reset_index(name='outlier_percent')
        
        fig, ax = plt.subplots(figsize=(10,6))
        sns.set_style("whitegrid")
    
        sns.lineplot(data=outlier_summary, x='Day', y='outlier_percent', hue='Group', marker='s', palette=palette, ax=ax)
        
        #ax.set_title(f'Percentage of Outliers in {metric_label} (MAD > {mad_threshold})')
        ax.set_ylabel('Outliers (%)')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        
        plt.savefig(os.path.join(plot_dir, f"{feature}_outlier_percentage.png"), bbox_inches='tight')
        plt.close()

    print(f"Variability and Outlier plots saved to {plot_dir}")
        

def _legacy_create_detection_analysis(base_dir, project_dir, batches_complete_info, batches_info, group_by="batch"):
    """
    Creates separate 96-well and 24-well detection matrices per batch.
    Aggregates 'Undetected' counts based on num_plates * 24 for the 24-well phase.
    """
    qc_dir = os.path.join(base_dir, project_dir, 'plots', 'quality_control')
    batch_matrix_dir = os.path.join(qc_dir, 'batch_matrices')
    os.makedirs(batch_matrix_dir, exist_ok=True)

    # Map batches_info by name for easy lookup
    info_lookup = {b['batch']: b for b in batches_info}
    
    all_batch_summaries = []

    for entry in batches_complete_info:
        batch = entry['batch']
        label = entry.get(group_by, batch)
        b_meta = info_lookup.get(batch)
        
        if not b_meta:
            print(f"⚠️ Metadata for {batch} not found in batches_info. Skipping.")
            continue

        days_96 = sorted(b_meta.get("days_96", []))
        days_24 = sorted(b_meta.get("days_24", []))
        num_expected_24 = b_meta.get("num_plates", 0) * 24

        # --- Helper to process a specific phase (96 or 24) ---
        def process_phase(phase_days, phase_suffix, expected_override=None):
            if not phase_days:
                return 0
            
            # 1. Identify unique Organoid_IDs seen in this phase
            unique_ids = set()
            for day in phase_days:
                path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_metrics', 'mask_metrics.csv')
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    if 'Organoid_ID' in df.columns:
                        unique_ids.update(df['Organoid_ID'].unique())
            
            if not unique_ids and not expected_override:
                return 0

            # 2. Create Matrix
            matrix = pd.DataFrame(0, index=sorted(list(unique_ids)), columns=phase_days)
            
            for day in phase_days:
                path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_metrics', 'mask_metrics.csv')
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df_f = filter_organoids(df, day)
                    if not df_f.empty and 'Organoid_ID' in df_f.columns:
                        detected = df_f['Organoid_ID'].unique()
                        # Only mark IDs that exist in our index
                        valid_detected = [d for d in detected if d in matrix.index]
                        matrix.loc[valid_detected, day] = 1

            # 3. Save Matrix
            matrix.to_csv(os.path.join(batch_matrix_dir, f"{batch}_{phase_suffix}_matrix.csv"))

            # 4. Summary Stats
            for day in phase_days:
                detected_count = matrix[day].sum() if not matrix.empty else 0
                # Use provided math for 24-well, use unique observed for 96-well
                total_expected = expected_override if expected_override else len(matrix)
                
                all_batch_summaries.append({
                    'Batch': batch,
                    'Group': label,
                    'Day': day,
                    'Phase': phase_suffix,
                    'Detected': detected_count,
                    'Undetected': max(0, total_expected - detected_count),
                    'Total_Expected': total_expected
                })
            return len(matrix)

        # Process both phases
        process_phase(days_96, "96well")
        process_phase(days_24, "24well", expected_override=num_expected_24)

    # --- AGGREGATION AND PLOTTING ---
    summary_df = pd.DataFrame(all_batch_summaries)
    if summary_df.empty:
        print("❌ No data processed.")
        return None, qc_dir

    global_summary = summary_df.groupby(['Group', 'Day']).agg({
        'Detected': 'sum',
        'Undetected': 'sum',
        'Total_Expected': 'sum'
    }).reset_index()

    global_summary['Undetected_Pct'] = (global_summary['Undetected'] / global_summary['Total_Expected']) * 100
    
    global_summary.to_csv(os.path.join(qc_dir, "undetected_tiles_global_summary.csv"), index=False)

    return global_summary, qc_dir

def _legacy_plot_undetected_summary(global_summary, output_dir):
    """
    Takes the aggregated summary from create_detection_analysis and 
    generates the QC plot and summary CSV using percentages.
    """
    
    # 3. Plotting
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Use the calculated percentage column
    ax = sns.lineplot(
        data=global_summary, 
        x='Day', 
        y='Undetected_Pct', 
        hue='Group', 
        palette="husl",
        marker='o',    # Added markers to make the specific days clearer
        linewidth=2
    )
    
    #plt.title('Percentage of Undetected or Filtered-out Tiles per Condition Across Days', fontsize=14, pad=15)
    plt.ylabel('Undetected Organoids (%)')
    plt.xlabel('Day')
    plt.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "undetected_tiles_percentage_plot.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()
    
    print(f"✅ Quality Control percentage plot and summary saved in: {output_dir}")


def _to_numeric_series(values):
    return pd.to_numeric(pd.Series(values), errors='coerce').dropna()


def _safe_pct(numerator, denominator):
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return np.nan
    return (numerator / denominator) * 100


def _resolve_std_multiplier(manual_std_thresholds, feature=None, group=None, day=None, default=3.0):
    """Resolve a manual mean +/- N SD threshold from a scalar or lookup dict."""
    if manual_std_thresholds is None:
        return default

    if isinstance(manual_std_thresholds, (int, float)):
        return float(manual_std_thresholds)

    if not isinstance(manual_std_thresholds, dict):
        return default

    lookup_keys = [
        (feature, group, day),
        (feature, day),
        (group, day),
        feature,
        str(day),
        day,
        "default",
    ]
    for key in lookup_keys:
        if key in manual_std_thresholds:
            value = manual_std_thresholds[key]
            if isinstance(value, dict):
                value = value.get(day, value.get(str(day), value.get("default", default)))
            return float(value)

    if feature in manual_std_thresholds and isinstance(manual_std_thresholds[feature], dict):
        feature_thresholds = manual_std_thresholds[feature]
        return float(feature_thresholds.get(day, feature_thresholds.get(str(day), feature_thresholds.get("default", default))))

    return default


def calculate_outlier_thresholds(
    values,
    method="iqr",
    iqr_multiplier=1.5,
    std_multiplier=3.0,
    percentile_bounds=(0.05, 0.95),
):
    """
    Calculate lower/upper outlier thresholds and robust descriptive statistics.

    The default is Tukey's IQR rule: Q1 - 1.5*IQR and Q3 + 1.5*IQR.
    The 5th/95th percentiles are reported for context, not used as default
    cutoffs unless method="percentile".
    """
    vals = _to_numeric_series(values)
    if vals.empty:
        return {
            "method": method,
            "lower_threshold": np.nan,
            "upper_threshold": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "q05": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "q95": np.nan,
            "iqr": np.nan,
            "n": 0,
        }

    q05 = vals.quantile(0.05)
    q25 = vals.quantile(0.25)
    median = vals.quantile(0.50)
    q75 = vals.quantile(0.75)
    q95 = vals.quantile(0.95)
    iqr = q75 - q25
    mean = vals.mean()
    std = vals.std(ddof=1)
    method_key = str(method).lower()

    if method_key in {"iqr", "tukey", "tukey_iqr"}:
        lower_threshold = q25 - iqr_multiplier * iqr
        upper_threshold = q75 + iqr_multiplier * iqr
        method_label = f"Tukey IQR ({iqr_multiplier}x)"
    elif method_key in {"mean_sd", "sd", "std", "standard_deviation"}:
        lower_threshold = mean - std_multiplier * std
        upper_threshold = mean + std_multiplier * std
        method_label = f"Mean +/- {std_multiplier} SD"
    elif method_key in {"percentile", "quantile"}:
        lower_q, upper_q = percentile_bounds
        lower_threshold = vals.quantile(lower_q)
        upper_threshold = vals.quantile(upper_q)
        method_label = f"Percentile ({lower_q:.0%}-{upper_q:.0%})"
    else:
        raise ValueError("Unknown outlier method. Use 'iqr', 'mean_sd', or 'percentile'.")

    return {
        "method": method_label,
        "lower_threshold": lower_threshold,
        "upper_threshold": upper_threshold,
        "mean": mean,
        "std": std,
        "median": median,
        "q05": q05,
        "q25": q25,
        "q75": q75,
        "q95": q95,
        "iqr": iqr,
        "n": len(vals),
    }


def _classify_outliers(df, feature, thresholds):
    values = pd.to_numeric(df[feature], errors='coerce')
    lower = thresholds["lower_threshold"]
    upper = thresholds["upper_threshold"]

    if pd.isna(lower) or pd.isna(upper):
        outlier_mask = pd.Series(False, index=df.index)
    else:
        outlier_mask = (values < lower) | (values > upper)

    directions = pd.Series("", index=df.index, dtype=object)
    directions[values < lower] = "low"
    directions[values > upper] = "high"

    return values, outlier_mask, directions


def plot_variability_and_outliers(
    base_dir,
    project_dir,
    batches_complete_info,
    features,
    metric_names,
    group_by="batch",
    outlier_method="iqr",
    iqr_multiplier=1.5,
    std_multiplier=3.0,
    manual_std_thresholds=None,
    percentile_bounds=(0.05, 0.95),
    mad_threshold=None,
    aggregation_level="object",
):
    """
    Plot variability and quantify morphology outliers.

    Default outlier rule: Tukey IQR fences, Q1 - 1.5*IQR and Q3 + 1.5*IQR.
    For manual mean/SD thresholding, call with outlier_method="mean_sd" and set
    std_multiplier or manual_std_thresholds.
    """
    if mad_threshold is not None and outlier_method == "mad":
        print("MAD outlier mode is no longer used here; using Tukey IQR instead.")
        outlier_method = "iqr"

    plot_dir = os.path.join(base_dir, project_dir, 'plots')
    outlier_dir = os.path.join(plot_dir, 'quality_control', 'outliers')
    os.makedirs(outlier_dir, exist_ok=True)

    all_summary_rows = []
    all_outlier_rows = []

    for i, feature in enumerate(features):
        all_records = []
        metric_label = metric_names[i] if i < len(metric_names) else feature

        for entry in batches_complete_info:
            batch = entry['batch']
            label = entry.get(group_by, batch)
            for day in entry['days']:
                path = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_metrics', 'mask_metrics.csv')
                if not os.path.exists(path):
                    continue

                df = pd.read_csv(path)
                df_f = prepare_morphology_records(
                    df,
                    day,
                    [feature],
                    aggregation_level=aggregation_level,
                )
                if df_f.empty or feature not in df_f.columns:
                    continue

                df_f = df_f.copy()
                if 'Organoid_ID' not in df_f.columns:
                    df_f['Organoid_ID'] = [f"{batch}_day{day}_object{idx:04d}" for idx in range(1, len(df_f) + 1)]

                df_f['Batch'] = batch
                df_f['Group'] = label
                df_f['Day'] = day
                all_records.append(df_f[['Batch', 'Group', 'Day', 'Organoid_ID', feature]])

        if not all_records:
            continue

        df_all = pd.concat(all_records, ignore_index=True)
        unique_groups = sorted(df_all['Group'].dropna().unique())
        palette = dict(zip(unique_groups, sns.color_palette("husl", len(unique_groups))))

        sd_summary = df_all.groupby(['Group', 'Day'])[feature].std().reset_index()
        fig, ax = plt.subplots(figsize=(10,6))
        sns.set_style("whitegrid")
        sns.lineplot(data=sd_summary, x='Day', y=feature, hue='Group', marker='o', palette=palette, ax=ax)
        ax.set_ylabel(f'Standard Deviation - {metric_label}')
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plot_dir, f"{feature}_{aggregation_level}_variability_sd.png"), bbox_inches='tight', dpi=300)
        plt.close()

        feature_summary_rows = []
        feature_outlier_rows = []
        for (group, day), group_df in df_all.groupby(['Group', 'Day']):
            resolved_std_multiplier = _resolve_std_multiplier(
                manual_std_thresholds,
                feature=feature,
                group=group,
                day=day,
                default=std_multiplier,
            )
            thresholds = calculate_outlier_thresholds(
                group_df[feature],
                method=outlier_method,
                iqr_multiplier=iqr_multiplier,
                std_multiplier=resolved_std_multiplier,
                percentile_bounds=percentile_bounds,
            )
            values, outlier_mask, directions = _classify_outliers(group_df, feature, thresholds)
            total_count = len(group_df)
            outlier_count = int(outlier_mask.sum())
            low_count = int((directions == "low").sum())
            high_count = int((directions == "high").sum())

            summary_row = {
                'Feature': feature,
                'Aggregation_Level': aggregation_level,
                'Group': group,
                'Day': day,
                'Outlier_Method': thresholds['method'],
                'Lower_Threshold': thresholds['lower_threshold'],
                'Upper_Threshold': thresholds['upper_threshold'],
                'Mean': thresholds['mean'],
                'Std_Dev': thresholds['std'],
                'Median': thresholds['median'],
                'Q05': thresholds['q05'],
                'Q25': thresholds['q25'],
                'Q75': thresholds['q75'],
                'Q95': thresholds['q95'],
                'IQR': thresholds['iqr'],
                'Total_Objects': total_count,
                'Outlier_Count': outlier_count,
                'Low_Outlier_Count': low_count,
                'High_Outlier_Count': high_count,
                'Outlier_Percent': _safe_pct(outlier_count, total_count),
            }
            feature_summary_rows.append(summary_row)
            all_summary_rows.append(summary_row)

            outliers_df = group_df.loc[outlier_mask].copy()
            if not outliers_df.empty:
                outliers_df['Feature'] = feature
                outliers_df['Aggregation_Level'] = aggregation_level
                outliers_df['Value'] = values.loc[outlier_mask].values
                outliers_df['Outlier_Direction'] = directions.loc[outlier_mask].values
                outliers_df['Outlier_Method'] = thresholds['method']
                outliers_df['Lower_Threshold'] = thresholds['lower_threshold']
                outliers_df['Upper_Threshold'] = thresholds['upper_threshold']
                outliers_df['Distance_From_Threshold'] = np.where(
                    outliers_df['Outlier_Direction'] == "low",
                    thresholds['lower_threshold'] - outliers_df['Value'],
                    outliers_df['Value'] - thresholds['upper_threshold'],
                )
                outliers_df = outliers_df[
                    [
                        'Feature', 'Batch', 'Group', 'Day', 'Organoid_ID',
                        'Aggregation_Level',
                        'Value', 'Outlier_Direction', 'Distance_From_Threshold',
                        'Outlier_Method', 'Lower_Threshold', 'Upper_Threshold',
                    ]
                ]
                feature_outlier_rows.extend(outliers_df.to_dict('records'))
                all_outlier_rows.extend(outliers_df.to_dict('records'))

        outlier_summary = pd.DataFrame(feature_summary_rows)
        outlier_summary.to_csv(os.path.join(outlier_dir, f"{feature}_outlier_summary.csv"), index=False)

        outlier_objects = pd.DataFrame(feature_outlier_rows)
        outlier_objects.to_csv(os.path.join(outlier_dir, f"{feature}_outlier_objects.csv"), index=False)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set_style("whitegrid")
        sns.lineplot(
            data=outlier_summary,
            x='Day',
            y='Outlier_Percent',
            hue='Group',
            marker='s',
            palette=palette,
            ax=ax,
        )
        #ax.set_title(f'Percentage of Outliers in {metric_label} ({outlier_method})')
        ax.set_ylabel('Outliers (%)')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(plot_dir, f"{feature}_{aggregation_level}_outlier_percentage.png"), bbox_inches='tight', dpi=300)
        plt.close()

    if all_summary_rows:
        pd.DataFrame(all_summary_rows).to_csv(
            os.path.join(outlier_dir, "outlier_summary_all_features.csv"),
            index=False,
        )
    if all_outlier_rows:
        pd.DataFrame(all_outlier_rows).to_csv(
            os.path.join(outlier_dir, "outlier_objects_all_features.csv"),
            index=False,
        )

    print(f"Variability and outlier QC saved to {plot_dir}")


def plot_well_organoid_counts_over_time(base_dir, project_dir, batches_complete_info, group_by="batch"):
    """Plot the number of detected organoids per image/well over time."""
    qc_dir = os.path.join(base_dir, project_dir, 'plots', 'quality_control')
    os.makedirs(qc_dir, exist_ok=True)

    records = []
    for entry in batches_complete_info:
        batch = entry['batch']
        label = entry.get(group_by, batch)
        for day in entry['days']:
            metrics_dir = os.path.join(
                base_dir,
                project_dir,
                'processed_data',
                f'Day {day}',
                batch,
                'inference_metrics',
            )
            well_summary_path = os.path.join(metrics_dir, 'well_metrics_summary.csv')
            metrics_path = os.path.join(metrics_dir, 'mask_metrics.csv')

            if os.path.exists(well_summary_path):
                well_df = pd.read_csv(well_summary_path)
                count_col = (
                    'Detected_Organoid_Count'
                    if 'Detected_Organoid_Count' in well_df.columns
                    else 'Object_Count_In_Image'
                )
                if count_col not in well_df.columns:
                    continue
                for _, row in well_df.iterrows():
                    records.append({
                        'Batch': batch,
                        'Group': label,
                        'Day': day,
                        'Image_ID': row.get('Image_ID', row.get('Well_ID')),
                        'Well_ID': row.get('Well_ID', row.get('Image_ID')),
                        'Plate_Index': row.get('Plate_Index', np.nan),
                        'Well_Index': row.get('Well_Index', np.nan),
                        'Organoid_Count': row[count_col],
                    })
                continue

            if not os.path.exists(metrics_path):
                continue

            metrics_df = _ensure_metric_identity_columns(pd.read_csv(metrics_path))
            count_df = (
                metrics_df
                .groupby(['Image_ID', 'Well_ID'], dropna=False)
                .agg(Organoid_Count=('Organoid_ID', 'count'))
                .reset_index()
            )
            for _, row in count_df.iterrows():
                records.append({
                    'Batch': batch,
                    'Group': label,
                    'Day': day,
                    'Image_ID': row['Image_ID'],
                    'Well_ID': row['Well_ID'],
                    'Plate_Index': np.nan,
                    'Well_Index': np.nan,
                    'Organoid_Count': row['Organoid_Count'],
                })

    if not records:
        print("No per-well organoid count data found.")
        return pd.DataFrame()

    count_df = pd.DataFrame(records)
    count_df.to_csv(os.path.join(qc_dir, "organoid_counts_by_well.csv"), index=False)

    summary = (
        count_df
        .groupby(['Group', 'Day'], dropna=False)['Organoid_Count']
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )
    summary['sem'] = summary['std'] / np.sqrt(summary['count'])
    summary.to_csv(os.path.join(qc_dir, "organoid_counts_by_group_day.csv"), index=False)

    unique_groups = sorted(count_df['Group'].dropna().unique())
    palette = dict(zip(unique_groups, sns.color_palette("husl", len(unique_groups))))

    fig, ax = plt.subplots(figsize=(10,6))
    sns.set_style("whitegrid")
    for group in unique_groups:
        group_df = summary[summary['Group'] == group]
        ax.errorbar(
            group_df['Day'],
            group_df['mean'],
            yerr=group_df['sem'],
            label=group,
            marker='o',
            capsize=4,
            color=palette[group],
            linewidth=2,
        )
    ax.set_xlabel('Day')
    ax.set_ylabel('Detected organoids per well')
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(qc_dir, "organoid_counts_per_well_over_time.png"), bbox_inches='tight', dpi=300)
    plt.close()

    return count_df


def _get_detection_runs(batch_metadata):
    if "get_segmentation_runs" in globals():
        return get_segmentation_runs(batch_metadata)

    runs = []
    if batch_metadata.get("days_96"):
        runs.append({
            "name": "days_96",
            "method": "dl",
            "days": batch_metadata["days_96"],
            "organoid_mode": "single",
            "num_plates": batch_metadata.get("num_plates", 1),
            "stitched_images_per_batch": 24,
            "images_per_plate": 24,
        })
    if batch_metadata.get("days_24"):
        runs.append({
            "name": "days_24",
            "method": "cl",
            "days": batch_metadata["days_24"],
            "organoid_mode": "single",
            "num_plates": batch_metadata.get("num_plates", 1),
            "stitched_images_per_batch": 24,
            "images_per_plate": 24,
        })
    return runs


def _read_metrics(base_dir, project_dir, day, batch):
    metrics_csv = os.path.join(
        base_dir,
        project_dir,
        'processed_data',
        f'Day {day}',
        batch,
        'inference_metrics',
        'mask_metrics.csv',
    )
    if not os.path.exists(metrics_csv):
        return pd.DataFrame(), metrics_csv
    return pd.read_csv(metrics_csv), metrics_csv


def _list_file_ids(folder, extensions):
    if not os.path.isdir(folder):
        return []
    extensions = tuple(ext.lower() for ext in extensions)
    ids = []
    for filename in os.listdir(folder):
        if filename.startswith("."):
            continue
        if filename.lower().endswith(extensions):
            ids.append(os.path.splitext(filename)[0])
    return sorted(set(ids))


def _format_expected_tile_ids(count):
    return [f"Tile{idx:06d}" for idx in range(1, int(count) + 1)]


def _expected_image_ids(base_dir, project_dir, day, batch, run):
    raw_dir = os.path.join(base_dir, project_dir, 'raw_data', f'Day {day}', batch)
    image_ids = _list_file_ids(raw_dir, ('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
    if image_ids:
        return image_ids, "raw_data"

    mask_dir = os.path.join(base_dir, project_dir, 'processed_data', f'Day {day}', batch, 'inference_npy')
    image_ids = _list_file_ids(mask_dir, ('.npy',))
    if image_ids:
        return image_ids, "inference_npy"

    if run.get("organoid_mode") == "single" and run.get("method") == "cl":
        images_per_plate = run.get("images_per_plate", run.get("stitched_images_per_batch", 24))
        expected_count = int(run.get("num_plates", 0)) * int(images_per_plate)
        if expected_count > 0:
            return _format_expected_tile_ids(expected_count), "metadata"

    return [], "unknown"


def _image_id_from_organoid_id(organoid_id):
    return str(organoid_id).split("_object", 1)[0]


def _metric_image_ids(metrics_df):
    if metrics_df.empty:
        return set()
    if 'Image_ID' in metrics_df.columns:
        return set(metrics_df['Image_ID'].dropna().astype(str))
    return set(_image_id_from_organoid_id(organoid_id) for organoid_id in metrics_df['Organoid_ID'].astype(str))


def create_detection_analysis(base_dir, project_dir, batches_complete_info, batches_info, group_by="batch"):
    """
    Report detected, undetected, and detected-but-filtered organoids.

    For single-organoid images, expected objects are inferred from raw image IDs,
    saved masks, or classical 24-well metadata. For multiple-organoid images, the
    expected object count is not knowable from metadata, so the report focuses on
    detected and not-considered objects and separately tracks expected images.
    """
    qc_dir = os.path.join(base_dir, project_dir, 'plots', 'quality_control')
    batch_matrix_dir = os.path.join(qc_dir, 'batch_matrices')
    os.makedirs(batch_matrix_dir, exist_ok=True)

    info_lookup = {batch['batch']: batch for batch in batches_info}
    all_batch_summaries = []

    for entry in batches_complete_info:
        batch = entry['batch']
        label = entry.get(group_by, batch)
        batch_metadata = info_lookup.get(batch)

        if not batch_metadata:
            print(f"Metadata for {batch} not found in batches_info. Skipping.")
            continue

        for run in _get_detection_runs(batch_metadata):
            run_name = run.get("name", run.get("method", "run"))
            run_days = sorted(run.get("days", []))
            organoid_mode = run.get("organoid_mode", "single")
            status_by_day = {}
            expected_id_union = set()

            for day in run_days:
                metrics_df, metrics_csv = _read_metrics(base_dir, project_dir, day, batch)
                expected_ids, expected_source = _expected_image_ids(base_dir, project_dir, day, batch, run)
                expected_id_set = set(expected_ids)
                expected_id_union.update(expected_id_set)

                if metrics_df.empty:
                    metrics_df = pd.DataFrame(columns=['Organoid_ID'])
                elif 'Organoid_ID' not in metrics_df.columns:
                    metrics_df = metrics_df.copy()
                    metrics_df['Organoid_ID'] = [f"{batch}_day{day}_object{idx:04d}" for idx in range(1, len(metrics_df) + 1)]

                considered_df = filter_organoids(metrics_df, day) if not metrics_df.empty else pd.DataFrame()

                if organoid_mode == "single":
                    detected_ids = set(metrics_df['Organoid_ID'].astype(str))
                    considered_ids = set(considered_df['Organoid_ID'].astype(str)) if not considered_df.empty else set()
                    if expected_id_set:
                        detected_ids = detected_ids & expected_id_set
                        considered_ids = considered_ids & expected_id_set

                    expected_objects = len(expected_id_set) if expected_id_set else np.nan
                    detected_count = len(detected_ids)
                    considered_count = len(considered_ids)
                    not_considered_count = max(0, detected_count - considered_count)
                    undetected_count = max(0, expected_objects - detected_count) if not pd.isna(expected_objects) else np.nan

                    if expected_id_set:
                        status = pd.Series("undetected", index=sorted(expected_id_set), dtype=object)
                        detected_not_considered = detected_ids - considered_ids
                        status.loc[sorted(detected_not_considered)] = "detected_not_considered"
                        status.loc[sorted(considered_ids)] = "considered"
                        status_by_day[day] = status
                else:
                    detected_count = len(metrics_df)
                    considered_count = len(considered_df)
                    not_considered_count = max(0, detected_count - considered_count)
                    expected_objects = np.nan
                    undetected_count = np.nan

                detected_image_ids = _metric_image_ids(metrics_df)
                considered_image_ids = _metric_image_ids(considered_df)
                expected_images = len(expected_id_set) if expected_id_set else np.nan
                undetected_images = (
                    max(0, expected_images - len(detected_image_ids & expected_id_set))
                    if expected_id_set else np.nan
                )

                excluded_count = not_considered_count
                if not pd.isna(undetected_count):
                    excluded_count += undetected_count

                all_batch_summaries.append({
                    'Batch': batch,
                    'Group': label,
                    'Day': day,
                    'Run': run_name,
                    'Method': run.get("method"),
                    'Organoid_Mode': organoid_mode,
                    'Expected_Source': expected_source,
                    'Expected_Images': expected_images,
                    'Detected_Images': len(detected_image_ids),
                    'Considered_Images': len(considered_image_ids),
                    'Undetected_Images': undetected_images,
                    'Expected_Objects': expected_objects,
                    'Detected': detected_count,
                    'Considered': considered_count,
                    'Not_Considered': not_considered_count,
                    'Undetected': undetected_count,
                    'Excluded_From_Analysis': excluded_count,
                    'Metrics_Path': metrics_csv,
                })

            if status_by_day:
                status_matrix = pd.DataFrame(status_by_day).reindex(sorted(expected_id_union))
                status_matrix.to_csv(os.path.join(batch_matrix_dir, f"{batch}_{run_name}_status_matrix.csv"))
                numeric_matrix = status_matrix.replace({
                    "undetected": 0,
                    "detected_not_considered": -1,
                    "considered": 1,
                })
                numeric_matrix.to_csv(os.path.join(batch_matrix_dir, f"{batch}_{run_name}_numeric_matrix.csv"))

    summary_df = pd.DataFrame(all_batch_summaries)
    if summary_df.empty:
        print("No detection QC data processed.")
        return None, qc_dir

    summary_df['Detection_Rate_Pct'] = summary_df.apply(
        lambda row: _safe_pct(row['Detected'], row['Expected_Objects']),
        axis=1,
    )
    summary_df['Considered_Rate_Pct'] = summary_df.apply(
        lambda row: _safe_pct(row['Considered'], row['Expected_Objects']),
        axis=1,
    )
    summary_df['Undetected_Pct'] = summary_df.apply(
        lambda row: _safe_pct(row['Undetected'], row['Expected_Objects']),
        axis=1,
    )
    summary_df['Not_Considered_Pct'] = summary_df.apply(
        lambda row: _safe_pct(row['Not_Considered'], row['Expected_Objects']),
        axis=1,
    )
    summary_df['Not_Considered_Of_Detected_Pct'] = summary_df.apply(
        lambda row: _safe_pct(row['Not_Considered'], row['Detected']),
        axis=1,
    )
    summary_df['Excluded_From_Analysis_Pct'] = summary_df.apply(
        lambda row: _safe_pct(row['Excluded_From_Analysis'], row['Expected_Objects']),
        axis=1,
    )

    summary_df.to_csv(os.path.join(qc_dir, "detection_qc_by_batch_day.csv"), index=False)

    def sum_with_nan(series):
        return series.sum(min_count=1)

    global_summary = summary_df.groupby(['Group', 'Day'], dropna=False).agg({
        'Expected_Images': sum_with_nan,
        'Detected_Images': sum_with_nan,
        'Considered_Images': sum_with_nan,
        'Undetected_Images': sum_with_nan,
        'Expected_Objects': sum_with_nan,
        'Detected': 'sum',
        'Considered': 'sum',
        'Not_Considered': 'sum',
        'Undetected': sum_with_nan,
        'Excluded_From_Analysis': sum_with_nan,
    }).reset_index()

    global_summary['Total_Expected'] = global_summary['Expected_Objects']
    global_summary['Detection_Rate_Pct'] = global_summary.apply(
        lambda row: _safe_pct(row['Detected'], row['Expected_Objects']),
        axis=1,
    )
    global_summary['Considered_Rate_Pct'] = global_summary.apply(
        lambda row: _safe_pct(row['Considered'], row['Expected_Objects']),
        axis=1,
    )
    global_summary['Undetected_Pct'] = global_summary.apply(
        lambda row: _safe_pct(row['Undetected'], row['Expected_Objects']),
        axis=1,
    )
    global_summary['Not_Considered_Pct'] = global_summary.apply(
        lambda row: _safe_pct(row['Not_Considered'], row['Expected_Objects']),
        axis=1,
    )
    global_summary['Not_Considered_Of_Detected_Pct'] = global_summary.apply(
        lambda row: _safe_pct(row['Not_Considered'], row['Detected']),
        axis=1,
    )
    global_summary['Excluded_From_Analysis_Pct'] = global_summary.apply(
        lambda row: _safe_pct(row['Excluded_From_Analysis'], row['Expected_Objects']),
        axis=1,
    )

    global_summary.to_csv(os.path.join(qc_dir, "detection_qc_global_summary.csv"), index=False)
    global_summary.to_csv(os.path.join(qc_dir, "undetected_tiles_global_summary.csv"), index=False)

    return global_summary, qc_dir


def plot_undetected_summary(global_summary, output_dir):
    """
    Plot detection QC percentages from create_detection_analysis.

    The plot separates true undetected objects from detected objects that were
    later excluded by morphology filters.
    """
    if global_summary is None or global_summary.empty:
        print("No detection QC summary available for plotting.")
        return

    percentage_columns = {
        'Undetected_Pct': 'Undetected (% expected)',
        'Not_Considered_Of_Detected_Pct': 'Not considered (% detected)',
        'Excluded_From_Analysis_Pct': 'Excluded from analysis (% expected)',
    }
    available_columns = [
        column for column in percentage_columns
        if column in global_summary.columns and not global_summary[column].isna().all()
    ]

    if not available_columns:
        print("No percentage columns available for detection QC plotting.")
        return

    plot_df = global_summary.melt(
        id_vars=['Group', 'Day'],
        value_vars=available_columns,
        var_name='QC_Metric',
        value_name='Percent',
    ).dropna(subset=['Percent'])
    plot_df['QC_Metric'] = plot_df['QC_Metric'].map(percentage_columns)

    plt.figure(figsize=(10,6))
    sns.set_style("whitegrid")
    ax = sns.lineplot(
        data=plot_df,
        x='Day',
        y='Percent',
        hue='Group',
        style='QC_Metric',
        markers=True,
        dashes=False,
        palette="husl",
        linewidth=2,
    )

    plt.ylabel('Organoids (%)')
    plt.xlabel('Day')
    ax.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "detection_qc_percentage_plot.png")
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()

    undetected_only = global_summary.dropna(subset=['Undetected_Pct'])
    if not undetected_only.empty:
        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        ax = sns.lineplot(
            data=undetected_only,
            x='Day',
            y='Undetected_Pct',
            hue='Group',
            palette="husl",
            marker='o',
            linewidth=2,
        )
        plt.ylabel('Undetected Organoids (%)')
        plt.xlabel('Day')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "undetected_tiles_percentage_plot.png"), bbox_inches='tight', dpi=300)
        plt.close()

    print(f"Quality control plots and summaries saved in: {output_dir}")
