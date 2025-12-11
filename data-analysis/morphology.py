from utils_morphology import *

import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
import pandas as pd
import os


features = [
    'Area_um2', 'Diameter_um', 'Perimeter_um', 'Circularity',
    'Skeleton_Length_um', 'Solidity', 'Eccentricity', 'Roughness_Index'
]

features_names = [
    'Area (um²)', 'Diameter (um)', 'Perimeter (um)', 'Circularity',
    'Skeleton Length (um)', 'Solidity', 'Eccentricity', 'Roughness Index'
]

features = [
    'Area_um2', 'Diameter_um'
]

features_names = [
    'Area (um²)', 'Diameter (um)'
]

plates_info_dict = {
    batch["batch"]: {
        "days": batch["days"],
        "condition": batch["condition"]
    }
    for batch in batches_complete_info
}

plates_info_dict = {
    batch["batch"]: {
        "days": batch["days"],
        "condition": batch["condition"]
    }
    for batch in batches_complete_info
}

days = [32, 33, 34, 35, 41, 45]

groups = ['condition']

for i, feature in enumerate(features):
    for group in groups:
        plot_feature_over_time(base_dir, project_dir, batches_complete_info, features, features_names, group_by=group)
        
        #plot_pca_grid_by_group(base_dir, project_dir, days=days, batches_complete_info=batches_complete_info, group_by=group)
        #plot_violin_grid_by_group(base_dir,project_dir, days=days, batches_complete_info=batches_complete_info, metric=feature, metric_name=features_names[i], group_by=group)
        #plot_barplot_grid_by_group(base_dir, project_dir, days=days, batches_complete_info=batches_complete_info, metric=feature, metric_name=features_names[i], group_by=group)

        #if group == 'condition' and feature == 'Area_um2':        
        #    plot_outlier_rate_over_days(merged_df, group_by=group, feature=feature)
        #    plot_outlier_distribution(merged_df, group_by=group, feature=feature)
        
        
        if group == 'condition' and feature == 'Area_um2':
            for day in days:
                plate_day_pairs = [(batch["batch"], day) for batch in batches_complete_info if day in batch["days"]]
                plot_feature_scatter_with_stats(base_dir, project_dir, feature, features_names[i], i, plate_day_pairs, plates_info_dict, group_by=group, title_prefix=f"Growth Comparison at Day {day}", filename=f"feature_comparison_scatter_{day}.svg")
            #    plot_feature_comparison_with_stats(base_dir, project_dir, feature, features_names[i], plate_day_pairs, plates_info_dict, group_by=group, title=f'{features_names[i]} Comparison between {group} at day {str(day)}', filename=f'comparison_{features_names[i]}_{group}_d{str(day)}.png')
            
            '''n_days = len(days)
            n_cols = 10  # adjust number of columns for layout
            n_rows = (n_days + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
            axes = axes.flatten()  # flatten in case of a grid

            for j, day in enumerate(days):
                plate_day_pairs = [(batch["batch"], day) for batch in batches_complete_info if day in batch["days"]]
                plot_feature_on_ax(axes[j], base_dir, project_dir, metric=feature, metric_name="Area (um²)",
                    plate_day_pairs=plate_day_pairs, batches_complete_info=plates_info_dict, group_by=group, title=f'Day {day}')

            for k in range(j+1, len(axes)):
                fig.delaxes(axes[k])

            path = os.path.join(plot_dir, 'comparison_Area_condition.png')
            fig.savefig(path, dpi=300)
            #plt.tight_layout()
            plt.show() '''

print("✅ All processing and plotting completed.")