from utils_morphology import *

features = [
    'Area_um2', 'Diameter_um', 'Perimeter_um', 'Circularity',
    'Skeleton_Length_um', 'Solidity', 'Eccentricity', 'Roughness_Index'
]

features_names = [
    'Area (um²)', 'Diameter (um)', 'Perimeter (um)', 'Circularity',
    'Skeleton Length (um)', 'Solidity', 'Eccentricity', 'Roughness Index'
]

plates_info_dict = {
    batch["batch"]: {
        "days": batch["days"],
        "condition": batch["condition"],
        "cell-line": batch["cell-line"]
    }
    for batch in batches_complete_info
}

days = [1, 22, 32]
groups = ['batch', 'condition', 'cell-line']
for i, feature in enumerate(features):
    for group in groups:
        plot_feature_over_time(base_dir, project_dir, batches_complete_info, features, features_names, group_by=group)
        plot_pca_grid_by_group(base_dir, project_dir, days=days, batches_complete_info=batches_complete_info, group_by=group)
        plot_violin_grid_by_group(base_dir,project_dir, days=days, batches_complete_info=batches_complete_info, metric=feature, metric_name=features_names[i], group_by=group)
        plot_barplot_grid_by_group(base_dir, project_dir, days=days, batches_complete_info=batches_complete_info, metric=feature, metric_name=features_names[i], group_by=group)
        
        #for day in days:
        #    plate_day_pairs = [(batch["batch"], day) for batch in batches_complete_info if day in batch["days"]]
        #    plot_feature_comparison_with_stats(base_dir, project_dir, feature, features_names[i], plate_day_pairs, plates_info_dict, group_by='cell-line', title=f'{features_names[i]} Comparison between {group} at day {str(day)}', filename=f'comparison_{features_names[i]}_{group}_d{str(day)}.png')

print("✅ All processing and plotting completed.")
