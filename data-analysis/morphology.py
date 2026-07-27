from utils_morphology import *

import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
import pandas as pd
import os

features = [
    'Area_um2', 'Diameter_um', 'Perimeter_um',
    'Feret_Diameter_Max_um', 'Aspect_Ratio', 'Circularity',
    'Roundness', 'Solidity', 'Eccentricity', 'Skeleton_Length_um'
]

features_names = [
    'Area (um²)', 'Diameter (um)', 'Perimeter (um)',
    'Feret Diameter (um)', 'Aspect Ratio', 'Circularity',
    'Roundness', 'Solidity', 'Eccentricity', 'Skeleton Length (um)'
]


plates_info_dict = {
    batch["batch"]: {
        "days": batch["days"]
    }
    for batch in batches_complete_info
}

days = [19,44,62]

groups = ['sample', 'condition']

outlier_method = "iqr"
iqr_multiplier = 1.5
# For manual mean +/- N SD thresholding, set outlier_method = "mean_sd"
# and change std_multiplier or provide manual_std_thresholds.
std_multiplier = 3.0
manual_std_thresholds = None
plot_aggregation_levels = ["object"]
outlier_aggregation_level = "object"

for group in groups:
    for aggregation_level in plot_aggregation_levels:
        plot_feature_over_time(
            base_dir,
            project_dir,
            batches_complete_info,
            features,
            features_names,
            group_by=group,
            aggregation_level=aggregation_level,
        )
    #plot_pca_grid_by_group(base_dir, project_dir, days=days, batches_complete_info=batches_complete_info, group_by=group)
    
    plot_feature_grid_scatters(base_dir,  project_dir,  days=days, batches_complete_info=plates_info_dict, features=features, metric_names=features_names, group_by=group)
    plot_well_organoid_counts_over_time(base_dir, project_dir, batches_complete_info, group_by=group)
    plot_variability_and_outliers(
        base_dir,
        project_dir,
        batches_complete_info,
        features=features,
        metric_names=features_names,
        group_by=group,
        outlier_method=outlier_method,
        iqr_multiplier=iqr_multiplier,
        std_multiplier=std_multiplier,
        manual_std_thresholds=manual_std_thresholds,
        aggregation_level=outlier_aggregation_level,
    )
    
    #detection_summary, out_path = create_detection_analysis(base_dir, project_dir, batches_complete_info, batches_info, group_by=group)
    #plot_undetected_summary(detection_summary, out_path)

print("✅ All processing and plotting completed.")
