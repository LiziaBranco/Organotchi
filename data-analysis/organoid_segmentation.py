from utils import (
    organoid_segmentation_inference, 
    analyze_organoid_morphology, 
    apply_filtering_to_segmentations, 
    process_24wellplate_images, 
    process_24wellplate_segmentation,
    detect_outliers_morphology
)
from experiment_info import *

# Process each plate
for batch in batches_info:
    batch_name = batch["batch"]
    
    # Process 96-well plates
    if "days_96" in batch and batch["days_96"]:
        print(f"\n🧠 Processing Plate {batch_name} (96-well plates)")
        organoid_segmentation_inference(checkpoint, base_dir, project_dir, batch["days_96"], [batch_name])
        apply_filtering_to_segmentations(base_dir, project_dir, batch["days_96"], [batch_name], pixel_to_micron["96_well"]["x"], pixel_to_micron["96_well"]["y"])
        analyze_organoid_morphology(base_dir, project_dir, batch["days_96"], [batch_name], pixel_to_micron["96_well"]["x"], pixel_to_micron["96_well"]["y"])
        detect_outliers_morphology(base_dir = base_dir, project_dir=project_dir, days=batch["days_96"], batches=[batch_name], manual_thresholds=None, features=features, mad_multiplier=3)
    
    # Process 24-well plates
    if "days_24" in batch and batch["days_24"]:
        print(f"\n🧠 Processing Plate {batch_name} (24-well plates)")
        process_24wellplate_images(base_dir, project_dir, batch["days_24"], [batch_name], [batch["num_plates"]])
        process_24wellplate_segmentation(base_dir, project_dir, batch["days_24"], [batch_name], pixel_to_micron["24_well"]["x"], pixel_to_micron["24_well"]["y"])
        analyze_organoid_morphology(base_dir, project_dir, batch["days_24"], [batch_name], pixel_to_micron["24_well"]["x"], pixel_to_micron["24_well"]["y"])
        detect_outliers_morphology(base_dir = base_dir, project_dir=project_dir, days=batch["days_24"], batches=[batch_name], manual_thresholds=None, features=features, mad_multiplier=3)
print("Processing completed for all plates.")