import os
import pandas as pd
from radiomics import featureextractor # https://pyradiomics.readthedocs.io/en/latest/index.html
from PIL import Image

# Set the paths for the TIF masks and the output CSV file
tif_masks_dir = 'robust_monitoring_organoid_growth/work_dirs/split-0/inference_tif'  # Path to .tif masks
output_csv = 'shape2D_features.csv'  # Output CSV file to save the results

# Initialize the PyRadiomics feature extractor for 2D shape features
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.enableFeatureClassByName('shape2D')  # Enable only the shape2D feature class

# List to store the results
results = []

# Iterate through each tif mask in the directory
for mask_filename in sorted(os.listdir(tif_masks_dir)):
    # Process only .tif files
    if mask_filename.lower().endswith('.tif'):
        # Load the mask
        mask_path = os.path.join(tif_masks_dir, mask_filename)
        
        # PyRadiomics expects an image and a mask, but we can pass the mask as both for 2D shape extraction
        try:
            # Extract features
            features = extractor.execute(mask_path, mask_path)

            # Collect the features into a dictionary
            image_results = {"image": mask_filename}
            for feature_name, feature_value in features.items():
                if feature_name.startswith('shape2D'):
                    image_results[feature_name] = feature_value

            # Add the image results to the list
            results.append(image_results)
            print(f"Processed {mask_filename}")

        except Exception as e:
            print(f"Error processing {mask_filename}: {e}")

# Convert the results list into a Pandas DataFrame
df = pd.DataFrame(results)

# Save the results to a CSV file
df.to_csv(output_csv, index=False)
print(f"Shape2D features saved to {output_csv}")