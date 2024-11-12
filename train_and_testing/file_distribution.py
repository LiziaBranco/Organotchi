import os
import shutil
import numpy as np
from PIL import Image  # To convert and save images
import pandas as pd

# Root directories for images and masks
images_root_dir = 'data/imgs'  # Replace with the actual root folder for images
masks_root_dir = 'data/labels'  # Replace with the actual root folder for masks
output_dir = 'robust_monitoring_organoid_growth/data'  # Replace with the actual output directory for split data

# Load the CSV file
csv_file = 'splits.csv'  # Replace with the actual path
df = pd.read_csv(csv_file)

# Clear existing output directories if they exist
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)  # Remove the entire output directory

# Recreate the root output directory
os.makedirs(output_dir, exist_ok=True)

# Desired image and mask dimensions
target_size = (1024, 768)

# Iterate through each row of the CSV file
for index, row in df.iterrows():
    img_filename = row['File']
    img_path = os.path.join(images_root_dir, img_filename)

    # Check if the file is a .jpg or .tif
    if img_filename.lower().endswith('.jpg'):
        # If it's already a .jpg, no need to convert
        new_img_filename = img_filename
    elif img_filename.lower().endswith('.tif'):
        # If it's a .tif, convert it to .jpg
        img = Image.open(img_path)
        img = img.resize(target_size, Image.Resampling.LANCZOS)  # Resize the image to 1024x768
        new_img_filename = img_filename.rsplit('.', 1)[0] + '.jpg'
        new_img_path = os.path.join(images_root_dir, new_img_filename)
        img.save(new_img_path, "JPEG")  # Save as JPEG
        img_path = new_img_path  # Use the new .jpg image path
    else:
        # Skip non-JPG and non-TIF files
        print(f"Skipping unsupported file format: {img_filename}")
        continue  # Skip files that are not .jpg or .tif

    # Mask filename (same name but with .npy extension)
    mask_filename = img_filename.rsplit('.', 1)[0] + '.npy'
    mask_path = os.path.join(masks_root_dir, mask_filename)

    # Check if the mask file exists
    if not os.path.exists(mask_path):
        print(f"Mask file not found for {img_filename}, skipping...")
        continue  # Skip files without corresponding mask

    # Iterate through the splits (split0 to split4)
    for split in ['split0', 'split1', 'split2', 'split3', 'split4']:
        split_type = row[split]

        # Create image and mask directories for the current split (train/val/test)
        split_img_dir = os.path.join(output_dir, f'images/{split}_{split_type}')
        split_mask_dir = os.path.join(output_dir, f'masks/{split}_{split_type}')

        # Create the directories if they don't exist
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_mask_dir, exist_ok=True)

        # Copy and resize the image to the split image directory
        img = Image.open(img_path)
        img = img.resize(target_size, Image.Resampling.LANCZOS)  # Resize image to 1024x768
        img.save(os.path.join(split_img_dir, new_img_filename))

        # Load and resize the mask .npy file
        mask_array = np.load(mask_path)

        # Normalize the mask values: ensure that the mask contains only 0 for background and 1 for organoids
        binary_mask = np.where(mask_array > 0, 1, 0).astype(np.uint8)  # Ensure binary mask (0 or 1)

        # Create a PIL image from the binary mask
        mask_img = Image.fromarray(binary_mask, mode='L')  # 'L' mode for grayscale

        # Resize the mask to the target size
        mask_img = mask_img.resize(target_size, Image.NEAREST)  # Resize mask to 1024x768

        # Save the mask as a TIFF image in the split mask directory
        mask_tif_filename = mask_filename.rsplit('.', 1)[0] + '.tif'  # Change extension to .tiff
        mask_img.save(os.path.join(split_mask_dir, mask_tif_filename), format='TIFF')  # Save as TIFF

print("Output directories have been created, and only listed JPG and converted TIF files, along with their masks, have been copied, resized to 1024x768, and processed.")
