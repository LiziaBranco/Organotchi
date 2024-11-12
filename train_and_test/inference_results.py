import os
import numpy as np
from PIL import Image

def load_and_convert_npy_image(npy_path):
    """Load .npy file, convert 0s and 1s to 255 for visualization."""
    array = np.load(npy_path)
    img = Image.fromarray((array * 255).astype(np.uint8))  # Convert binary mask to an image
    return img

def load_jpg_image(jpg_path):
    """Load a jpg image."""
    return Image.open(jpg_path)

def save_concatenated_image(ground_truth_path, inference_path, original_images, output_folder, image_name):
    # Define paths
    ground_truth_path = os.path.join(ground_truth_path, f"{image_name}.npy")
    inference_path = os.path.join(inference_path, f"{image_name}.npy")
    jpg_path = os.path.join(original_images, f"{image_name}.jpg")

    # Check if all the required files exist
    if not (os.path.exists(ground_truth_path) and os.path.exists(inference_path) and os.path.exists(jpg_path)):
        print(f"Skipping {image_name} due to missing files.")
        return

    # Load images
    img1 = load_and_convert_npy_image(ground_truth_path)
    img2 = load_and_convert_npy_image(inference_path)
    img3 = load_jpg_image(jpg_path)

    # Resize images to the same size (optional, adjust as necessary)
    img2 = img2.resize(img1.size)
    img3 = img3.resize(img1.size)

    # Concatenate the images horizontally
    total_width = img1.width + img2.width + img3.width
    max_height = max(img1.height, img2.height, img3.height)
    
    # Create a new blank image to hold the concatenated result
    concatenated_image = Image.new('RGB', (total_width, max_height))

    # Paste the images side by side
    concatenated_image.paste(img1.convert('RGB'), (0, 0))  # Convert to RGB for consistency
    concatenated_image.paste(img2.convert('RGB'), (img1.width, 0))
    concatenated_image.paste(img3.convert('RGB'), (img1.width + img2.width, 0))

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save the concatenated image
    output_path = os.path.join(output_folder, f"{image_name}_concatenated.jpg")
    concatenated_image.save(output_path)
    print(f"Concatenated image saved at {output_path}")

def process_all_npy_images(ground_truth_path, inference_path, original_images, output_folder):
    """Process all npy files in npy_folder2 and save concatenated images in output_folder."""
    for npy_file in os.listdir(inference_path):
        if npy_file.endswith('.npy'):
            # Extract the base image name (without the .npy extension)
            image_name = os.path.splitext(npy_file)[0]

            # Call the function to process and save the concatenated image
            save_concatenated_image(ground_truth_path, inference_path, original_images, output_folder, image_name)

# Example usage
if __name__ == "__main__":

    for split in range(0, 5):  # Adjust split range as needed
        ground_truth_path = 'data/labels'  # Folder with original npy masks
        inference_path = f"Training_Results/work_dirs/split-{split}/inference_npy"  # Folder with new npy masks
        original_images = 'data/imgs'  # Folder with jpg original images
        output_folder = f"Training_Results/work_dirs/split-{split}/inference_vis"  # Folder to save concatenated images

        process_all_npy_images(ground_truth_path, inference_path, original_images, output_folder)
