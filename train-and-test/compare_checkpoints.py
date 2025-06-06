import os
import numpy as np
from PIL import Image
import pandas as pd
from mmengine.config import Config
from mmseg.apis import inference_model, init_model
from torch import nn
from os.path import join as opj
from utils import *

base_directory = '../../../AGMayer/Lizia/Organoid_Segmentation/2025_03_05-Organoid-Segmentation'

project_directory = f'{base_directory}/2024_02_22-Organoid-Segmentation-lancaster'

checkpoints = {
    'split0': f'{project_directory}/Training-Results/work_dirs/split-0/iter_144000.pth',
    'split1': f'{project_directory}/Training-Results/work_dirs/split-1/iter_128000.pth',
    'split2': f'{project_directory}/Training-Results/work_dirs/split-2/iter_160000.pth',
    'split3': f'{project_directory}/Training-Results/work_dirs/split-3/iter_144000.pth',
    'split4': f'{project_directory}/Training-Results/work_dirs/split-4/iter_112000.pth'
}

for split in range(0,5):  # Loop through splits 0 to 4
    img_dir = f'{project_directory}/Training-Results/data/images/split{split}_test'
    mask_dir = f'{project_directory}/Training-Results/data/masks/split{split}_test'  # Ground truth masks
    ckp_dir = os.path.dirname(checkpoints[f'split{split}'])
    outdir_inference = opj(ckp_dir, 'inference_npy')
    outdir_tif = opj(ckp_dir, 'inference_tif')
    os.makedirs(outdir_inference, exist_ok=True)
    os.makedirs(outdir_tif, exist_ok=True)

    # Load the model config and checkpoint for CPU
    ckp = checkpoints[f'split{split}']
    cfg = Config.fromfile(f'{project_directory}/Training-Results/work_dirs/split-{split}/split-{split}.py')
    cfg.load_from = ckp
    cfg.model.decode_head.norm_cfg = dict(type='BN', requires_grad=True)

    # Initialize the model for CPU (using map_location='cpu' to force loading on CPU)
    model = init_model(cfg, checkpoint=ckp, device='cpu')
    model.apply(all_eval_except_dropout)

    # Prepare CSV to store the metrics
    metrics_csv_path = f'{project_directory}/Training-Results/work_dirs/split-{split}/split{split}_metrics.csv'
    metrics_data = []

    # Perform inference and calculate metrics for each image
    for img_loc in sorted(os.listdir(img_dir)):
        img = os.path.join(img_dir, img_loc)
        img_id = os.path.basename(img).replace('.jpg', '')

        # Run inference on the CPU
        result = inference_model(model, img)

        # Extract the prediction mask from the SegDataSample object
        pred = result.pred_sem_seg.data[0].cpu().numpy()

        # Save the predicted mask in npy format
        npy_pred_path = opj(outdir_inference, f'{img_id}.npy')
        np.save(npy_pred_path, pred)

        # Save the predicted mask in tif format
        tif_pred_path = opj(outdir_tif, f'{img_id}.tif')
        save_as_tif(pred, tif_pred_path)

            # Load the ground truth mask as a TIF image
        gt_path = os.path.join(mask_dir, f'{img_id}.tif')
        if not os.path.exists(gt_path):
            print(f"Ground truth mask for {img_id} not found, skipping...")
            continue

        # Use PIL to open the tif image and convert it to a numpy array
        gt_image = Image.open(gt_path)
        gt = np.array(gt_image)

        # Calculate Dice, IoU, and accuracy
        dice = dice_coefficient(pred, gt)
        iou = iou_score(pred, gt)
        acc = accuracy(pred, gt)

        # Store the metrics for this image in a list
        metrics_data.append([img_id, dice, iou, acc])

        print(f"Processed {img_id} - Dice: {dice:.4f}, IoU: {iou:.4f}, Accuracy: {acc:.4f}")

    # Create a DataFrame for the metrics and save it as CSV
    metrics_df = pd.DataFrame(metrics_data, columns=['Image_ID', 'Dice', 'IoU', 'Accuracy'])
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Metrics saved to {metrics_csv_path}")

for split in range(0, 5):  # Adjust split range as needed
    ground_truth_path = f'{project_directory}/Training-Data/labels'  # Folder with original npy masks
    inference_path = f'{project_directory}/Training-Results/work_dirs/split-{split}/inference_npy'  # Folder with new npy masks
    original_images = f'{project_directory}/Training-Data/imgs'  # Folder with jpg original images
    output_folder = f'{project_directory}/Training-Results/work_dirs/split-{split}/inference_vis'  # Folder to save concatenated images

    process_all_npy_images(ground_truth_path, inference_path, original_images, output_folder)