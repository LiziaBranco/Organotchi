import os
import numpy as np
import pandas as pd
from mmengine.config import Config
from mmseg.apis import inference_model, init_model
from glob import glob

from utils import *  # Ensure you have the necessary utility functions

def main():
    # Dictionary to store metrics (means, stds, etc.) for plotting
    metrics = {
        'Dice': {'means': [], 'stds': [], 'mins': [], 'maxs': [], 'marker': 'o', 'color': 'b'},
        'IoU': {'means': [], 'stds': [], 'mins': [], 'maxs': [], 'marker': 'x', 'color': 'g'},
        'Acc': {'means': [], 'stds': [], 'mins': [], 'maxs': [], 'marker': '^', 'color': 'r'}
    }

    all_checkpoints = [[] for _ in range(5)]  # Store checkpoints for each split

    # Loop through all splits (0 to 4 for now)
    for split in range(0, 5):  # Adjust split range as needed
        img_dir = f'robust_monitoring_organoid_growth/data/images/split{split}_test'
        mask_dir = f'robust_monitoring_organoid_growth/data/masks/split{split}_test'  # Ground truth masks
        ckp_dir = f'robust_monitoring_organoid_growth/work_dirs/split-{split}'
        outdir_inference = os.path.join(ckp_dir, 'inference_npy')
        outdir_tif = os.path.join(ckp_dir, 'inference_tif')

        # Ensure directories for inference output are created
        create_directories([outdir_inference, outdir_tif])

        # List all .pth files (checkpoints) in the checkpoint directory
        checkpoint_files = sorted(glob(os.path.join(ckp_dir, '*.pth')))

        # Prepare CSV for storing overall metrics of all checkpoints
        summary_csv_path = os.path.join(ckp_dir, f'split{split}_summary_metrics.csv')
        summary_csv_columns = ['Iteration', 'Dice_Min', 'Dice_Max', 'Dice_Mean', 'Dice_Std',
                               'IoU_Min', 'IoU_Max', 'IoU_Mean', 'IoU_Std',
                               'Acc_Min', 'Acc_Max', 'Acc_Mean', 'Acc_Std']
        summary_data = []
        checkpoints = []

        # Create the checkpoint_metrics directory inside the split directory
        checkpoint_metrics_dir = os.path.join(ckp_dir, 'checkpoint_metrics')
        os.makedirs(checkpoint_metrics_dir, exist_ok=True)

        # Loop over each checkpoint
        for ckp in checkpoint_files:
            checkpoint_name = os.path.basename(ckp)
            try:
                iter_number = int(checkpoint_name.split('_')[1].replace('iter', '').replace('.pth', ''))
                checkpoints.append(iter_number)
            except ValueError:
                print(f"Skipping file {ckp}, invalid format for iteration number.")
                continue

            # Load the model config and checkpoint for CPU
            cfg = Config.fromfile(f'robust_monitoring_organoid_growth/work_dirs/split-{split}/split-{split}.py')
            cfg.load_from = ckp
            cfg.model.decode_head.norm_cfg = dict(type='BN', requires_grad=True)

            # Initialize the model (forcing loading on CPU)
            model = init_model(cfg, checkpoint=ckp, device='cpu')
            model.apply(all_eval_except_dropout)

            # Store per-image metrics for this checkpoint
            checkpoint_metrics_data = []

            # Perform inference and calculate metrics for each image in the test set
            for img_loc in sorted(os.listdir(img_dir)):
                img_path = os.path.join(img_dir, img_loc)
                img_id = os.path.basename(img_path).replace('.jpg', '')

                # Run inference on the image
                result = inference_model(model, img_path)

                # Extract the predicted mask from the result
                pred = result.pred_sem_seg.data[0].cpu().numpy()

                # Save the predicted mask as .npy and .tif files
                npy_pred_path = os.path.join(outdir_inference, f'{img_id}.npy')
                np.save(npy_pred_path, pred)

                tif_pred_path = os.path.join(outdir_tif, f'{img_id}.tif')
                save_as_tif(pred, tif_pred_path)

                # Load the ground truth mask
                gt_path = os.path.join(mask_dir, f'{img_id}.tif')
                gt = load_image_as_array(gt_path)
                if gt is None:
                    continue

                # Calculate metrics: Dice, IoU, and Accuracy
                dice = dice_coefficient(pred, gt)
                iou = iou_score(pred, gt)
                acc = accuracy(pred, gt)

                # Store metrics for this image
                checkpoint_metrics_data.append([img_id, dice, iou, acc])

                print(f"Processed {img_id} - Dice: {dice:.4f}, IoU: {iou:.4f}, Accuracy: {acc:.4f}")

            # Save per-image metrics to a CSV file in the checkpoint_metrics folder
            checkpoint_metrics_csv = os.path.join(checkpoint_metrics_dir, f'{iter_number}_metrics.csv')
            checkpoint_df = pd.DataFrame(checkpoint_metrics_data, columns=['Image_ID', 'Dice', 'IoU', 'Accuracy'])
            checkpoint_df.to_csv(checkpoint_metrics_csv, index=False)
            print(f"Per-image metrics saved to {checkpoint_metrics_csv}")

            # Aggregate metrics (mean, std, min, max) for the current checkpoint
            stats = aggregate_metrics(checkpoint_metrics_data)

            # Store the aggregated statistics in the summary data
            summary_data.append([iter_number, stats['Dice_Min'], stats['Dice_Max'], stats['Dice_Mean'], stats['Dice_Std'],
                                 stats['IoU_Min'], stats['IoU_Max'], stats['IoU_Mean'], stats['IoU_Std'],
                                 stats['Acc_Min'], stats['Acc_Max'], stats['Acc_Mean'], stats['Acc_Std']])

            # Update metric stats for plotting
            for metric_name, metric_values in metrics.items():
                metric_values['means'].append(stats[f'{metric_name}_Mean'])
                metric_values['stds'].append(stats[f'{metric_name}_Std'])
                metric_values['mins'].append(stats[f'{metric_name}_Min'])
                metric_values['maxs'].append(stats[f'{metric_name}_Max'])

        all_checkpoints[split] = checkpoints  # Store checkpoints for the current split

        # Save the summary of metrics for all checkpoints into a CSV file
        summary_df = pd.DataFrame(summary_data, columns=summary_csv_columns)
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"Summary metrics saved to {summary_csv_path}")

        # Call the plot function for the current split
        plot_metrics_from_csv(summary_csv_path, split)

if __name__ == "__main__":
    main()
