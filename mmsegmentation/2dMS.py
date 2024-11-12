# dataset settings
dataset_type = 'MS2D' # Dataset type, this will be used to define the dataset.
data_root = 'robust_monitoring_organoid_growth/data/' # Root path of data.

crop_size = (768, 1024) #(height, width)
resize_scale = (1536, 1024) 
random_crop = (768, 1024)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=False
)

train_pipeline = [  # Training pipeline.
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path.
    dict(type='LoadAnnotations', reduce_zero_label=False),  # Second pipeline to load annotations for current image.
    dict(type='RandomResize',  # Augmentation pipeline that resize the images and their annotations.
        scale=resize_scale,  # The scale of image.
        ratio_range=(0.8, 2.0),  # The augmented scale range as ratio.
        keep_ratio=True),  # Whether to keep the aspect ratio when resizing the image.
    dict(type='RandomCrop',  # Augmentation pipeline that randomly crop a patch from current image.
        crop_size=random_crop,  # The crop size of patch.
        cat_max_ratio=0.75),  # The max area ratio that could be occupied by single category.
    dict(type='RandomFlip',  # Augmentation pipeline that flip the images and their annotations
        prob=0.5),  # The ratio or probability to flip
    dict(type='PhotoMetricDistortion'),  # Augmentation pipeline that distort current image with several photo metric methods.
    #dict(type='Normalize', **img_norm_cfg),
    dict(type='PackSegInputs')  # Pack the inputs data for the semantic segmentation.
]
test_pipeline = [
    dict(type='LoadImageFromFile'),  # First pipeline to load images from file path
    dict(type='Resize',  # Use resize augmentation
        scale=resize_scale,  # Images scales for resizing.
        keep_ratio=True),  # Whether to keep the aspect ratio when resizing the image.
        # add loading annotation after ``Resize`` because ground truth
        # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=False),  # Load annotations for semantic segmentation provided by dataset.
    #dict(type='Normalize', **img_norm_cfg),
    dict(type='PackSegInputs')  # Pack the inputs data for the semantic segmentation.
]

train_dataloader = dict( # Train dataloader config
    batch_size=1, # Number of data samples (e.g., images, inputs) that are processed simultaneously on a single GPU/CPU during one forward and backward pass of the neural network
    num_workers=4, # Worker to pre-fetch data for each single GPU/CPU
    persistent_workers=True, # Shut down the worker processes after an epoch end, which can accelerate training speed.
    sampler=dict(type='InfiniteSampler', shuffle=True), # Randomly shuffle during training. #shuffle = True
    dataset=dict( # Train dataset config
        type=dataset_type, # Type of dataset, refer to mmseg/datasets/ for details.
        data_root=data_root, # The root of dataset.
        data_prefix=dict( # Prefix for training data.
            img_path='images/split0_train', seg_map_path='masks/split0_train'),
        pipeline=train_pipeline)) # Processing pipeline. This is passed by the train_pipeline created before.

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False), # Not shuffle during validation and testing.
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/split0_val', seg_map_path='masks/split0_val'),
        pipeline=test_pipeline))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/split0_test', seg_map_path='masks/split0_test'),
        pipeline=test_pipeline))

# The metric to measure the accuracy. Here, we use IoUMetric.
# iou_metrics=['mDice'] and ignore_index=0, following the author's implementation
val_evaluator = dict(type='IoUMetric', iou_metrics = ['mDice']) #iou_metrics=['mIoU', 'mDice', 'accuracy'])
test_evaluator = dict(type='IoUMetric', iou_metrics = ['mDice']) #iou_metrics=['mIoU', 'mDice', 'accuracy'])

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'])
