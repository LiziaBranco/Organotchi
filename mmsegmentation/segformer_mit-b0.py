# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True) # Segmentation usually uses SyncBN
data_preprocessor = dict( # The config of data preprocessor, usually includes image normalization and augmentation
    type='SegDataPreProcessor', # The type of data preprocessor.
    mean=[123.675, 116.28, 103.53], # Mean values used for normalizing the input images.
    std=[58.395, 57.12, 57.375], # Standard variance used for normalizing the input images.
    bgr_to_rgb=False, # Whether to convert image from BGR to RGB. #The images are already in RGB, so I'm using False
    pad_val=0, # Padding value of image. #0
    seg_pad_val=255 # Padding value of segmentation map. #255
)
model = dict(
    type='EncoderDecoder', # Name of segmentor
    data_preprocessor=data_preprocessor,
    pretrained=None,  # The ImageNet pretrained backbone to be loaded
    backbone=dict(
        type='MixVisionTransformer', # The type of backbone. Please refer to mmseg/models/backbones/resnet.py for details.
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHead', # Type of decode head. Please refer to mmseg/models/decode_heads for available options.
        in_channels=[32, 64, 160, 256], # Input channel of decode head.
        in_index=[0, 1, 2, 3], # The index of feature map to select.
        channels=256, # The intermediate channels of decode head.
        dropout_ratio=0.1, # The dropout ratio before final classification layer.
        num_classes=19, # Number of segmentation class. Usually 19 for cityscapes, 21 for VOC, 150 for ADE20k.
        norm_cfg=norm_cfg, # The configuration of norm layer.
        align_corners=False, # The align_corners argument for resize in decoding.
        loss_decode=dict( # Config of loss function for the decode_head.
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)), # Type of loss used for segmentation.
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
