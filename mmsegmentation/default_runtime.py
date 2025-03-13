# Set the default scope of the registry to mmseg.
default_scope = 'mmseg'
# environment
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

#vis_backends = [dict(type='LocalVisBackend')]
#visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None # Load checkpoint from file.
resume = False # Whether to resume from existed model.

#tta_model = dict(type='SegTTAModel')
