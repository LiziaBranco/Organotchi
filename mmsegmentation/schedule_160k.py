# optimizer
max_iters = 160000
val_interval = 16000
checkpoints_interval = 16000

optimizer = dict(type='SGD', # Type of optimizers, refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/optimizer/default_constructor.py for more details
                 lr=0.01, # Learning rate of optimizers #0.01
                 momentum=0.9, # Momentum
                 weight_decay=0.0005) # Weight decay of SGD #0.0005

optim_wrapper = dict(type='OptimWrapper', # Optimizer wrapper provides a common interface for updating parameters.
                     optimizer=optimizer, # Optimizer used to update model parameters.
                     clip_grad=None) # If ``clip_grad`` is not None, it will be the arguments of ``torch.nn.utils.clip_grad``.
# learning policy
param_scheduler = [
    dict(
        type='PolyLR', # The policy of scheduler, also support Step, CosineAnnealing, Cyclic, etc. Refer to details of supported LrUpdater from https://github.com/open-mmlab/mmengine/blob/main/mmengine/optim/scheduler/lr_scheduler.py
        eta_min=1e-4, # Minimum learning rate at the end of scheduling.
        power=0.9, # The power of polynomial decay.
        begin=0, # Step at which to start updating the parameters.
        end=max_iters, # Step at which to stop updating the parameters.
        by_epoch=False) # Whether count by epoch or not.
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'), # Log the time spent during iteration.
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False), # Collect and write logs from different components of ``Runner``.
    param_scheduler=dict(type='ParamSchedulerHook'), # update some hyper-parameters in optimizer, e.g., learning rate.
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=checkpoints_interval), # Save checkpoints periodically.
    sampler_seed=dict(type='DistSamplerSeedHook')) # Data-loading sampler for distributed training.
