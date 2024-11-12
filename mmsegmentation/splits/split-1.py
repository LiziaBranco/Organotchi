_base_ = [    
'../../_base_/models/segformer_mit-b0.py', '../../_base_/datasets/2dMS.py',    
'../../_base_/default_runtime.py', '../../_base_/schedules/schedule_160k.py'    
]    

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'   

crop_size = (768, 1024) #(height, width)
data_preprocessor = dict(size=crop_size)

model = dict(data_preprocessor=data_preprocessor, 
             init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
             decode_head=dict(num_classes=2,
                              out_channels = 2,
                              loss_decode=[    
                              dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1, use_sigmoid=False, avg_non_ignore=True),
                              dict(type='DiceLoss', loss_name='loss_dice', loss_weight=10) # Sobre a utilização destes parametros: https://mmsegmentation.readthedocs.io/en/main/notes/faq.html 
                                      ]))
    
# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=5.)
        }))
            
evaluation = dict(outdir='robust_monitoring_organoid_growth/work_dirs/split-1/eval_imgs')
            
work_dir='robust_monitoring_organoid_growth/work_dirs/split-1'

train_dataloader = dict(
    dataset=dict(
            data_prefix=dict(
                img_path='images/split1_train', seg_map_path='masks/split1_train')))

val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img_path='images/split1_val', seg_map_path='masks/split1_val')))

test_dataloader_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img_path='images/split1_test', seg_map_path='masks/split1_test')))