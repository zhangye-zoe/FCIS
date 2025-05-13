dataset_type = 'FCISDSBDataset'
data_root = '/mnt/eternus/users/Ye/DSB2018/data256/mmseg'
train_processes = [
    dict(
        type='Affine',
        scale=(0.8, 1.2),
        shear=5,
        rotate_degree=[-180, 180],
        translate_frac=(0, 0.01)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomCrop', crop_size=(256, 256)),
    dict(type='Pad', pad_size=(256, 256)),
    dict(type='RandomBlur'),
    dict(
        type='ColorJitter',
        hue_delta=8,
        saturation_range=(0.8, 1.2),
        brightness_delta=26,
        contrast_range=(0.75, 1.25)),
    dict(
        type='Normalize',
        mean=[0.68861804, 0.46102882, 0.61138992],
        std=[0.19204499, 0.20979484, 0.1658672],
        if_zscore=False),
    dict(type='UNetLabelMake'),
    dict(
        type='Formatting',
        data_keys=['img'],
        label_keys=[
            'sem_gt', 'sem_gt_inner', 'loss_weight_map', 'inst_gt', 'adj_gt'
        ])
]
test_processes = [
    dict(
        type='Normalize',
        mean=[0.68861804, 0.46102882, 0.61138992],
        std=[0.19204499, 0.20979484, 0.1658672],
        if_zscore=False),
    dict(type='Formatting', data_keys=['img'], label_keys=[])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type='FCISDSBDataset',
        data_root='/mnt/eternus/users/Ye/DSB2018/data256/mmseg',
        img_dir='images/train',
        ann_dir='fcis_inst/train',
        adj_dir='adjacency/train',
        dis_dir='dis/train',
        split='train.txt',
        processes=[
            dict(
                type='Affine',
                scale=(0.8, 1.2),
                shear=5,
                rotate_degree=[-180, 180],
                translate_frac=(0, 0.01)),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='RandomFlip', prob=0.5, direction='vertical'),
            dict(type='RandomCrop', crop_size=(256, 256)),
            dict(type='Pad', pad_size=(256, 256)),
            dict(type='RandomBlur'),
            dict(
                type='ColorJitter',
                hue_delta=8,
                saturation_range=(0.8, 1.2),
                brightness_delta=26,
                contrast_range=(0.75, 1.25)),
            dict(
                type='Normalize',
                mean=[0.68861804, 0.46102882, 0.61138992],
                std=[0.19204499, 0.20979484, 0.1658672],
                if_zscore=False),
            dict(type='UNetLabelMake'),
            dict(
                type='Formatting',
                data_keys=['img'],
                label_keys=[
                    'sem_gt', 'sem_gt_inner', 'loss_weight_map', 'inst_gt',
                    'adj_gt'
                ])
        ]),
    val=dict(
        type='FCISDSBDataset',
        data_root='/mnt/eternus/users/Ye/DSB2018/data256/mmseg',
        img_dir='images/test',
        ann_dir='fcis_inst/test',
        adj_dir='adjacency/test',
        dis_dir='dis/test',
        split='test.txt',
        processes=[
            dict(
                type='Normalize',
                mean=[0.68861804, 0.46102882, 0.61138992],
                std=[0.19204499, 0.20979484, 0.1658672],
                if_zscore=False),
            dict(type='Formatting', data_keys=['img'], label_keys=[])
        ]),
    test=dict(
        type='FCISDSBDataset',
        data_root='/mnt/eternus/users/Ye/DSB2018/data256/mmseg',
        img_dir='images/test',
        ann_dir='fcis_inst/test',
        adj_dir='adjacency/test',
        dis_dir='dis/test',
        split='test.txt',
        processes=[
            dict(
                type='Normalize',
                mean=[0.68861804, 0.46102882, 0.61138992],
                std=[0.19204499, 0.20979484, 0.1658672],
                if_zscore=False),
            dict(type='Formatting', data_keys=['img'], label_keys=[])
        ]))
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
runner = dict(type='EpochBasedRunner', max_epochs=200)
evaluation = dict(
    interval=1,
    custom_intervals=[1],
    custom_milestones=[95],
    by_epoch=True,
    metric='all',
    save_best='imwAji',
    rule='greater')
checkpoint_config = dict(by_epoch=True, interval=5, max_keep_ckpts=5)
optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.0005)
optimizer_config = dict()
lr_config = dict(
    policy='step',
    by_epoch=True,
    step=[70],
    gamma=0.1,
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1e-06)
model = dict(
    type='FCISNet5',
    num_classes=5,
    train_cfg=dict(),
    test_cfg=dict(
        mode='split',
        radius=1,
        crop_size=(256, 256),
        overlap_size=(40, 40),
        rotate_degrees=[0, 90],
        flip_directions=['none', 'horizontal', 'vertical', 'diagonal']))
work_dir = '/mnt/data/ISAS.DE/ye.zhang/Tissue-Image-Segmentation/z_visual/DSB2018_4chan_test'
gpu_ids = range(0, 1)
