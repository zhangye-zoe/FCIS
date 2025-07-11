# dataset settings
dataset_type = 'FCISPanNukeDataset'
data_root = '/mnt/data/ISAS.DE/ye.zhang/FCIS/data/PanNuke/data256/mmseg'
train_processes = [
    dict(type='Affine', scale=(0.8, 1.2), shear=5, rotate_degree=[-180, 180], translate_frac=(0, 0.01)),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    # dict(type='Resize', scale_factor=2, resize_mode='scale'),
    dict(type='RandomCrop', crop_size=(256, 256)),
    dict(type='Pad', pad_size=(256, 256)),
    dict(type='RandomBlur'),
    dict(
        type='ColorJitter', hue_delta=8, saturation_range=(0.8, 1.2), brightness_delta=26, contrast_range=(0.75, 1.25)),
    dict(
        type='Normalize',
        mean=[0.68861804, 0.46102882, 0.61138992],
        std=[0.19204499, 0.20979484, 0.1658672],
        if_zscore=False),
    dict(type='UNetLabelMake'),# wc={2:10, 3:10, 4:10}),
    dict(type='Formatting', data_keys=['img'], label_keys=['sem_gt', 'sem_gt_inner', 'loss_weight_map', 'inst_gt', 'adj_gt'])
]
test_processes = [
    # dict(type='Resize', scale_factor=2, resize_mode='scale'),
    dict(
        type='Normalize',
        mean=[0.68861804, 0.46102882, 0.61138992],
        std=[0.19204499, 0.20979484, 0.1658672],
        if_zscore=False),
    # dict(type='UNetLabelMake'),#wc={2:10, 3:10, 4:10}),
    dict(type='Formatting', data_keys=['img'], label_keys=[])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=32,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train4pub',
        ann_dir='fcis_inst/train4pub',
        adj_dir='adjacency/train4pub',
        dis_dir='dis/train4pub',
        split='train4pub.txt',
        processes=train_processes),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='fcis_inst/val',
        adj_dir='adjacency/val',
        dis_dir='dis/val',
        split='val.txt',
        processes=test_processes),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/test',
        ann_dir='fcis_inst/test',
        adj_dir='adjacency/test',
        dis_dir='dis/test',
        split='test.txt',
        processes=test_processes),
)
