_base_ = [
    './conic.py',
    '../_base_/default_runtime.py',
]

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)

evaluation = dict(
    interval=20,
    custom_intervals=[1],
    custom_milestones=[95],
    by_epoch=True,
    metric='all',
    save_best='mDice',
    rule='greater',
)

checkpoint_config = dict(
    by_epoch=True,
    interval=5,
    max_keep_ckpts=5,
)

optimizer = dict(type='Adam', lr=0.0001, weight_decay=0.0005)
optimizer_config = dict()

# NOTE: poly learning rate decay
# lr_config = dict(
#     policy='poly', warmup='linear', warmup_iters=100, warmup_ratio=1e-6, power=1.0, min_lr=0.0, by_epoch=False)

# NOTE: fixed learning rate decay
# lr_config = dict(policy='fixed', warmup='linear', warmup_iters=100, warmup_ratio=1e-6, by_epoch=False)

# NOTE: step learning rate decay
lr_config = dict(
    policy='step', by_epoch=True, step=[70], gamma=0.1, warmup='linear', warmup_iters=100, warmup_ratio=1e-6)

# model settings
model = dict(
    type='UNet',
    # model training and testing settings
    num_classes=7,
    train_cfg=dict(),
    test_cfg=dict(
        mode='split',
        radius=1,
        crop_size=(256, 256),
        overlap_size=(40, 40),
        rotate_degrees=[0, 90],
        flip_directions=['none', 'horizontal', 'vertical', 'diagonal'],
    ),
)
