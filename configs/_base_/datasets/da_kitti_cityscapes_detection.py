dataset_type = 'CityscapesDataset'
data_root_s = 'data/kitti/'
data_root_t = 'data/cityscapes/'
classes = ('car',)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    #dict(type='Resize', img_scale= (2048, 1024), keep_ratio=True),
    dict(type='Resize', img_scale= (1200, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        #img_scale=(2048, 1024),
        img_scale=(1200, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data_s = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root_s + 'annotations/instance_train.json',
        img_prefix=data_root_s + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root_s + 'annotations/instance_train.json',
        img_prefix=data_root_s + 'train/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root_s + 'annotations/instances_filtered_filtered_grFine_val.json',
        img_prefix=data_root_s + 'leftImg8bit/val/',
        pipeline=test_pipeline))

data_t = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root_t + 'annotations/instancesonly_filtered_gtFine_train.json',
        img_prefix=data_root_t + 'leftImg8bit/train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root_t + 'annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix=data_root_t + 'leftImg8bit/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root_t + 'annotations/instances_filtered_filtered_gtFine_val.json',
        img_prefix=data_root_t + 'leftImg8bit/val/',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')
