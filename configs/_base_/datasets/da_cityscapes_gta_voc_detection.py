dataset_type = 'VOCCityscapesDataset'
data_root_s = 'data/cityscapes_voc/'
data_root_t = 'data/sim10k_coco/'
classes = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
           'bicycle' )
#classes = ('car',)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(2048, 1024), keep_ratio=True),
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
        img_scale=(2048, 1024),
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
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root_s + 'VOC2007/ImageSets/Main/source_trainval.txt',
        img_prefix=data_root_s + 'VOC2007/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root_s + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root_s + 'VOC2007/',
        pipeline=test_pipeline),
    test=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root_s + 'VOC2007/ImageSets/Main/train_test.txt',
        img_prefix=data_root_s + 'VOC2007/',
        pipeline=test_pipeline))
data_t = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root_t + 'VOC2012/ImageSets/Main/train.txt',
        img_prefix=data_root_t + 'VOC2012/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root_t + 'VOC2012/ImageSets/Main/val.txt',
        img_prefix=data_root_t + 'VOC2012/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root_t + 'VOC2012/ImageSets/Main/val.txt',
        img_prefix=data_root_t + 'VOC2012/',
        pipeline=test_pipeline))


evaluation = dict(interval=1, metric='mAP')
