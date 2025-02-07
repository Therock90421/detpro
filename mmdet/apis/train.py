import random

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DAHOOKS, DADistSamplerSeedHook, DAEpochBasedRunner,
                         DAOptimizerHook, build_optimizer, EpochBasedRunner, DistSamplerSeedHook, DAHOOKS, OptimizerHook)
from mmcv.utils import build_from_cfg

from mmdet.core import DistEvalHook, EvalHook, Fp16OptimizerHook, DADistEvalHook, DAEvalHook
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.utils import get_root_logger, convert_splitbn_model
import pdb


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        # find_unused_parameters = cfg.get('find_unused_parameters', True)
        find_unused_parameters = True
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    runner = EpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))
        """
        val_dataset_t = build_dataset(cfg.data_t.val, dict(test_mode=True))
        val_dataloader_t = build_dataloader(
            val_dataset_t,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader_t, **eval_cfg))
        """

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def da_train_detector(model,
                      dataset_s,
                      dataset_t,
                      cfg,
                      distributed=False,
                      validate=False,
                      timestamp=None,
                      meta=None):
    logger = get_root_logger(cfg.log_level)

    dataset_s = dataset_s if isinstance(dataset_s, (list, tuple)) else [dataset_s]
    dataset_t = dataset_t if isinstance(dataset_t, (list, tuple)) else [dataset_t]
    if 'imgs_per_gpu' in cfg.data_s:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data_s:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data_s.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data_s.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data_s.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data_s.imgs_per_gpu} in this experiments')
        cfg.data_s.samples_per_gpu = cfg.data_s.imgs_per_gpu

    data_loaders_s = [
          build_dataloader(
              ds,
              cfg.data_s.samples_per_gpu,
              cfg.data_s.workers_per_gpu,
              # cfg.gpus will be ignored if distributed
              len(cfg.gpu_ids),
              dist=distributed,
              seed=cfg.seed) for ds in dataset_s
    ]
    data_loaders_t = [
          build_dataloader(
              ds,
              cfg.data_t.samples_per_gpu,
              cfg.data_t.workers_per_gpu,
              # cfg.gpus will be ignored if distributed
              len(cfg.gpu_ids),
              dist=distributed,
              seed=cfg.seed) for ds in dataset_t
    ]
    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # build runner
    optimizer = {}
    for name, module in model.module.named_children():
        if 'backbone' in name:
            optimizer.update({name: build_optimizer(module, cfg.optimizer_backbone)})
        elif 'dis' in name:
            optimizer.update({name: build_optimizer(module, cfg.optimizer_discriminator)})
        else:
            optimizer.update({name: build_optimizer(module, cfg.optimizer)})
    #optimizer = build_optimizer(model, cfg.optimizer)
    runner = DAEpochBasedRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    #if 'Aux' in cfg.model.backbone.type:
    #    runner.model.module.backbone = convert_splitbn_model(runner.model.module.backbone)
    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = DAFp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = DAOptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        runner.register_hook(DADistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_dataset = build_dataset(cfg.data_t.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data_t.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        print('***********************************************************')
        print(eval_cfg)

        eval_hook = DADistEvalHook if distributed else DAEvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))  
        """ 
        val_dataset_s = build_dataset(cfg.data_s.val, dict(test_mode=True))
        val_dataloader_s = build_dataloader(
            val_dataset_s,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data_s.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_hook = DADistEvalHook if distributed else DAEvalHook
        runner.register_hook(eval_hook(val_dataloader_s, **eval_cfg))
        """

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, DAHOOKS)
            runner.register_hook(hook, priority=priority)
    
    if cfg.resume_from:
        if 'Aux' in cfg.model.backbone.type:
            runner.model.module.backbone = convert_splitbn_model(runner.model.module.backbone)
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
        if 'Aux' in cfg.model.backbone.type:
            runner.model.module.backbone = convert_splitbn_model(runner.model.module.backbone)
    else:
        if 'Aux' in cfg.model.backbone.type:
            runner.model.module.backbone = convert_splitbn_model(runner.model.module.backbone)

    runner.run(data_loaders_s, data_loaders_t, cfg.workflow, cfg.total_epochs)
