# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings

import torch

import mmcv
from .da_base_runner import DABaseRunner
from .checkpoint import save_checkpoint
from .utils import get_host_info

import numpy as np
import math


class DAEpochBasedRunner(DABaseRunner):
    """Epoch-based Runner.

    This runner train models epoch by epoch.
    """

    def train(self, data_loader_s, data_loader_t, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader_t = data_loader_t
        self.data_loader_s = data_loader_s
        self._max_iter_per_epoch = min(len(self.data_loader_s), len(self.data_loader_t))
        print("training dataset size:", self._max_iter_per_epoch)
        #self._max_iter_per_epoch = 100
        self._max_iters = self._max_epochs * self._max_iter_per_epoch
        self.call_hook('before_train_epoch')
        self.model.train()

        #quit()
        self.iter_s = iter(self.data_loader_s)
        self.iter_t = iter(self.data_loader_t)
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        '''
            for name, param in self.model.named_parameters():
                print(name, param.requires_grad)
            count_idx += 1
            if (count_idx == 2):
                quit()
            print('#########################')
        '''
        for i in range(self._max_iter_per_epoch):
            self._inner_iter = i
            self.call_hook('before_train_iter')
            if self.batch_processor is None:
                try:
                    input_data_s = self.iter_s.__next__()
                except:
                    self.iter_s = iter(self.data_loader_s)
                    input_data_s = self.iter_s.__next__()
                try:
                    input_data_t = self.iter_t.__next__()
                except:
                    self.iter_t = iter(self.data_loader_t)
                    input_data_t = self.iter_t.__next__()
                outputs = self.model.train_step(input_data_s, torch.tensor([0]).to("cuda:0"), input_data_t, torch.tensor([1]).to("cuda:0"), self.optimizer, **kwargs)
            else:#TODO
                pass
                #outputs = self.batch_processor(
                #    self.model, data_batch, train_mode=True, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('"batch_processor()" or "model.train_step()"'
                                ' must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_train_iter')
            self._iter += 1
        self.call_hook('after_train_epoch')
        self._epoch += 1

    def val(self, data_loader_t, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader_t
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        gt_domain = torch.tensor([1])
        for i, data_batch in enumerate(self.data_loader_t):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            with torch.no_grad():
                if self.batch_processor is None:
                    outputs = self.model.val_step(data_batch, self.optimizer,
                                                  gt_domain, **kwargs)
                else:
                    outputs = self.batch_processor(
                        self.model, data_batch, train_mode=False, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('"batch_processor()" or "model.val_step()"'
                                ' must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loader_s, data_loader_t, workflow, max_epochs, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
            max_epochs (int): Total training epochs.
        """
        assert isinstance(data_loader_s, list)
        assert isinstance(data_loader_t, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loader_s) == len(workflow)

        self._max_epochs = max_epochs
        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * max(len(data_loader_t[i]), len(data_loader_s[i]))
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loader_s[i], data_loader_t[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        """Save the checkpoint.

        Args:
            out_dir (str): The directory that checkpoints are saved.
            filename_tmpl (str, optional): The checkpoint filename template,
                which contains a placeholder for the epoch number.
                Defaults to 'epoch_{}.pth'.
            save_optimizer (bool, optional): Whether to save the optimizer to
                the checkpoint. Defaults to True.
            meta (dict, optional): The meta information to be saved in the
                checkpoint. Defaults to None.
            create_symlink (bool, optional): Whether to create a symlink
                "latest.pth" to point to the latest checkpoint.
                Defaults to True.
        """
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        # in some environments, `os.symlink` is not supported, you may need to
        # set `create_symlink` to False
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filename, dst_file)


class DARunner(DAEpochBasedRunner):
    """Deprecated name of EpochBasedRunner."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            'Runner was deprecated, please use EpochBasedRunner instead')
        super().__init__(*args, **kwargs)
