# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:20
@file: trainer.py
@author: zj
@description: 
"""

import torch
from torch.utils.data import DataLoader

from .datasets.build import build_dataset
from .transforms.build import build_transform
from .samplers import IterationBasedBatchSampler
from .batch_collate import BatchCollator


def build_dataloader(data_dir, max_iter=None, train=True):
    transform = build_transform(train=train)
    dataset = build_dataset(data_dir, transform, train=train)

    if train:
        # 训练阶段使用随机采样器
        sampler = torch.utils.data.RandomSampler(dataset)
        batch_size = 24
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
        batch_size = 24

    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
    if train:
        batch_sampler = IterationBasedBatchSampler(batch_sampler, num_iterations=max_iter, start_iter=0)

    data_loader = DataLoader(dataset, num_workers=8, batch_sampler=batch_sampler,
                             collate_fn=BatchCollator(), pin_memory=True)

    return data_loader
