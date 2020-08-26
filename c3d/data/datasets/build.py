# -*- coding: utf-8 -*-

"""
@date: 2020/8/26 上午10:14
@file: build.py
@author: zj
@description: 
"""

from torchvision.datasets import HMDB51


def build_dataset(data_dir, transform=None, train=True):
    if train:
        return HMDB51('data/hmdb51_org', 'data/testTrainMulti_7030_splits', 16,
                      step_between_clips=16, train=train, transform=transform)
    else:
        return HMDB51('data/hmdb51_org', 'data/testTrainMulti_7030_splits', 16,
                      step_between_clips=16, train=train, transform=transform)
