# -*- coding: utf-8 -*-

"""
@date: 2020/8/26 上午10:15
@file: build.py
@author: zj
@description: 
"""

from .train_transform import TrainTransform
from .test_transform import TestTransform


def build_transform(cfg, train=True):
    if train:
        return TrainTransform(cfg)
    else:
        return TestTransform(cfg)
