# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:30
@file: trainer.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn

from .c3d import C3D_BN


def build_model(num_classes=1000):
    model = C3D_BN(num_classes=num_classes)
    return model


def build_criterion():
    return nn.CrossEntropyLoss()
