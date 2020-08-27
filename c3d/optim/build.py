# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:55
@file: trainer.py
@author: zj
@description: 
"""

import torch.optim as optim


def build_optimizer(cfg, model):
    lr = cfg.OPTIMIZER.LR
    momentum = cfg.OPTIMIZER.MOMENTUM
    weight_decay = cfg.OPTIMIZER.WEIGHT_DECAY

    return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    # return optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-6)


def build_lr_scheduler(cfg, optimizer):
    milestones = cfg.LR_SCHEDULER.MILESTONES

    return optim.lr_scheduler.MultiStepLR(optimizer, milestones)
