# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:52
@file: trainer.py
@author: zj
@description: 
"""

import os
import torch

from c3d.data.build import build_dataset, build_transform, build_dataloader
from c3d.model.build import build_model, build_criterion
from c3d.optim.build import build_optimizer, build_lr_scheduler
from c3d.engine.trainer import do_train
from c3d.util.checkpoint import CheckPointer
from c3d.util.logger import setup_logger

if __name__ == '__main__':
    output_dir = './outputs'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model_name = 'C3D'
    logger = setup_logger(model_name, save_dir=output_dir)

    epoches = 10
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_transform, test_transform = build_transform()
    data_dir = './data/'
    data_sets, data_sizes = build_dataset(data_dir, train_transform, test_transform)
    data_loaders = build_dataloader(data_sets)

    criterion = build_criterion()
    model = build_model(num_classes=51).to(device)
    optimizer = build_optimizer(model)
    lr_scheduler = build_lr_scheduler(optimizer)

    checkpointer = CheckPointer(model, optimizer=optimizer, scheduler=lr_scheduler, save_dir=output_dir,
                                save_to_disk=True, logger=logger)

    do_train('C3D', model, criterion, optimizer, lr_scheduler, data_loaders, data_sizes, checkpointer, logger,
             epoches=epoches, device=device)
