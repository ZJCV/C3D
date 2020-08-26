# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午7:52
@file: trainer.py
@author: zj
@description: 
"""

import os
import torch

from c3d.data.build import build_dataloader
from c3d.model.build import build_model, build_criterion
from c3d.optim.build import build_optimizer, build_lr_scheduler
from c3d.engine.trainer import do_train
from c3d.util.checkpoint import CheckPointer
from c3d.util.logger import setup_logger

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    output_dir = './outputs'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    logger = setup_logger('C3D.train', save_dir=output_dir)

    max_iter = 10000
    data_loader = build_dataloader('', max_iter, train=True)
    criterion = build_criterion()
    model = build_model(num_classes=51).to(device)
    optimizer = build_optimizer(model)
    lr_scheduler = build_lr_scheduler(optimizer)

    checkpointer = CheckPointer(model, optimizer=optimizer, scheduler=lr_scheduler, save_dir=output_dir,
                                save_to_disk=True, logger=logger)

    arguments = {"iteration": 0}
    do_train(arguments,
             model, criterion, optimizer, lr_scheduler, data_loader,
             checkpointer, logger, max_iter, device=device)
