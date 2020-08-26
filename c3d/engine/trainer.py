# -*- coding: utf-8 -*-

"""
@date: 2020/8/21 下午8:00
@file: trainer.py
@author: zj
@description: 
"""

import time
import datetime
import copy
import torch

from c3d.util.metrics import topk_accuracy
from c3d.util.metric_logger import MetricLogger


def do_train(model, criterion, optimizer, lr_scheduler, data_loader,
             checkpointer, logger, max_iter, device=None):
    logger.info("Start training ...")
    meters = MetricLogger()

    model.train()

    iteration = 0
    log_step = 10
    save_step = 10
    eval_step = 2500

    start_training_time = time.time()
    end = time.time()
    # Iterate over data.
    for inputs, labels in data_loader:
        iteration += 1

        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            # print(outputs.shape)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # compute top-k accuray
            topk_list = topk_accuracy(outputs, labels, topk=(1, 5))
            meters.update(loss=loss / len(labels), acc_1=topk_list[0], acc_5=topk_list[1])

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time)

        if iteration % log_step == 0:
            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                meters.delimiter.join([
                    "iter: {iter:06d}",
                    "lr: {lr:.5f}",
                    '{meters}',
                    "eta: {eta}",
                    'mem: {mem}M',
                ]).format(
                    iter=iteration,
                    lr=optimizer.param_groups[0]['lr'],
                    meters=str(meters),
                    eta=eta_string,
                    mem=round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0),
                )
            )

    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))
