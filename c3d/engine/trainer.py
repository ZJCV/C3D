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

from .inference import do_evaluation


def do_train(cfg, arguments,
             model, criterion, optimizer, lr_scheduler, data_loader,
             checkpointer, logger, device=None):
    logger.info("Start training ...")
    meters = MetricLogger()

    model.train()

    start_iter = arguments['iteration']
    max_iter = cfg.TRAIN.MAX_ITER
    log_step = cfg.TRAIN.LOG_STEP
    save_step = cfg.TRAIN.SAVE_STEP
    eval_step = cfg.TRAIN.EVAL_STEP

    start_training_time = time.time()
    end = time.time()
    # Iterate over data.
    for iteration, (inputs, labels) in enumerate(data_loader, start_iter):
        iteration += 1
        arguments['iteration'] = iteration

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
        if iteration % save_step == 0:
            checkpointer.save("model_{:06d}".format(iteration), **arguments)
        if iteration % eval_step == 0:
            do_evaluation(cfg, model, device, iteration=iteration)
            model.train()

    checkpointer.save("model_final", **arguments)

    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))

    return model
