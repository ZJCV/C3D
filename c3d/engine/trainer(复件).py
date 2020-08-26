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


def do_train(model_name, model, criterion, optimizer, lr_scheduler, data_loaders, data_sizes, checkpointer, logger,
             epoches=100, device=None):
    logger.info("Start training ...")
    meters = MetricLogger()

    # best_model_weights = copy.deepcopy(model.state_dict())
    # best_acc = 0.0

    # loss_dict = {'train': [], 'test': []}
    # acc_dict = {'train': [], 'test': []}
    max_iter = epoches * len(data_loaders['train'])
    start_training_time = time.time()
    end = time.time()
    for epoch in range(epoches):
        logger.info('{} - Epoch {}/{}'.format(model_name, epoch, epoches - 1))
        logger.info('-' * 10)

        # Each epoch has a training and test phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # running_loss = 0.0
            # running_acc = 0.0

            iteration = 0
            # Iterate over data.
            for inputs, labels in data_loaders[phase]:
                # print(inputs.shape, labels.shape)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print(outputs.shape)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # compute top-k accuray
                    topk_list = topk_accuracy(outputs, labels, topk=(1,))
                    # running_acc += topk_list[0]
                    meters.update(loss=loss / len(labels), acc=topk_list[0])

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step()

                batch_time = time.time() - end
                end = time.time()
                meters.update(time=batch_time)

                iteration += 1
                if iteration % 10 == 0:
                    iteration = 0
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

            # statistics
            # running_loss += loss.item() * inputs.size(0)
            # print(f'loss: {running_loss}, acc: {running_acc}')
            # if phase == 'train':
            #     lr_scheduler.step()

            # epoch_loss = running_loss / data_sizes[phase]
            # epoch_acc = running_acc / len(data_loaders[phase])

            # loss_dict[phase].append(epoch_loss)
            # acc_dict[phase].append(epoch_acc)

            # logger.info('{} Loss: {:.4f} Top-1 Acc: {:.4f}'.format(
            #     phase, epoch_loss, epoch_acc))

            # deep copy the model
            # if phase == 'test' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_weights = copy.deepcopy(model.state_dict())

        # 每训练一轮就保存
        # checkpointer.save("model_{:06d}".format(epoch))

    total_training_time = int(time.time() - start_training_time)
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info("Total training time: {} ({:.4f} s / it)".format(total_time_str, total_training_time / max_iter))

    # time_elapsed = time.time() - since
    # logger.info('Training {} complete in {:.0f}m {:.0f}s'.format(model_name, time_elapsed // 60, time_elapsed % 60))
    # logger.info('Best test Top-1 Acc: {:4f}'.format(best_acc))

    # load best model weights
    # model.load_state_dict(best_model_weights)
    # return model, loss_dict, acc_dict
