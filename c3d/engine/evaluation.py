# -*- coding: utf-8 -*-

"""
@date: 2020/8/26 上午11:09
@file: evaluation.py
@author: zj
@description: 
"""

import torch
from tqdm import tqdm
import numpy as np

from c3d.util.metrics import topk_accuracy
from c3d.util.logger import setup_logger
from c3d.data.build import build_dataloader


def compute_on_dataset(model, data_loader, device):
    results_dict = {}
    cate_acc_dict = {}
    acc_top1 = list()
    acc_top5 = list()

    for batch in tqdm(data_loader):
        images, targets = batch
        cpu_device = torch.device("cpu")

        with torch.no_grad():
            outputs = model(images.to(device)).to(cpu_device)
            # outputs = torch.stack([o.to(cpu_device) for o in outputs])

            topk_list = topk_accuracy(outputs, targets, topk=(1, 5))
            acc_top1.append(topk_list[0].item())
            acc_top5.append(topk_list[1].item())

            outputs = outputs.numpy()
            preds = np.argmax(outputs, 1)
            targets = targets.numpy()
            for target, pred in zip(targets, preds):
                results_dict.update({
                    str(target):
                        results_dict.get(str(target), 0) + 1
                })
                cate_acc_dict.update({
                    str(target):
                        cate_acc_dict.get(str(target), 0) + int(target == pred)
                })

    return results_dict, cate_acc_dict, acc_top1, acc_top5


def inference(model, data_loader, device):
    logger = setup_logger('C3D.evaluation')

    dataset = data_loader.dataset
    logger.info("Evaluating {} dataset({} video clips):".format('hmdb51', len(dataset)))
    results_dict, cate_acc_dict, acc_top1, acc_top5 = compute_on_dataset(model, data_loader, device)

    logger.info('totoal - top_1 acc: {:.3f}, top_5 acc: {:.3f}'.format(np.mean(acc_top1), np.mean(acc_top5)))
    for key in sorted(results_dict.keys(), key=lambda x: int(x)):
        total_num = results_dict[key]
        acc_num = cate_acc_dict[key]

        if total_num != 0:
            logger.info('cate: {} - acc: {:.3f}'.format(key, acc_num / total_num))
        else:
            logger.info('cate: {} - acc: 0.0'.format(key, acc_num / total_num))


@torch.no_grad()
def do_evaluation(model, device):
    model.eval()

    data_loaders_val = build_dataloader('', train=False)
    inference(model, data_loaders_val, device)
