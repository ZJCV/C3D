# -*- coding: utf-8 -*-

"""
@date: 2020/8/25 下午4:38
@file: batch_collate.py
@author: zj
@description: 
"""

from torch.utils.data._utils.collate import default_collate


class BatchCollator:

    def __call__(self, batches):
        videos = list()
        labels = list()
        for batch in batches:
            video, audio, label = batch
            videos.append(video)
            labels.append(label)
        return default_collate(videos), default_collate(labels)
