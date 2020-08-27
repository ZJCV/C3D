# -*- coding: utf-8 -*-

"""
@date: 2020/8/26 上午10:14
@file: build.py
@author: zj
@description: 
"""

from torchvision.datasets import HMDB51
from torchvision.datasets import UCF101


def build_dataset(cfg, transform=None, train=True):
    if train:
        data_dir = cfg.DATASETS.TRAIN.VIDEO_DIR
        annotation_dir = cfg.DATASETS.TRAIN.ANNOTATION_DIR
        frames_per_clip = cfg.DATASETS.TRAIN.FRAMES_PER_CLIP
        step_between_clips = cfg.DATASETS.TRAIN.STEP_BETWEEN_CLIPS

        dataset_name = cfg.DATASETS.TRAIN.NAME
    else:
        data_dir = cfg.DATASETS.TEST.VIDEO_DIR
        annotation_dir = cfg.DATASETS.TEST.ANNOTATION_DIR
        frames_per_clip = cfg.DATASETS.TEST.FRAMES_PER_CLIP
        step_between_clips = cfg.DATASETS.TEST.STEP_BETWEEN_CLIPS

        dataset_name = cfg.DATASETS.TEST.NAME
    if dataset_name == 'HMDB51':
        return HMDB51(data_dir, annotation_dir, frames_per_clip,
                      step_between_clips=step_between_clips, train=train, transform=transform)
    else:
        return UCF101(data_dir, annotation_dir, frames_per_clip,
                      step_between_clips=step_between_clips, train=train, transform=transform)
