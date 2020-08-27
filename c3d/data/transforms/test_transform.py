# -*- coding: utf-8 -*-

"""
@date: 2020/8/25 下午4:17
@file: test_transform.py
@author: zj
@description: 
"""

import numpy as np
import torch
import torchvision.transforms as transforms


class TestTransform:

    def __init__(self, cfg):
        input_size = cfg.MODEL.INPUT_SIZE
        H, W, C = input_size

        mean = cfg.TRANSFORM.MEAN
        std = cfg.TRANSFORM.STD

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((H, W)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.torch_size = (C, H, W)

    def __call__(self, video):
        assert isinstance(video, torch.Tensor) and len(video.shape) == 4
        T, H, W, C = video.shape

        C, H, W = self.torch_size
        res_video = torch.from_numpy(np.zeros((T, C, H, W)))
        for i in range(T):
            img = video[i].numpy().astype(np.uint8)
            res_video[i] = self.transform(img)
        res_video = res_video.transpose(0, 1)

        return res_video.float()
