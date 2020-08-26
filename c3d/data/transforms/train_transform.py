# -*- coding: utf-8 -*-

"""
@date: 2020/8/25 下午4:06
@file: train_transform.py
@author: zj
@description: 
"""

import numpy as np
import torch
import torchvision.transforms as transforms


class TrainTransform:

    def __init__(self):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # transforms.RandomErasing()
        ])

    def __call__(self, video):
        assert isinstance(video, torch.Tensor) and len(video.shape) == 4
        T, H, W, C = video.shape

        res_video = torch.from_numpy(np.zeros((T, 3, 112, 112)))
        for i in range(T):
            img = video[i].numpy().astype(np.uint8)
            res_video[i] = self.transform(img)
        res_video = res_video.transpose(0, 1)

        return res_video.type(torch.FloatTensor)
