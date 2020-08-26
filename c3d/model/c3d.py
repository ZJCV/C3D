# -*- coding: utf-8 -*-

"""
@date: 2020/8/24 下午8:07
@file: c3d.py
@author: zj
@description: 
"""

import torch
import torch.nn as nn


class C3D_BN(nn.Module):
    """
    References
    ----------
    [1] Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks."
    Proceedings of the IEEE international conference on computer vision. 2015.
    [2] Ioffe, Surgey, et al. "Batch Normalization: Accelerating deep network training
    by reducing internal covariate shift."
    arXiv:1502.03167v2 [cs.LG] 13 Feb 2015

    参考C3D with Batch Normalization for Video Classification
    https://naadispeaks.wordpress.com/2018/12/25/c3d-with-batch-normalization-for-video-classification/

    input: NxCxLxHxW(片段数x通道数x帧数x高x宽)
    kernel: DxKxK(时间长度x空间大小)

    conv1  in:3*16*112*112   kernel: 64*3*3*3  stride: 1x1x1 padding: 1x1x1  out:16*64*112*112
    pool1  in:64*16*112*112  kernel: 64*3*3*3  stride: 1x2x2                 out:64*16*56*56

    conv2  in:64*16*56*56    kernel: 128*3*3*3 stride: 1x1x1 padding: 1x1x1  out:128*16*56*56
    pool2  in:128*16*56*56   kernel: 128*3*3*3 stride: 2x2x2                 out:128*8*28*28

    conv3a in:128*8*28*28    kernel: 256*3*3*3 stride: 1x1x1 padding: 1x1x1  out:256*8*28*28
    conv3b in:256*8*28*28    kernel: 256*3*3*3 stride: 1x1x1 padding: 1x1x1  out:256*8*28*28
    pool3  in:256*8*28*28    kernel: 256*3*3*3 stride: 2x2x2　                out:256*4*14*14

    conv4a in:256*4*14*14    kernel: 512*3*3*3 stride: 1x1x1 padding: 1x1x1  out:512*4*14*14
    conv4b in:512*4*14*14    kernel: 512*3*3*3 stride: 1x1x1 padding: 1x1x1  out:512*4*14*14
    pool4  in:512*4*14*14    kernel: 512*3*3*3 stride: 2x2x2                 out:512*2*7*7

    conv5a in:512*2*7*7      kernel: 512*3*3*3 stride: 1x1x1 padding: 1x1x1  out:512*2*7*7
    conv5b in:512*2*7*7      kernel: 512*3*3*3 stride: 1x1x1 padding: 1x1x1  out:512*2*7*7
    pool5  in:512*2*7*7      kernel: 512*3*3*3 stride: 2x2x2                 out:512*1*4*4

    512*1*4*4 -> 1*8192

    fc1    in: 1*8192        kernel: 8291*4096                               out: 1*4096
    fc2    in: 1*4096        kernel: 4096*4096                               out: 1*4096

    softmax分类
    """

    def __init__(self, in_features=3, num_classes=8):
        super(C3D_BN, self).__init__()

        self.conv1 = nn.Conv3d(in_features, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv1_bn = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv2_bn = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3a_bn = nn.BatchNorm3d(256)
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3b_bn = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv4a_bn = nn.BatchNorm3d(512)
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv4b_bn = nn.BatchNorm3d(512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv5a_bn = nn.BatchNorm3d(512)
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv5b_bn = nn.BatchNorm3d(512)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.conv1_bn(self.conv1(x)))
        h = self.pool1(h)

        h = self.relu(self.conv2_bn(self.conv2(h)))
        h = self.pool2(h)

        h = self.relu(self.conv3a_bn(self.conv3a(h)))
        h = self.relu(self.conv3b_bn(self.conv3b(h)))
        h = self.pool3(h)

        h = self.relu(self.conv4a_bn(self.conv4a(h)))
        h = self.relu(self.conv4b_bn(self.conv4b(h)))
        h = self.pool4(h)

        h = self.relu(self.conv5a_bn(self.conv5a(h)))
        h = self.relu(self.conv5b_bn(self.conv5b(h)))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.relu(self.fc7(h))
        h = self.fc8(h)
        return h


if __name__ == '__main__':
    data = torch.randn((1, 3, 16, 112, 112))
    model = C3D_BN()
    outputs = model(data)
    print(outputs.shape)
