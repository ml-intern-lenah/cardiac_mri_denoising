#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 23:13:12 2025

@author: na19
"""
from imports import *


# Original DnCNN

class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=17):
        super(DnCNN, self).__init__()
        layers = []

        # First layer
        layers.append(nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(64, 64, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))

        # Last layer
        layers.append(nn.Conv2d(64, channels, 3, padding=1, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out
    

# define class U-Net model
dnet = BasicUNet(spatial_dims=2, in_channels=1, out_channels=1,features=(64, 128, 256, 512, 1024, 128),act=('ReLU', {'inplace': True}),norm=('batch', {'affine': True}),bias=False,dropout=0.5, upsample='nontrainable')
dnet = dnet.to(device)
print(dnet)
