#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 22:56:43 2025

@author: na19
"""
# All imports required
from __future__ import annotations
from collections.abc import Sequence
import skimage.measure as measure
from scipy.io import savemat, loadmat
import scipy
from pathlib import Path
import os
import numpy as np
import math
import pydicom 
import torch
import copy
import time
import matplotlib.pyplot as plt


from glob import glob
import imageio
from PIL import Image
from skimage.transform import resize
from skimage.util import view_as_blocks


# torch libraries
import torch
import torchvision
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision import transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch import optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.nn.modules.loss import L1Loss
from torch.nn.modules.loss import MSELoss
from torch.utils.tensorboard import SummaryWriter
from monai.networks.nets import BasicUNet


# Optuna libraries
import optuna
from optuna.distributions import BaseDistribution

# Call all other utils functions
#rom utility_non_complex import Patch, NumpyDataset, NumpyDataset_non_mri,AllPatches,hyper_params, standardize_img,convert_to_torchfloat, objective, train_model, DnCNN
from utils import Patch, AllPatches,hyper_params, standardize_img,convert_to_torchfloat

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()


