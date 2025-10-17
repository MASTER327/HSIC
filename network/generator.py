import torch
import torch.nn as nn
import torch.nn.functional as F
from .discriminator import *
import cv2
from mtutils import min_max_normalize
import random

from timm.layers import DropPath, to_2tuple, trunc_normal_
import torch.fft
import random
from math import sqrt
import numpy as np

class SpaRandomization(nn.Module):
    def __init__(self, num_features, eps=1e-5, device=0):
        super().__init__()
        self.eps = eps
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True).to(
            device)  

    def forward(self, x, ):
        N, C, H, W = x.size()
        # x = self.norm(x)
        if self.training:
            x = x.view(N, C, -1)
            mean = x.mean(-1, keepdim=True)
            var = x.var(-1, keepdim=True)

            x = (x - mean) / (var + self.eps).sqrt()

            idx_swap = torch.randperm(N)
            alpha = torch.rand(N, 1, 1)  
            mean = self.alpha * mean + (1 - self.alpha) * mean[idx_swap]  
            var = self.alpha * var + (1 - self.alpha) * var[idx_swap]

            x = x * (var + self.eps).sqrt() + mean
            x = x.view(N, C, H, W)

        return x, idx_swap

class Generator_double_branch_CBM2D(nn.Module):
    def __init__(self, imdim=3, imsize=[13, 13], dim1=128, dim2=64, device=0):
        super().__init__()

        self.patch_size = imsize[0]

        # for Intrinsic capturing branch
        self.conv1 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.mp1 = nn.MaxPool2d(3)
        # self.mp = nn.AvgPool2d(3)

        self.conv2 = nn.Conv2d(dim1, dim2, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(dim2)
        self.relu = nn.ReLU(inplace=True)

        self.d_conv2 = nn.ConvTranspose2d(dim2, dim1, kernel_size=1, stride=1, padding=0)
        self.d_mp1 = nn.ConvTranspose2d(dim1, dim1, kernel_size=3, stride=3, padding=0)
        self.d_conv1 = nn.ConvTranspose2d(dim1, imdim, kernel_size=5, stride=1, padding=0)

        self.outR = nn.Conv2d(imdim, imdim, kernel_size=1, stride=1, padding=0)

        # for Variation capturing branch
        self.conv3 = nn.Conv2d(imdim, dim1, kernel_size=5, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(dim1)
        self.relu = nn.ReLU(inplace=True)

        self.mp2 = nn.MaxPool2d(3)

        self.conv4 = nn.Conv2d(dim1, dim2, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(dim2)
        self.relu = nn.ReLU(inplace=True)

        self.d_conv4 = nn.ConvTranspose2d(dim2, dim1, kernel_size=1, stride=1, padding=0)
        self.d_mp2 = nn.ConvTranspose2d(dim1, dim1, kernel_size=3, stride=3, padding=0)
        self.d_conv3 = nn.ConvTranspose2d(dim1, imdim, kernel_size=5, stride=1, padding=0)

        self.outS = nn.Conv2d(imdim, imdim, kernel_size=1, stride=1, padding=0)

        #  Asymmetric augmentation decoupling
        self.spaRandom = SpaRandomization(dim2, device=device)

    def forward(self, x):
        in_size = x.size(0)

        r = self.relu(self.bn1(self.conv1(x)))
        r = self.mp1(r)
        r = self.relu(self.bn2(self.conv2(r)))

        r = self.d_conv1(self.d_mp1(self.d_conv2(r)))
        x_r = self.outR(r)

        # --------------------------------------------------------------------------------

        s = self.relu(self.bn3(self.conv3(x)))
        s = self.mp2(s)
        s = self.relu(self.bn4(self.conv4(s)))

        # s = self.SpaRandomization_CrossInC(s)  # Channel Shuffle
        # s = self.perturbation(s)  # Channel Perturbation
        # s = self.speRandom(s)  # Spectral Random

        s, idx_swap = self.spaRandom(s)  

        s = self.d_conv3(self.d_mp2(self.d_conv4(s)))
        x_s = torch.sigmoid(self.outS(s))

        return x_r, x_s, x_r * x_s
