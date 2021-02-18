from torch.nn import init
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import math

from utils.options import args as parser_args
import numpy as np
import pdb

LearnedBatchNorm = nn.BatchNorm2d


class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)

DenseConv = nn.Conv2d

class Block16L1Conv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = nn.Parameter(torch.ones(self.weight.shape),requires_grad=False)

    def forward(self, x):
        sparseWeight = self.mask * self.weight
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def get_mask(self, pr_rate):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape
        w = w.permute(1, 0, 2, 3)
        w = w.contiguous().view(-1,16*k_1*k_2) 
        prune = int(w.size(0)*pr_rate)
        w = torch.sum(torch.abs(w), 1)
        _, indice = torch.topk(w, prune, largest=False)
        m = torch.ones(w.size(0))
        m[indice] = 0
        m = torch.unsqueeze(m, 1)
        m = m.repeat(1, 16*k_1*k_2)
        m = m.view(c_in, c_out, k_1, k_2)
        m = m.permute(1, 0, 2, 3)
        self.mask = nn.Parameter(m,requires_grad=False)

class Block16UnfoldConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = nn.Parameter(torch.ones(self.weight.shape),requires_grad=False)

    def forward(self, x):
        sparseWeight = self.mask * self.weight
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def get_mask(self, pr_indices):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape
        w = w.permute(1, 0, 2, 3)
        w = w.contiguous().view(-1,16*k_1*k_2) 

        m = torch.ones(w.size(0))
        m[pr_indices] = 0
        m = torch.unsqueeze(m, 1)
        m = m.repeat(1, 16*k_1*k_2)
        m = m.view(c_in, c_out, k_1, k_2)
        m = m.permute(1, 0, 2, 3)
        self.mask = nn.Parameter(m,requires_grad=False)

class Block16RandomConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = nn.Parameter(torch.ones(self.weight.shape),requires_grad=False)

    def forward(self, x):
        sparseWeight = self.mask * self.weight
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def get_mask(self, pr_rate):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape
        w = w.permute(1, 0, 2, 3)
        w = w.contiguous().view(-1,16*k_1*k_2) 
        preserve = int(w.size(0)*pr_rate)
        indice = torch.randint(w.size(0),(preserve,))
        m = torch.ones(w.size(0))
        m[indice] = 0
        m = torch.unsqueeze(m, 1)
        m = m.repeat(1, 16*k_1*k_2)
        m = m.view(c_in, c_out, k_1, k_2)
        m = m.permute(1, 0, 2, 3)
        self.mask = nn.Parameter(m,requires_grad=False)


