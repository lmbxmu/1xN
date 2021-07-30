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


def rearrange_w(w, N, pr_rate):
    c_out, c_in, k_1, k_2 = w.shape
    rearrange_w = w.view(c_out,-1)
    w_score = torch.sum(torch.abs(rearrange_w), 1)
    _, index = torch.sort(w_score)
    w = w[index]
    return w, index

class BlockL1Conv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = nn.Parameter(torch.ones(self.weight.shape),requires_grad=False)

    def forward(self, x):
        sparseWeight = self.mask * self.weight
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x
    
    def sparse_weight(self):
        sparseWeight = self.mask * self.weight
        return sparseWeight
        
    def left_weight(self):
        sparseWeight = self.mask.cpu() * self.weight.cpu()
        l1_value = torch.sum(torch.abs(sparseWeight))
        return l1_value


    def get_rearr_mask(self, pr_rate, N):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape
        w, rearrange_indice = rearrange_w(w, N, pr_rate)
        w = w.permute(1, 0, 2, 3)
        w = w.contiguous().view(-1,N*k_1*k_2) 
        prune = int(w.size(0)*pr_rate)
        w = torch.sum(torch.abs(w), 1)
        _, indice = torch.topk(w, prune, largest=False)
        m = torch.ones(w.size(0))
        m[indice] = 0
        m = torch.unsqueeze(m, 1)
        m = m.repeat(1, N*k_1*k_2)
        m = m.view(c_in, c_out, k_1, k_2)
        m = m.permute(1, 0, 2, 3)
        re_m = torch.zeros_like(m)
        for i in range(c_out):
            re_m[rearrange_indice[i]] = m[i]
        self.bias.requires_grad = False
        self.mask = nn.Parameter(re_m, requires_grad=False)
    
    def get_mask(self, pr_rate, N):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape
        w = w.permute(1, 0, 2, 3)
        w = w.contiguous().view(-1,N*k_1*k_2) 
        prune = int(w.size(0)*pr_rate)
        w = torch.sum(torch.abs(w), 1)
        _, indice = torch.topk(w, prune, largest=False)
        m = torch.ones(w.size(0))
        m[indice] = 0
        m = torch.unsqueeze(m, 1)
        m = m.repeat(1, N*k_1*k_2)
        m = m.view(c_in, c_out, k_1, k_2)
        m = m.permute(1, 0, 2, 3)
        self.bias.requires_grad = False
        self.mask = nn.Parameter(m, requires_grad=False)


class BlockRandomConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = nn.Parameter(torch.ones(self.weight.shape),requires_grad=False)

    def forward(self, x):
        sparseWeight = self.mask * self.weight
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def get_mask(self, pr_rate, N):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape
        w = w.permute(1, 0, 2, 3)
        w = w.contiguous().view(-1,N*k_1*k_2) 
        preserve = int(w.size(0)*pr_rate)
        indice = torch.randint(w.size(0),(preserve,))
        m = torch.ones(w.size(0))
        m[indice] = 0
        m = torch.unsqueeze(m, 1)
        m = m.repeat(1, N*k_1*k_2)
        m = m.view(c_in, c_out, k_1, k_2)
        m = m.permute(1, 0, 2, 3)
        self.bias.requires_grad = False
        self.mask = nn.Parameter(m,requires_grad=False)
        
class UnstructureConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = nn.Parameter(torch.ones(self.weight.shape))

    def forward(self, x):
        sparseWeight = self.mask * self.weight
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def get_mask(self, prune_rate):
        self.prune_rate = prune_rate
        w = self.weight.detach().cpu()
        w = w.view(-1) #c_out * (c_in * k * k) -> 4 * (c_out * c_in * k * k / 4)
        m = self.mask.detach().cpu()
        m = m.view(-1)
        _, indice = torch.topk(torch.abs(w), int(w.size(0)*prune_rate), largest=False)
        m[indice] = 0 
        self.bias.requires_grad = False
        self.mask = nn.Parameter(m.view(self.weight.shape))


class StructureConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = nn.Parameter(torch.ones(self.weight.shape))

    def forward(self, x):
        sparseWeight = self.mask * self.weight
        x = F.conv2d(
            x, sparseWeight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def get_mask(self, pr_rate):
        w = self.weight.detach().cpu()
        c_out, c_in, k_1, k_2 = w.shape
        w = w.contiguous().view(-1,c_in*k_1*k_2) 
        prune = int(w.size(0)*pr_rate)
        w = torch.sum(torch.abs(w), 1)
        _, indice = torch.topk(w, prune, largest=False)
        m = torch.ones(w.size(0))
        m[indice] = 0
        m = torch.unsqueeze(m, 1)
        m = m.repeat(1, c_in*k_1*k_2)
        m = m.view(c_out, c_in, k_1, k_2)
        self.bias.requires_grad = False
        self.mask = nn.Parameter(m,requires_grad=False)

