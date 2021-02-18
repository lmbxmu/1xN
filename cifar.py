import torch
import torch.nn as nn
import torch.optim as optim
from utils.options import args
import utils.common as utils
import os
import time
import copy
import sys
import random
import numpy as np
import heapq
from data import cifar10, cifar100, imagenet
from utils.common import *
from importlib import import_module
from thop import profile

from utils.scheduler import get_policy
from utils.builder import get_builder
from utils.conv_type import *

import models

checkpoint = utils.checkpoint(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
logger = utils.get_logger(os.path.join(args.job_dir, 'logger'+now+'.log'))
device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'

if args.label_smoothing is None:
    loss_func = nn.CrossEntropyLoss().cuda()
else:
    loss_func = LabelSmoothing(smoothing=args.label_smoothing)

# Data
print('==> Loading Data..')
if args.data_set == 'cifar10':
    loader = cifar10.Data(args)
elif args.data_set == 'cifar100':
    loader = cifar100.Data(args)

def train(model, optimizer, trainLoader, args, epoch):

    model.train()
    losses = utils.AverageMeter(':.4e')
    accurary = utils.AverageMeter(':6.3f')
    print_freq = len(trainLoader.dataset) // args.train_batch_size // 10
    start_time = time.time()
    for batch, (inputs, targets) in enumerate(trainLoader):

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_func(output, targets)
        loss.backward()
        losses.update(loss.item(), inputs.size(0))
        optimizer.step()

        prec1 = utils.accuracy(output, targets)
        accurary.update(prec1[0], inputs.size(0))

        if batch % print_freq == 0 and batch != 0:
            current_time = time.time()
            cost_time = current_time - start_time
            logger.info(
                'Epoch[{}] ({}/{}):\t'
                'Loss {:.4f}\t'
                'Accurary {:.2f}%\t\t'
                'Time {:.2f}s'.format(
                    epoch, batch * args.train_batch_size, len(trainLoader.dataset),
                    float(losses.avg), float(accurary.avg), cost_time
                )
            )
            start_time = current_time

def validate(model, testLoader):
    global best_acc
    model.eval()

    losses = utils.AverageMeter(':.4e')
    accurary = utils.AverageMeter(':6.3f')

    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testLoader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            predicted = utils.accuracy(outputs, targets)
            accurary.update(predicted[0], inputs.size(0))

        current_time = time.time()
        logger.info(
            'Test Loss {:.4f}\tAccurary {:.2f}%\t\tTime {:.2f}s\n'
            .format(float(losses.avg), float(accurary.avg), (current_time - start_time))
        )
    return accurary.avg

def get_prune_rate(model, pr_cfg):
    all_params = 0
    prune_params = 0

    i = 0
    for name, module in model.named_modules():
        if hasattr(module, "mask"):
            w = module.weight.data.detach().cpu()
            params = w.size(0) * w.size(1) * w.size(2) * w.size(3)
            all_params = all_params + params
            prune_params += int(params * pr_cfg[i])
            i += 1

    print('Conv1x1 Params Compress Rate: %.2f M/%.2f M(%.2f%%)' % ((all_params-prune_params)/1000000, all_params/1000000, 100. * prune_params / all_params))

def get_model(args):
    
    cfg_len = {
        'vgg': 17,
        'resnet32': 31,
    }
    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]().to(device)
    ckpt = torch.load(args.pretrained_model, map_location=device)
    model.load_state_dict(ckpt['state_dict'],strict=False)
    pr_cfg = []

    if args.layerwise == 'l1':
        weights = []
        for name, module in model.named_modules():
            if hasattr(module, "mask"):
                w = module.weight.data.detach().cpu()
                c_out, c_in, k_1, k_2 = w.shape
                w = w.permute(1, 0, 2, 3)
                w = w.contiguous().view(-1,16*k_1*k_2) 
                w = torch.sum(torch.abs(w), 1)
                weights.append(w.view(-1))  

        all_weights = torch.cat(weights,0)
        preserve_num = int(all_weights.size(0) * (1-args.pr_target))
        preserve_weight, _ = torch.topk(all_weights, preserve_num)
        threshold = preserve_weight[preserve_num-1]

        #Based on the pruning threshold, the prune cfg of each layer is obtained
        for weight in weights:
            pr_cfg.append(torch.sum(torch.lt(weight,threshold)).item()/weight.size(0))  

    elif args.layerwise == 'uniform':
        pr_cfg = [args.pr_target] * cfg_len[args.arch]

    else:
        pass
        #pr_cfg = manual_pr_cfg[args.layerwise]


    return model, pr_cfg

def main():
    start_epoch = 0
    best_acc = 0.0

    model, pr_cfg = get_model(args)
    validate(model, loader.testLoader)

    i = 0

    if args.conv_type == 'Block16UnfoldConv':
        feature_gather = []
        model_tbd = models.__dict__[args.arch+'_ori']().to(device)
        for batch, (inputs, targets) in enumerate(loader.trainLoader):
            output = inputs
            for feature in model_tbd.features:
                if isinstance(feature, nn.Conv2d):
                    w = feature.weight
                    pdb.set_trace()
                    c_out, c_in, kw, kh = w.shape
                    x_unf = torch.nn.functional.unfold(output, (kw, kh), dilation = feature.dilation, padding = feature.padding, stride = feature.stride).to(device)
                    x_unf = x_unf.view(x_unf.size(0),c_in,kh*kw,-1).transpose(2, 3)
                    w_unf = w.transpose(0, 1).view(c_in, c_out, -1).transpose(1,2) 
                    out_cw = x_unf.matmul(w_unf).transpose(2,3)#batch * c_in * c_out * (h*w) channel wise
                    feature_gather.append(torch.sum(out_cw, dim = 0))
                    out_unf = torch.sum(out_cw, dim = 1) #c_out * (h*w)
                    output = out_unf.view(out_unf.size(0), out_unf.size(1),int((output.size(2)+feature.padding[0]*2-kw)/feature.stride[0]+1), -1)
                else:
                    output = feature(output)
            break
        
        pr_indices = []
        for i, feature_cw in enumerate(feature_gather):
            f = feature_cw.view(-1, 16 * feature_cw.size(2)) #16*1
            prune = int(f.size(0)*pr_cfg[i])
            f = torch.sum(torch.abs(f), 1)
            _, indice = torch.topk(f, prune, largest=False)
            pr_indices.append(indice)
            #基于每一个卷积核的cw维度的featur map，c_in * c_out * (h*w)得到每个channel对应的16个
        i = 0
        for name, module in model.named_modules():
            if hasattr(module, "mask"):
                module.get_mask(pr_indices[i])
                i += 1
        del feature_gather, model_tbd, pr_indices, out_cw, output, out_unf
    else:
        for name, module in model.named_modules():
            if hasattr(module, "mask"):
                module.get_mask(pr_cfg[i])
                i += 1

    get_prune_rate(model, pr_cfg)

    model = model.to(device)
    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)
    validate(model, loader.testLoader)

    optimizer = get_optimizer(args, model)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_decay_step, gamma=0.1)

    if args.resume == True:
        start_epoch, best_acc = resume(args, model, optimizer)

    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)

    for epoch in range(start_epoch, args.num_epochs):
        
        train(model, optimizer, loader.trainLoader, args, epoch)
        test_acc = validate(model, loader.testLoader)
        scheduler.step()

        is_best = best_acc < test_acc
        best_acc = max(best_acc, test_acc)

        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'cfg': pr_cfg,
        }

        checkpoint.save_model(state, epoch + 1, is_best)
        
    logger.info('Best accurary: {:.3f}'.format(float(best_acc)))

def resume(args, model, optimizer):
    if os.path.exists(args.job_dir+'/checkpoint/model_last.pt'):
        print(f"=> Loading checkpoint ")

        checkpoint = torch.load(args.job_dir+'/checkpoint/model_last.pt')

        start_epoch = checkpoint["epoch"]

        best_acc = checkpoint["best_acc"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> Loaded checkpoint (epoch) {checkpoint['epoch']})")

        return start_epoch, best_acc
    else:
        print(f"=> No checkpoint found at '{args.job_dir}' '/checkpoint/")

def get_optimizer(args, model):
    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and ("sparseThreshold" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

    return optimizer

if __name__ == '__main__':
    main()