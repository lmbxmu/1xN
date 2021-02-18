import torch
import torch.nn as nn
import torch.optim as optim
from utils.options import args
import utils.common as utils
from utils.common import *

import os
import copy
import time
import math
import sys
import numpy as np
import heapq
import random
from utils.scheduler import get_policy
from utils.conv_type import Block16UnfoldConv

from models.resnet import Bottleneck
from models.mobilenet_v2 import InvertedResidual

if args.use_dali:
    from data import imagenet_dali
else:
    from data import imagenet
from importlib import import_module

if args.debug:
    import pdb

import models

checkpoint = utils.checkpoint(args)
now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
logger = utils.get_logger(os.path.join(args.job_dir, 'logger'+now+'.log'))
device = torch.device(f"cuda:{args.gpus[0]}") if torch.cuda.is_available() else 'cpu'

if args.label_smoothing is None:
    loss_func = nn.CrossEntropyLoss().cuda()
else:
    loss_func = LabelSmoothing(smoothing=args.label_smoothing)

# load training data
print('==> Preparing data..')
if args.use_dali:
    def get_data_set(type='train'):
        if type == 'train':
            return imagenet_dali.get_imagenet_iter_dali('train', args.data_path, args.train_batch_size,
                                                       num_threads=4, crop=224, device_id=args.gpus[0], num_gpus=1)
        else:
            return imagenet_dali.get_imagenet_iter_dali('val', args.data_path, args.eval_batch_size,
                                                       num_threads=4, crop=224, device_id=args.gpus[0], num_gpus=1)
    train_loader = get_data_set('train')
    val_loader = get_data_set('test')
else:
    data_tmp = imagenet.Data(args)
    train_loader = data_tmp.trainLoader
    val_loader = data_tmp.testLoader

def train(epoch, train_loader, model, criterion, optimizer):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    data_time = utils.AverageMeter('Data', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    model.train()
    end = time.time()

    if args.use_dali:
        num_iter = train_loader._size // args.train_batch_size
    else:
        num_iter = len(train_loader)

    print_freq = num_iter // 10
    i = 0 
    if args.use_dali:
        for batch_idx, batch_data in enumerate(train_loader):
            if args.debug:
                if i > 5:
                    break
                i += 1
            images = batch_data[0]['data'].cuda()
            targets = batch_data[0]['label'].squeeze().long().cuda()
            data_time.update(time.time() - end)

            adjust_learning_rate(optimizer, epoch, batch_idx, num_iter)

            # compute output
            logits = model(images)
            loss = loss_func(logits, targets)

            # measure accuracy and record loss
            prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)   #accumulated loss
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % print_freq == 0 and batch_idx != 0:
                logger.info(
                    'Epoch[{0}]({1}/{2}): '
                    'Loss {loss.avg:.4f} '
                    'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                        epoch, batch_idx, num_iter, loss=losses,
                        top1=top1, top5=top5))
    else:
        for batch_idx, (images, targets) in enumerate(train_loader):
            if args.debug:
                if i > 5:
                    break
                i += 1
            images = images.cuda()
            targets = targets.cuda()
            data_time.update(time.time() - end)

            adjust_learning_rate(optimizer, epoch, batch_idx, num_iter)

            # compute output
            logits = model(images)
            loss = loss_func(logits, targets)

            # measure accuracy and record loss
            prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
            n = images.size(0)
            losses.update(loss.item(), n)  # accumulated loss
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % print_freq == 0 and batch_idx != 0:
                logger.info(
                    'Epoch[{0}]({1}/{2}): '
                    'Loss {loss.avg:.4f} '
                    'Prec@1(1,5) {top1.avg:.2f}, {top5.avg:.2f}'.format(
                        epoch, batch_idx, num_iter, loss=losses,
                        top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

def validate(val_loader, model, criterion, args):

    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    top1 = utils.AverageMeter('Acc@1', ':6.2f')
    top5 = utils.AverageMeter('Acc@5', ':6.2f')

    if args.use_dali:
        num_iter = val_loader._size // args.eval_batch_size
    else:
        num_iter = len(val_loader)

    model.eval()
    with torch.no_grad():
        end = time.time()
        i = 0
        if args.use_dali:
            for batch_idx, batch_data in enumerate(val_loader):
                if args.debug:
                    if i > 5:
                        break
                    i += 1
                images = batch_data[0]['data'].cuda()
                targets = batch_data[0]['label'].squeeze().long().cuda()

                # compute output
                logits = model(images)
                loss = criterion(logits, targets)

                # measure accuracy and record loss
                pred1, pred5 = utils.accuracy(logits, targets, topk=(1, 5))
                n = images.size(0)
                losses.update(loss.item(), n)
                top1.update(pred1[0], n)
                top5.update(pred5[0], n)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
        else:
            for batch_idx, (images, targets) in enumerate(val_loader):
                if args.debug:
                    if i > 5:
                        break
                    i += 1
                images = images.cuda()
                targets = targets.cuda()

                # compute output
                logits = model(images)
                loss = criterion(logits, targets)

                # measure accuracy and record loss
                pred1, pred5 = utils.accuracy(logits, targets, topk=(1, 5))
                n = images.size(0)
                losses.update(loss.item(), n)
                top1.update(pred1[0], n)
                top5.update(pred5[0], n)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        logger.info(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                    .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg

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

    model = models.__dict__[args.arch]().to(device)
    ckpt = torch.load(args.pretrained_model, map_location=device)
    model.load_state_dict(ckpt, strict = False)
    print('==> Testing Baseline Model..')
    #validate(val_loader, model, loss_func, args)

    manual_pr_cfg = {
        'm1': [0.1]+[0.2]*10+[0.3]*8+[0.4]*6+[0.5]*8+[0.7],
        'm2': [0.1]+[0.1]*2+[0.2]*2+[0.1]*2+[0.3]*4+[0.1]*2+[0.3]*12+[0.1]*2+[0.55]*6+[0.8],
        'm3': [0.3]+[0.3]*10+[0.4]*8+[0.6]*6+[0.65]*8+[0]
    }

    if args.layerwise == 'l1':
        pr_cfg = []
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
        pr_cfg = [args.pr_target] * 34
    else:
        pr_cfg = manual_pr_cfg[args.layerwise]

    return model, pr_cfg

def MyUnfold(output, feature):
    w = feature.weight
    c_out, c_in, kw, kh = w.shape
    x_unf = torch.nn.functional.unfold(output, (kw, kh), dilation = feature.dilation, padding = feature.padding, stride = feature.stride)
    x_unf = x_unf.view(x_unf.size(0),c_in,kh*kw,-1).transpose(2, 3) 
    w_unf = w.transpose(0, 1).view(c_in, c_out, -1).transpose(1,2) 
    out_cw = x_unf.matmul(w_unf).transpose(2,3)#batch * c_in * c_out * (h*w) channel wise
    inchannel_feature = torch.sum(out_cw, dim = 0)
    out_unf = torch.sum(out_cw, dim = 1) #c_out * (h*w)
    output = out_unf.view(out_unf.size(0), out_unf.size(1),int((output.size(2)+feature.padding[0]*2-kw)/feature.stride[0]+1), -1)
    return output, inchannel_feature

def main():
    start_epoch = 0
    best_acc = 0.0
    best_acc_top1 = 0.0

    model, pr_cfg = get_model(args)
    
    optimizer = get_optimizer(args, model)

    i = 0
    print('==> Generating Pruned Model..')
    if args.conv_type == 'Block16L1Conv' or args.conv_type == 'Block16RandomConv':
        for name, module in model.named_modules():
            if hasattr(module, "mask"):
                module.get_mask(pr_cfg[i])
                i += 1
    elif args.conv_type == 'Block16UnfoldConv':
        feature_gather = []
        if args.use_dali:
            for batch_idx, batch_data in enumerate(train_loader):
                images = batch_data[0]['data'].cuda()
                targets = batch_data[0]['label'].squeeze().long().cuda()
                output = images
                for feature in model.features:
                    if isinstance(feature, InvertedResidual) or isinstance(feature, nn.Sequential):
                        for sub_feature in feature:
                            if isinstance(sub_feature, Block16UnfoldConv):
                                output, inchannel_feature = MyUnfold(output, sub_feature)#每一个卷积层进行unfold，提取input channel对应的featruemap
                                feature_gather.append(inchannel_feature)
                            else:
                                output = sub_feature(output) 
                    elif isinstance(feature, Block16UnfoldConv):
                        output, inchannel_feature = MyUnfold(output, feature)
                        feature_gather.append(inchannel_feature)
                    else:
                        output = feature(output)
                break
        else:
            for batch_idx, (images, targets) in enumerate(train_loader):
                output = images
                for feature in model.features:
                    if isinstance(feature, InvertedResidual):
                        for sub_feature in feature.conv:
                            x = output
                            if isinstance(sub_feature, Block16UnfoldConv):
                                output, inchannel_feature = MyUnfold(output, sub_feature)
                                feature_gather.append(inchannel_feature)
                            else:
                                output = sub_feature(output) 
                        if feature.use_res_connect:
                            output = x + output
                    elif isinstance(feature, nn.Sequential):
                        for sub_feature in feature:
                            if isinstance(sub_feature, Block16UnfoldConv):
                                output, inchannel_feature = MyUnfold(output, sub_feature)
                                feature_gather.append(inchannel_feature)
                            else:
                                output = sub_feature(output) 
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
        i = 0

        for name, module in model.named_modules():
            if hasattr(module, "mask"):
                module.get_mask(pr_indices[i])
                i += 1
        del feature_gather, model_tbd, optimizer_tbd, pr_indices
    else:
        pass

    get_prune_rate(model, pr_cfg)

    model = model.to(device)
    if args.resume == True:
        start_epoch, best_acc = resume(args, model, optimizer)

    if len(args.gpus) != 1:
        model = nn.DataParallel(model, device_ids=args.gpus)

    for epoch in range(start_epoch, args.num_epochs):
        train_obj, train_acc_top1,  train_acc = train(epoch,  train_loader, model, loss_func, optimizer)
        valid_obj, test_acc_top1, test_acc = validate(val_loader, model, loss_func, args)
        if args.use_dali:
            train_loader.reset()
            val_loader.reset()

        is_best = best_acc < test_acc
        best_acc_top1 = max(best_acc_top1, test_acc_top1)
        best_acc = max(best_acc, test_acc)

        model_state_dict = model.module.state_dict() if len(args.gpus) > 1 else model.state_dict()

        state = {
            'state_dict': model_state_dict,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }
        checkpoint.save_model(state, epoch + 1, is_best)

    logger.info('Best accurary(top5): {:.3f} (top1): {:.3f}'.format(float(best_acc),float(best_acc_top1)))
        
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

def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    #Warmup
    if args.lr_policy == 'step':
        factor = epoch // 30

        if epoch >= 80:
            factor = factor + 1

        lr = args.lr * (0.1 ** factor)
    elif args.lr_policy == 'cos':  # cos without warm-up
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * (epoch - 5) / (args.num_epochs - 5)))
    elif args.lr_policy == 'exp':
        step = 1
        decay = 0.96
        lr = args.lr * (decay ** (epoch // step))
    elif args.lr_policy == 'fixed':
        lr = args.lr
    else:
        raise NotImplementedError
    if epoch < 5:
            lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    if args.twolr:
        for idx,param_group in enumerate(optimizer.param_groups):
            if idx==2:
                param_group['lr'] = 2*lr
            else:
                param_group['lr'] = lr

    if step == 0:
        print('current learning rate:{0}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_optimizer(args, model):
    if args.optimizer == "sgd":
        if args.twolr:
            # split the weight parameter that need weight decay
            print("use 2 * learning rate trick")
            all_parameters = model.parameters()
            weight_parameters = []
            bias_parameters = []
            if args.arch=='resnet':
                for pname, p in model.named_parameters():
                    if 'fc' in pname or 'conv' in pname or 'downsample.0' in pname:
                        if 'weight' in pname:
                            weight_parameters.append(p)
                        if 'bias' in pname:
                            bias_parameters.append(p)
            elif args.arch=='mobilenet_v2':
                for pname, p in model.named_parameters():
                    if 'classifier' in pname or '0.weight' in pname or '3.weight' in pname or '6.weight' in pname:
                        if 'weight' in pname:
                            weight_parameters.append(p)
                        if 'bias' in pname:
                            bias_parameters.append(p)
            elif args.arch=='mobilenet_v1':
                for pname, p in model.named_parameters():
                    if 'classifier' in pname or '0.weight' in pname or '3.weight' in pname:
                        if 'weight' in pname:
                            weight_parameters.append(p)
                        if 'bias' in pname:
                            bias_parameters.append(p)
            else:
                raise NotImplementedError
            wb = weight_parameters+bias_parameters
            wb_parameters_id = list(map(id, wb))
            other_parameters = list(filter(lambda p: id(p) not in wb_parameters_id, all_parameters))

            # define the optimizer
            optimizer = torch.optim.SGD(
                [
                    {'params' : other_parameters},
                    {'params' : weight_parameters, 'lr': args.lr, 'weight_decay' : args.weight_decay},
                    {'params': bias_parameters, 'lr': 2*args.lr}
                ],
                args.lr,
                momentum=args.momentum,
                )
        else:
            parameters = list(model.named_parameters())
            bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
            rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
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
