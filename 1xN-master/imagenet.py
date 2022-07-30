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
from utils.conv_type import *

from utils.common import *
from models.mobilenet_v2 import InvertedResidual

if args.use_dali:
    from data import imagenet_dali
else:
    from data import imagenet
from importlib import import_module

if args.debug:
    import pdb

import models

visible_gpus_str = ','.join(str(i) for i in args.gpus)
os.environ['CUDA_VISIBLE_DEVICES'] = visible_gpus_str
args.gpus = [i for i in range(len(args.gpus))]
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

def get_model(args):

    model = models.__dict__[args.arch]().to(device)
    ckpt = torch.load(args.pretrained_model, map_location=device)
    if args.arch == 'mobilenet_v2':
        fc_weight = ckpt['classifier.1.weight']
        ckpt['classifier.1.weight'] = fc_weight.view(
            fc_weight.size(0), fc_weight.size(1), 1, 1)
        model.load_state_dict(ckpt, strict = False)
    elif args.arch == 'mobilenet_v1':
        fc_weight = ckpt['state_dict']['classifier.0.weight']
        ckpt['state_dict']['classifier.0.weight'] = fc_weight.view(
            fc_weight.size(0), fc_weight.size(1), 1, 1)
        model.load_state_dict(ckpt['state_dict'], strict = False)
    elif args.arch == 'mobilenet_v3_small' or args.arch == 'mobilenet_v3_large':
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d): 
                if ckpt[name+'.weight'].dim() == 2:
                    weight = ckpt[name+'.weight']
                    ckpt[name+'.weight'] = weight.view(weight.size(0), weight.size(1), 1, 1)
        model.load_state_dict(ckpt, strict = False)

    print('==> Testing Baseline Model..')
    validate(val_loader, model, loss_func, args)
    if args.use_dali:
        val_loader.reset()

    cfg_len = {
        'mobilenet_v1': 14,
        'mobilenet_v2': 35,
        'mobilenet_v3_small': 42,
        'mobilenet_v3_large': 48,
    }

    pr_cfg = [args.pr_target] * cfg_len[args.arch]
    
    if args.full == False:
        pr_cfg[-1] = 0
        if args.arch == 'mobilenet_v3_small' or args.arch == 'mobilenet_v3_large':
            pr_cfg[-2] = 0

    return model, pr_cfg    

def main():
    if args.pruned_model == None:
        start_epoch = 0
        best_acc = 0.0
        best_acc_top1 = 0.0

        model, pr_cfg = get_model(args)
        optimizer = get_optimizer(args, model)
            
        i = 0
        flag = True
        print('==> Generating Pruned Model..')
        for name, module in model.named_modules():
            if hasattr(module, "mask"):
                if args.rearrange == False:
                    module.get_mask(pr_cfg[i],args.N)
                else:
                    if args.arch == 'mobilenet_v1':
                        module.get_rearr_mask(pr_cfg[i],args.N)
                    else:
                        if flag == False:
                            module.get_rearr_mask(pr_cfg[i],args.N)
                            flag = True
                        else:
                            module.get_mask(pr_cfg[i],args.N)
                            if 'fc' in name:
                                flag = True
                            else:
                                flag = False
                i += 1
    
        model = model.to(device)
        
        if args.resume == True:
            start_epoch, best_acc = resume(args, model, optimizer)

        if len(args.gpus) != 1:
            model = nn.DataParallel(model, device_ids=args.gpus)

        validate(val_loader, model, loss_func, args)
        if args.use_dali:
            val_loader.reset()

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
        
        if args.export_onnx == True:
            import torch.onnx
            print('==> Exporting Onnx Model..')
            args.conv_type = 'DenseConv'
            converted_model = models.__dict__[args.arch]().to(device)
            for name, module in model.named_modules():
                if hasattr(module, "mask"):
                    sparseWeight = module.sparse_weight()
                    module.weight.data = sparseWeight
            converted_model.load_state_dict(model.state_dict(), strict=False)

            dummy_input = torch.randn(1, 3, 224, 224, device=device)
            ckpt_dir = Path(args.job_dir)  / 'checkpoint'
            torch.onnx.export(model, dummy_input,f'{ckpt_dir}/best_model.onnx')

        logger.info('Best accurary(top5): {:.3f} (top1): {:.3f}'.format(float(best_acc),float(best_acc_top1)))
    
    else:
        import torch
        model = models.__dict__[args.arch]().to(device)
        ckpt = torch.load(args.pruned_model, map_location=device)
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print('==> Evaluating Pruned Model..')
        validate(val_loader, model, loss_func, args)

        if args.export_onnx == True:
            import torch.onnx
            print('==> Exporting Onnx Model..')
            args.conv_type = 'DenseConv'
            converted_model = models.__dict__[args.arch]().to(device)
            for name, module in model.named_modules():
                if hasattr(module, "mask"):
                    sparseWeight = module.sparse_weight()
                    module.weight.data = sparseWeight
            
            converted_model.load_state_dict(model.state_dict(), strict=False)
            validate(val_loader, converted_model, loss_func, args)    

            dummy_input = torch.randn(1, 3, 224, 224, device=device)
            ckpt_dir = Path(args.job_dir) / 'checkpoint'
            torch.onnx.export(model, dummy_input,f'{ckpt_dir}/best_model.onnx')

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
    elif args.lr_policy == 'cos':  # cos with warm-up
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

    if step == 0:
        print('current learning rate:{0}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_optimizer(args, model):
    if args.optimizer == "sgd":

        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": args.weight_decay,
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
