import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torch


class Data:
    def __init__(self, args):
        pin_memory = False
        if args.gpus is not None:
            pin_memory = True

        #scale_size = 299 if args.student_model.startswith('inception') else 224
        scale_size = 224

        traindir = os.path.join(args.data_path, 'ILSVRC2012_img_train')
        valdir = os.path.join(args.data_path, 'val')
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        trainset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(scale_size),
                transforms.ToTensor(),
                normalize,
            ]))

        self.trainLoader = DataLoader(
            trainset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=pin_memory)

        testset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(scale_size),
                transforms.ToTensor(),
                normalize,
            ]))

        self.testLoader = DataLoader(
            testset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True)
