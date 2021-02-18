import torch
import torch.nn as nn
import pdb
from utils.builder import get_builder

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

cfg_mask = {
    'vgg16': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
}

class_cfg = []

class VGG(nn.Module):
    def __init__(self, builder, vgg_name, layer_cfg=None, num_classes=10):
        super(VGG, self).__init__()
        self.layer_cfg = layer_cfg
        self.cfg_index = 0
        self.features = self._make_layers(builder,cfg[vgg_name])
        self.classifier = nn.Linear(512 if self.layer_cfg is None else self.layer_cfg[-1], num_classes)
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, builder, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                x = x if self.layer_cfg is None else self.layer_cfg[self.cfg_index]
                layers += [builder.conv2d(in_channels,
                                     x,
                                     kernel_size=3,
                                     padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
                self.cfg_index += 1
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VGG_ori(nn.Module):
    def __init__(self, vgg_name, layer_cfg=None, num_classes=10):
        super(VGG_ori, self).__init__()
        self.layer_cfg = layer_cfg
        self.cfg_index = 0
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512 if self.layer_cfg is None else self.layer_cfg[-1], num_classes)
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                x = x if self.layer_cfg is None else self.layer_cfg[self.cfg_index]
                layers += [nn.Conv2d(in_channels,
                                     x,
                                     kernel_size=3,
                                     padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
                self.cfg_index += 1
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def vgg():
    return VGG(get_builder(),vgg_name='vgg16')

def vgg_ori():
    return VGG_ori(vgg_name='vgg16')