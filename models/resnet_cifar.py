import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.builder import get_builder

norm_mean, norm_var = 0.0, 1.0

def conv3x3(builder, in_planes, out_planes, stride=1):
    return builder.conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class ResBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, builder, inplanes, planes, filter_num, stride=1):
        super(ResBasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(builder, inplanes, filter_num, stride)
        self.bn1 = nn.BatchNorm2d(filter_num)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(builder, filter_num, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.shortcut = nn.Sequential()
        #pdb.set_trace()
        if stride != 1 or inplanes != planes:
            if stride!=1:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2],
                                    (0, 0, 0, 0, (planes-inplanes)//2, planes-inplanes-(planes-inplanes)//2), "constant", 0))
            else:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, :, :],
                                    (0, 0, 0, 0, (planes-inplanes)//2, planes-inplanes-(planes-inplanes)//2), "constant", 0))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, builder, block, num_layers, layer_cfg=None, block_cfg=None, num_classes=10):
        super(ResNet, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6

        if block_cfg == None:
            block_cfg = [16, 32, 64]
        self.layer_cfg = layer_cfg
        self.cfg_index = 0
        self.inplanes = block_cfg[0]
        self.conv1 = builder.conv2d(3,self.inplanes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, builder, block_cfg[0], blocks=n, stride=1)
        self.layer2 = self._make_layer(block, builder, block_cfg[1], blocks=n, stride=2)
        self.layer3 = self._make_layer(block, builder,block_cfg[2], blocks=n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(block_cfg[2] * block.expansion, num_classes)

        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, builder, block, planes, blocks, stride):
        layers = []

        layers.append(block(builder, self.inplanes, planes, filter_num=
                    self.layer_cfg[self.cfg_index] if self.layer_cfg != None else planes,
                                        stride=stride))
        self.cfg_index += 1

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(builder, self.inplanes, planes, filter_num=
                    self.layer_cfg[self.cfg_index] if self.layer_cfg != None else planes))
            self.cfg_index += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet56(builder, layer_cfg=None, block_cfg = None, **kwargs):
    return ResNet(ResBasicBlock, 56, layer_cfg, block_cfg, **kwargs)

def resnet110(builder, layer_cfg=None,block_cfg = None,  **kwargs):
    return ResNet(ResBasicBlock, 110, layer_cfg, block_cfg, **kwargs)

def resnet(builder, arch, layer_cfg=None, block_cfg=None,**kwargs):
    if arch == 'resnet56':
        return resnet56(layer_cfg, block_cfg, **kwargs)
    elif arch == 'resnet110':
        return resnet110(layer_cfg, block_cfg, **kwargs)

