import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from utils.builder import get_builder
from utils.conv_type import *



def conv_bn(builder, inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_dw(builder, inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        builder.conv2d(inp, oup, 1, 1, 0, bias=True),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

class MobileNet(nn.Module):
    def __init__(self, builder, n_class):
        super(MobileNet, self).__init__()

        in_planes = 32
        cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]

        self.conv1 = conv_bn(builder, 3, in_planes, stride=2)

        self.features = self._make_layers(builder, in_planes, cfg, conv_dw)

        self.classifier = nn.Sequential(builder.conv1x1_fc(cfg[-1], n_class))
        
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = F.avg_pool2d(x, 7)
        #x = x.view(-1, 1024)
        #x = x.mean(3).mean(2)  # global average pooling

        x = self.classifier(x)
        return x.flatten(1)

    def _make_layers(self, builder, in_planes, cfg, layer):
        layers = []
        for x in cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(layer(builder, in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, BlockL1Conv) or isinstance(m, BlockRandomConv):
                m.bias.data.zero_()

def mobilenet_v1():
    return MobileNet(get_builder(), n_class=1000)
