import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.builder import get_builder

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, builder, in_planes, planes, channels_1, channels_2, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = builder.conv1x1(in_planes, channels_1)
        self.bn1 = nn.BatchNorm2d(channels_1)
        self.conv2 = builder.conv3x3(channels_1, channels_2, stride=stride)
        self.bn2 = nn.BatchNorm2d(channels_2)
        self.conv3 = builder.conv1x1(channels_2, self.expansion*planes)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                builder.conv1x1(in_planes, self.expansion*planes, stride=stride),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, builder, block, num_blocks, block_cfg, layer_cfg, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.layer_cfg = layer_cfg
        self.current_conv = 0
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layers(builder, block, block_cfg[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layers(builder, block, block_cfg[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layers(builder, block, block_cfg[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layers(builder, block, block_cfg[3], num_blocks[3], stride=2)

        self.avgpool = nn.Sequential(nn.AvgPool2d(7))
        #self.fc = nn.Linear(block_cfg[3] * block.expansion, num_classes)
        self.fc = builder.conv1x1_fc(block_cfg[3] * block.expansion, num_classes)
        
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        '''


    def _make_layers(self, builder, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(builder, self.in_planes, planes,
                self.layer_cfg[self.current_conv], self.layer_cfg[self.current_conv+1], stride))
            self.current_conv +=2
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        #out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out.flatten(1)

def resnet50():
    block_cfg = [64, 128, 256, 512]
    layer_cfg = [64, 64, 64, 64, 64, 64,
                    128, 128, 128, 128, 128, 128, 128, 128, 
                    256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
                    512, 512, 512, 512, 512, 512]
    return ResNet(get_builder(), Bottleneck, [3,4,6,3], block_cfg = block_cfg, layer_cfg = layer_cfg, num_classes =1000)




