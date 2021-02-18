import torch
import torch.nn as nn
import math
from utils.builder import get_builder

def conv_bn(builder, inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(builder, inp, oup):
    return nn.Sequential(
        builder.conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, builder, inp, hidden_dim, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup
        #print(self.use_res_connect)

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                builder.conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                builder.conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                builder.conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, builder, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        layer_index = 0
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 1, 2],
            [6, 24, 1, 1],
            [6, 32, 1, 2],
            [6, 32, 1, 1],
            [6, 32, 1, 1],
            [6, 64, 1, 2],
            [6, 64, 1, 1],
            [6, 64, 1, 1],
            [6, 64, 1, 1],
            [6, 96, 1, 1],
            [6, 96, 1, 1],
            [6, 96, 1, 1],
            [6, 160, 1, 2],
            [6, 160, 1, 1],
            [6, 160, 1, 1],
            [6, 320, 1, 1],
        ]
        
        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(builder, 3, input_channel, 2)]
        # building inverted residual blocks
        lastc = 32
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            hidden_dim = input_channel * t
            self.features.append(block(builder, input_channel,hidden_dim,output_channel, s, expand_ratio=t))
            input_channel = output_channel
            layer_index += 1
            lastc = c

        # building last several layers
        self.features.append(conv_1x1_bn(builder,input_channel, self.last_channel))
        
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(0),
            nn.Linear(self.last_channel, n_class),
            )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
def mobilenet_v2():
    return MobileNetV2(get_builder())


if __name__ == "__main__":
    '''
    model = mobilenet_v2()
    import thop
    from thop import profile

    Input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(Input, ))

    print('--------------Model--------------')
    print('Params: %.2f M '%(params/1000000))
    print('FLOPS: %.2f M '%(flops/1000000))


    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            for param_tensor in module.state_dict():
                print(param_tensor,'\t',module.state_dict()[param_tensor].size())

    ckpt = torch.load('/Users/zhangyuxin/Documents/MAC/pretrain_model/mobilenet_v2.pth.tar',map_location='cpu')
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())
    model.load_state_dict(ckpt)
    '''
