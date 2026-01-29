# src/models/mobilenetv2_cifar.py
from typing import Optional
import torch
from torch import nn
from . import register_model
from .utils import load_checkpoint_, set_bn_opts_

def conv_3x3_bn(inp, oup, stride, bn_eps, bn_momentum):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup, eps=bn_eps, momentum=bn_momentum),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup, bn_eps, bn_momentum):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup, eps=bn_eps, momentum=bn_momentum),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, bn_eps, bn_momentum):
        super().__init__()
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = (stride == 1 and inp == oup)
        layers = []
        if expand_ratio != 1:
            layers.append(conv_1x1_bn(inp, hidden_dim, bn_eps, bn_momentum))
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim, eps=bn_eps, momentum=bn_momentum),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup, eps=bn_eps, momentum=bn_momentum),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2CIFAR(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0, bn_eps=1e-5, bn_momentum=0.1):
        super().__init__()
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],  # stride 1 en CIFAR (no 2)
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * max(1.0, width_mult))

        self.features = [conv_3x3_bn(3, input_channel, 1, bn_eps, bn_momentum)]
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, t, bn_eps, bn_momentum))
                input_channel = output_channel
        self.features.append(conv_1x1_bn(input_channel, last_channel, bn_eps, bn_momentum))
        self.features = nn.Sequential(*self.features)

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Linear(last_channel, num_classes)

        nn.init.normal_(self.head.weight, 0, 0.01); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

@register_model("mobilenetv2_cifar")
def mobilenetv2_cifar(*, num_classes=10, width: float = 1.0, bn_eps=1e-5, bn_momentum=0.1, checkpoint: Optional[str]=""):
    m = MobileNetV2CIFAR(num_classes=num_classes, width_mult=width, bn_eps=bn_eps, bn_momentum=bn_momentum)
    set_bn_opts_(m, eps=bn_eps, momentum=bn_momentum)
    load_checkpoint_(m, checkpoint=checkpoint, strict=True)
    return m
