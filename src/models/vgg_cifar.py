# src/models/vgg_cifar.py
from typing import Optional
import torch
from torch import nn
from . import register_model
from .utils import load_checkpoint_

cfgs = {
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
}

def make_layers(cfg, batch_norm=True, bn_eps=1e-5, bn_momentum=0.1):
    layers, in_ch = [], 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_ch, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v, eps=bn_eps, momentum=bn_momentum), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_ch = v
    return nn.Sequential(*layers)

class VGG_CIFAR(nn.Module):
    def __init__(self, num_classes=10, bn_eps=1e-5, bn_momentum=0.1):
        super().__init__()
        self.features = make_layers(cfgs["VGG16"], True, bn_eps, bn_momentum)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(512, num_classes)
        nn.init.normal_(self.head.weight, 0, 0.01); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

@register_model("vgg16_cifar")
def vgg16_cifar(*, num_classes=10, bn_eps=1e-5, bn_momentum=0.1, checkpoint: Optional[str]=""):
    m = VGG_CIFAR(num_classes=num_classes, bn_eps=bn_eps, bn_momentum=bn_momentum)
    load_checkpoint_(m, checkpoint=checkpoint, strict=True)
    return m
