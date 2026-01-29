# src/models/resnet_cifar.py
from typing import Optional, Type, List
import torch
from torch import nn
from . import register_model
from .utils import load_checkpoint_, set_bn_opts_

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_eps=1e-5, bn_momentum=0.1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=bn_eps, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNetCIFAR(nn.Module):
    """
    ResNet para imágenes 32x32 (CIFAR/SVHN). Stem 3x3, sin maxpool.
    ResNet-18: [2,2,2,2]
    Expone .head (nn.Linear).
    """
    def __init__(self, block: Type[BasicBlock], layers: List[int],
                 num_classes: int = 10, bn_eps: float = 1e-5, bn_momentum: float = 0.1, width: float = 1.0):
        super().__init__()
        w = lambda c: max(16, int(c * width))
        self.inplanes = w(64)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        # No maxpool, resol 32x32

        self.layer1 = self._make_layer(block, w(64),  layers[0], stride=1, bn_eps=bn_eps, bn_momentum=bn_momentum)
        self.layer2 = self._make_layer(block, w(128), layers[1], stride=2, bn_eps=bn_eps, bn_momentum=bn_momentum)  # 16x16
        self.layer3 = self._make_layer(block, w(256), layers[2], stride=2, bn_eps=bn_eps, bn_momentum=bn_momentum)  # 8x8
        self.layer4 = self._make_layer(block, w(512), layers[3], stride=2, bn_eps=bn_eps, bn_momentum=bn_momentum)  # 4x4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(w(512) * block.expansion, num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def _make_layer(self, block, planes, blocks, stride=1, bn_eps=1e-5, bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, eps=bn_eps, momentum=bn_momentum),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, bn_eps=bn_eps, bn_momentum=bn_momentum))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, bn_eps=bn_eps, bn_momentum=bn_momentum))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.head(x)

@register_model("resnet18_cifar")
def resnet18_cifar(*, num_classes: int = 10, bn_eps: float = 1e-5, bn_momentum: float = 0.1,
                   width: float = 1.0, checkpoint: Optional[str] = ""):
    model = ResNetCIFAR(BasicBlock, [2, 2, 2, 2],
                        num_classes=num_classes, bn_eps=bn_eps, bn_momentum=bn_momentum, width=width)
    # Ajustar eps/momentum por si se pasan por kwargs
    # (las capas internas ya lo usan; esto asegura consistencia si añades más BNs en el futuro)
    set_bn_opts_(model, eps=bn_eps, momentum=bn_momentum)
    load_checkpoint_(model, checkpoint=checkpoint, strict=True)
    return model
