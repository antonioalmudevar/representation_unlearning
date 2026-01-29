# src/models/wrn_cifar.py

from typing import Optional
from torch import nn
from . import register_model
from .utils import load_checkpoint_, set_bn_opts_


# -------------------------------------------------------------
# Basic utilities
# -------------------------------------------------------------
def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)


# -------------------------------------------------------------
# WideResNet components
# -------------------------------------------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride, bn_eps, bn_momentum, drop_rate=0.0):
        super().__init__()

        self.equalInOut = (in_ch == out_ch)

        self.relu1 = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(in_ch, eps=bn_eps, momentum=bn_momentum)
        self.conv1 = conv3x3(in_ch, out_ch, 1 if self.equalInOut else stride)

        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_ch, eps=bn_eps, momentum=bn_momentum)
        self.conv2 = conv3x3(out_ch, out_ch, 1)

        self.dropout = nn.Dropout(p=drop_rate) if drop_rate > 0 else nn.Identity()

        self.shortcut = (
            nn.Identity()
            if in_ch == out_ch and stride == 1
            else nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
        )

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        out = self.conv1(out if self.equalInOut else x)
        out = self.relu2(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        return out + (x if self.equalInOut else self.shortcut(x))


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_ch, out_ch, stride, bn_eps, bn_momentum, drop_rate):
        super().__init__()
        layers = []
        for i in range(nb_layers):
            layers.append(
                BasicBlock(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    stride if i == 0 else 1,
                    bn_eps,
                    bn_momentum,
                    drop_rate,
                )
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# -------------------------------------------------------------
# Main WideResNet definition
# -------------------------------------------------------------
class WideResNet(nn.Module):
    """
    WideResNet for CIFAR inputs.
    depth must be 6n + 4. widen_factor scales the channel widths.
    """

    def __init__(
        self,
        depth=28,
        widen_factor=10,
        num_classes=10,
        bn_eps=1e-5,
        bn_momentum=0.1,
        drop_rate=0.0,
    ):
        super().__init__()

        assert (depth - 4) % 6 == 0, "WRN depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]

        # Stem
        self.conv1 = conv3x3(3, nStages[0])

        # Groups
        self.block1 = NetworkBlock(n, nStages[0], nStages[1], 1, bn_eps, bn_momentum, drop_rate)
        self.block2 = NetworkBlock(n, nStages[1], nStages[2], 2, bn_eps, bn_momentum, drop_rate)
        self.block3 = NetworkBlock(n, nStages[2], nStages[3], 2, bn_eps, bn_momentum, drop_rate)

        # Head
        self.bn = nn.BatchNorm2d(nStages[3], eps=bn_eps, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(nStages[3], num_classes)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(self.bn(x))
        x = self.avgpool(x).flatten(1)
        return self.head(x)


# -------------------------------------------------------------
# Factory functions
# -------------------------------------------------------------
@register_model("wrn_28_10")
def wrn_28_10(
    *,
    num_classes=10,
    bn_eps=1e-5,
    bn_momentum=0.1,
    drop_rate=0.0,
    checkpoint: Optional[str] = "",
):
    m = WideResNet(
        depth=28,
        widen_factor=10,
        num_classes=num_classes,
        bn_eps=bn_eps,
        bn_momentum=bn_momentum,
        drop_rate=drop_rate,
    )
    set_bn_opts_(m, eps=bn_eps, momentum=bn_momentum)
    load_checkpoint_(m, checkpoint=checkpoint, strict=True)
    return m


@register_model("wrn_16_8")
def wrn_16_8(
    *,
    num_classes=10,
    bn_eps=1e-5,
    bn_momentum=0.1,
    drop_rate=0.0,
    checkpoint: Optional[str] = "",
):
    """
    WideResNet-16-8 for CIFAR-style inputs (3×32×32).
    depth = 16 → n = (16 - 4) // 6 = 2 blocks per stage.
    widen_factor = 8 → typical for SVHN/CIFAR.
    """
    m = WideResNet(
        depth=16,
        widen_factor=8,
        num_classes=num_classes,
        bn_eps=bn_eps,
        bn_momentum=bn_momentum,
        drop_rate=drop_rate,
    )
    set_bn_opts_(m, eps=bn_eps, momentum=bn_momentum)
    load_checkpoint_(m, checkpoint=checkpoint, strict=True)
    return m
