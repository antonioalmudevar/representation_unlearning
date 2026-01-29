# src/models/vit_tiny.py
from typing import Optional
import math, torch
from torch import nn
from . import register_model
from .utils import load_checkpoint_

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2
    def forward(self, x):
        x = self.proj(x)          # [B, D, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)), nn.GELU(), nn.Dropout(drop),
            nn.Linear(int(dim*mlp_ratio), dim), nn.Dropout(drop),
        )
    def forward(self, x):
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x + h
        h = x
        x = self.norm2(x)
        x = self.mlp(x) + h
        return x

class ViTTiny32(nn.Module):
    def __init__(self, num_classes=10, img_size=32, patch=4, dim=192, depth=12, heads=3, drop=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch, 3, dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n + 1, dim))
        self.pos_drop = nn.Dropout(p=drop)
        self.blocks = nn.Sequential(*[TransformerEncoder(dim, heads, 4.0, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x[:, 0])
        return self.head(x)

@register_model("vit_tiny_patch4_32")
def vit_tiny_patch4_32(*, num_classes=10, checkpoint: Optional[str]=""):
    m = ViTTiny32(num_classes=num_classes, img_size=32, patch=4, dim=192, depth=12, heads=3, drop=0.0)
    load_checkpoint_(m, checkpoint=checkpoint, strict=True)
    return m
