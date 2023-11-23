from functools import partial, reduce

import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, _cfg

from .vits_suppix import ViTSuperPixel, ConvStem
from .tome.patch import timm as tome_timm

__all__ = [
    'tome_small',
    'tome_base',
]


def tome_small(**kwargs):
    # minus one ViT block
    model = ViTSuperPixel(
        patch_size=8, embed_dim=384, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    tome_timm(model)
    model.r = 16
    return model


def tome_base(**kwargs):
    # minus one ViT block
    model = ViTSuperPixel(
        patch_size=8, embed_dim=768, depth=11, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    tome_timm(model)
    model.r = 16
    return model
