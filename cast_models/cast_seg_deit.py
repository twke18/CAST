"""Define CAST model for segmentation following DeiT convention.

Modified from:
    https://github.com/facebookresearch/moco-v3/blob/main/vits.py
    https://github.com/facebookresearch/deit/blob/main/models.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial, reduce
from operator import mul

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import PatchEmbed
from timm.models.layers import trunc_normal_

from cast_models.utils import segment_mean_nd
from cast_models.graph_pool import GraphPooling
from cast_models.modules import Pooling, ConvStem, BlockFusion

__all__ = [
    'cast_small',
    'cast_small_deep',
    'cast_base',
    'cast_base_deep',
]


class CAST(VisionTransformer):
    def __init__(self, *args, **kwargs):
        depths = kwargs['depth']
        # These entries do not exist in timm.VisionTransformer
        num_clusters = kwargs.pop('num_clusters', [64, 32, 16, 8])
        kwargs['depth'] = sum(kwargs['depth'])
        super().__init__(**kwargs)

        self.d_model = kwargs['embed_dim']
        self.patch_size = kwargs['patch_size']
        self.distilled = False

        # Do not tackle dist_token.
        assert self.dist_token is None, 'dist_token is not None.'
        assert self.head_dist is None, 'head_dist is not None.'

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

        # --------------------------------------------------------------------------
        # Encoder specifics
        cumsum_depth = [0]
        for d in depths:
            cumsum_depth.append(d + cumsum_depth[-1])

        blocks = []
        pools = []
        for ind, depth in enumerate(depths):

            # Build Attention Blocks.
            blocks.append(self.blocks[cumsum_depth[ind]:cumsum_depth[ind+1]])

            # Build Pooling layers
            pool = Pooling(
                pool_block=GraphPooling(
                    num_clusters=num_clusters[ind],
                    d_model=kwargs['embed_dim'],
                    l2_normalize_for_fps=False))
            pools.append(pool)

        self.blocks1, self.pool1 = blocks[0], pools[0]
        self.blocks2, self.pool2 = blocks[1], pools[1]
        self.blocks3, self.pool3 = blocks[2], pools[2]
        self.blocks4, self.pool4 = blocks[3], pools[3]

        # The output block specifics.
        self.block_fusion = BlockFusion(kwargs['embed_dim'], True, True)
        # --------------------------------------------------------------------------


    def _block_operations(self, x, cls_token, x_pad_mask,
                          nn_block, pool_block, norm_block):
        """Wrapper to define operations per block.
        """
        # Forward nn block with cls_token and x
        cls_x = torch.cat([cls_token, x], dim=1)
        cls_x = nn_block(cls_x).type_as(x)
        cls_token, x = cls_x[:, :1, :], cls_x[:, 1:, :]

        # Perform pooling only on x
        cls_token, pool_logit, centroid, pool_pad_mask, pool_inds = (
            pool_block(cls_token, x, x_pad_mask)
        )

        # Generate output by cls_token
        if norm_block is not None:
            out = norm_block(cls_x)
        else:
            out = cls_x

        return (x, cls_token, pool_logit, centroid,
                pool_pad_mask, pool_inds, out)

    def forward_features(self, x, y):
        x = self.patch_embed(x) # NxHxWxC
        N, H, W, C = x.shape
        yH, yW = y.shape[-2:]

        # Collect features within each segment.
        x = F.interpolate(x.permute(0, 3, 1, 2),
                          size=(yH, yW),
                          mode='bilinear').permute(0, 2, 3, 1)
        x = segment_mean_nd(x, y)


        # Create padding mask.
        ones = torch.ones((N, yH, yW, 1), dtype=x.dtype, device=x.device)
        avg_ones = segment_mean_nd(ones, y).squeeze(-1)
        x_padding_mask = avg_ones <= 0.5

        # Collect positional encodings within each segment.
        pos_embed = self.pos_embed[:, 1:]
        pos_embed = pos_embed.view(1,
                                   self.patch_embed.grid_size[0],
                                   self.patch_embed.grid_size[1],
                                   self.embed_dim)
        pos_embed = F.interpolate(pos_embed.permute(0, 3, 1, 2),
                                  size=(yH, yW), mode='bicubic', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).contiguous()
        pos_embed = pos_embed.expand(N, -1, -1, -1)
        pos_embed = segment_mean_nd(pos_embed, y)

        # Add positional encodings
        x = self.pos_drop(x + pos_embed)

        # Add class token.
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        cls_token = cls_token + self.pos_embed[:, :1]

        # Block1
        (block1, cls_token1, pool_logit1, centroid1,
         pool_padding_mask1, pool_inds1, out1) = self._block_operations(
            x, cls_token, x_padding_mask,
            self.blocks1, self.pool1, None)

        # Block2
        (block2, cls_token2, pool_logit2, centroid2,
         pool_padding_mask2, pool_inds2, out2) = self._block_operations(
            centroid1, cls_token1, pool_padding_mask1,
            self.blocks2, self.pool2, None)

        # Block3
        (block3, cls_token3, pool_logit3, centroid3,
         pool_padding_mask3, pool_inds3, out3) = self._block_operations(
            centroid2, cls_token2, pool_padding_mask2,
            self.blocks3, self.pool3, None)

        # Block4
        (block4, cls_token4, pool_logit4, centroid4,
         pool_padding_mask4, pool_inds4, out4) = self._block_operations(
            centroid3, cls_token3, pool_padding_mask3,
            self.blocks4, self.pool4, self.norm)

        out_block, out_cls_token = self.block_fusion(
            block1, block2, block3, block4,
            cls_token1, cls_token2, cls_token3, cls_token4,
            pool_logit1, pool_logit2, pool_logit3)

        out = torch.cat([out_cls_token, out_block], dim=1)
        out = self.pre_logits(out)

        return out

    def forward(self, x, y):
        y = y.long()
        x = self.forward_features(x, y)

        return x


@register_model
def cast_small(pretrained=False, **kwargs):
    # minus one ViT block
    model = CAST(
        patch_size=8, embed_dim=384, num_clusters=[320, 160, 80, 40],
        depth=[3, 3, 3, 2], num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def cast_small_deep(pretrained=False, **kwargs):
    # minus one ViT block
    model = CAST(
        patch_size=8, embed_dim=384, num_clusters=[320, 160, 80, 40],
        depth=[6, 3, 3, 3], num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def cast_base(pretrained=False, **kwargs):
    # minus one ViT block
    model = CAST(
        patch_size=8, embed_dim=768, num_clusters=[320, 160, 80, 40],
        depth=[3, 3, 3, 2], num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def cast_base_deep(pretrained=False, **kwargs):
    # minus one ViT block
    model = CAST(
        patch_size=8, embed_dim=768, num_clusters=[320, 160, 80, 40],
        depth=[6, 3, 3, 3], num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model
