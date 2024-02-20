"""Define CAST model for classification.

Modified from:
    https://github.com/facebookresearch/moco-v3/blob/main/vits.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial, reduce
from operator import mul

from timm.models.vision_transformer import VisionTransformer, Block, _cfg
from timm.models.layers import PatchEmbed

from cast_models.utils import segment_mean_nd
from cast_models.graph_pool import GraphPooling
from cast_models.modules import Pooling, ConvStem

__all__ = [
    'cast_small',
    'cast_small_deep',
    'cast_base',
    'cast_base_deep',
]


class CAST(VisionTransformer):
    def __init__(self, stop_grad_conv1=False, **kwargs):
        depths = kwargs['depth']
        # These entries do not exist in timm.VisionTransformer
        num_clusters = kwargs.pop('num_clusters', [64, 32, 16, 8])
        kwargs['depth'] = sum(kwargs['depth'])
        super().__init__(**kwargs)

        # Do not tackle dist_token
        assert self.dist_token is None, 'dist_token is not None.'
        assert self.head_dist is None, 'head_dist is not None.'

        # ----------------------------------------------------------------------
        # Encoder specifics
        # del self.blocks # overwrite with new blocks
        # dpr = [x.item() for x in torch.linspace(0, 0, sum(depths))]
        # dpr = dpr[::-1]
        cumsum_depth = [0]
        for d in depths:
            cumsum_depth.append(d + cumsum_depth[-1])

        blocks = []
        pools = []
        for ind, depth in enumerate(depths):

            # Build Attention Blocks
            blocks.append(self.blocks[cumsum_depth[ind]:cumsum_depth[ind+1]])
            # block = []
            # for _ in range(depth):
            #     block.append(Block(dim=kwargs['embed_dim'],
            #                        num_heads=kwargs['num_heads'],
            #                        mlp_ratio=kwargs['mlp_ratio'],
            #                        qkv_bias=kwargs['qkv_bias'],
            #                        drop_path=dpr.pop(),
            #                        norm_layer=kwargs['norm_layer']))
            # blocks.append(nn.Sequential(*block))

            # Build Pooling layers
            pool = Pooling(
                pool_block=GraphPooling(
                    num_clusters=num_clusters[ind],
                    d_model=kwargs['embed_dim'],
                    l2_normalize_for_fps=False))
            # Last graph pooling is not needed
            if ind == len(depths) - 1:
                for param in pool.pool_block.fc1.parameters():
                    param.requires_grad = False
                for param in pool.pool_block.fc2.parameters():
                    param.requires_grad = False
                for param in pool.pool_block.centroid_fc.parameters():
                    param.requires_grad = False
            pools.append(pool)

        self.blocks1, self.pool1 = blocks[0], pools[0]
        self.blocks2, self.pool2 = blocks[1], pools[1]
        self.blocks3, self.pool3 = blocks[2], pools[2]
        self.blocks4, self.pool4 = blocks[3], pools[3]
        # ----------------------------------------------------------------------

        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(pos_emb)
        self.pos_embed.requires_grad = False
        self.pe_token = nn.Parameter(pe_token)
        self.pe_token.requires_grad = False

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
            out = norm_block(cls_x)[:, 0]
        else:
            out = cls_x[:, 0]

        return (x, cls_token, pool_logit, centroid,
                pool_pad_mask, pool_inds, out)

    def forward_features(self, x, y, upscale_features=False):
        x = self.patch_embed(x) # NxHxWxC
        N, H, W, C = x.shape

        # Collect features within each segment
        if upscale_features:
            yH, yW = y.shape[-2:]
            x = F.interpolate(x.permute(0, 3, 1, 2),
                              size=(yH, yW),
                              mode='bilinear').permute(0, 2, 3, 1)
            x = segment_mean_nd(x, y)

            # Create padding mask.
            ones = torch.ones((N, yH, yW, 1), dtype=x.dtype, device=x.device)
            avg_ones = segment_mean_nd(ones, y).squeeze(-1)
            x_padding_mask = avg_ones <= 0.5

            # Collect positional encodings within each segment.
            pos_embed = self.pos_embed
            pos_embed = pos_embed.view(1, H, W, C)
            pos_embed = F.interpolate(
                pos_embed.permute(0, 3, 1, 2),
                size=(yH, yW), mode='bicubic', align_corners=False
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).contiguous()
            pos_embed = pos_embed.expand(N, -1, -1, -1)
            pos_embed = segment_mean_nd(pos_embed, y)
        else:
            y = y.unsqueeze(1).float()
            y = F.interpolate(y, x.shape[1:3], mode='nearest')
            y = y.squeeze(1).long()
            x = segment_mean_nd(x, y)

            # Create padding mask
            ones = torch.ones((N, H, W, 1), dtype=x.dtype, device=x.device)
            avg_ones = segment_mean_nd(ones, y).squeeze(-1)
            x_padding_mask = avg_ones <= 0.5

            # Collect positional encodings within each segment
            pos_embed = self.pos_embed.view(1, H, W, C).expand(N, -1, -1, -1)
            pos_embed = segment_mean_nd(pos_embed, y)

        # Add positional encodings
        x = self.pos_drop(x + pos_embed)

        # Intermediate results
        intermediates = {}

        # Add class token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        cls_token = cls_token + self.pe_token.expand(x.shape[0], -1, -1)

        # Block1
        (block1, cls_token1, pool_logit1, centroid1,
         pool_padding_mask1, pool_inds1, out1) = self._block_operations(
            x, cls_token, x_padding_mask,
            self.blocks1, self.pool1, None)

        intermediates1 = {
            'logit1': pool_logit1, 'centroid1': centroid1, 'block1': block1,
            'padding_mask1': x_padding_mask, 'sampled_inds1': pool_inds1,
        }
        intermediates.update(intermediates1)

        # Block2
        (block2, cls_token2, pool_logit2, centroid2,
         pool_padding_mask2, pool_inds2, out2) = self._block_operations(
            centroid1, cls_token1, pool_padding_mask1,
            self.blocks2, self.pool2, None)

        intermediates2 = {
            'logit2': pool_logit2, 'centroid2': centroid2, 'block2': block2,
            'padding_mask2': pool_padding_mask1, 'sampled_inds2': pool_inds2,
        }
        intermediates.update(intermediates2)

        # Block3
        (block3, cls_token3, pool_logit3, centroid3,
         pool_padding_mask3, pool_inds3, out3) = self._block_operations(
            centroid2, cls_token2, pool_padding_mask2,
            self.blocks3, self.pool3, None)

        intermediates3 = {
            'logit3': pool_logit3, 'centroid3': centroid3, 'block3': block3,
            'padding_mask3': pool_padding_mask2, 'sampled_inds3': pool_inds3,
        }
        intermediates.update(intermediates3)

        # Block4
        (block4, cls_token4, pool_logit4, centroid4,
         pool_padding_mask4, pool_inds4, out4) = self._block_operations(
            centroid3, cls_token3, pool_padding_mask3,
            self.blocks4, self.pool4, self.norm)
        out4 = self.pre_logits(out4)

        intermediates4 = {
            'logit4': pool_logit4, 'centroid4': centroid4, 'block4': block4,
            'padding_mask4': pool_padding_mask3, 'out4': out4, 'sampled_inds4': pool_inds4,
        }
        intermediates.update(intermediates4)

        return intermediates

    def forward(self, x, y, upscale_features=False):
        intermediates = self.forward_features(
            x, y, upscale_features=upscale_features
        )
        x = self.head(intermediates['out4'])

        return x


def cast_small(pretrained=True, **kwargs):
    # minus one ViT block
    model = CAST(
        patch_size=8, embed_dim=384, num_clusters=[64, 32, 16, 8],
        depth=[3, 3, 3, 2], num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model


def cast_small_deep(pretrained=True, **kwargs):
    # minus one ViT block
    model = CAST(
        patch_size=8, embed_dim=384, num_clusters=[64, 32, 16, 8],
        depth=[6, 3, 3, 3], num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model


def cast_base(pretrained=True, **kwargs):
    # minus one ViT block
    model = CAST(
        patch_size=8, embed_dim=768, num_clusters=[64, 32, 16, 8],
        depth=[3, 3, 3, 2], num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model


def cast_base_deep(pretrained=True, **kwargs):
    # minus one ViT block
    model = CAST(
        patch_size=8, embed_dim=768, num_clusters=[64, 32, 16, 8],
        depth=[6, 3, 3, 3], num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), embed_layer=ConvStem, **kwargs)
    model.default_cfg = _cfg()
    return model
