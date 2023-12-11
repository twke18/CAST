"""Define CAST model for segmentation.

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
from timm.models.layers.helpers import to_2tuple
from timm.models.layers import PatchEmbed

from cast_models.utils import segment_mean_nd
from cast_models.graph_pool import GraphPooling

__all__ = [
    'cast_small',
    'cast_small_deep',
    'cast_base',
    'cast_base_deep',
]


class Pooling(nn.Module):

    def __init__(self, pool_block):
        super(Pooling, self).__init__()
        self.pool_block = pool_block

    def forward(self, cls_token, x, padding_mask=None):
        """Perform Pooling module.
        
        Args:
            cls_token: A `float` tensor of shape `[batch_size, 1, channels]`
            x: A `float` tensor of shape `[batch_size, num_nodes, channels]`
            padding_mask: A `binary` tensor of shape `[batch_size, num_nodes]`,
                         where `True` indicates the entry is padded; otherwise,
                         should be `False`

        Returns:
            cls_token: A `float` tensor of shape
                `[batch_size, 1, channels]`
            pool_logit: A `float` tensor of shape
                `[batch_size, num_nodes, num_pooled_nodes]`
            centroid: A `float` tensor of shape
                `[batch_size, num_pooled_nodes, channels]`
            pool_padding_mask: A `binary` tensor of shape
                `[batch_size, num_pooled_nodes]`
            sampled_x_inds: A `integer` tensor of shape
                `[batch_size, num_pooled_nodes]`
        """
        cls_token, centroid, pool_logit, sampled_x_inds = self.pool_block(
            cls_token=cls_token, src=x, mask=padding_mask)

        pool_padding_mask = torch.zeros(
            (pool_logit.shape[0], pool_logit.shape[-1]),
            dtype=torch.bool,
            device=pool_logit.device)

        return cls_token, pool_logit, centroid, pool_padding_mask, sampled_x_inds


class _BatchNorm1d(nn.Module):
  """This class is specific for 3D inputs of shape
  [length, batch_size, channels].
  """
  def __init__(self, num_features, eps=1e-5, momentum=0.1,
               affine=True, track_running_stats=True):
    super(_BatchNorm1d, self).__init__()
    self.norm = nn.BatchNorm1d(num_features=num_features,
                               eps=eps,
                               momentum=momentum,
                               affine=affine,
                               track_running_stats=track_running_stats)

  def forward(self, x):
    x_t = x.transpose(1, 2)
    x_t = self.norm(x_t)
    x_t = x_t.transpose(1, 2)
    return x_t


class BlockFusion(nn.Module):

    def __init__(self, dim, block4_identity=True, discrete=True):
        super(BlockFusion, self).__init__()

        self.proj_blocks = self._make_proj_block(dim * 4, dim)

        self._discrete = discrete

    def _make_proj_block(self, in_dim, out_dim):
        return nn.Sequential(_BatchNorm1d(in_dim),
                             nn.Dropout(0.2),
                             nn.Linear(in_dim, out_dim, bias=True))

    def _unpool(self, block, label):
        """Helper function to unpool
        """
        bs, ns, cs = block.shape
        label = label.unsqueeze(2).expand(-1, -1, cs)
        unpool_block = torch.gather(block, 1, label)

        return unpool_block

    def _proj_block_operations(self, x, cls_token, proj_block):
        """Helper function to perform concat and split with proj_block
        """
        cls_x = torch.cat([cls_token, x], dim=1)
        cls_x = proj_block(cls_x).type_as(x)
        cls_token, x = cls_x[:, :1, :], cls_x[:, 1:, :]

        return x, cls_token

    def forward(self, block1, block2, block3, block4,
                cls_token1, cls_token2, cls_token3, cls_token4,
                logit1_2, logit2_3, logit3_4):
        """Unpool block representations as `block1`, given
        pooling logit from low-to-high level.

        Args:
            block1: A `float` tensor of shape `[batch_size, num_node_1, channels]`.
            block2: A `float` tensor of shape `[batch_size, num_node_2, channels]`.
            block3: A `float` tensor of shape `[batch_size, num_node_3, channels]`.
            block4: A `float` tensor of shape `[batch_size, num_node_4, channels]`.
            logit1_2: A `float` tensor of shape `[batch_size, num_node_1, num_node_2]`.
            logit2_3: A `float` tensor of shape `[batch_size, num_node_2, num_node_3]`.
            logit3_4: A `float` tensor of shape `[batch_size, num_node_3, num_node_4]`.
        """
        if self._discrete:
            label1_2 = torch.argmax(logit1_2, dim=-1)
            label2_3 = torch.argmax(logit2_3, dim=-1)
            label3_4 = torch.argmax(logit3_4, dim=-1)
            label1_3 = torch.gather(label2_3, 1, label1_2)
            label1_4 = torch.gather(label3_4, 1, label1_3)
        else:
            raise NotImplementedError("Only support discrete unpooling")

        block2 = self._unpool(block2, label1_2)
        block3 = self._unpool(block3, label1_3)
        block4 = self._unpool(block4, label1_4)

        out_block = torch.cat([block1, block2, block3, block4], dim=-1)
        out_cls_token = torch.cat([cls_token1, cls_token2, cls_token3, cls_token4], dim=-1)
        out_block, out_cls_token = self._proj_block_operations(
            out_block, out_cls_token, self.proj_blocks)

        return out_block, out_cls_token


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
        del self.norm

        # ----------------------------------------------------------------------
        # Encoder specifics

        # Split intermediate attentional blocks
        del self.blocks # overwrite with new blocks
        dpr = [x.item() for x in torch.linspace(0, 0, sum(depths))]
        dpr = dpr[::-1]

        blocks = []
        pools = []
        for ind, depth in enumerate(depths):

            # Build Attention Blocks
            block = []
            for _ in range(depth):
                block.append(Block(dim=kwargs['embed_dim'],
                                   num_heads=kwargs['num_heads'],
                                   mlp_ratio=kwargs['mlp_ratio'],
                                   qkv_bias=kwargs['qkv_bias'],
                                   drop_path=dpr.pop(),
                                   norm_layer=kwargs['norm_layer']))
            blocks.append(nn.Sequential(*block))

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

    def _block_operations(self, x, cls_token, x_pad_mask, nn_block, pool_block=None, norm_block=None):
        """Wrapper to define operations per block.
        """
        # Forward nn block with cls_token and x
        cls_x = torch.cat([cls_token, x], dim=1)
        cls_x = nn_block(cls_x).type_as(x)
        cls_token, x = cls_x[:, :1, :], cls_x[:, 1:, :]

        # Perform pooling only on x.
        if pool_block is not None:
            (cls_token, pool_logit, centroid,
             pool_pad_mask, pool_inds) = pool_block(cls_token, x, x_pad_mask)
        else:
            pool_logit, centroid, pool_pad_mask, pool_inds = None, None, None, None

        # Generate output by cls_token
        if norm_block is not None:
            out = norm_block(cls_x)[:, 1:].mean(dim=1)
        else:
            out = cls_x[:, 1:].mean(dim=1)

        return x, cls_token, pool_logit, centroid, pool_pad_mask, pool_inds, out

    def forward_features(self, x, y):
        x = self.patch_embed(x) # NxHxWxC
        N, H, W, C = x.shape

        # Collect features within each segment
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
            self.blocks4, self.pool4, None)

        intermediates4 = {
            'logit4': pool_logit4, 'centroid4': centroid4, 'block4': block4,
            'padding_mask4': pool_padding_mask3, 'sampled_inds4': pool_inds4,
        }
        intermediates.update(intermediates4)

        # Final block.
        out_block, out_cls_token = self.block_fusion(
            block1, block2, block3, block4,
            cls_token1, cls_token2, cls_token3, cls_token4,
            pool_logit1, pool_logit2, pool_logit3)
        out = out_block.mean(dim=1)
        out = self.pre_logits(out)
        intermediates.update({'final_out': out})

        return intermediates

    def forward(self, x, y):
        intermediates = self.forward_features(x, y)
        intermediates['final_out'] = self.head(intermediates['final_out'])
        x = intermediates['final_out']

        return x


class ConvStem(nn.Module):
    """ 
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """
    def __init__(self, img_size=224, patch_size=8, in_chans=3, embed_dim=768, norm_layer=None, flatten=False):
        super().__init__()

        assert patch_size == 8, 'ConvStem only supports patch size of 8'
        assert embed_dim % 8 == 0, 'Embed dimension must be divisible by 2 for ConvStem'

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(4):
            stride = 2 if l < 3 else 1
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        else:
            x = x.permute(0, 2, 3, 1) # BxCxHxW -> BxHxWXC
        x = self.norm(x)
        return x


def cast_small(pretrained=True, **kwargs):
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
