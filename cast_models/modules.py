"""Define shared modules"""
import torch
import torch.nn as nn

from timm.models.layers.helpers import to_2tuple


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


class ConvStem(nn.Module):
    """ 
    ConvStem, from Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881

    Copied from https://github.com/facebookresearch/moco-v3/blob/main/vits.py
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
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        else:
            x = x.permute(0, 2, 3, 1) # BxCxHxW -> BxHxWXC
        x = self.norm(x)
        return x


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
        """Helper function to unpool.
        """
        bs, ns, cs = block.shape
        label = label.unsqueeze(2).expand(-1, -1, cs)
        unpool_block = torch.gather(block, 1, label)

        return unpool_block

    def _proj_block_operations(self, x, cls_token, proj_block):
        """Helper function to perform concat and split with proj_block.
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
