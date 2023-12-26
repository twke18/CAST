"""Build segmentation model with FCN."""

import torch
import torch.nn as nn
import torch.nn.functional as F

import spml.models.utils as model_utils
import spml.utils.general.common as common_utils
import spml.utils.segsort.common as segsort_common
from spml.models.embeddings.base_model import ResnetBase
from spml.models.embeddings.local_model import LocationColorNetwork

import cast_models.cast_seg


class ViT(ResnetBase):
    
  def __init__(self, hrch, img_size, load_head, config):
    """Build FCN using ResNet as backbone network.

    Args:
      backbone_depth: A list of integers indicate the number
        of residual layers in each block.
      strides: A list of intergers indicate the stride.
      dilations: A list of integers indicate the dilations.
      config: An easydict of the network configurations.
    """

    super(ViT, self).__init__()

    # Build Backbone Network.
    self.vit_backbone = cast_models.cast_seg.__dict__[hrch](img_size=img_size)

    # Build Feature Pyramid Network.
    dim = self.vit_backbone.head.weight.shape[1]
    if load_head:
        self.vit_backbone.head = self._build_mlp(3, dim, 4096, 256)
    else:
        self.vit_backbone.head = nn.Identity()

    # Build Local Feature Network.
    self.lfn = LocationColorNetwork(use_color=False, use_location=True,
                                    norm_color=False, smooth_ksize=None)

    # Parameters for VMF clustering.
    self.label_divisor = config.network.label_divisor
    self.num_classes = config.dataset.num_classes

    self.semantic_ignore_index = config.dataset.semantic_ignore_index

    self.kmeans_num_clusters = config.network.kmeans_num_clusters
    self.kmeans_iterations = config.network.kmeans_iterations

    self.initialize()

  def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    """Follow Moco-v3 to build MLP-based output head.
    """
    mlp = []
    for l in range(num_layers):
      dim1 = input_dim if l == 0 else mlp_dim
      dim2 = output_dim if l == num_layers - 1 else mlp_dim

      mlp.append(nn.Linear(dim1, dim2, bias=False))

      if l < num_layers - 1:
        mlp.append(nn.BatchNorm1d(dim2))
        mlp.append(nn.ReLU(inplace=True))
      elif last_bn:
        # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
        # for simplicity, we further removed gamma in BN
        mlp.append(nn.BatchNorm1d(dim2, affine=False))

    return nn.Sequential(*mlp)

  def generate_embeddings(self, datas, targets=None, resize_as_input=False):
    """Feed-forward segmentation model to generate pixel-wise embeddings
    and location & RGB features.

    Args:
      datas: A dict with an entry `image`, which is a 4-D float tensor
        of shape `[batch_size, channels, height, width]`.
      targets: A dict with an entry `semantic_label` and `instance_label`,
        which are 3-D long tensors of shape `[batch_size, height, width]`.
      resize_as_input: enable/disable resize_as_input to upscale the 
        embeddings to the same size as the input image.

    Return:
      A dict with entry `embedding` and `local_feature` of shape
      `[batch_size, channels, height, width]`.
    """
    slic_labels = targets['slic_label']

    # Generate embeddings.
    spix_embeddings, intermediates = self.vit_backbone(datas['image'], slic_labels, True)

    if resize_as_input:
      img_h, img_w = datas['image'].shape[-2:]
      slic_labels = F.interpolate(slic_labels.unsqueeze(1),
                                  (img_h, img_w),
                                  mode='nearest').squeeze(1)

    n, _, c = spix_embeddings.shape
    embeddings = torch.gather(
        spix_embeddings,
        1,
        slic_labels.view(n, -1, 1).expand(-1, -1, c))
    _, h, w = slic_labels.shape
    embeddings = embeddings.view(n, h, w, c).permute(0, 3, 1, 2).contiguous()

    outputs = {'embedding': embeddings, 'slic_label': slic_labels, 'spix_embedding': spix_embeddings}
    outputs.update(intermediates)

    return outputs

  def generate_clusters(self, logits, slic_labels, embeddings):
    """Recover clustering indices from logits.

    Args:
      logits: A list of 3-D `float` tensor, each is of
        shape `[batch_size, cur_num_nodes, next_num_nodes]`.
      slic_labels: A 3-D long tensor of shape
        `[batch_size, height, width]`.
      embeddings: A 3-D long tensor of shape
        `[batch_size, num_superpixels, channels]`.

    Return:
      cluster_index: A 3-D long tensor of shape
        `[batch_size, height, width]`.
    """
    cluster_index = None
    for logit in logits:
      lab = torch.argmax(logit, dim=-1)
      if cluster_index is None:
        cluster_index = lab
      else:
        cluster_index = torch.gather(lab, 1, cluster_index)

    flat_labels = slic_labels.flatten(1, 2)
    cluster_indices = torch.gather(cluster_index, 1, flat_labels)
    n, h, w = slic_labels.shape
    cluster_indices = cluster_indices.view(n, h, w)

    # Split by batch.
    batch_indices = torch.arange(n, dtype=cluster_indices.dtype,
                                 device=cluster_indices.device)
    lab_div = cluster_indices.max() + 1
    cluster_indices = batch_indices.view(n, 1, 1) * lab_div + cluster_indices
    cluster_indices = cluster_indices.view(-1)
    cluster_labs, cluster_indices = torch.unique(
        cluster_indices, return_inverse=True)
    cluster_batch_indices = torch.gather(cluster_labs, 0, cluster_indices)
    cluster_batch_indices = cluster_batch_indices // lab_div

    cluster_embeddings = F.normalize(embeddings, dim=1)
    cluster_embeddings = (cluster_embeddings.permute(0, 2, 3, 1)
                                            .contiguous()
                                            .flatten(0, 2))

    return cluster_embeddings, cluster_indices, cluster_batch_indices

  def forward(self, datas, targets=None, resize_as_input=None):
    """Generate pixel-wise embeddings and Spherical Kmeans clustering
    within each image.
    """

    targets = targets if targets is not None else {}

    # Generaet embeddings.
    outputs = self.generate_embeddings(datas, targets, resize_as_input)

    # Generate clusterings.
    (cluster_embeddings,
     cluster_indices,
     cluster_batch_indices) = self.generate_clusters(
        [outputs['logit1'], outputs['logit2'], outputs['logit3'], outputs['logit4']],
        outputs['slic_label'],
        outputs['embedding'])

    outputs.update({'cluster_index': cluster_indices,
                    'cluster_embedding': cluster_embeddings,
                    'cluster_batch_index': cluster_batch_indices})

    return outputs

  def initialize(self):
    pass

  def get_params_lr(self):
    """Helper function to adjust learning rate for each sub modules.
    """
    # Specify learning rate for each sub modules.
    ret = []
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          ['vit_backbone'],
          ['weight'])],
      'lr': 1})
    ret.append({
      'params': [n for n in model_utils.get_params(
          self,
          ['vit_backbone'],
          ['bias'])],
      'lr': 2,
      'weight_decay': 0})

    return ret

  def name_mapping(self, name, resume=False):
    if resume:
      return name if not name.startswith('module.') else name[len('module.'):]

    return name.replace('module.base_encoder', 'vit_backbone')


def interpolate_pos_embed(model, checkpoint_model):
  for k in checkpoint_model:
    if 'pos_embed' in k:
      pos_embed_checkpoint = checkpoint_model[k]
      embedding_size = pos_embed_checkpoint.shape[-1]
      num_patches = model.patch_embed.num_patches
      num_extra_tokens = model.pos_embed.shape[-2] - num_patches
      # height (== width) for the checkpoint position embedding
      orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
      # height (== width) for the new position embedding
      new_size = int(num_patches ** 0.5)
      # class_token and dist_token are kept unchanged
      if orig_size != new_size:
        print("Position interpolate from %dx%d to %dx%d, with key name %s" % (orig_size, orig_size, new_size, new_size, k))
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model[k] = new_pos_embed


def cast_small_p512_finetune(config):
  """CAST-S for fine-tuning
  """
  return ViT('cast_small', 512, False, config)


def cast_small_p512_retrieval(config):
  """CAST-S for retrievals
  """
  return ViT('cast_small', 512, True, config)

