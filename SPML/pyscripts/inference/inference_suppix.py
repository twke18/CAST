"""Inference script for semantic segmentation by nearest neighbor retrievals.
"""
from __future__ import print_function, division
import os
import math

import PIL.Image as Image
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import spml.data.transforms as transforms
import spml.utils.general.vis as vis_utils
import spml.utils.general.common as common_utils
import spml.utils.segsort.others as segsort_others
from spml.data.datasets.spix_seeds_base_dataset import SPixListDataset
from spml.config.default import config
from spml.config.parse_args import parse_args
import spml.utils.segsort.common as segsort_common
import spml.models.embeddings.cast as cast
from spml.models.predictions.segsort import segsort

cudnn.enabled = True
cudnn.benchmark = True


def separate_comma(str_comma):
  ints = [int(i) for i in str_comma.split(',')]
  return ints


def main():
  """Inference for semantic segmentation.
  """
  # Retreve experiment configurations.
  args = parse_args('Inference for semantic segmentation.')
  config.network.kmeans_num_clusters = separate_comma(args.kmeans_num_clusters)
  config.network.label_divisor = args.label_divisor

  # Create directories to save results.
  semantic_dir = os.path.join(args.save_dir, 'semantic_gray')
  semantic_rgb_dir = os.path.join(args.save_dir, 'semantic_color')

  # Create color map.
  color_map = vis_utils.load_color_map(config.dataset.color_map_path)
  color_map = color_map.numpy()

  # Create data loaders.
  test_dataset = SPixListDataset(
      data_dir=args.data_dir,
      data_list=args.data_list,
      img_mean=config.network.pixel_means,
      img_std=config.network.pixel_stds,
      size=None,
      random_crop=False,
      random_scale=False,
      random_mirror=False,
      training=False,
      n_segments=config.network.num_superpixels,
      slic_compactness=10,
      slic_scale_factor=1.0)
  test_image_paths = test_dataset.image_paths

  # Create models.
  if config.network.backbone_types == 'cast_small_p512_retrieval':
    embedding_model = cast.cast_small_p512_retrieval(config).cuda()
  else:
    raise ValueError('Not support ' + config.network.backbone_types)

  if config.network.prediction_types == 'segsort':
    prediction_model = segsort(config)
  else:
    raise ValueError('Not support ' + config.network.prediction_types)

  embedding_model = embedding_model.to("cuda:0")
  prediction_model = prediction_model.to("cuda:0")
  embedding_model.eval()
  prediction_model.eval()
      
  # Load trained weights.
  model_path_template = os.path.join(args.snapshot_dir, 'model-{:d}.pth')
  save_iter = config.train.max_iteration - 1
  checkpoint = torch.load(model_path_template.format(save_iter))['state_dict']
  cast.interpolate_pos_embed(embedding_model.vit_backbone, checkpoint)
  embedding_model.load_state_dict(checkpoint, resume=False)

  # Load memory prototypes.
  semantic_memory_prototypes, semantic_memory_prototype_labels = None, None
  if args.semantic_memory_dir is not None:
    semantic_memory_prototypes, semantic_memory_prototype_labels = (
      segsort_others.load_memory_banks(args.semantic_memory_dir))
    semantic_memory_prototypes = semantic_memory_prototypes.to("cuda:0")
    semantic_memory_prototype_labels = semantic_memory_prototype_labels.to("cuda:0")

    # Remove ignore class.
    valid_prototypes = torch.ne(
        semantic_memory_prototype_labels,
        config.dataset.semantic_ignore_index).nonzero()
    valid_prototypes = valid_prototypes.view(-1)
    semantic_memory_prototypes = torch.index_select(
        semantic_memory_prototypes,
        0,
        valid_prototypes)
    semantic_memory_prototype_labels = torch.index_select(
        semantic_memory_prototype_labels,
        0,
        valid_prototypes)

  # Start inferencing.
  for data_index in tqdm(range(len(test_dataset))):
    # Image path.
    image_path = test_image_paths[data_index]
    base_name = os.path.basename(image_path).replace('.jpg', '.png')

    # Image resolution.
    image_batch, label_batch, _ = test_dataset[data_index]
    image_h, image_w = image_batch['image'].shape[-2:]

    # Resize the input image.
    if config.test.image_size > 0:
      image_batch['image'] = transforms.resize_with_interpolation(
          image_batch['image'].transpose(1, 2, 0),
          config.test.image_size,
          method='bilinear').transpose(2, 0, 1)
      for lab_name in ['semantic_label', 'instance_label', 'slic_label']:
        label_batch[lab_name] = transforms.resize_with_interpolation(
            label_batch[lab_name],
            config.test.image_size,
            method='nearest')
    resize_image_h, resize_image_w = image_batch['image'].shape[-2:]

    # Crop and Pad the input image.
    #image_batch['image'] = transforms.resize_with_pad(
    #    image_batch['image'].transpose(1, 2, 0),
    #    config.test.crop_size,
    #    image_pad_value=0).transpose(2, 0, 1)
    image_batch['image'] = torch.FloatTensor(
        image_batch['image'][np.newaxis, ...]).to("cuda:0")
    #pad_image_h, pad_image_w = image_batch['image'].shape[-2:]
    pad_image_h, pad_image_w = resize_image_h, resize_image_w

    # Put label batch to gpu 1.
    for k, v in label_batch.items():
      label_batch[k] = torch.LongTensor(v[np.newaxis, ...]).to("cuda:0")

    # Create the fake labels where clustering ignores 255.
    fake_label_batch = {}
    for label_name in ['semantic_label', 'instance_label']:
      lab = np.zeros((resize_image_h, resize_image_w),
                     dtype=np.uint8)
      #lab = transforms.resize_with_pad(
      #    lab,
      #    config.test.crop_size,
      #    image_pad_value=config.dataset.semantic_ignore_index)

      fake_label_batch[label_name] = torch.LongTensor(
          lab[np.newaxis, ...]).to("cuda:0")
    fake_label_batch['slic_label'] = label_batch['slic_label']

    # Create place holder for full-resolution embeddings.
    with torch.no_grad():
      embeddings = embedding_model(image_batch, fake_label_batch, resize_as_input=True)

    # Generate predictions.
    outputs = prediction_model(
        embeddings,
        {'semantic_memory_prototype': semantic_memory_prototypes,
         'semantic_memory_prototype_label': semantic_memory_prototype_labels},
        with_loss=False, with_prediction=True)


    # Save semantic predictions.
    semantic_pred = outputs.get('semantic_prediction', None)
    if semantic_pred is not None:
      semantic_pred = (semantic_pred.view(resize_image_h, resize_image_w)
                                    .cpu()
                                    .data
                                    .numpy()
                                    .astype(np.uint8))
      semantic_pred = cv2.resize(
          semantic_pred,
          (image_w, image_h),
          interpolation=cv2.INTER_NEAREST)

      semantic_pred_name = os.path.join(
          semantic_dir, base_name)
      if not os.path.isdir(os.path.dirname(semantic_pred_name)):
        os.makedirs(os.path.dirname(semantic_pred_name))
      Image.fromarray(semantic_pred, mode='L').save(semantic_pred_name)

      semantic_pred_rgb = color_map[semantic_pred]
      semantic_pred_rgb_name = os.path.join(
          semantic_rgb_dir, base_name)
      if not os.path.isdir(os.path.dirname(semantic_pred_rgb_name)):
        os.makedirs(os.path.dirname(semantic_pred_rgb_name))
      Image.fromarray(semantic_pred_rgb, mode='RGB').save(
          semantic_pred_rgb_name)


if __name__ == '__main__':
  main()
