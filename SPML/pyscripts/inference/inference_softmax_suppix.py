"""Inference script for semantic segmentation by softmax classifier.
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
from spml.data.datasets.base_dataset import ListDataset
from spml.data.datasets.spix_seeds_base_dataset import SPixListDataset
from spml.config.default import config
from spml.config.parse_args import parse_args
import spml.models.embeddings.cast as cast
from spml.models.predictions.softmax_classifier import softmax_classifier

cudnn.enabled = True
cudnn.benchmark = True


def main():
  """Inference for semantic segmentation.
  """
  # Retreve experiment configurations.
  args = parse_args('Inference for semantic segmentation.')

  # Create directories to save results.
  semantic_dir = os.path.join(args.save_dir, 'semantic_gray')
  semantic_rgb_dir = os.path.join(args.save_dir, 'semantic_color')
  if not os.path.isdir(semantic_dir):
    os.makedirs(semantic_dir)
  if not os.path.isdir(semantic_rgb_dir):
    os.makedirs(semantic_rgb_dir)

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
  if config.network.backbone_types == 'cast_small_p512_finetune':
    embedding_model = cast.cast_small_p512_finetune(config).cuda()
  else:
    raise ValueError('Not support ' + config.network.backbone_types)

  prediction_model = softmax_classifier(config).cuda()
  embedding_model.eval()
  prediction_model.eval()
      
  # Load trained weights.
  model_path_template = os.path.join(args.snapshot_dir, 'model-{:d}.pth')
  save_iter = config.train.max_iteration - 1
  embedding_model.load_state_dict(
      torch.load(model_path_template.format(save_iter))['embedding_model'],
      resume=True)
  prediction_model.load_state_dict(
      torch.load(model_path_template.format(save_iter))['prediction_model'])


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
    image_batch['image'] = torch.FloatTensor(
        image_batch['image'][np.newaxis, ...]).to("cuda:0")
    pad_image_h, pad_image_w = resize_image_h, resize_image_w

    # Put label batch to gpu 1.
    for k, v in label_batch.items():
      label_batch[k] = torch.LongTensor(v[np.newaxis, ...]).to("cuda:0")

    # Create the fake labels where clustering ignores 255.
    fake_label_batch = {}
    for label_name in ['semantic_label', 'instance_label']:
      lab = np.zeros((resize_image_h, resize_image_w),
                     dtype=np.uint8)

      fake_label_batch[label_name] = torch.LongTensor(
          lab[np.newaxis, ...]).to("cuda:0")
    fake_label_batch['slic_label'] = label_batch['slic_label']

    # Create place holder for full-resolution embeddings.
    with torch.no_grad():
      embeddings = embedding_model(image_batch, fake_label_batch, resize_as_input=True)
      outputs = prediction_model(embeddings)

    # Save semantic predictions.
    semantic_logits = outputs.get('semantic_logit', None)
    if semantic_logits is not None:
      semantic_pred = torch.argmax(semantic_logits, 1)
      semantic_pred = (semantic_pred.view(pad_image_h, pad_image_w)
                                    .cpu()
                                    .data
                                    .numpy()
                                    .astype(np.uint8))
      semantic_pred = semantic_pred[:resize_image_h, :resize_image_w]
      semantic_pred = cv2.resize(
          semantic_pred,
          (image_w, image_h),
          interpolation=cv2.INTER_NEAREST)

      semantic_pred_name = os.path.join(semantic_dir, base_name)
      Image.fromarray(semantic_pred, mode='L').save(semantic_pred_name)

      semantic_pred_rgb = color_map[semantic_pred]
      semantic_pred_rgb_name = os.path.join(semantic_rgb_dir, base_name)
      Image.fromarray(semantic_pred_rgb, mode='RGB').save(
          semantic_pred_rgb_name)

      # Clean GPU memory cache to save more space.
      outputs = {}
      crop_embeddings = {}
      crop_outputs = {}
      torch.cuda.empty_cache()


if __name__ == '__main__':
  main()
