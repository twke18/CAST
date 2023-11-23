"""Genereate pseudo labels from CAM by random walk and CRF.
"""
from __future__ import print_function, division
import os
import math

import PIL.Image as Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import spml.data.transforms as transforms
import spml.utils.general.vis as vis_utils
import spml.utils.general.others as other_utils
from spml.data.datasets.base_dataset import ListDataset
from spml.config.default import config
from spml.config.parse_args import parse_args
from spml.models.embeddings.resnet_pspnet import resnet_101_pspnet
from spml.models.embeddings.resnet_deeplab import resnet_101_deeplab
from spml.models.crf import DenseCRF

cudnn.enabled = True
cudnn.benchmark = True

ALPHA=6
WALK_STEPS=6


def separate_comma(str_comma):
  ints = [int(i) for i in str_comma.split(',')]
  return ints


def main():
  """Generate pseudo labels from CAM by random walk and CRF.
  """
  # Retreve experiment configurations.
  args = parse_args('Generate pseudo labels from CAM by random walk and CRF.')
  config.network.kmeans_num_clusters = separate_comma(args.kmeans_num_clusters)
  config.network.label_divisor = args.label_divisor

  # Create directories to save results.
  semantic_dir = os.path.join(args.save_dir, 'semantic_gray')
  semantic_rgb_dir = os.path.join(args.save_dir, 'semantic_color')

  # Create color map.
  color_map = vis_utils.load_color_map(config.dataset.color_map_path)
  color_map = color_map.numpy()

  # Create data loaders.
  test_dataset = ListDataset(
      data_dir=args.data_dir,
      data_list=args.data_list,
      img_mean=config.network.pixel_means,
      img_std=config.network.pixel_stds,
      size=None,
      random_crop=False,
      random_scale=False,
      random_mirror=False,
      training=False)
  test_image_paths = test_dataset.image_paths

  # Define CRF.
  postprocessor = DenseCRF(
      iter_max=args.crf_iter_max,
      pos_xy_std=args.crf_pos_xy_std,
      pos_w=args.crf_pos_w,
      bi_xy_std=args.crf_bi_xy_std,
      bi_rgb_std=args.crf_bi_rgb_std,
      bi_w=args.crf_bi_w,)

  # Create models.
  if config.network.backbone_types == 'panoptic_pspnet_101':
    embedding_model = resnet_101_pspnet(config)
  elif config.network.backbone_types == 'panoptic_deeplab_101':
    embedding_model = resnet_101_deeplab(config).cuda()
  else:
    raise ValueError('Not support ' + config.network.backbone_types)

  embedding_model = embedding_model.to("cuda:0")
  embedding_model.eval()

  # Load trained weights.
  model_path_template = os.path.join(args.snapshot_dir, 'model-{:d}.pth')
  save_iter = config.train.max_iteration - 1
  embedding_model.load_state_dict(
      torch.load(model_path_template.format(save_iter))['embedding_model'],
      resume=True)

  # Start inferencing.
  for data_index in tqdm(range(len(test_dataset))):
    # Image path.
    image_path = test_image_paths[data_index]
    base_name = os.path.basename(image_path).replace('.jpg', '.png')

    # Image resolution.
    image_batch, label_batch, _ = test_dataset[data_index]
    image_h, image_w = image_batch['image'].shape[-2:]

    # Load cam
    sem_labs = np.unique(label_batch['semantic_label'])
    #cam = np.load(os.path.join('/home/twke/repos/SEAM/outputs/train+/cam', base_name.replace('.png', '.npy')), allow_pickle=True).item()
    cam = np.load(os.path.join(args.cam_dir, base_name.replace('.png', '.npy')),
                  allow_pickle=True).item()
    cam_full_arr = np.zeros((21, image_h, image_w), np.float32)
    for k, v in cam.items():
      cam_full_arr[k+1] = v
    cam_full_arr[0] = np.power(1 - np.max(cam_full_arr[1:], axis=0, keepdims=True), ALPHA)
    cam_full_arr = torch.from_numpy(cam_full_arr).cuda()

    # Image resolution.
    batches = other_utils.create_image_pyramid(
        image_batch, label_batch,
        scales=[1],
        is_flip=True)

    affs = []
    for image_batch, label_batch, data_info in batches:
      resize_image_h, resize_image_w = image_batch['image'].shape[-2:]
      # Crop and Pad the input image.
      image_batch['image'] = transforms.resize_with_pad(
          image_batch['image'].transpose(1, 2, 0),
          config.test.crop_size,
          image_pad_value=0).transpose(2, 0, 1)
      for lab_name in ['semantic_label', 'instance_label']:
        label_batch[lab_name] = transforms.resize_with_pad(
            label_batch[lab_name],
            config.test.crop_size,
            image_pad_value=255)
      image_batch['image'] = torch.FloatTensor(
          image_batch['image'][np.newaxis, ...]).to("cuda:0")
      for k, v in label_batch.items():
        label_batch[k] = torch.LongTensor(v[np.newaxis, ...]).to("cuda:0")
      pad_image_h, pad_image_w = image_batch['image'].shape[-2:]

      with torch.no_grad():
        embeddings = embedding_model(image_batch, label_batch, resize_as_input=True)
        embs = embeddings['embedding'][:, :, :resize_image_h, :resize_image_w]
        if data_info['is_flip']:
          embs = torch.flip(embs, dims=[3])
        embs = F.interpolate(embs, size=(image_h//8, image_w//8), mode='bilinear')
        embs = embs / torch.norm(embs, dim=1)
        embs_flat = embs.view(embs.shape[1], -1)
        aff = torch.matmul(embs_flat.t(), embs_flat).mul_(5).add_(-5).exp_()
        affs.append(aff)

    aff = torch.mean(torch.stack(affs, dim=0), dim=0)
    cam_full_arr = F.interpolate(
        cam_full_arr.unsqueeze(0), scale_factor=1/8., mode='bilinear').squeeze(0)
    cam_shape = cam_full_arr.shape[-2:]

    # Start random walk.
    aff_mat = aff ** 20

    trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
    for _ in range(WALK_STEPS):
      trans_mat = torch.matmul(trans_mat, trans_mat)

    cam_vec = cam_full_arr.view(21, -1)
    cam_rw = torch.matmul(cam_vec, trans_mat)
    cam_rw = cam_rw.view(21, cam_shape[0], cam_shape[1])

    cam_rw = cam_rw.data.cpu().numpy()
    cam_rw = cv2.resize(cam_rw.transpose(1, 2, 0),
                        dsize=(image_w, image_h),
                        interpolation=cv2.INTER_LINEAR)
    cam_rw_pred = np.argmax(cam_rw, axis=-1).astype(np.uint8)

    # CRF
    image = image_batch['image'].data.cpu().numpy().astype(np.float32)
    image = image[0, :, :image_h, :image_w].transpose(1, 2, 0)
    image *= np.reshape(config.network.pixel_stds, (1, 1, 3))
    image += np.reshape(config.network.pixel_means, (1, 1, 3))
    image = image * 255
    image = image.astype(np.uint8)
    cam_rw = postprocessor(image, cam_rw.transpose(2,0,1))

    cam_rw_pred = np.argmax(cam_rw, axis=0).astype(np.uint8)

    # Save semantic predictions.
    semantic_pred = cam_rw_pred

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
