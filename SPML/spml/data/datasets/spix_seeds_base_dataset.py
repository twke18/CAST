"""Classes for Dataset which generate superpixel.
"""

import cv2
import numpy as np

from spml.data.datasets.base_dataset import ListDataset


class SPixListDataset(ListDataset):

  def __init__(self,
               data_dir,
               data_list,
               img_mean=(0, 0, 0),
               img_std=(1, 1, 1),
               size=None,
               random_crop=False,
               random_scale=False,
               random_mirror=False,
               training=False,
               n_segments=784,
               slic_compactness=10.0,
               slic_scale_factor=1.0):
    """Base class for Dataset.

    Args:
      data_dir: A string indicates root directory of images and labels.
      data_list: A list of strings which indicate path of paired images
        and labels. 'image_path semantic_label_path instance_label_path'.
      img_mean: A list of scalars indicate the mean image value per channel.
      img_std: A list of scalars indicate the std image value per channel.
      size: A tuple of scalars indicate size of output image and labels.
        The output resolution remain the same if `size` is None.
      random_crop: enable/disable random_crop for data augmentation.
        If True, adopt randomly cropping as augmentation.
      random_scale: enable/disable random_scale for data augmentation.
        If True, adopt adopt randomly scaling as augmentation.
      random_mirror: enable/disable random_mirror for data augmentation.
        If True, adopt adopt randomly mirroring as augmentation.
      training: enable/disable training to set dataset for training and
        testing. If True, set to training mode.
      n_segments: A scalar indicates the number of superpixels.
      slic_compactness: A scalar indicates the compactness ratio used
        in SLIC algorithm.
      slic_scale_factor: A scalar indicates the resizing ration of
        input image for SLIC algorithm.
    """
    super(SPixListDataset, self).__init__(
        data_dir,
        data_list,
        img_mean,
        img_std,
        size,
        random_crop,
        random_scale,
        random_mirror,
        training)
    self.n_segments = n_segments
    self.slic_compactness = slic_compactness
    self.slic_scale_factor = slic_scale_factor

  def _generate_superpixels(self, image):
    """Generate superpixels.
    """
    image = (image * 255).astype(np.uint8)
    if self.slic_scale_factor != 1:
      dsize = (int(image.shape[0] * self.slic_scale_factor),
               int(image.shape[1] * self.slic_scale_factor))
      image = cv2.resize(
          image, dsize=dsize, interpolation=cv2.INTER_LINEAR)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    seeds = cv2.ximgproc.createSuperpixelSEEDS(
        image.shape[1], image.shape[0], 3, num_superpixels=self.n_segments, num_levels=1, prior=2,
        histogram_bins=5, double_step=False);
    seeds.iterate(image, num_iterations=15);
    segment = seeds.getLabels()

    return segment

  def __getitem__(self, idx):
    """Retrive image and label by index.
    """
    if self.training:
      image, semantic_label, instance_label = self._training_preprocess(idx)
    else:
      image, semantic_label, instance_label = self._eval_preprocess(idx)

    segment = self._generate_superpixels(image)

    image = image - np.array(self.img_mean, dtype=image.dtype)
    image = image / np.array(self.img_std, dtype=image.dtype)

    inputs = {'image': image.transpose(2, 0, 1)}
    labels = {'semantic_label': semantic_label,
              'instance_label': instance_label,
              'slic_label': segment}

    return inputs, labels, idx

