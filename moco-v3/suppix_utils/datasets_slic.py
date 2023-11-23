"""Define datasets which generate superpixels using SLIC."""
from typing import Optional, Callable, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.datasets.folder as folder
from skimage.segmentation import slic

import moco.loader


class ImageFolder(datasets.ImageFolder):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 normalize: Optional[Callable] = None,
                 loader: Callable[[str], Any] = folder.default_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 n_segments: int = 256,
                 compactness: float = 10.0,
                 blur_ops: Optional[Callable] = None,
                 slic_scale_factor=1.0):
        super(ImageFolder, self).__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file)
        self.normalize = normalize
        self.n_segments = n_segments
        self.compactness = compactness
        self.blur_ops = blur_ops
        self.slic_scale_factor = slic_scale_factor

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # Prepare arguments when multi-view pipeline is adopted
        compactness = self.compactness
        blur_ops = self.blur_ops
        n_segments = self.n_segments
        slic_scale_factor = self.slic_scale_factor
        if isinstance(sample, (list, tuple)):
            if not isinstance(compactness, (list, tuple)):
                compactness = [compactness] * len(sample)

            if not isinstance(n_segments, (list, tuple)):
                n_segments = [n_segments] * len(sample)

            if not isinstance(blur_ops, (list, tuple)):
                blur_ops = [blur_ops] * len(sample)

            if not isinstance(slic_scale_factor, (list, tuple)):
                slic_scale_factor = [slic_scale_factor] * len(sample)


        # Generate superpixels
        if isinstance(sample, (list, tuple)):
            segments = []
            for samp, comp, n_seg, blur_op, scale in zip(sample, compactness, n_segments, blur_ops, slic_scale_factor):
                if blur_op is not None:
                    samp = blur_op(samp)
                samp = F.interpolate(samp.unsqueeze(0),
                                     scale_factor=scale,
                                     mode='bilinear').squeeze(0)
                _comp = comp
                while True:
                  segment = slic(samp.data.numpy().transpose(1, 2, 0),
                                 n_segments=n_seg,
                                 compactness=_comp,
                                 convert2lab=True,
                                 start_label=0)
                  if np.unique(segment).size < n_seg // 2:
                      _comp *= 3
                  else:
                      break
                segment = torch.LongTensor(segment)
                segments.append(segment)
        else:
            if blur_ops is not None:
                samp = blur_ops(sample)
            samp = F.interpolate(samp.unsqueeze(0),
                                 scale_factor=slic_scale_factor,
                                 mode='bilinear').squeeze(0)
            _comp = compactness
            while True:
              segments = slic(samp.data.numpy().transpose(1, 2, 0),
                              n_segments=n_segments,
                              compactness=_comp,
                              convert2lab=True,
                              start_label=0)
              if np.unique(segments).size < n_segments // 2:
                  _comp *= 3
              else:
                  break
            segments = torch.LongTensor(segments)

        # Normalize the images
        if self.normalize is not None:
          if isinstance(sample, (list, tuple)):
              sample = [self.normalize(samp) for samp in sample]
          else:
              sample = self.normalize(sample)

        return sample, segments, target
