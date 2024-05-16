import torch
import torchvision.transforms.functional as F
import numpy as np
import yaml
from pathlib import Path

import cv2
import skimage.color as sk_color
import skimage.morphology as sk_morph


IGNORE_LABEL = 255
STATS = {
    #"vit": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    "deit": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    "vit": {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
}


def seg_to_rgb(seg, colors):
    im = torch.zeros((seg.shape[0], seg.shape[1], seg.shape[2], 3)).float()
    cls = torch.unique(seg)
    for cl in cls:
        color = colors[int(cl)]
        if len(color.shape) > 1:
            color = color[0]
        im[seg == cl] = color
    return im


def dataset_cat_description(path, cmap=None):
    desc = yaml.load(open(path, "r"), Loader=yaml.FullLoader)
    colors = {}
    names = []
    for i, cat in enumerate(desc):
        names.append(cat["name"])
        if "color" in cat:
            colors[cat["id"]] = torch.tensor(cat["color"]).float() / 255
        else:
            colors[cat["id"]] = torch.tensor(cmap[cat["id"]]).float()
    colors[IGNORE_LABEL] = torch.tensor([0.0, 0.0, 0.0]).float()
    return names, colors


def rgb_normalize(x, stats):
    """
    x : C x *
    x \in [0, 1]
    """
    return F.normalize(x, stats["mean"], stats["std"])


def rgb_denormalize(x, stats, inplace=True):
    """
    x : N x C x *
    x \in [-1, 1]
    """
    if not inplace:
        x = x.clone()
    mean = torch.tensor(stats["mean"])
    std = torch.tensor(stats["std"])
    for i in range(3):
        x[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return x


def generate_superpixels(image, n_segments, slic_scale_factor=1.0):
    """Generate superpixels.
    """
    image = (image * 255).astype(np.uint8)
    if slic_scale_factor != 1:
        dsize = (int(image.shape[0] * slic_scale_factor),
                 int(image.shape[1] * slic_scale_factor))
        image = cv2.resize(
            image, dsize=dsize, interpolation=cv2.INTER_LINEAR)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    seeds = cv2.ximgproc.createSuperpixelSEEDS(
        image.shape[1], image.shape[0], 3, num_superpixels=n_segments, num_levels=1, prior=2,
        histogram_bins=5, double_step=False);
    seeds.iterate(image, num_iterations=15);
    segment = seeds.getLabels()

    return segment


def generate_contour(label, size=2, is_symmetric=True):
    if is_symmetric:
        label = np.pad(label, ((1, 1), (1, 1)), mode='symmetric')
    else:
        label = np.pad(label, ((1, 1), (1, 1)), mode='constant', constant_values=0)
    l_diff = label[1:-1, 1:-1] != label[1:-1, :-2]
    r_diff = label[1:-1, 1:-1] != label[1:-1, 2:]
    t_diff = label[1:-1, 1:-1] != label[:-2, 1:-1]
    b_diff = label[1:-1, 1:-1] != label[2:, 1:-1]

    edge = (l_diff + r_diff + t_diff + b_diff).astype(label.dtype)

    # dilation
    if size > 0:
        disk = sk_morph.disk(size)
        edge = edge.astype(np.int32)
        edge = sk_morph.dilation(edge, disk).astype(label.dtype)

    return edge


def label2color(label, img):
    out = sk_color.label2rgb(label, img, kind='avg', bg_label=-1)
    edge = generate_contour(label, 0).astype(np.bool)
    out[edge] = 1
    return out

