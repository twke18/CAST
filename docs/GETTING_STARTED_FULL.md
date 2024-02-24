# Getting started with fully-supervised learning of CAST
Our CAST can also be trained with fully-supervised learning.  In the paper, we fine-tune pre-trained models on ADE20K and Pascal Context for semantic segmentation.  We pre-train models for fully-supervised classification with [DeiT](https://github.com/facebookresearch/deit) framework.

We provide the bashscripts for running fully-supervised experiments.  By default, we use `CAST-S`.  You can use larger models, e.g. `CAST-B` by replacing `--model cast_small` with `--model cast_base` in the bashscripts.

### Pre-train on ImageNet for classification

1. fully-supervised learning of CAST on ImageNet-1K:
```
> bash scripts/deit/train_imagenet1k_cast.sh
```
