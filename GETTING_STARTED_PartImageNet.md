# Getting started with part segmentation of CAST on PartImageNet


We employ open-vocabulary segmentation to predict parts and wholes labels based on the CAST segments. In this paper, we apply the OVSeg framework [OVSeg](https://github.com/facebookresearch/ov-seg), which predicts labels for masked images, except we did not fine-tune CLIP on these masked images.

We provide jupyter notebooks for predicting segmentation maps and conducting evaluations. We save the segmentations first and reuse them in subsequent evaluations.

### Pre-train on ImageNet for classification

1. Save hierarchical segmentation:
- [CAST](notebooks/save_segmentation_cast.ipynb)
- [ViT](notebooks/save_segmentation_vit.ipynb)

2. Visualize open-vocabulary segmentation:
- [CAST/ViT](notebooks/visualize_ovseg_cast.ipynb)
- [SAM](notebooks/visualize_ovseg_sam.ipynb)

3. Evaluate open-vocabulary segmentation:
- [CAST/ViT](notebooks/eval_ovseg_cast.ipynb)
- [SAM](notebooks/eval_ovseg_sam.ipynb)