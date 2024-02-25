# Model overview
In this code base, we provide our implementation of [CAST](../cast_models/cast.py).  We provide an overview of input and output of both models.

## Input format
Our CAST takes input as images and the corresponding superpixels.  CAST first aggregates pixel features into tokens within each superpixel, and groups the segment tokens hierarchically.  The final hierarchical segmentaiton can be derived by looking up the fine-to-coarse grouping indices at each level.

1. `image`: a float tensor of shape (batch_size, 3, H, W).  The pixel values are normalized RGB.
2. `superpixel`: a long tensor of shape (batch_size, H, W).  The pixel values are superpixel indices.


## Model variants and output format
We provide four variants of CAST for each experiment reported in the paper.

1. `cast_models/cast.py`: the model is self-supervised trained on ImageNet.  The model outputs the [CLS] token of shape (batch_size, C).  We report the linear probing results in Table 4.
2. `cast_models/cast_seg.py`: the model is self-supervised trained on COCO.  The model outputs the patch-wise tokens of shape (batch_size, H*W, C).  We report the semantic segmentation results before and after fine-tuning on Pascal VOC 2012 in Table 3a.
3. `cast_models/cast_deit.py`: the model is supervised trained on ImageNet.  The model outputs the [CLS] token of shape (batch_size, C).  We use such pre-trained models for fine-tuning on ADE20K and Pascal Context.
4. `cast_models/cast_seg_deit.py`: the model is fine-tuned on ADE20K and Pascal Context.  The model outputs the [CLS] and patch-wise tokens of shape (batch_size, 1 + H*W, C).  We report the semantic segmentation results in Table 3b and 3c.

We also provide a baseline which uses superpixel tokens and [ToMe](https://arxiv.org/abs/2210.09461) for hierarchical grouping.  See [moco-v3/models/tome_suppix.py](../moco-v3/models/tome_suppix.py)


### Usage 
For training and testing, all you need is to prepare superpixel segmentation for an input image.
```
# import CAST
from cast_models.cast import cast_small
model = cast_small()

# prepare superpixels
> n_segments = 196
> image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
> seeds = cv2.ximgproc.createSuperpixelSEEDS(image.shape[1],
                                             image.shape[0],
                                             3,
                                             num_superpixels=n_segments,
                                             num_levels=1,
                                             prior=2,
                                             histogram_bins=15,
                                             double_step=False);
> seeds.iterate(image, num_iterations=20);
> segments = seeds.getLabels()
> segments = torch.LongTensor(segments)

# feed image and superpixels to our model
> output = model.forward(torch.as_tensor(image).unsqueeze(0).permute(0, 3, 1, 2), # NxCxHxW
                         segments.unsqueeze(0))
```
