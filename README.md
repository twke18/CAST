# Learning Hierarchical Image Segmentation For Recognition and By Recognition
By [Tsung-Wei Ke](https://twke18.github.io/), [Sangwoo Mo](https://sites.google.com/view/sangwoomo) and [Stella X. Yu](https://web.eecs.umich.edu/~stellayu/)

Image segmentation and recognition occur simultaneously, with recognition relying on the underlying segmentation to form a continuous visual grouping hierarchy. For example, the same object can be parsed into different part-to-whole structures, resulting in varying recognitions. Despite this, most prior works treated segmentation and recognition as separate tasks. In this paper, we aim to devise a learning framework that involves segmentation in the recognition process, utilizing hierarchical segmentation for recognition, which is learned by recognition. Specifically, we propose CAST, which realizes this concept through designs inspired by vision transformers, enabling concurrent segmentation and recognition with a single model. The core idea of CAST is to employ adaptive segment tokens that group the finest pixels into coarser segments, using the latest embedding to represent the entire image for recognition. Trained solely on image recognition objectives, CAST automatically discovers the hierarchy of segments. Our experiments demonstrate that CAST provides consistent hierarchical segmentation and recognition, which is impossible with state-of-the-art segmentation methods such as SAM. Additionally, CAST offers several advantages over the standard ViT, including improved semantic segmentation, computational efficiency, and object-centric attention.


## Installation

Create a conda environment with the following command:
```
# initiate conda env
> conda update conda
> conda env create -f environment.yaml
> conda activate cast

# install dgl (https://www.dgl.ai/pages/start.html)
> pip install  dgl -f https://data.dgl.ai/wheels/cu118/repo.html
> pip install  dglgo -f https://data.dgl.ai/wheels-test/repo.html

# install
> pip install -e .
```

## Data Preparation

See [Preparing datasets](DATASETS.md).

## Getting Started

### Representation pre-training / Image classification

See [Getting started with self-supervised learning of CAST](GETTING_STARTED_SELF.md)

See [Getting started with fully-supervised learning of CAST](GETTING_STARTED_FULL.md)

### Fine-tuning on Semantic segmentation

See [Getting started with fine-tuning of CAST on Pascal Context and ADE20K](GETTING_STARTED_ADE_CONTEXT.md)

See [Getting started with unsupervised segementation and fine-tuning of CAST on VOC](GETTING_STARTED_VOC.md)

### Part segmentation on PartImageNet


## Citation
If you find this code useful for your research, please consider citing our paper Hierarchical Vision Transformers with Adaptive Segment Tokens.
```
```

## License
CAST is released under the MIT License (refer to the LICENSE file for details).

## Acknowledgement
This release of code is based on [MOCO-v3](https://github.com/facebookresearch/moco-v3), [DeiT](https://github.com/facebookresearch/deit), [SegFormer](https://github.com/NVlabs/SegFormer), [SPML](https://github.com/twke18/SPML), and [OV-Seg](https://github.com/facebookresearch/ov-seg).