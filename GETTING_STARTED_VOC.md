# Getting started with unsupervised/fully-supervised segmentation on Pascal VOC
Our CAST can do unsupervised/fully-supervised segmentation on Pascal VOC.  For unsupervised segmentation, we generate segmentations and extract segment features using CAST, which is self-supervisedly trained on COCO.  For fully-supervised segmentation, we fine-tune CAST with ground-truth pixel labels on VOC.  All experiments are based on [SPML](https://github.com/twke18/SPML) framework.


We provide the bashscripts for running all experiments.  By default, we use MoCo-trained `CAST-S`.

See [MODEL_ZOO.md](MODEL_ZOO.md) for downloading our (pre-)trained model weights.

1. Unsupervised segmentation on VOC
```
bash scripts/spml/nn_cast.sh
```

2. Fine-tuning CAST on Pascal Context
```
bash scripts/spml/finetune_cast.sh
```
