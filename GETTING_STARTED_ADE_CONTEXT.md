# Getting started with fine-tuning CAST on ADE20K and Pascal Context
Our CAST can also be fine-tuned for fully-supervised semantic segmentation.  In the paper, we use [Segmenter](https://github.com/rstrudel/segmenter) framework for all experiments.

We provide the bashscripts for running fine-tuning experiments.  By default, we use Deit-trained `CAST-S`.  You can use larger models, e.g. `CAST-B` by replacing `--backbone deit_cast_small` with `--backbone deit_cast_base` in the bashscripts.

See [MODEL_ZOO.md](MODEL_ZOO.md) for downloading our (pre-)trained model weights.

1. Fine-tuning CAST on ADE20K
```
bash scripts/segmenter/train_ade20k_cast.sh
```

2. Fine-tuning CAST on Pascal Context
```
bash scripts/segmenter/train_context_cast.sh
```
