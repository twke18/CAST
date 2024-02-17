#!/usr/bin/bash

export PYTHONPATH=deit/:$PYTHONPATH

python -W ignore -m torch.distributed.launch \
  --nproc_per_node=8 \
  --use_env deit/main_suppix.py \
  --model cast_small \
  --batch-size 128 \
  --num-superpixels 196 \
  --num_workers 4 \
  --data-set IMNET-SUPERPIXEL \
  --data-path data/ILSVRC2014/Img \
  --distributed \
  --dist-eval \
  --eval \
  --resume snapshots/deit/imagenet1k/cast_small/best_checkpoint.pth \
  --output_dir snapshots/deit/imagenet1k/cast_small
