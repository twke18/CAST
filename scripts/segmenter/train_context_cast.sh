#/usr/bin/bash

SNAPSHOTS=snapshots/segmenter/pascal_context/cast_small

export PYTHONPATH=$(pwd)/segmenter/:$PYTHONPATH
export DATASET=$(pwd)/data/

python -m segm.train_cast \
  --log-dir ${SNAPSHOTS} \
  --pretrained snapshots/deit/imagenet1k/cast_small/best_checkpoint.pth \
  --dataset pascal_context \
  --backbone deit_cast_small \
  --decoder mask_transformer

python -m segm.eval.miou_cast ${SNAPSHOTS}/checkpoint.pth pascal_context --singlescale --save-images --no-blend

