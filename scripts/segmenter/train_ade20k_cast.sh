#/usr/bin/bash

SNAPSHOTS=snapshots/segmenter/ade20k/cast_small

export PYTHONPATH=segmenter/:$PYTHONPATH

python -m segm.train_cast \
  --log-dir ${SNAPSHOTS} \
  --dataset ade20k \
  --backbone deit_cast_small \
  --decoder mask_transformer

export DATASET=./data/
python -m segm.eval.miou_cast ${SNAPSHOTS}/checkpoint.pth ade20k --singlescale --save-images --no-blend
