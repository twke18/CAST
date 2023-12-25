#/usr/bin/bash

SNAPSHOTS=snapshots/segmenter/pascal_context/cast_small

export PYTHONPATH=segmenter/:$PYTHONPATH

python -m segm.train_cast \
  --log-dir ${SNAPSHOTS} \
  --dataset pascal_context \
  --backbone deit_cast_small \
  --decoder mask_transformer

export DATASET=./data/
python -m segm.eval.miou_cast ${SNAPSHOTS}/checkpoint.pth pascal_context --singlescale --save-images --no-blend

