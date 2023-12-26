#!/usr/bin/bash
SNAPSHOTS=snapshots/moco/coco/cast_small
mkdir -p ${SNAPSHOTS}

export PYTHONPATH=moco-v3/:$PYTHONPATH

python -W ignore moco-v3/main_moco_suppix.py \
    -a cast_seg_small_pretrain -b 256 \
    --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
    --epochs=400 --warmup-epochs=40 \
    --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
    --dist-url 'tcp://localhost:10001' \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    ./data/coco
mv checkpoint_0*99.pth.tar checkpoint_0*49.pth.tar ${SNAPSHOTS}
