#!/usr/bin/bash
SNAPSHOTS=snapshots/coco/cast_small
mkdir -p ${SNAPSHOTS}

python -W ignore main_moco_cast.py \
    -a cast_small -b 256 \
    --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
    --epochs=400 --warmup-epochs=40 \
    --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
    --dist-url 'tcp://localhost:10001' \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    ./data/coco
mv checkpoint_0*99.pth.tar checkpoint_0*49.pth.tar ${SNAPSHOTS}
