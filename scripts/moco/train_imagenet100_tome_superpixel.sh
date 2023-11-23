#!/usr/bin/bash
SNAPSHOTS=snapshots/moco/imagenet100/tome_small
mkdir -p ${SNAPSHOTS}
mkdir -p ${SNAPSHOTS}/lincls

export PYTHONPATH=moco-v3/:$PYTHONPATH

python -W ignore moco-v3/main_moco_suppix.py \
    -a tome_small -b 256 \
    --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
    --epochs=200 --warmup-epochs=20 \
    --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
    --dist-url 'tcp://localhost:10001' \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    ./data/ILSVRC2014/Img-100/
mv checkpoint_0199.pth.tar ${SNAPSHOTS}

python -W ignore moco-v3/main_lincls_suppix.py \
    -a tome_small --lr 0.8 -b 256 \
    --dist-url 'tcp://localhost:10001' \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --pretrained ${SNAPSHOTS}/checkpoint_0199.pth.tar \
    ./data/ILSVRC2014/Img-100/ &> lincls_log
mv checkpoint.pth.tar model_best.pth.tar lincls_log ${SNAPSHOTS}/lincls

rm *.pth.tar
