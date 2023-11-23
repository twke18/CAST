#!/usr/bin/bash
SNAPSHOTS=snapshots/moco/imagenet1k/vit_conv_small
mkdir -p ${SNAPSHOTS}
mkdir -p ${SNAPSHOTS}/lincls

export PYTHONPATH=moco-v3/:$PYTHONPATH

python -W ignore moco-v3/main_moco.py \
    -a vit_conv_small -b 256 \
    --optimizer=adamw --lr=1.5e-4 --weight-decay=.1 \
    --epochs=100 --warmup-epochs=10 \
    --stop-grad-conv1 --moco-m-cos --moco-t=.2 \
    --dist-url 'tcp://localhost:10001' \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    ./data/ILSVRC2014/Img/
mv checkpoint_0099.pth.tar ${SNAPSHOTS}

python -W ignore moco-v3/main_lincls.py \
    -a vit_conv_small --lr 30 -b 1024 \
    --dist-url 'tcp://localhost:10001' \
    --multiprocessing-distributed --world-size 1 --rank 0 \
    --pretrained ${SNAPSHOTS}/checkpoint_0099.pth.tar \
    ./data/ILSVRC2014/Img/ &> lincls_log
mv checkpoint.pth.tar model_best.pth.tar lincls_log ${SNAPSHOTS}/lincls

rm *.pth.tar
