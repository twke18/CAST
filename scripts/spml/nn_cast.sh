#!/bin/bash
# This script is used for inference and benchmarking
# the pre-trained CAST on PASCAL VOC 2012 using 
# nearest neighbor search. Users could also modify from
# this script for their use case.
#

# Set up parameters for network.
BACKBONE_TYPES=cast_small_p512_retrieval
EMBEDDING_DIM=256

# Set up parameters for training.
PREDICTION_TYPES=segsort
TRAIN_SPLIT=train+
GPUS=0,1,2,3
LR_POLICY=poly
USE_SYNCBN=true
SNAPSHOT_STEP=20000
MAX_ITERATION=20000
WARMUP_ITERATION=100
LR=3e-3
WD=5e-4
BATCH_SIZE=4
CROP_SIZE=512
MEMORY_BANK_SIZE=0
KMEANS_ITERATIONS=15
KMEANS_NUM_CLUSTERS=4
SEM_ANN_LOSS_TYPES=segsort # segsort / none
SEM_OCC_LOSS_TYPES=none # segsort / none
IMG_SIM_LOSS_TYPES=none # segsort / none
FEAT_AFF_LOSS_TYPES=none # segsort / none
SEM_ANN_CONCENTRATION=0
SEM_OCC_CONCENTRATION=0
IMG_SIM_CONCENTRATION=0
FEAT_AFF_CONCENTRATION=0
SEM_ANN_LOSS_WEIGHT=1.0
SEM_OCC_LOSS_WEIGHT=0.0
IMG_SIM_LOSS_WEIGHT=0.0
FEAT_AFF_LOSS_WEIGHT=0.0

# Set up parameters for inference.
INFERENCE_SPLIT=val
INFERENCE_IMAGE_SIZE=512
INFERENCE_CROP_SIZE_H=512
INFERENCE_CROP_SIZE_W=512
INFERENCE_STRIDE=512

# Set up path for saving models.
SNAPSHOT_ROOT=snapshots/moco/coco/cast_small
SNAPSHOT_DIR=${SNAPSHOT_ROOT}/nn_test
echo ${SNAPSHOT_DIR}

mkdir -p ${SNAPSHOT_DIR}
cp ${SNAPSHOT_ROOT}/checkpoint_0149.pth.tar ${SNAPSHOT_DIR}/model-$((${MAX_ITERATION}-1)).pth

# Set up the procedure pipeline.
IS_CONFIG=1
IS_INFERENCE=1

# Update PYTHONPATH.
export PYTHONPATH=SPML/:$PYTHONPATH

# Set up the data directory and file list.
DATAROOT=./data/VOCdevkit
TEST_DATA_LIST=SPML/datasets/voc12/panoptic_${INFERENCE_SPLIT}.txt
MEMORY_DATA_LIST=SPML/datasets/voc12/panoptic_${TRAIN_SPLIT}_hed.txt


# Build configuration file for training embedding network.
if [ ${IS_CONFIG} -eq 1 ]; then
  if [ ! -d ${SNAPSHOT_DIR} ]; then
    mkdir -p ${SNAPSHOT_DIR}
  fi

  sed -e "s/TRAIN_SPLIT/${TRAIN_SPLIT}/g"\
    -e "s/BACKBONE_TYPES/${BACKBONE_TYPES}/g"\
    -e "s/PREDICTION_TYPES/${PREDICTION_TYPES}/g"\
    -e "s/EMBEDDING_DIM/${EMBEDDING_DIM}/g"\
    -e "s/GPUS/${GPUS}/g"\
    -e "s/BATCH_SIZE/${BATCH_SIZE}/g"\
    -e "s/LABEL_DIVISOR/2048/g"\
    -e "s/USE_SYNCBN/${USE_SYNCBN}/g"\
    -e "s/LR_POLICY/${LR_POLICY}/g"\
    -e "s/SNAPSHOT_STEP/${SNAPSHOT_STEP}/g"\
    -e "s/MAX_ITERATION/${MAX_ITERATION}/g"\
    -e "s/WARMUP_ITERATION/${WARMUP_ITERATION}/g"\
    -e "s/LR/${LR}/g"\
    -e "s/WD/${WD}/g"\
    -e "s/MEMORY_BANK_SIZE/${MEMORY_BANK_SIZE}/g"\
    -e "s/KMEANS_ITERATIONS/${KMEANS_ITERATIONS}/g"\
    -e "s/KMEANS_NUM_CLUSTERS/${KMEANS_NUM_CLUSTERS}/g"\
    -e "s/TRAIN_CROP_SIZE/${CROP_SIZE}/g"\
    -e "s/TEST_SPLIT/${INFERENCE_SPLIT}/g"\
    -e "s/TEST_IMAGE_SIZE/${INFERENCE_IMAGE_SIZE}/g"\
    -e "s/TEST_CROP_SIZE_H/${INFERENCE_CROP_SIZE_H}/g"\
    -e "s/TEST_CROP_SIZE_W/${INFERENCE_CROP_SIZE_W}/g"\
    -e "s/TEST_STRIDE/${INFERENCE_STRIDE}/g"\
    -e "s#PRETRAINED#${PRETRAINED}#g"\
    -e "s/SEM_ANN_LOSS_TYPES/${SEM_ANN_LOSS_TYPES}/g"\
    -e "s/SEM_OCC_LOSS_TYPES/${SEM_OCC_LOSS_TYPES}/g"\
    -e "s/IMG_SIM_LOSS_TYPES/${IMG_SIM_LOSS_TYPES}/g"\
    -e "s/FEAT_AFF_LOSS_TYPES/${FEAT_AFF_LOSS_TYPES}/g"\
    -e "s/SEM_ANN_CONCENTRATION/${SEM_ANN_CONCENTRATION}/g"\
    -e "s/SEM_OCC_CONCENTRATION/${SEM_OCC_CONCENTRATION}/g"\
    -e "s/IMG_SIM_CONCENTRATION/${IMG_SIM_CONCENTRATION}/g"\
    -e "s/FEAT_AFF_CONCENTRATION/${FEAT_AFF_CONCENTRATION}/g"\
    -e "s/SEM_ANN_LOSS_WEIGHT/${SEM_ANN_LOSS_WEIGHT}/g"\
    -e "s/SEM_OCC_LOSS_WEIGHT/${SEM_OCC_LOSS_WEIGHT}/g"\
    -e "s/IMG_SIM_LOSS_WEIGHT/${IMG_SIM_LOSS_WEIGHT}/g"\
    -e "s/FEAT_AFF_LOSS_WEIGHT/${FEAT_AFF_LOSS_WEIGHT}/g"\
    SPML/configs/voc12_template.yaml > ${SNAPSHOT_DIR}/config.yaml

  cat ${SNAPSHOT_DIR}/config.yaml
fi


# Train for the embedding.
if [ ${IS_INFERENCE} -eq 1 ]; then
  python3 SPML/pyscripts/inference/prototype_suppix.py\
    --data_dir ${DATAROOT}\
    --data_list ${MEMORY_DATA_LIST}\
    --snapshot_dir ${SNAPSHOT_DIR}\
    --save_dir ${SNAPSHOT_DIR}/results/${TRAIN_SPLIT} \
    --kmeans_num_clusters 6,6\
    --label_divisor 2048\
    --cfg_path ${SNAPSHOT_DIR}/config.yaml

  python3 SPML/pyscripts/inference/inference_suppix.py\
    --data_dir ${DATAROOT}\
    --data_list ${TEST_DATA_LIST}\
    --snapshot_dir ${SNAPSHOT_DIR}\
    --save_dir ${SNAPSHOT_DIR}/results/${INFERENCE_SPLIT} \
    --semantic_memory_dir ${SNAPSHOT_DIR}/results/${TRAIN_SPLIT}/semantic_prototype\
    --kmeans_num_clusters 6,6\
    --label_divisor 2048\
    --cfg_path ${SNAPSHOT_DIR}/config.yaml

  python3 SPML/pyscripts/benchmark/benchmark_by_mIoU.py\
    --pred_dir ${SNAPSHOT_DIR}/results/${INFERENCE_SPLIT}/semantic_gray\
    --gt_dir ${DATAROOT}/VOC2012/segcls\
    --num_classes 21
fi
