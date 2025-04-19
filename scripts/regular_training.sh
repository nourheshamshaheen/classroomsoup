#!/bin/bash

# ========== USER-DEFINED VARIABLES ==========
DATASET="CIFAR100"
BATCH_SIZE=128
MODEL_NAME="resnet50"
NUM_CLASSES=100
LR=1e-04
MAX_EPOCHS=200
SAVE_PATH="./checkpoints"
EXPERIMENT_NAME="regular_training_${DATASET}_${MODEL_NAME}_lr${LR}_b${BATCH_SIZE}"
# ============================================

# Run regular training
python src/reg_train.py \
  --dataset $DATASET \
  --batch_size $BATCH_SIZE \
  --model_name $MODEL_NAME \
  --num_classes $NUM_CLASSES \
  --lr $LR \
  --max_epochs $MAX_EPOCHS \
  --save_path $SAVE_PATH \
  --experiment_name "$EXPERIMENT_NAME" \
  --use_comet --pretrained
