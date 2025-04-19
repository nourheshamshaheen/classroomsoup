#!/bin/bash
# Task Curriculum Experiment - Stage 1 (Coarse)
# This script trains a model using coarse labels.
# Experiment name includes hyperparameter values for logging.

# ========== USER-DEFINED VARIABLES ==========
DATASET="CIFAR100"
BATCH_SIZE=128
MODEL_NAME="vit_small_patch16_224"
NUM_CLASSES=20               # Coarse stage: fewer classes
LR=1e-04
MAX_EPOCHS=100
VARIANT="staged"
TASK_CURR_STAGE=1
SAVE_PATH="./checkpoints/task_curriculum"
EXPERIMENT_NAME="task_coarse_${DATASET}_${MODEL_NAME}_${VARIANT}_b${BATCH_SIZE}_lr${LR}_ep${MAX_EPOCHS}"
VAL_INDICES_PATH="./split/cifar100_val_split.npz"
EVERY_K=5
# ============================================

# Run experiment
python src/main.py \
  --curriculum_type task \
  --dataset $DATASET \
  --batch_size $BATCH_SIZE \
  --model_name $MODEL_NAME \
  --num_classes $NUM_CLASSES \
  --lr $LR \
  --max_epochs $MAX_EPOCHS \
  --variant $VARIANT \
  --task_curr_stage $TASK_CURR_STAGE \
  --save_path $SAVE_PATH \
  --experiment_name "$EXPERIMENT_NAME" \
  --use_comet \
  --pretrained \
  --val_indices_path $VAL_INDICES_PATH \
  --every_k $EVERY_K
