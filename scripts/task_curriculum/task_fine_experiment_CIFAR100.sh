#!/bin/bash
# Task Curriculum Experiment - Stage 2 (Fine)
# This script fine-tunes a model from a coarse checkpoint.
# The experiment name is enriched with hyperparameter details for logging.

# ========== USER-DEFINED VARIABLES ==========
DATASET="CIFAR100"
BATCH_SIZE=128
MODEL_NAME="vit_small_patch16_224"
NUM_CLASSES=100
LR=1e-04
MAX_EPOCHS=100
VARIANT="staged"
TASK_CURR_STAGE=2
COARSE_CHECKPOINT_PATH="[PUT_COARSE_CHECKPOINT_FILENAME_HERE]"
SAVE_PATH="./checkpoints/task_curriculum"
EXPERIMENT_NAME="task_fine_frombest100_${DATASET}_${MODEL_NAME}_${VARIANT}_b${BATCH_SIZE}_lr${LR}_ep${MAX_EPOCHS}_finetune"
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
  --prev_checkpoint "${SAVE_PATH}/${COARSE_CHECKPOINT_PATH}" \
  --save_path $SAVE_PATH \
  --experiment_name "$EXPERIMENT_NAME" \
  --use_comet \
  --val_indices_path $VAL_INDICES_PATH \
  --every_k $EVERY_K
