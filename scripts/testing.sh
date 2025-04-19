#!/usr/bin/env bash

# ========== USER-DEFINED VARIABLES ==========
CHECKPOINT="[PUT_CHECKPOINT_PATH_HERE]"   # ← Set your trained model checkpoint path here
MODEL_NAME="resnet50"
DATASET="CIFAR100"
NUM_CLASSES=100
# ============================================

# Run test
python src/test.py \
  --checkpoint "$CHECKPOINT" \
  --model_name $MODEL_NAME \
  --dataset $DATASET \
  --num_classes $NUM_CLASSES
