#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export MODEL_NAME="${MODEL_NAME:-stabilityai/stable-diffusion-xl-base-1.0}"
export INSTANCE_DIR="${INSTANCE_DIR:-my_dataset}"
export OUTPUT_DIR="${OUTPUT_DIR:-trained-lora}"

export PLACEHOLDER_TOKEN="${PLACEHOLDER_TOKEN:-katyushkins}"
export CLASS_NAME="${CLASS_NAME:-woman}"
export LORA_RANK="${LORA_RANK:-4}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1000}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-100}"
export RESOLUTION="${RESOLUTION:-1024}"
export MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
export SEED="${SEED:-0}"
export NUM_VAL_IMGS_PER_PROMPT="${NUM_VAL_IMGS_PER_PROMPT:-1}"

export VALIDATION_PROMPTS="${VALIDATION_PROMPTS:-a photo of ${PLACEHOLDER_TOKEN} ${CLASS_NAME}#portrait photo of ${PLACEHOLDER_TOKEN} ${CLASS_NAME}#${PLACEHOLDER_TOKEN} ${CLASS_NAME} driving a car#${PLACEHOLDER_TOKEN} ${CLASS_NAME} swimming in a pool#${PLACEHOLDER_TOKEN} ${CLASS_NAME} dressed as a ballerina}"

if [[ ! -d "$INSTANCE_DIR" ]]; then
  echo "Dataset directory '$INSTANCE_DIR' not found. Put 3-5 photos into $INSTANCE_DIR before training."
  exit 1
fi

echo "Starting training"
echo "  model:              $MODEL_NAME"
echo "  dataset:            $INSTANCE_DIR"
echo "  output:             $OUTPUT_DIR"
echo "  placeholder_token:  $PLACEHOLDER_TOKEN"
echo "  class_name:         $CLASS_NAME"
echo "  lora_rank:          $LORA_RANK"
echo "  train_epochs:       $NUM_TRAIN_EPOCHS"
echo "  checkpoint_every:   $CHECKPOINTING_STEPS"
echo "  resolution:         $RESOLUTION"
echo "  mixed_precision:    $MIXED_PRECISION"
echo

accelerate config default >/dev/null 2>&1 || true

PYTHONUNBUFFERED=1 accelerate launch SDXL_LoRA/train.py \
  --pretrained_model_name_or_path="$MODEL_NAME" \
  --train_data_dir="$INSTANCE_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --mixed_precision="$MIXED_PRECISION" \
  --num_train_epochs="$NUM_TRAIN_EPOCHS" \
  --checkpointing_steps="$CHECKPOINTING_STEPS" \
  --resolution="$RESOLUTION" \
  --validation_prompts="$VALIDATION_PROMPTS" \
  --num_val_imgs_per_prompt="$NUM_VAL_IMGS_PER_PROMPT" \
  --placeholder_token="$PLACEHOLDER_TOKEN" \
  --class_name="$CLASS_NAME" \
  --seed="$SEED" \
  --lora_rank="$LORA_RANK"
