#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export OUTPUT_DIR="${OUTPUT_DIR:-trained-lora}"
export CHECKPOINT_IDX="${CHECKPOINT_IDX:-latest}"
export NUM_IMAGES_PER_PROMPT="${NUM_IMAGES_PER_PROMPT:-4}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-25}"
export GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
export VERSION="${VERSION:-0}"
export SEED="${SEED:-0}"

export PLACEHOLDER_TOKEN="${PLACEHOLDER_TOKEN:-katyushkins}"
export CLASS_NAME="${CLASS_NAME:-woman}"
export PROMPTS="${PROMPTS:-a photo of ${PLACEHOLDER_TOKEN} ${CLASS_NAME}#portrait photo of ${PLACEHOLDER_TOKEN} ${CLASS_NAME} in cinematic lighting#${PLACEHOLDER_TOKEN} ${CLASS_NAME} riding a bike#${PLACEHOLDER_TOKEN} ${CLASS_NAME} as a wizard in fantasy style#${PLACEHOLDER_TOKEN} ${CLASS_NAME} in anime style#${PLACEHOLDER_TOKEN} ${CLASS_NAME} in a business suit in an office skyscraper#${PLACEHOLDER_TOKEN} ${CLASS_NAME} on a tropical beach#${PLACEHOLDER_TOKEN} ${CLASS_NAME} in a snowy forest#${PLACEHOLDER_TOKEN} ${CLASS_NAME} in cyberpunk city at night#${PLACEHOLDER_TOKEN} ${CLASS_NAME} oil painting portrait}"

CONFIG_PATH=$(find "$OUTPUT_DIR" -path "*/logs/hparams.yml" | sort | tail -n 1)

if [[ -z "${CONFIG_PATH:-}" ]]; then
  echo "Could not find any training config in '$OUTPUT_DIR'. Run training first."
  exit 1
fi

EXP_DIR=$(dirname "$(dirname "$CONFIG_PATH")")

if [[ "$CHECKPOINT_IDX" == "latest" ]]; then
  CHECKPOINT_IDX=$(find "$EXP_DIR" -maxdepth 1 -type d -name "checkpoint-*" | sed 's|.*/checkpoint-||' | sort -n | tail -n 1)
fi

if [[ -z "${CHECKPOINT_IDX:-}" ]]; then
  echo "Could not find checkpoints inside '$EXP_DIR'."
  exit 1
fi

echo "Starting inference"
echo "  config:              $CONFIG_PATH"
echo "  checkpoint_idx:      $CHECKPOINT_IDX"
echo "  num_images_prompt:   $NUM_IMAGES_PER_PROMPT"
echo "  batch_size:          $BATCH_SIZE"
echo "  inference_steps:     $NUM_INFERENCE_STEPS"
echo "  guidance_scale:      $GUIDANCE_SCALE"
echo

PYTHONUNBUFFERED=1 python SDXL_LoRA/inference.py \
  --config_path="$CONFIG_PATH" \
  --checkpoint_idx="$CHECKPOINT_IDX" \
  --prompts="$PROMPTS" \
  --num_images_per_prompt="$NUM_IMAGES_PER_PROMPT" \
  --batch_size="$BATCH_SIZE" \
  --num_inference_steps="$NUM_INFERENCE_STEPS" \
  --guidance_scale="$GUIDANCE_SCALE" \
  --version="$VERSION" \
  --seed="$SEED"
