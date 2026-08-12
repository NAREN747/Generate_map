#!/usr/bin/env bash
# train_facade_lora.sh
# ---------------------
# Fine-tunes a LoRA on your prepared facade dataset using diffusers'
# official training script (not reinvented here — their script is
# well-tested and handles the training loop, checkpointing, and mixed
# precision correctly).
#
# Prerequisites:
#   1. Run prepare_lora_dataset.py first to get a dataset/ folder with
#      images + metadata.jsonl
#   2. pip install diffusers[training] accelerate datasets peft
#   3. accelerate config   (run once, answer the prompts for your GPU setup)
#
# Usage:
#   ./train_facade_lora.sh <dataset_dir> <output_dir> [style_tag]
#
# Example:
#   ./train_facade_lora.sh lora_dataset/ facade_lora_output/ "bengaluru concrete facade"
#
# Training time: on a single consumer GPU (e.g. RTX 3060/4070 class),
# expect roughly 1-3 hours for a 100-300 image dataset at the defaults
# below. CPU training is not practical for this step — this one genuinely
# needs a GPU.

set -e

DATASET_DIR="${1:?Usage: ./train_facade_lora.sh <dataset_dir> <output_dir> [style_tag]}"
OUTPUT_DIR="${2:?Usage: ./train_facade_lora.sh <dataset_dir> <output_dir> [style_tag]}"
STYLE_TAG="${3:-facade texture}"

# diffusers ships this training script in their examples/ folder — download
# it once if not already present locally
TRAIN_SCRIPT="train_text_to_image_lora.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "Fetching diffusers' official LoRA training script..."
    curl -sL -o "$TRAIN_SCRIPT" \
        "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/text_to_image/train_text_to_image_lora.py"
fi

echo "Starting LoRA training..."
echo "  Dataset:    $DATASET_DIR"
echo "  Output:     $OUTPUT_DIR"
echo "  Style tag:  $STYLE_TAG"
echo ""

accelerate launch "$TRAIN_SCRIPT" \
  --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
  --train_data_dir="$DATASET_DIR" \
  --caption_column="text" \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --num_train_epochs=15 \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --checkpointing_steps=200 \
  --validation_prompt="$STYLE_TAG, building facade texture, photorealistic" \
  --seed=42 \
  --output_dir="$OUTPUT_DIR"

echo ""
echo "Done. LoRA weights in $OUTPUT_DIR/"
echo "Use them in facade_texture_gen.py by loading the LoRA into the pipeline"
echo "before generation (see load_lora_weights() note in that script)."
