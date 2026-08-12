#!/usr/bin/env python3
"""
prepare_lora_dataset.py
------------------------
Prepares a folder of real facade photos into the format diffusers' LoRA
training script expects: resized/cropped images + a captions file.

This is the data-curation step for closing the visual gap — a LoRA
fine-tuned on real photos of your target architectural style will produce
far more consistent, style-matched facades than the generic base SD model
we started with.

Where to get source photos (you curate these — not automated here):
  - Your own photos of the target city/style (best option — matches your
    actual bboxes)
  - Wikimedia Commons (search "building facade <city name>", CC-licensed)
  - Flickr, filtered to Creative Commons license
  - Openly-licensed architecture photo datasets (e.g. some academic
    facade-parsing datasets are released for research/commercial reuse —
    check each dataset's license before using)

Aim for 100-300 images of one consistent style for a good LoRA — more
isn't always better if the style is inconsistent; consistency matters
more than volume for LoRA fine-tunes.

Usage:
    python prepare_lora_dataset.py --in raw_photos/ --out lora_dataset/ \\
        --style-tag "bengaluru concrete residential facade"
"""

import argparse
import os
import sys
import json

from PIL import Image

TARGET_SIZE = 512   # standard SD training resolution
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def resize_and_crop(img, size=TARGET_SIZE):
    """Center-crop to square, then resize — standard SD training prep."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.LANCZOS)
    return img.convert("RGB")


def process_dataset(in_dir, out_dir, style_tag):
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(in_dir)
             if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]

    if not files:
        sys.exit(f"No images found in {in_dir} (looked for {VALID_EXTENSIONS})")

    if len(files) < 50:
        print(f"Warning: only {len(files)} images found. LoRA fine-tunes typically "
              f"want 100-300 consistent images — results may be weak or overfit "
              f"below ~50. Consider gathering more before training.")

    captions = {}
    n_ok, n_failed = 0, 0

    for fname in files:
        try:
            path = os.path.join(in_dir, fname)
            img = Image.open(path)
            processed = resize_and_crop(img)

            out_name = f"facade_{n_ok:04d}.png"
            processed.save(os.path.join(out_dir, out_name))

            # caption = style tag applied uniformly; this is what teaches the
            # LoRA to associate the tag with this visual style during training
            captions[out_name] = style_tag

            n_ok += 1
        except Exception as e:
            print(f"  Skipped {fname}: {e}")
            n_failed += 1

    # diffusers' train_text_to_image_lora.py expects a metadata.jsonl
    # (one {"file_name": ..., "text": ...} per line) alongside the images
    metadata_path = os.path.join(out_dir, "metadata.jsonl")
    with open(metadata_path, "w") as f:
        for fname, caption in captions.items():
            f.write(json.dumps({"file_name": fname, "text": caption}) + "\n")

    print(f"\nProcessed: {n_ok} images ({n_failed} skipped)")
    print(f"Output: {out_dir}/")
    print(f"Captions: {metadata_path}")

    if n_ok < 50:
        sys.exit(1)  # non-zero so a calling script/CI can catch "dataset too small"


def main():
    parser = argparse.ArgumentParser(description="Prepare a facade photo dataset for LoRA training")
    parser.add_argument("--in", dest="in_dir", required=True, help="folder of raw source photos")
    parser.add_argument("--out", required=True, help="output folder for the prepared dataset")
    parser.add_argument("--style-tag", required=True,
                         help='caption applied to every image, e.g. '
                              '"bengaluru concrete residential facade" — '
                              'use this exact phrase later as your generation style prompt')
    args = parser.parse_args()

    process_dataset(args.in_dir, args.out, args.style_tag)


if __name__ == "__main__":
    main()
