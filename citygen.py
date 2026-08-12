#!/usr/bin/env python3
"""
citygen.py
----------
Single entry point for indie devs: bbox in, textured .glb + terrain +
attribution file out. Wraps osm_to_gltf.py (geometry) and
facade_texture_gen.py (AI facades) into one command with sane defaults,
style presets, and automatic GPU detection so it degrades gracefully on
machines without a good GPU instead of just failing.

Usage:
    python citygen.py --bbox 12.9716,77.5946,12.9816,77.6046 --out mycity \\
        --style downtown --terrain

    python citygen.py --list-styles

Output (in a folder named after --out):
    mycity/city.glb           -- final textured, terrain-correct scene
    mycity/CREDITS.txt        -- required OSM attribution, auto-filled
    mycity/manifest.json      -- bbox, style, generation settings used
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

STYLE_PRESETS = {
    "residential": "weathered concrete residential apartment block, muted colors",
    "downtown": "dense downtown commercial buildings, glass and concrete, signage",
    "suburban": "low-rise suburban houses, brick and siding, front gardens",
    "european": "European stone townhouse facades, shutters, tiled roofs",
    "industrial": "industrial warehouse buildings, corrugated metal, concrete",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def detect_gpu():
    """Return 'cuda', 'rocm', or None. Best-effort — never crashes the pipeline."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    # torch reports ROCm devices through the same torch.cuda.* API on Linux,
    # so the check above covers both NVIDIA and AMD when torch is installed
    # with the right backend. If torch isn't installed at all, we have no GPU path.
    return None


def write_credits(out_dir, bbox, style):
    """
    OSM's ODbL license requires attribution for any use of the data,
    including derivative 3D content. This file makes that a solved
    problem for whoever uses this tool, instead of an easy-to-miss step.
    """
    text = f"""Map data generated with citygen.

Contains information from OpenStreetMap, which is made available under
the Open Database License (ODbL): https://opendatacommons.org/licenses/odbl/

© OpenStreetMap contributors: https://www.openstreetmap.org/copyright

Bounding box used: {bbox}
Style preset: {style}
Generated: {datetime.now(timezone.utc).isoformat()}

If you ship a game or app using this data, keep this attribution
visible somewhere reasonable (credits screen, about page, or README) —
that's the one condition attached to using OSM data for free.
"""
    path = os.path.join(out_dir, "CREDITS.txt")
    with open(path, "w") as f:
        f.write(text)
    return path


def run_step(cmd, description):
    print(f"\n--- {description} ---")
    print(" ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"Step failed: {description} (exit code {result.returncode})")


def main():
    parser = argparse.ArgumentParser(description="Generate a game-ready city from a real bounding box")
    parser.add_argument("--bbox", help="min_lat,min_lon,max_lat,max_lon")
    parser.add_argument("--out", help="output folder name")
    parser.add_argument("--style", default="residential", choices=list(STYLE_PRESETS.keys()),
                         help="facade style preset")
    parser.add_argument("--terrain", action="store_true", help="use real elevation data")
    parser.add_argument("--terrain-resolution", type=int, default=20)
    parser.add_argument("--skip-ai", action="store_true",
                         help="skip AI facade pass — just the procedural window grid, "
                              "useful with no GPU or for fast iteration")
    parser.add_argument("--force-cpu", action="store_true", help="force CPU even if a GPU is found")
    parser.add_argument("--lora", default=None,
                         help="path to a LoRA fine-tuned via train_facade_lora.sh for a "
                              "custom, more consistent facade style")
    parser.add_argument("--list-styles", action="store_true", help="list available style presets and exit")
    parser.add_argument("--spec", default=None,
                         help="path to a build_spec.json from ai_director.py — overrides "
                              "--style/--terrain with the director's decisions. Manual flags "
                              "you pass explicitly still take precedence over the spec.")
    args = parser.parse_args()

    if args.spec:
        with open(args.spec) as f:
            spec = json.load(f)
        print(f"Loaded build spec from {args.spec}: style={spec['style']}, "
              f"use_terrain={spec['use_terrain']}")
        print(f"  Director reasoning: {spec.get('notes', '(none)')}")
        # spec values are defaults; only fill in what the user didn't explicitly set
        if args.style == parser.get_default("style"):
            args.style = spec["style"]
        if not args.terrain:
            args.terrain = spec["use_terrain"]
        if not args.bbox:
            args.bbox = spec.get("bbox")

    if args.list_styles:
        print("Available style presets:")
        for name, prompt in STYLE_PRESETS.items():
            print(f"  {name:12s} - {prompt}")
        return

    if not args.bbox or not args.out:
        parser.error("--bbox and --out are required (unless using --list-styles)")

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    raw_glb = os.path.join(out_dir, "city_raw.glb")
    final_glb = os.path.join(out_dir, "city.glb")

    # --- Step 1: geometry + terrain ---
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "osm_to_gltf.py"),
           "--bbox", args.bbox, "--out", raw_glb]
    if args.terrain:
        cmd += ["--terrain", "--terrain-resolution", str(args.terrain_resolution)]
    run_step(cmd, "Fetching map data and building geometry")

    # --- Step 2: facades ---
    gpu = None if args.force_cpu else detect_gpu()
    skip_ai = args.skip_ai or gpu is None
    if skip_ai and not args.skip_ai:
        print("\nNo compatible GPU detected — falling back to procedural facades "
              "(--skip-ai). Install a CUDA or ROCm build of torch for AI-textured "
              "facades, or pass --skip-ai to silence this note.")

    cmd = [sys.executable, os.path.join(SCRIPT_DIR, "facade_texture_gen.py"),
           "--in", raw_glb, "--out", final_glb,
           "--style", STYLE_PRESETS[args.style]]
    if skip_ai:
        cmd.append("--skip-ai")
    elif gpu == "cpu" or args.force_cpu:
        cmd.append("--cpu")
    if args.lora:
        cmd += ["--lora", args.lora]
    run_step(cmd, "Generating facade textures")

    # --- Step 3: attribution + manifest (always required, always automatic) ---
    credits_path = write_credits(out_dir, args.bbox, args.style)
    manifest = {
        "bbox": args.bbox,
        "style": args.style,
        "terrain": args.terrain,
        "ai_facades": not skip_ai,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Output in {out_dir}/")
    print(f"  {final_glb}")
    print(f"  {credits_path}  (keep this — it's your OSM attribution)")
    print(f"  {manifest_path}")


if __name__ == "__main__":
    main()
