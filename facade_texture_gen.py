#!/usr/bin/env python3
"""
facade_texture_gen.py
----------------------
Stage 2 of the pipeline: takes the city.glb produced by osm_to_gltf.py and
gives every building a real facade texture instead of flat gray.

Two-step approach per building:
  1. Procedural control image: draw a window/floor grid based on the
     building's actual width and height (from its footprint + level count).
     This is deterministic and testable offline (no GPU/network needed).
  2. AI stylization pass: run that control image through a local Stable
     Diffusion img2img pipeline with a style prompt, so the flat grid
     becomes a plausible facade (materials, weathering, window variation)
     while keeping the window layout the img2img was conditioned on.

Install deps (Arch, venv from the previous step):
    pip install torch diffusers transformers accelerate trimesh pillow

    GPU notes for Arch:
    - NVIDIA: install torch with CUDA support (check pytorch.org for the
      current command — Arch's official torch package is CPU-only).
    - AMD: install ROCm-enabled torch (pytorch.org has a rocm build), or
      run on CPU with --cpu (slow, but works for testing the pipeline).

Usage:
    python facade_texture_gen.py --in city.glb --out city_textured.glb \\
        --style "weathered concrete apartment block, Bengaluru street"

    # test the control-image + UV logic without touching the GPU:
    python facade_texture_gen.py --in city.glb --out city_textured.glb --skip-ai
"""

import argparse
import sys

import numpy as np
from PIL import Image, ImageDraw
import trimesh

TEXTURE_PX_PER_METER = 32   # control image resolution
MIN_TEXTURE_SIZE = 64
MAX_TEXTURE_SIZE = 768       # cap so SD img2img stays fast on modest GPUs
WINDOW_MARGIN_FRAC = 0.15    # inset of each window within its grid cell


def mesh_footprint_dims(mesh):
    """Approximate a building's (width, height) in meters from its bounds."""
    bounds = mesh.bounds  # [[minx,miny,minz],[maxx,maxy,maxz]]
    dx = bounds[1][0] - bounds[0][0]
    dy = bounds[1][1] - bounds[0][1]
    dz = bounds[1][2] - bounds[0][2]
    width = max(dx, dy)   # widest horizontal footprint edge, used as facade width
    height = dz
    return width, height


def generate_control_image(width_m, height_m, level_height_m=3.0):
    """
    Draw a simple window/floor grid scaled to the building's real
    proportions. This becomes the img2img conditioning image, so the
    final AI-generated facade keeps correct window placement and floor
    count instead of hallucinating a random layout.
    """
    px_w = int(np.clip(width_m * TEXTURE_PX_PER_METER, MIN_TEXTURE_SIZE, MAX_TEXTURE_SIZE))
    px_h = int(np.clip(height_m * TEXTURE_PX_PER_METER, MIN_TEXTURE_SIZE, MAX_TEXTURE_SIZE))

    img = Image.new("RGB", (px_w, px_h), color=(180, 175, 165))  # base wall tone
    draw = ImageDraw.Draw(img)

    n_floors = max(1, round(height_m / level_height_m))
    floor_px = px_h / n_floors
    n_windows_per_floor = max(1, round(width_m / 2.5))  # ~1 window per 2.5m of facade
    window_px = px_w / n_windows_per_floor

    for f in range(n_floors):
        y0 = f * floor_px
        y1 = (f + 1) * floor_px
        wy0 = y0 + floor_px * WINDOW_MARGIN_FRAC
        wy1 = y1 - floor_px * WINDOW_MARGIN_FRAC
        for w in range(n_windows_per_floor):
            x0 = w * window_px
            x1 = (w + 1) * window_px
            wx0 = x0 + window_px * WINDOW_MARGIN_FRAC
            wx1 = x1 - window_px * WINDOW_MARGIN_FRAC
            if wx1 > wx0 and wy1 > wy0:
                draw.rectangle([wx0, wy0, wx1, wy1], fill=(90, 110, 120))

    return img


def build_box_uvs(mesh):
    """
    Simple box/planar UV projection: pick the dominant facing axis per
    face (via normal) and project vertices onto the other two axes.
    Good enough for blockout-style extruded buildings; not a substitute
    for a real unwrap on complex geometry.
    """
    verts = mesh.vertices
    normals = mesh.face_normals
    uvs = np.zeros((len(verts), 2))
    counts = np.zeros(len(verts))

    for face_idx, face in enumerate(mesh.faces):
        n = normals[face_idx]
        dominant = np.argmax(np.abs(n))  # 0=x,1=y,2=z is "up", so walls dominate x or y
        for vi in face:
            v = verts[vi]
            if dominant == 2:
                u, w = v[0], v[1]   # top/bottom faces: project to xy
            elif dominant == 0:
                u, w = v[1], v[2]   # x-facing walls: project to yz
            else:
                u, w = v[0], v[2]   # y-facing walls: project to xz
            uvs[vi] += [u, w]
            counts[vi] += 1

    counts[counts == 0] = 1
    uvs /= counts[:, None]

    # normalize to 0-1 range
    uv_min, uv_max = uvs.min(axis=0), uvs.max(axis=0)
    span = np.where(uv_max - uv_min > 1e-6, uv_max - uv_min, 1.0)
    uvs = (uvs - uv_min) / span
    return uvs


def stylize_with_ai(control_img, style_prompt, device="cuda", lora_path=None):
    """
    Run the control image through a local Stable Diffusion img2img pass.
    Imports are deferred so --skip-ai works without torch/diffusers installed.

    lora_path: optional path to a LoRA fine-tuned via train_facade_lora.sh.
    Using one noticeably improves style consistency vs. the generic base
    model — worth doing once you've settled on a target aesthetic.
    """
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline

    pipe = stylize_with_ai._pipe
    loaded_lora = stylize_with_ai._loaded_lora
    if pipe is None or loaded_lora != lora_path:
        model_id = "runwayml/stable-diffusion-v1-5"
        dtype = torch.float16 if device != "cpu" else torch.float32
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype)
        pipe = pipe.to(device)
        if lora_path:
            pipe.load_lora_weights(lora_path)
            print(f"Loaded LoRA weights from {lora_path}")
        stylize_with_ai._pipe = pipe
        stylize_with_ai._loaded_lora = lora_path

    prompt = f"{style_prompt}, building facade texture, photorealistic, flat orthographic view"
    result = pipe(
        prompt=prompt,
        image=control_img.resize((512, 512)),
        strength=0.55,   # keep window layout from control image, restyle materials
        guidance_scale=7.5,
    ).images[0]
    return result.resize(control_img.size)


stylize_with_ai._pipe = None
stylize_with_ai._loaded_lora = None


def process_scene(in_path, out_path, style_prompt, skip_ai=False, device="cuda", lora_path=None):
    scene = trimesh.load(in_path)
    if not isinstance(scene, trimesh.Scene):
        sys.exit("Input file did not load as a trimesh Scene — check the .glb path.")

    n_processed = 0
    for name, geom in list(scene.geometry.items()):
        if not name.startswith("building_"):
            continue

        width_m, height_m = mesh_footprint_dims(geom)
        control_img = generate_control_image(width_m, height_m)

        if skip_ai:
            final_img = control_img
        else:
            final_img = stylize_with_ai(control_img, style_prompt, device=device, lora_path=lora_path)

        uvs = build_box_uvs(geom)
        geom.visual = trimesh.visual.texture.TextureVisuals(
            uv=uvs, image=final_img
        )
        n_processed += 1
        print(f"Textured {name} ({width_m:.1f}m x {height_m:.1f}m)")

    print(f"Total buildings textured: {n_processed}")
    scene.export(out_path)
    print(f"Exported {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Add facade textures to a city.glb")
    parser.add_argument("--in", dest="in_path", required=True, help="input .glb from osm_to_gltf.py")
    parser.add_argument("--out", required=True, help="output .glb with textured buildings")
    parser.add_argument("--style", default="weathered concrete residential block",
                         help="style prompt for the AI stylization pass")
    parser.add_argument("--skip-ai", action="store_true",
                         help="only apply the procedural control image (no GPU/model needed) — "
                              "use this to test UV mapping and window-grid logic first")
    parser.add_argument("--cpu", action="store_true", help="run Stable Diffusion on CPU (slow)")
    parser.add_argument("--lora", default=None,
                         help="path to LoRA weights from train_facade_lora.sh, for a "
                              "style fine-tuned on your own facade photos")
    args = parser.parse_args()

    device = "cpu" if args.cpu else "cuda"
    process_scene(args.in_path, args.out, args.style, skip_ai=args.skip_ai, device=device,
                  lora_path=args.lora)


if __name__ == "__main__":
    main()
