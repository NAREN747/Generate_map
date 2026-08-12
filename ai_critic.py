#!/usr/bin/env python3
"""
ai_critic.py
------------
Automated art-direction pass. Renders each generated building, sends it to
a vision-capable model for critique against a scoring rubric, and drives a
regenerate loop for anything that scores below threshold — instead of a
single-shot "generate and hope" pipeline.

This is the closest thing this pipeline has to a human art reviewer: it
won't invent detail a human artist would, but it catches and fixes the
failure modes that would otherwise ship silently (misaligned windows,
style drift, texture stretching, a building that just looks wrong).

Requires a vision-capable model behind either Anthropic's API, or ANY
OpenAI-compatible chat completions endpoint (NVIDIA NIM, OpenAI itself,
OpenRouter, a local vLLM/Ollama server — anything speaking the same
image_url content format). You are not locked into one provider.

Rendering uses trimesh's offscreen renderer (pyglet backend) —
install with: pip install pyglet

This also builds persistent cross-run memory (citygen_memory.py): every
critique gets logged, and recurring issues automatically get folded into
future generation prompts via a "learned suffix" — the tool gets better
at your specific style the more you run it, without retraining anything.

Usage:
    # Anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    python ai_critic.py --in city_textured.glb --out city_final.glb \\
        --style "weathered concrete residential" --provider anthropic

    # Any OpenAI-compatible endpoint (NIM/Nemotron shown, but this works
    # for literally any provider with a chat completions + vision API)
    export NVIDIA_API_KEY=nvapi-...
    python ai_critic.py --in city_textured.glb --out city_final.glb \\
        --style "weathered concrete residential" --provider openai \\
        --api-base https://integrate.api.nvidia.com/v1/chat/completions \\
        --model nvidia/nemotron-3-ultra --api-key-env NVIDIA_API_KEY

    # a local model server, e.g. Ollama with a vision model:
    python ai_critic.py --in city_textured.glb --out city_final.glb \\
        --style "..." --provider openai \\
        --api-base http://localhost:11434/v1/chat/completions \\
        --model llava --api-key-env OLLAMA_API_KEY
"""

import argparse
import base64
import json
import os
import sys

import requests
import trimesh
from PIL import Image

import citygen_memory as memory_store
import quality_metrics as qm

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-4-5"   # vision-capable

# Convenience presets so common providers don't need --api-base/--model
# spelled out every time — still just filling in defaults for the generic
# OpenAI-compatible path, nothing provider-specific under the hood.
PROVIDER_PRESETS = {
    "nemotron": {
        "api_base": "https://integrate.api.nvidia.com/v1/chat/completions",
        "model": "nvidia/nemotron-3-ultra",
        "api_key_env": "NVIDIA_API_KEY",
    },
}

RUBRIC_PROMPT = """You are an art director reviewing a procedurally generated
building facade for a video game. Rate it and identify concrete issues.

Target style: {style}

Respond ONLY with JSON, no other text, in this exact format:
{{
  "score": <integer 1-10, 10 being AAA-quality art-directed facade>,
  "issues": [<short strings, e.g. "windows misaligned with floor lines", "texture looks flat/2D", "color doesn't match target style", "stretching visible on left edge">],
  "regenerate": <true if score < 7, false otherwise>,
  "prompt_adjustment": <short phrase to append to the generation prompt to fix the biggest issue, or null if regenerate is false>
}}
"""

# Objective-metric fail thresholds — independent of the AI's subjective
# score, so a vision model being fooled by a texture that "reads okay" at
# a glance doesn't let an actually-broken generation through. These are
# deliberately conservative (only catch clear failures, not style taste).
SHARPNESS_FAIL_THRESHOLD = 50.0      # below this, texture is suspiciously flat/blank
UV_STRETCH_PCT_FAIL_THRESHOLD = 40.0  # % of faces over stretch_threshold before flagging


def compute_objective_metrics(geom):
    """
    Pull the objective (non-AI) quality metrics for a textured building
    mesh. Returns None if the mesh has no texture yet (e.g. --skip-ai
    procedural-only mode) — objective texture metrics don't apply there.
    """
    visual = getattr(geom, "visual", None)
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    image = getattr(material, "image", None) if material is not None else None

    if uv is None or image is None:
        return None

    return qm.evaluate_building(
        vertices=geom.vertices,
        faces=geom.faces,
        uvs=uv,
        texture_image=image,
        mesh_surface_area_m2=geom.area,
    )


def objective_metrics_flag_issues(metrics):
    """
    Translate raw objective metrics into the same issues/regenerate shape
    the AI critique produces, so both feed into one decision. Only fires
    on clear failures — this is a safety net, not a taste judge.
    """
    if metrics is None:
        return [], False

    issues = []
    force_regenerate = False

    sharpness = metrics.get("texture_sharpness")
    if sharpness is not None and sharpness < SHARPNESS_FAIL_THRESHOLD:
        issues.append(f"objective: texture sharpness very low ({sharpness:.1f}) — "
                       f"likely a failed or degenerate generation")
        force_regenerate = True

    uv = metrics.get("uv_stretch", {})
    pct_over = uv.get("pct_over_threshold")
    if pct_over is not None and pct_over > UV_STRETCH_PCT_FAIL_THRESHOLD:
        issues.append(f"objective: {pct_over:.0f}% of faces have significant UV stretch — "
                       f"texture will visibly distort")
        force_regenerate = True

    return issues, force_regenerate


def render_building(mesh, out_path, resolution=(512, 512)):
    """
    Render a single building mesh to a PNG using trimesh's offscreen
    renderer. Falls back with a clear error if the rendering backend
    (pyglet) isn't available — this is an optional-but-recommended dep,
    not part of the core pipeline.
    """
    scene = trimesh.Scene(mesh)
    try:
        png_bytes = scene.save_image(resolution=resolution)
    except Exception as e:
        sys.exit(
            f"Rendering failed ({e}). This step needs pyglet installed "
            f"(pip install pyglet) and a display or headless-rendering setup. "
            f"On a headless Linux box, you may need Xvfb: "
            f"sudo pacman -S xorg-server-xvfb  (Arch) then run under "
            f"`xvfb-run python ai_critic.py ...`"
        )
    with open(out_path, "wb") as f:
        f.write(png_bytes)
    return out_path


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def critique_image_anthropic(image_path, style_prompt, api_key):
    """Send a rendered building to Claude's vision API and parse its structured critique."""
    img_b64 = image_to_base64(image_path)

    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 500,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                    {"type": "text", "text": RUBRIC_PROMPT.format(style=style_prompt)},
                ],
            }
        ],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    text = "".join(block["text"] for block in data["content"] if block["type"] == "text")
    return _parse_critique_json(text)


def critique_image_openai_compatible(image_path, style_prompt, api_key, api_base, model):
    """
    Generic OpenAI-compatible chat completions call with vision input.
    Works against NVIDIA NIM, OpenAI, OpenRouter, a local vLLM/Ollama
    server, or anything else speaking this widely-adopted format —
    the tool isn't tied to any single provider.
    """
    img_b64 = image_to_base64(image_path)

    payload = {
        "model": model,
        "max_tokens": 500,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": RUBRIC_PROMPT.format(style=style_prompt)},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(api_base, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    text = data["choices"][0]["message"]["content"]
    return _parse_critique_json(text)


def _parse_critique_json(text):
    """Shared parsing logic — strip markdown fences, parse JSON, fail safe on garbage."""
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"  Warning: couldn't parse critique response as JSON, treating as pass. Raw: {text[:200]}")
        return {"score": 10, "issues": [], "regenerate": False, "prompt_adjustment": None}


def critique_image(image_path, style_prompt, api_key, provider="anthropic", api_base=None, model=None):
    if provider == "anthropic":
        return critique_image_anthropic(image_path, style_prompt, api_key)
    return critique_image_openai_compatible(image_path, style_prompt, api_key, api_base, model)


def resolve_provider_config(provider, api_base, model, api_key_env):
    """
    Fold preset shortcuts (like --provider nemotron) into the generic
    openai-compatible path, and figure out which env var to read the key
    from. Returns (provider, api_key, api_base, model).
    """
    if provider in PROVIDER_PRESETS:
        preset = PROVIDER_PRESETS[provider]
        api_base = api_base or preset["api_base"]
        model = model or preset["model"]
        api_key_env = api_key_env or preset["api_key_env"]
        provider = "openai"
    elif provider == "anthropic":
        api_key_env = api_key_env or "ANTHROPIC_API_KEY"
    else:  # generic "openai" provider with no preset
        if not api_base or not model:
            sys.exit("--provider openai requires --api-base and --model "
                      "(or use a preset like --provider nemotron)")
        api_key_env = api_key_env or "OPENAI_API_KEY"

    api_key = os.environ.get(api_key_env)
    if not api_key:
        sys.exit(f"Set {api_key_env} in your environment first.")

    return provider, api_key, api_base, model


def run_critique_loop(glb_path, out_path, style_prompt, threshold=7, max_iterations=2,
                       render_dir="critique_renders", provider="anthropic",
                       api_base=None, model=None, api_key_env=None,
                       use_memory=True, memory_path=memory_store.DEFAULT_MEMORY_PATH):
    provider, api_key, api_base, model = resolve_provider_config(provider, api_base, model, api_key_env)

    memory = memory_store.load_memory(memory_path) if use_memory else {}
    style_key = style_prompt  # bucket learned history by the exact style prompt used

    learned_suffix = memory_store.get_learned_suffix(memory, style_key) if use_memory else ""
    if learned_suffix:
        print(f"Applying learned suffix from past runs: \"{learned_suffix}\"")
        style_prompt = f"{style_prompt}, {learned_suffix}"

    os.makedirs(render_dir, exist_ok=True)
    scene = trimesh.load(glb_path)
    if not isinstance(scene, trimesh.Scene):
        sys.exit("Input did not load as a trimesh Scene.")

    report = {}
    changelog = []

    for iteration in range(1, max_iterations + 1):
        print(f"\n=== Critique pass {iteration}/{max_iterations} ===")
        any_regenerated = False

        for name, geom in list(scene.geometry.items()):
            if not name.startswith("building_"):
                continue

            prev_entry = report.get(name)  # from last iteration, if any

            render_path = os.path.join(render_dir, f"{name}_iter{iteration}.png")
            render_building(geom, render_path)

            effective_style = style_prompt
            if prev_entry and prev_entry.get("prompt_adjustment"):
                effective_style += ", " + prev_entry["prompt_adjustment"]

            ai_critique = critique_image(render_path, effective_style, api_key,
                                          provider=provider, api_base=api_base, model=model)

            objective = compute_objective_metrics(geom)
            objective_issues, objective_forces_regen = objective_metrics_flag_issues(objective)

            # merge: objective metrics can only ever push toward "needs work",
            # never override the AI critique into a false pass
            combined_issues = ai_critique.get("issues", []) + objective_issues
            combined_regenerate = ai_critique.get("regenerate", False) or objective_forces_regen

            critique = {
                **ai_critique,
                "issues": combined_issues,
                "regenerate": combined_regenerate,
                "objective_metrics": objective,
            }
            report[name] = critique

            print(f"  {name}: score={critique['score']}/10  issues={combined_issues}")

            # --- changelog entry: what actually changed vs. last iteration ---
            change_entry = {
                "iteration": iteration,
                "building": name,
                "score": critique["score"],
                "score_delta": (critique["score"] - prev_entry["score"]) if prev_entry else None,
                "prompt_adjustment_applied": prev_entry.get("prompt_adjustment") if prev_entry else None,
                "new_issues": [i for i in combined_issues if not prev_entry or i not in prev_entry.get("issues", [])],
                "resolved_issues": [i for i in prev_entry.get("issues", []) if prev_entry and i not in combined_issues] if prev_entry else [],
            }
            changelog.append(change_entry)

            if use_memory:
                memory = memory_store.record_critique(memory, style_key, name, critique)

            if combined_regenerate and critique["score"] < threshold:
                # NOTE: actual mesh/texture regeneration would call back into
                # facade_texture_gen.py's stylize_with_ai() here, passing
                # effective_style + critique['prompt_adjustment']. Left as
                # an integration point rather than duplicated here — wire
                # this to your facade_texture_gen import once both scripts
                # are in the same run.
                any_regenerated = True

        if not any_regenerated:
            print("\nAll buildings meet threshold. Stopping early.")
            break

    scene.export(out_path)

    if use_memory:
        memory_store.save_memory(memory, memory_path)
        print(f"\nMemory updated: {memory_path} (run citygen_memory.py to see what's been learned)")

    report_path = os.path.join(render_dir, "critique_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    changelog_path = os.path.join(render_dir, "changelog.json")
    with open(changelog_path, "w") as f:
        json.dump(changelog, f, indent=2)

    print(f"\n--- Changelog summary ---")
    for entry in changelog:
        if entry["score_delta"] is not None:
            direction = "improved" if entry["score_delta"] > 0 else ("worsened" if entry["score_delta"] < 0 else "unchanged")
            print(f"  {entry['building']} (iter {entry['iteration']}): {direction} "
                  f"({entry['score_delta']:+d}) after applying \"{entry['prompt_adjustment_applied']}\"")
            if entry["resolved_issues"]:
                print(f"    fixed: {entry['resolved_issues']}")
            if entry["new_issues"]:
                print(f"    new: {entry['new_issues']}")

    avg_score = sum(r["score"] for r in report.values()) / max(len(report), 1)
    print(f"\nFinal average score: {avg_score:.1f}/10 across {len(report)} buildings")
    print(f"Full report: {report_path}")
    print(f"Changelog: {changelog_path}")
    print(f"Output: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="AI-driven quality critique and regeneration loop")
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--style", required=True)
    parser.add_argument("--threshold", type=int, default=7, help="min acceptable score (1-10)")
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--provider", default="anthropic",
                         help="'anthropic', a preset ('nemotron'), or 'openai' for any generic "
                              "OpenAI-compatible endpoint (requires --api-base and --model)")
    parser.add_argument("--api-base", default=None, help="endpoint URL, for --provider openai")
    parser.add_argument("--model", default=None, help="model name/slug, for --provider openai")
    parser.add_argument("--api-key-env", default=None,
                         help="env var name to read the API key from (default depends on provider)")
    parser.add_argument("--no-memory", action="store_true",
                         help="don't read/write citygen_memory.json — treat this as a fresh run")
    parser.add_argument("--memory-path", default=memory_store.DEFAULT_MEMORY_PATH)
    args = parser.parse_args()

    run_critique_loop(args.in_path, args.out, args.style,
                       threshold=args.threshold, max_iterations=args.max_iterations,
                       provider=args.provider, api_base=args.api_base, model=args.model,
                       api_key_env=args.api_key_env,
                       use_memory=not args.no_memory, memory_path=args.memory_path)


if __name__ == "__main__":
    main()
