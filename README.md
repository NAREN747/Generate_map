# citygen

Generate a game-ready 3D city block from real-world map data — free, local,
and runnable on a normal desktop. Built for indie developers who need a
map that isn't a stock asset pack and can't justify CityEngine or a modeling team.

Point it at a bounding box, get back a textured `.glb` with real building
footprints, real heights, optional real terrain, and AI-generated facades —
in one command.

```bash
python citygen.py --bbox 12.9716,77.5946,12.9816,77.6046 --out mycity --style downtown --terrain
```

## What it actually does

1. **Pulls real geometry from OpenStreetMap** — building footprints, heights
   (from OSM tags where available), and road networks for any bounding box
   on Earth.
2. **Extrudes it into a 3D blockout** — buildings become real 3D meshes at
   real heights, roads become correctly-widthed ribbons.
3. **(Optional) Adds real terrain** — samples elevation data and displaces
   the ground mesh, lifting buildings and roads to sit correctly on slopes
   instead of floating on a flat plane.
4. **Generates facade textures** — a procedural window/floor grid (scaled to
   each building's real proportions) optionally gets run through a local
   Stable Diffusion pass to turn it into a plausible material — concrete,
   brick, glass, whatever style you pick.
5. **Exports one `.glb`** — drop it into Unity, Unreal, Godot, or any glTF-
   compatible engine.

## Why this exists

Indie devs currently choose between: hand-modeling a city (slow), buying an
asset pack that doesn't match any real place (generic), or licensing
CityEngine/Houdini (expensive, steep learning curve). This is a free,
scriptable middle path — not a AAA-quality result, but a real, explorable,
*your-choice-of-location* starting point you can hand-tune from there.

**Honest scope**: this produces stylized/blockout-quality results, not
photorealistic AAA output. See [Limitations](#limitations) below.

## Quickstart

```bash
git clone https://github.com/yourusername/citygen.git
cd citygen

# recommended: use a venv (required on Arch and other PEP 668 distros)
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
python check_env.py          # confirms what's installed / what's missing
```

`check_env.py` tells you exactly what to install next — run it first,
especially before your first real generation.

### GPU setup (optional, for AI facade generation)

The pipeline works without a GPU (`--skip-ai` gives procedural facades).
For AI-generated facades, install torch matching your hardware **before**
`requirements.txt`'s AI section:

```bash
# NVIDIA
pip install torch --index-url https://download.pytorch.org/whl/cu121
# AMD (ROCm)
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
# CPU only (slow, but works)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install diffusers transformers accelerate
```

## Usage

```bash
# list style presets
python citygen.py --list-styles

# basic generation, no terrain, no AI (fast, works anywhere)
python citygen.py --bbox <bbox> --out mycity --skip-ai

# full pipeline
python citygen.py --bbox <bbox> --out mycity --style european --terrain
```

`--bbox` format: `min_lat,min_lon,max_lat,max_lon`. Start with a small area
(a few blocks) — the free Overpass/Open-Elevation APIs this relies on will
time out or rate-limit large requests. [bboxfinder.com](http://bboxfinder.com)
is a quick way to draw a box and get these coordinates.

**Output** (in a folder named after `--out`):
- `city.glb` — the final scene
- `CREDITS.txt` — required OpenStreetMap attribution, auto-generated
- `manifest.json` — records exactly what was generated, for reproducibility

## Style presets

| Preset | Look |
|---|---|
| `residential` | weathered concrete apartment blocks, muted tones |
| `downtown` | dense commercial, glass and concrete, signage |
| `suburban` | low-rise houses, brick/siding, front gardens |
| `european` | stone townhouses, shutters, tiled roofs |
| `industrial` | warehouses, corrugated metal, concrete |

## Going further: fine-tuning your own facade style (LoRA)

The base Stable Diffusion model gives plausible-but-generic facades. For a
noticeably more consistent, art-directed look, fine-tune a LoRA on real
photos of your target style:

```bash
# 1. Curate 100-300 photos of one consistent architectural style
#    (your own photos, Wikimedia Commons, or other CC-licensed sources —
#    see prepare_lora_dataset.py's docstring for sourcing notes)

# 2. Prepare the dataset (resizes, crops, generates captions)
python prepare_lora_dataset.py --in raw_photos/ --out lora_dataset/ \
    --style-tag "your city concrete residential facade"

# 3. Train (needs a GPU; ~1-3 hours on a consumer card for 100-300 images)
pip install diffusers[training] accelerate datasets peft
accelerate config   # one-time setup, answer prompts for your hardware
./train_facade_lora.sh lora_dataset/ facade_lora_output/ "your city concrete residential facade"

# 4. Use it in generation
python citygen.py --bbox <bbox> --out mycity --style residential \
    --lora facade_lora_output/
```

This is the single highest-leverage step for closing the gap between
"generic AI texture" and "consistent, recognizable style" — worth doing
once you've settled on the aesthetic you're targeting.

## Architecture: instructions vs. execution

Think CPU/GPU: `ai_director.py` is the only place that makes judgment
calls (what style fits this area, does terrain matter here) — everything
else just executes whatever it's told, deterministically, with no
hidden decisions of its own.

```
ai_director.py          "CPU": reads real OSM stats for a bbox, decides style/terrain,
                         writes build_spec.json — auditable, human-readable, before anything runs
       |
       v
citygen.py               "GPU": orchestrates execution of that spec exactly
├── osm_to_gltf.py        Overpass query -> footprints/roads -> extruded geometry -> terrain
└── facade_texture_gen.py UV projection -> window-grid control image -> Stable Diffusion pass
       |
       v
ai_critic.py + quality_metrics.py   reviews the result (AI + objective metrics),
                                     writes changelog.json, feeds citygen_memory.json
```

```bash
# let the director decide style/terrain from real area context
python ai_director.py --bbox 12.9716,77.5946,12.9816,77.6046
python citygen.py --spec build_spec.json --out mycity
```

`build_spec.json` is plain, readable JSON — check it before running if
you want to sanity-check the director's reasoning:
```json
{
  "style": "downtown",
  "use_terrain": false,
  "notes": "High building density and commercial tags suggest a dense downtown core; flat road grid suggests minimal elevation change.",
  "bbox": "12.9716,77.5946,12.9816,77.6046",
  "context": { "n_buildings": 340, "avg_building_height_m": 28.4, ... }
}
```

Any flag you pass to `citygen.py` explicitly still overrides the spec —
the director sets defaults, it doesn't force decisions you've already made.

Each script also runs standalone if you want to hook into just one stage —
see the docstring at the top of each file for direct usage.

## Limitations

Being upfront about these so expectations are calibrated correctly:

- **Not photorealistic.** This is stylized/blockout quality — closer to an
  indie walking-sim aesthetic than a AAA open-world game. See the facade
  pass for what "AI-textured" actually means here.
- **UV mapping is a simplified box projection**, not a true unwrap. Fine for
  boxy buildings, will show stretching on complex geometry.
- **No vegetation, props, or road-marking detail yet** — see
  [Roadmap](#roadmap).
- **No LOD generation yet** — large scenes will be heavy in VR without
  manual optimization in-engine.
- **Depends on free public APIs** (Overpass, Open-Elevation) that rate-limit
  under load — keep bounding boxes small, especially at first.

## Data licensing — read this before shipping anything

- **Map data** comes from OpenStreetMap, licensed under **ODbL**. This tool
  auto-generates a `CREDITS.txt` per output — keep that attribution visible
  in anything you ship (credits screen, README, about page). This is the
  one real obligation attached to using this tool.
- **Do not** point this pipeline at Google Maps data — Google's terms of
  service restrict extracting/reconstructing 3D content from their
  platform outside their own tools. This project only uses OSM for exactly
  that reason.
- **Code license**: MIT (see `LICENSE`) — do whatever you want with the
  code itself.
- **Stable Diffusion**: check the license of whichever checkpoint you use;
  the default (`runwayml/stable-diffusion-v1-5`) permits commercial use as
  of writing, but verify current terms if you swap models.

## Going further: AI-driven quality critique loop

Instead of one-shot generation, `ai_critic.py` renders each building,
sends it to a vision-capable model with a scoring rubric, and flags
anything below quality threshold with a specific fix suggestion — an
automated stand-in for a human art-direction pass.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python ai_critic.py --in city_textured.glb --out city_final.glb \
    --style "weathered concrete residential" --threshold 7 --max-iterations 2

# any OpenAI-compatible endpoint works — not locked to one provider.
# nemotron is a built-in preset using the NVIDIA API key your JARVIS
# setup already has:
export NVIDIA_API_KEY=nvapi-...
python ai_critic.py --in city_textured.glb --out city_final.glb \
    --style "weathered concrete residential" --provider nemotron

# or point it at literally any other OpenAI-compatible vision model —
# OpenAI, OpenRouter, a local Ollama/vLLM server, anything:
python ai_critic.py --in city_textured.glb --out city_final.glb \
    --style "..." --provider openai \
    --api-base http://localhost:11434/v1/chat/completions \
    --model llava --api-key-env OLLAMA_API_KEY
```

Needs `pyglet` for offscreen rendering (`pip install pyglet`); on a
headless Linux box you'll also need Xvfb (`sudo pacman -S xorg-server-xvfb`
on Arch, then run under `xvfb-run python ai_critic.py ...`).

Output includes `critique_report.json` — a per-building score/issues
breakdown, useful for spotting systemic problems (e.g. every building
scoring low on "texture looks flat" tells you to revisit the LoRA, not
chase individual buildings one at a time).

**Note**: the regeneration call-back into `facade_texture_gen.py` is left
as an integration point in the script rather than duplicated — wire it up
once you're running both in the same session. This script's job is the
render → critique → decide loop; the actual regeneration reuses the
existing facade pipeline.

### Objective quality metrics, not just AI opinion

`quality_metrics.py` measures two things directly from geometry and
pixels — no model call, no subjectivity:

- **UV stretch** — per-triangle distortion in the box-projection UV
  mapping. Catches texture skew a vision model might not reliably flag.
- **Texture sharpness** — an edge-response variance measure. Catches a
  failed/blank AI generation automatically, before it ships.

These run inside `ai_critic.py` alongside the AI critique and act as a
safety net — they can only push a building toward "needs regeneration,"
never override a real problem into a false pass.

### Changelog: what actually changed between passes

`ai_critic.py` writes `changelog.json` alongside the critique report — a
per-building, per-iteration diff showing the score delta, which prompt
fix was applied, which issues got resolved, and which new issues (if
any) appeared. Printed as a human-readable summary at the end of each
run, so you can see whether a fix actually helped or just traded one
problem for another.

### It gets better the more you use it

Every critique gets logged to `citygen_memory.json` (via `citygen_memory.py`),
bucketed per style. Once an issue recurs 3+ times for a given style (e.g.
"windows too small" keeps getting flagged for your `residential` preset),
it automatically gets folded into a **learned suffix** applied to every
future generation with that style — before the critique loop even runs,
not just after.

```bash
python citygen_memory.py   # see what's been learned so far, per style
```

This is deliberately simple — a frequency count over past critique
feedback, not model retraining — but it compounds: the tool makes fewer
repeat mistakes the more you run it. Delete `citygen_memory.json` any
time to reset; the pipeline works identically with no memory file, it
just starts learning from zero again. Use `--no-memory` on `ai_critic.py`
to skip this for a one-off run.

## Roadmap

- [ ] Proper UV unwrapping (`xatlas`) instead of box projection
- [ ] Procedural roof shapes from OSM `roof:shape` tags
- [ ] Vegetation/prop scattering
- [ ] LOD generation for VR performance
- [ ] Road intersection blending + lane markings

## Contributing

Issues and PRs welcome. If you're extending a pipeline stage, keep the
scripts runnable standalone (not just through `citygen.py`) — that's what
keeps this hackable for other devs' specific needs.

## License

MIT — see `LICENSE`. Map data remains under OpenStreetMap's ODbL regardless
of this repo's license; see [Data licensing](#data-licensing--read-this-before-shipping-anything).
