#!/usr/bin/env python3
"""
ai_director.py
----------------
The "CPU" in the CPU/GPU split: looks at what's actually in a bbox (from
real OSM data — building density, height distribution, land-use tag mix)
and decides how it should be built. Outputs a structured build spec —
not code, not geometry — for the deterministic pipeline (osm_to_gltf.py,
facade_texture_gen.py) to execute exactly.

This keeps all AI judgment calls in one auditable place (a JSON file you
can read before it runs) instead of scattered through the generation
code. The rest of the pipeline never makes a style decision on its own —
it just executes whatever spec it's handed.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python ai_director.py --bbox 12.9716,77.5946,12.9816,77.6046

    # or any provider, same as ai_critic.py / citygen.py:
    python ai_director.py --bbox <bbox> --provider nemotron

Output: build_spec.json — feed this into citygen.py with --spec (once
wired) or read the fields manually into --style/--terrain flags.
"""

import argparse
import json
import sys

import citygen_providers as providers

VALID_STYLES = ["residential", "downtown", "suburban", "european", "industrial"]

DIRECTOR_PROMPT = """You are deciding build parameters for a procedurally
generated 3D city block, based on real OpenStreetMap statistics for one
area. You are the planning step only — you do not generate geometry or
textures yourself, you decide parameters that a separate deterministic
pipeline will execute exactly as specified.

Area statistics:
{stats}

Choose parameters from this fixed set — do not invent new style names:
  style: one of {styles}
  use_terrain: true/false — true if this area likely has meaningful elevation
               change (hilly, near water/valley), false for flat urban grids
  notes: one short sentence explaining your reasoning, for a human to audit

Respond ONLY with JSON, no other text:
{{
  "style": "<one of the valid styles>",
  "use_terrain": <true/false>,
  "notes": "<short reasoning>"
}}
"""


def analyze_bbox_context(nodes, buildings, roads):
    """
    Deterministic feature extraction from raw OSM data — this is prep
    work, not a judgment call, so it stays plain code rather than an AI
    call. The director only reasons over these summary stats, not the
    raw OSM payload, to keep its input small and its decisions auditable.
    """
    n_buildings = len(buildings)
    n_roads = len(roads)

    heights = []
    tag_counts = {}
    for b in buildings:
        tags = b.get("tags", {})
        btype = tags.get("building", "yes")
        tag_counts[btype] = tag_counts.get(btype, 0) + 1

        if "height" in tags:
            try:
                heights.append(float(str(tags["height"]).split()[0]))
            except ValueError:
                pass
        elif "building:levels" in tags:
            try:
                heights.append(float(tags["building:levels"]) * 3.0)
            except ValueError:
                pass

    highway_counts = {}
    for r in roads:
        htype = r.get("tags", {}).get("highway", "unknown")
        highway_counts[htype] = highway_counts.get(htype, 0) + 1

    return {
        "n_buildings": n_buildings,
        "n_roads": n_roads,
        "avg_building_height_m": (sum(heights) / len(heights)) if heights else None,
        "max_building_height_m": max(heights) if heights else None,
        "building_type_counts": dict(sorted(tag_counts.items(), key=lambda kv: -kv[1])[:8]),
        "road_type_counts": dict(sorted(highway_counts.items(), key=lambda kv: -kv[1])[:8]),
    }


def request_build_spec(context, provider="anthropic", api_base=None, model=None, api_key_env=None):
    provider, api_key, api_base, model = providers.resolve_provider_config(
        provider, api_base, model, api_key_env)

    prompt = DIRECTOR_PROMPT.format(
        stats=json.dumps(context, indent=2),
        styles=VALID_STYLES,
    )
    text = providers.call_text_model(prompt, api_key, provider, api_base=api_base, model=model)
    spec = providers.parse_json_response(text)

    if spec is None:
        print(f"Warning: director response wasn't valid JSON, falling back to defaults. Raw: {text[:200]}")
        return {"style": "residential", "use_terrain": False,
                "notes": "fallback default — director response was unparseable"}

    if spec.get("style") not in VALID_STYLES:
        print(f"Warning: director chose an invalid style '{spec.get('style')}', falling back to 'residential'.")
        spec["style"] = "residential"

    return spec


def main():
    parser = argparse.ArgumentParser(description="AI director: decide build parameters from real bbox context")
    parser.add_argument("--bbox", required=True, help="min_lat,min_lon,max_lat,max_lon")
    parser.add_argument("--out", default="build_spec.json")
    parser.add_argument("--provider", default="anthropic")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key-env", default=None)
    args = parser.parse_args()

    try:
        min_lat, min_lon, max_lat, max_lon = map(float, args.bbox.split(","))
    except ValueError:
        sys.exit("bbox must be 4 comma-separated numbers: min_lat,min_lon,max_lat,max_lon")

    # reuse the existing, already-tested Overpass fetch/parse from the geometry pipeline
    import osm_to_gltf

    print("Fetching OSM data for context analysis...")
    osm_json = osm_to_gltf.fetch_osm_data(min_lat, min_lon, max_lat, max_lon)
    nodes, buildings, roads = osm_to_gltf.parse_osm(osm_json)

    context = analyze_bbox_context(nodes, buildings, roads)
    print("Area context:")
    print(json.dumps(context, indent=2))

    print("\nRequesting build spec from director model...")
    spec = request_build_spec(context, provider=args.provider, api_base=args.api_base,
                               model=args.model, api_key_env=args.api_key_env)
    spec["bbox"] = args.bbox
    spec["context"] = context

    with open(args.out, "w") as f:
        json.dump(spec, f, indent=2)

    print(f"\nDecided: style={spec['style']}  use_terrain={spec['use_terrain']}")
    print(f"Reasoning: {spec['notes']}")
    print(f"\nSpec written to {args.out}")
    print(f"Run: python citygen.py --bbox {args.bbox} --out mycity "
          f"--style {spec['style']}{' --terrain' if spec['use_terrain'] else ''}")


if __name__ == "__main__":
    main()
