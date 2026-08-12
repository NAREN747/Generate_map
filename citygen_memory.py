#!/usr/bin/env python3
"""
citygen_memory.py
------------------
Persistent, local, no-GPU-needed memory of past critique results. Every
time ai_critic.py scores a building, the result gets recorded here, keyed
by style. Over enough runs, recurring issues surface as a "learned suffix"
that automatically gets appended to future generation prompts — so if
"windows too small" gets flagged repeatedly for your residential style,
future residential generations start with that fix already applied,
instead of the tool repeating the same mistake every run.

This is not model fine-tuning — it's a frequency-based heuristic over
critique feedback. It's cheap, fully local, explainable (you can read
citygen_memory.json and see exactly why a suffix got added), and it
compounds: the more you use the tool, the fewer times each issue needs
re-fixing.

Storage: a single JSON file (default citygen_memory.json) in the working
directory. Safe to delete any time to reset — the pipeline works fine
with no memory file, it just starts learning from zero again.
"""

import json
import os
import time

DEFAULT_MEMORY_PATH = "citygen_memory.json"


def load_memory(path=DEFAULT_MEMORY_PATH):
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def save_memory(memory, path=DEFAULT_MEMORY_PATH):
    with open(path, "w") as f:
        json.dump(memory, f, indent=2)


def record_critique(memory, style_key, building_name, critique):
    """
    Log one critique result. style_key should be a stable identifier for
    the style being generated (e.g. the style prompt string, or a preset
    name) — history is bucketed per style, since a fix that helps
    "downtown glass towers" isn't necessarily relevant to "suburban brick".
    """
    entry = memory.setdefault(style_key, {"runs": [], "issue_counts": {}, "adjustment_counts": {}})

    entry["runs"].append({
        "building": building_name,
        "score": critique.get("score"),
        "issues": critique.get("issues", []),
        "timestamp": time.time(),
    })

    for issue in critique.get("issues", []):
        entry["issue_counts"][issue] = entry["issue_counts"].get(issue, 0) + 1

    adjustment = critique.get("prompt_adjustment")
    if adjustment:
        entry["adjustment_counts"][adjustment] = entry["adjustment_counts"].get(adjustment, 0) + 1

    return memory


def get_learned_suffix(memory, style_key, min_frequency=3, max_terms=3):
    """
    Return a short phrase to append to the style prompt, built from the
    most frequently recurring prompt adjustments for this style. Only
    surfaces adjustments seen at least min_frequency times, so a one-off
    critique quirk doesn't permanently warp the style — this should
    reflect a genuine recurring pattern, not noise.
    """
    entry = memory.get(style_key)
    if not entry:
        return ""

    frequent = [
        adjustment for adjustment, count in
        sorted(entry["adjustment_counts"].items(), key=lambda kv: -kv[1])
        if count >= min_frequency
    ]
    return ", ".join(frequent[:max_terms])


def get_style_stats(memory, style_key):
    """Summary stats for a style — useful for a dashboard or just eyeballing progress."""
    entry = memory.get(style_key)
    if not entry or not entry["runs"]:
        return None

    scores = [r["score"] for r in entry["runs"] if r.get("score") is not None]
    top_issues = sorted(entry["issue_counts"].items(), key=lambda kv: -kv[1])[:5]

    return {
        "n_runs": len(entry["runs"]),
        "avg_score": sum(scores) / len(scores) if scores else None,
        "top_issues": top_issues,
        "learned_suffix": get_learned_suffix(memory, style_key),
    }


def print_summary(path=DEFAULT_MEMORY_PATH):
    """CLI entry point: show what the tool has learned so far, per style."""
    memory = load_memory(path)
    if not memory:
        print(f"No memory yet at {path} — run ai_critic.py a few times first.")
        return

    for style_key, entry in memory.items():
        stats = get_style_stats(memory, style_key)
        print(f"\nStyle: {style_key}")
        print(f"  Runs so far: {stats['n_runs']}")
        if stats["avg_score"] is not None:
            print(f"  Average score: {stats['avg_score']:.1f}/10")
        if stats["top_issues"]:
            print("  Most common issues:")
            for issue, count in stats["top_issues"]:
                print(f"    ({count}x) {issue}")
        if stats["learned_suffix"]:
            print(f"  Learned suffix (auto-applied to future generations): {stats['learned_suffix']}")
        else:
            print("  No learned suffix yet (needs recurring feedback, not just one-off issues)")


if __name__ == "__main__":
    print_summary()
