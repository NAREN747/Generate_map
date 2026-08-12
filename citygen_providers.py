#!/usr/bin/env python3
"""
citygen_providers.py
----------------------
Shared "any API key, any provider" plumbing used by both ai_critic.py
(vision critique) and ai_director.py (text-only planning). Factored out
so ai_director.py doesn't need to import trimesh/pyglet just to make a
text API call — the director is a lightweight planning step, not a
rendering step.
"""

import json
import os
import sys

import requests

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = "claude-sonnet-4-5"

PROVIDER_PRESETS = {
    "nemotron": {
        "api_base": "https://integrate.api.nvidia.com/v1/chat/completions",
        "model": "nvidia/nemotron-3-ultra",
        "api_key_env": "NVIDIA_API_KEY",
    },
}


def resolve_provider_config(provider, api_base, model, api_key_env):
    """Returns (provider, api_key, api_base, model). Exits with a clear message on misconfiguration."""
    if provider in PROVIDER_PRESETS:
        preset = PROVIDER_PRESETS[provider]
        api_base = api_base or preset["api_base"]
        model = model or preset["model"]
        api_key_env = api_key_env or preset["api_key_env"]
        provider = "openai"
    elif provider == "anthropic":
        api_key_env = api_key_env or "ANTHROPIC_API_KEY"
    else:
        if not api_base or not model:
            sys.exit("--provider openai requires --api-base and --model "
                      "(or use a preset like --provider nemotron)")
        api_key_env = api_key_env or "OPENAI_API_KEY"

    api_key = os.environ.get(api_key_env)
    if not api_key:
        sys.exit(f"Set {api_key_env} in your environment first.")

    return provider, api_key, api_base, model


def parse_json_response(text):
    """Shared fail-safe JSON parser: strips markdown fences, never raises on garbage input."""
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def call_text_model(prompt, api_key, provider, api_base=None, model=None, max_tokens=500):
    """
    Text-only (no image) call, routed to whichever provider was resolved.
    Returns the raw text response — caller decides how to parse it.
    """
    if provider == "anthropic":
        payload = {
            "model": ANTHROPIC_MODEL,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return "".join(block["text"] for block in data["content"] if block["type"] == "text")

    # generic OpenAI-compatible
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(api_base, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]
