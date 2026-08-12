#!/usr/bin/env python3
"""
check_env.py
------------
Run this first. Checks what's installed and what's missing, so a new user
gets one clear list instead of a stack trace three minutes into their
first run.

Usage:
    python check_env.py
"""

import importlib
import sys

CORE_DEPS = ["requests", "numpy", "shapely", "trimesh", "pyproj", "PIL"]
AI_DEPS = ["torch", "diffusers", "transformers", "accelerate"]


def check_module(name):
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return True, device_name
        return False, None
    except ImportError:
        return False, None


def main():
    print("citygen environment check\n" + "=" * 40)

    print("\nCore pipeline (required):")
    core_missing = []
    for dep in CORE_DEPS:
        ok = check_module(dep)
        print(f"  [{'OK' if ok else 'MISSING'}] {dep}")
        if not ok:
            core_missing.append(dep)

    print("\nAI facade generation (optional — enables --style texturing):")
    ai_missing = []
    for dep in AI_DEPS:
        ok = check_module(dep)
        print(f"  [{'OK' if ok else 'MISSING'}] {dep}")
        if not ok:
            ai_missing.append(dep)

    if not ai_missing:
        has_gpu, device_name = check_gpu()
        if has_gpu:
            print(f"\nGPU detected: {device_name} — AI facade generation will use it.")
        else:
            print("\nNo GPU detected by torch — AI facade generation will run on CPU "
                  "(slow) or you should pass --skip-ai / --force-cpu explicitly.")

    print("\n" + "=" * 40)
    if core_missing:
        print("Core dependencies missing. Install with:")
        print(f"  pip install {' '.join(core_missing)}")
        print("\nThe pipeline will not run until these are installed.")
        sys.exit(1)
    elif ai_missing:
        print("Core pipeline ready. AI facades unavailable until you install:")
        print(f"  pip install {' '.join(ai_missing)}")
        print("(See requirements.txt for hardware-specific torch install commands.)")
        print("\nUntil then, use --skip-ai for procedural (non-AI) facades.")
    else:
        print("Everything installed. You're ready to run citygen.py.")


if __name__ == "__main__":
    main()
