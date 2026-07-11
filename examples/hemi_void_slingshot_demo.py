#!/usr/bin/env python3
"""Nucleation → store → directional release of a hemi-void Hopf flux bubble.

Demonstrates resonant electro-vibrational control:
  dual charge envelopes → phase alignment → rear-hemi flywheel → slingshot.
"""

from pathlib import Path

from hfb.cli import load_config, run_slingshot_demo

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    run_slingshot_demo(cfg, root / "outputs")
