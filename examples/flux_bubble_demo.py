#!/usr/bin/env python3
"""Minimal Hopf Flux Bubble demo without the CLI."""

from pathlib import Path

from hfb.cli import load_config, run_bubble_demo

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    run_bubble_demo(cfg, root / "outputs")