#!/usr/bin/env python3
"""Transducer engine + craft/payload mission demo."""

from pathlib import Path

from hfb.cli import load_config, run_mission_demo

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs" / "default.yaml")
    run_mission_demo(cfg, root / "outputs")
