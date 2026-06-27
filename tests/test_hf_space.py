"""Smoke tests for the Hugging Face Space bundle."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "space" / "hopf-flux-bubble"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_hf_space_bundle_imports(tmp_path):
    sync = ROOT / "scripts" / "sync_hf_space.sh"
    assert sync.is_file()

    import subprocess

    subprocess.run(["bash", str(sync)], check=True, cwd=ROOT)

    assert (SPACE / "app.py").is_file()
    assert (SPACE / "demo_core.py").is_file()
    assert (SPACE / "hfb").is_dir()
    assert (SPACE / "configs" / "default.yaml").is_file()

    sys.path.insert(0, str(SPACE))
    try:
        demo_core = _load_module("hf_demo_core", SPACE / "demo_core.py")
        params = demo_core.default_run_params()
        assert "bubble_radius" in params
        assert demo_core.is_hf_space() is False

        pytest.importorskip("gradio")
        app_mod = _load_module("hf_app", SPACE / "app.py")
        assert hasattr(app_mod, "build_app")
        blocks = app_mod.build_app()
        assert blocks is not None
    finally:
        sys.path.remove(str(SPACE))
        for key in ("hf_demo_core", "hf_app", "demo_core", "build_info"):
            sys.modules.pop(key, None)