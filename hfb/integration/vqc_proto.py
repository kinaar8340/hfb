"""Optional bridge to vqc_proto orbital_braille SLM tooling."""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


def _candidate_vqc_roots() -> list[Path]:
    """Search order for orbital_braille on disk."""
    env = os.environ.get("VQC_PROTO_PATH")
    home = Path.home()
    return [
        Path(env) if env else None,
        home / "Projects" / "vqc_proto" / "space" / "orbital-braille",
        home / "Projects" / "vqc_proto" / "proto",
        home / "Projects" / "vqc_proto_full" / "space" / "orbital-braille",
        home / "Projects" / "vqc_proto_full" / "proto",
    ]  # type: ignore[list-item]


def resolve_vqc_root(hint: Path | None = None) -> Path | None:
    """Return first path containing orbital_braille/lg_modes.py."""
    if hint is not None and vqc_proto_available(hint):
        return hint
    for root in _candidate_vqc_roots():
        if root is not None and vqc_proto_available(root):
            return root
    return None


def _default_vqc_path() -> Path:
    resolved = resolve_vqc_root()
    if resolved is not None:
        return resolved
    return Path(os.environ.get("VQC_PROTO_PATH", Path.home() / "Projects" / "vqc_proto/space/orbital-braille"))


def _load_module_from_file(name: str, path: Path, package: str = "orbital_braille") -> ModuleType | None:
    """Load a single module file without executing orbital_braille/__init__.py."""
    if not path.is_file():
        return None
    try:
        if package not in sys.modules:
            pkg = ModuleType(package)
            pkg.__path__ = [str(path.parent)]  # type: ignore[attr-defined]
            sys.modules[package] = pkg
        spec = importlib.util.spec_from_file_location(f"{package}.{name}", path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"{package}.{name}"] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def vqc_proto_available(path: Path | None = None) -> bool:
    """Return True if orbital_braille LG modes are present on disk."""
    root = path or _default_vqc_path()
    return (root / "orbital_braille" / "lg_modes.py").is_file()


def vqc_slm_available(path: Path | None = None) -> bool:
    """Return True if slm_typehead can be loaded (no heavy optional deps)."""
    root = path or _default_vqc_path()
    mod = _load_module_from_file("slm_typehead", root / "orbital_braille" / "slm_typehead.py")
    return mod is not None and hasattr(mod, "save_phase_hologram")


@dataclass
class VQCProtoStatus:
    root: str | None
    lg_modes: bool
    slm_typehead: bool
    message: str


def vqc_proto_status() -> VQCProtoStatus:
    """Diagnostic summary for CLI and notebooks."""
    root = resolve_vqc_root()
    if root is None:
        return VQCProtoStatus(
            root=None,
            lg_modes=False,
            slm_typehead=False,
            message="vqc_proto not found — set VQC_PROTO_PATH or clone kinaar8340/vqc_proto",
        )
    lg = vqc_proto_available(root)
    slm = vqc_slm_available(root)
    return VQCProtoStatus(
        root=str(root),
        lg_modes=lg,
        slm_typehead=slm,
        message="full SLM coupling" if slm else "LG modes only (slm_typehead deps missing)",
    )


@dataclass
class VQCProtoBridge:
    """Lazy loader for vqc_proto SLM export utilities."""

    root: Path | None = None
    _lg_mod: ModuleType | None = None
    _slm_mod: ModuleType | None = None

    def __post_init__(self) -> None:
        self.root = resolve_vqc_root(self.root) or self.root or _default_vqc_path()
        if not vqc_proto_available(self.root):
            raise ImportError(
                "vqc_proto orbital_braille not found. Set VQC_PROTO_PATH or install from "
                "https://github.com/kinaar8340/vqc_proto"
            )

    def _lg(self) -> ModuleType:
        if self._lg_mod is None:
            mod = _load_module_from_file("lg_modes", self.root / "orbital_braille" / "lg_modes.py")
            if mod is None:
                raise ImportError("Failed to load orbital_braille.lg_modes")
            self._lg_mod = mod
        return self._lg_mod

    def _slm(self) -> ModuleType | None:
        if self._slm_mod is None:
            self._slm_mod = _load_module_from_file(
                "slm_typehead", self.root / "orbital_braille" / "slm_typehead.py"
            )
        return self._slm_mod

    def lg_mode(self) -> Callable:
        return self._lg().lg_mode

    def slm_config(self) -> Any:
        slm = self._slm()
        if slm is None:
            raise ImportError("slm_typehead unavailable (optional vqc_proto deps missing)")
        return slm.SLMConfig

    def save_phase_hologram(self) -> Callable | None:
        slm = self._slm()
        return slm.save_phase_hologram if slm is not None else None

    def phase_to_levels(self) -> Callable | None:
        slm = self._slm()
        return slm.phase_to_levels if slm is not None else None