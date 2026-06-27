"""SLM phase hologram export for flux-bubble vortex conduit arrays."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from hfb.optics.lg_modes import lg_mode_full
from hfb.integration.vqc_proto import VQCProtoBridge, vqc_proto_available, vqc_slm_available


@dataclass
class SLMExportConfig:
    resolution_x: int = 512
    resolution_y: int = 512
    extent_mm: float = 4.0
    wavelength_nm: float = 1550.0
    w0_mm: float = 0.8
    bit_depth: int = 8
    phase_wrap: Literal["0_2pi", "neg_pi_pi"] = "0_2pi"
    device_preset: str = "generic_512"

    @classmethod
    def from_vqc_preset(cls, preset: str = "generic_512", extent_mm: float = 4.0) -> SLMExportConfig:
        if not vqc_proto_available():
            return cls(extent_mm=extent_mm, device_preset=preset)
        bridge = VQCProtoBridge()
        SLMConfig = bridge.slm_config()
        cfg = SLMConfig.from_preset(preset, extent_mm=extent_mm)
        return cls(
            resolution_x=cfg.resolution_x,
            resolution_y=cfg.resolution_y,
            extent_mm=cfg.extent_mm,
            wavelength_nm=cfg.wavelength_nm,
            w0_mm=cfg.w0_mm,
            bit_depth=cfg.bit_depth,
            phase_wrap=cfg.phase_wrap,
            device_preset=preset,
        )


@dataclass
class VortexConduitSpec:
    """Ring array of LG vortices forming a flux-bubble wall."""

    ring_radius_mm: float = 1.2
    num_vortices: int = 12
    winding: int = 1
    amplitude: float = 1.0
    core_sigma_mm: float = 0.25


@dataclass
class HFBSLMManifest:
    payload: str = "hfb_flux_bubble"
    ring_radius_mm: float = 1.2
    num_vortices: int = 12
    winding: int = 1
    wavelength_nm: float = 1550.0
    device_preset: str = "generic_512"
    generator: str = "hfb/optics/slm_export.py"
    vqc_proto_coupled: bool = False
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def _slm_grid(cfg: SLMExportConfig) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    half = cfg.extent_mm / 2.0
    x = np.linspace(-half, half, cfg.resolution_x)
    y = np.linspace(-half, half, cfg.resolution_y)
    return np.meshgrid(x, y, indexing="xy")


def flux_bubble_vortex_field(
    spec: VortexConduitSpec,
    cfg: SLMExportConfig,
    lg_fn=None,
) -> NDArray[np.complexfloating]:
    """
    Superpose LG vortices on a ring — flux-bubble wall hologram target field.

    Uses vqc_proto lg_mode when available; falls back to local hfb.optics.lg_modes.
    """
    X, Y = _slm_grid(cfg)
    field = np.zeros_like(X, dtype=complex)
    sigma = spec.core_sigma_mm

    if lg_fn is None:
        if vqc_proto_available():
            lg_fn = VQCProtoBridge().lg_mode()
        else:
            lg_fn = None

    for i in range(spec.num_vortices):
        angle = 2.0 * np.pi * i / spec.num_vortices
        x0 = spec.ring_radius_mm * np.cos(angle)
        y0 = spec.ring_radius_mm * np.sin(angle)
        dx = X - x0
        dy = Y - y0
        rho = np.sqrt(dx**2 + dy**2)
        phi = np.arctan2(dy, dx)
        gauss = np.exp(-(rho**2) / (2.0 * sigma**2))
        if lg_fn is not None:
            carrier = lg_fn(spec.winding, rho, phi, w0=cfg.w0_mm)
        else:
            carrier = lg_mode_full(spec.winding, dx, dy, w0=cfg.w0_mm)
        field += spec.amplitude * gauss * carrier

    return field


def field_to_slm_phase(
    field: NDArray[np.complexfloating],
    wrap: Literal["0_2pi", "neg_pi_pi"] = "0_2pi",
) -> NDArray[np.floating]:
    """Extract phase map in radians for phase-only SLM upload."""
    phase = np.angle(field)
    if wrap == "0_2pi":
        return np.mod(phase, 2.0 * np.pi)
    return np.mod(phase + np.pi, 2.0 * np.pi) - np.pi


def phase_to_levels(phase: NDArray[np.floating], bit_depth: int = 8, wrap: str = "0_2pi") -> NDArray[np.uint8]:
    """Convert radians to SLM gray levels (local fallback)."""
    if vqc_slm_available():
        fn = VQCProtoBridge().phase_to_levels()
        if fn is not None:
            return fn(phase, bit_depth=bit_depth, wrap=wrap)
    if wrap == "0_2pi":
        norm = np.mod(phase, 2.0 * np.pi) / (2.0 * np.pi)
    else:
        norm = (np.mod(phase + np.pi, 2.0 * np.pi) - np.pi + np.pi) / (2.0 * np.pi)
    max_val = (1 << bit_depth) - 1
    return np.round(norm * max_val).astype(np.uint8)


def save_phase_hologram(
    phase: NDArray[np.floating],
    path: str | Path,
    bit_depth: int = 8,
    wrap: str = "0_2pi",
) -> None:
    """Save phase map as hardware-ready grayscale image."""
    if vqc_slm_available():
        save_fn = VQCProtoBridge().save_phase_hologram()
        if save_fn is not None:
            save_fn(phase, path, bit_depth=bit_depth, wrap=wrap)
            return
    from PIL import Image

    levels = phase_to_levels(phase, bit_depth=bit_depth, wrap=wrap)
    Image.fromarray(levels, mode="L").save(path)


def export_flux_bubble_hologram(
    out_dir: str | Path,
    spec: VortexConduitSpec | None = None,
    cfg: SLMExportConfig | None = None,
    num_frames: int = 1,
    phase_twist_per_frame: float = 0.0,
) -> dict:
    """
    Export SLM hologram package for a flux-bubble vortex ring.

    Writes phase PNG, manifest.json, and preview when matplotlib is available.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = spec or VortexConduitSpec()
    cfg = cfg or SLMExportConfig.from_vqc_preset()

    coupled = vqc_slm_available()
    lg_fn = VQCProtoBridge().lg_mode() if coupled else None
    field = flux_bubble_vortex_field(spec, cfg, lg_fn=lg_fn)

    ext = "png" if cfg.bit_depth <= 8 else "tiff"
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    stack = []
    for i in range(num_frames):
        twist = np.exp(1j * phase_twist_per_frame * i)
        phase = field_to_slm_phase(field * twist, wrap=cfg.phase_wrap)
        stack.append(phase)
        save_phase_hologram(
            phase,
            frames_dir / f"phase_{i:04d}.{ext}",
            bit_depth=cfg.bit_depth,
            wrap=cfg.phase_wrap,
        )

    meta = HFBSLMManifest(
        ring_radius_mm=spec.ring_radius_mm,
        num_vortices=spec.num_vortices,
        winding=spec.winding,
        wavelength_nm=cfg.wavelength_nm,
        device_preset=cfg.device_preset,
        vqc_proto_coupled=coupled,
    )
    (out_dir / "manifest.json").write_text(json.dumps(asdict(meta), indent=2))
    np.save(out_dir / "phase_stack.npy", np.stack(stack))

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, min(4, num_frames), figsize=(3 * min(4, num_frames), 3))
        if num_frames == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.imshow(stack[i], cmap="twilight", vmin=0, vmax=2 * np.pi)
            ax.set_title(f"frame {i}")
            ax.axis("off")
        fig.suptitle("HFB flux-bubble vortex ring SLM phase")
        fig.tight_layout()
        fig.savefig(out_dir / "preview.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass

    return {
        "out_dir": str(out_dir),
        "frames": num_frames,
        "resolution": f"{cfg.resolution_x}×{cfg.resolution_y}",
        "vqc_proto_coupled": coupled,
        "device": cfg.device_preset,
    }