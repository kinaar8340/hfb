"""Dual opposing charge envelopes and electro-vibrational boundary conditions.

Models concentric shells carrying opposing charge densities. The gap between
shells is a controllable capacitive layer whose electric field couples to
topological flux (Hopfions) and vibrational modes of the medium.

Analog only — electrostatic handles on effective metrics, not literal vacuum
engineering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class DualChargeConfig:
    """Parameters for dual opposing charge shells."""

    inner_radius: float = 0.85
    outer_radius: float = 1.15
    wall_width: float = 0.12
    charge_density: float = 1.0
    """|σ| on each shell (outer has opposite sign)."""
    vibration_amp: float = 0.15
    vibration_freq: float = 1.0
    """Drive frequency in units of the medium's resonant frequency ω₀."""
    vibration_phase: float = 0.0
    elongation: float = 1.35
    """Orthogonal (y) stretch for hemi/gourd bias; 1 = isotropic."""
    axis: str = "x"
    """Propagation / slingshot axis ('x' or 'y')."""


def dual_shell_masks(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    cfg: DualChargeConfig | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Return (inner_shell, outer_shell, gap) envelopes peaked on dual boundaries.

    Envelopes are smooth Gaussians in the radial (possibly elongated) coordinate.
    """
    cfg = cfg or DualChargeConfig()
    if cfg.axis == "y":
        x, y = y, x
    # Elongate orthogonal to propagation for hemi/gourd preference
    r = np.sqrt(x**2 + (y / cfg.elongation) ** 2)
    w = cfg.wall_width
    inner = np.exp(-((r - cfg.inner_radius) ** 2) / (2.0 * w**2))
    outer = np.exp(-((r - cfg.outer_radius) ** 2) / (2.0 * w**2))
    r_mid = 0.5 * (cfg.inner_radius + cfg.outer_radius)
    gap_width = max(0.5 * (cfg.outer_radius - cfg.inner_radius), w)
    gap = np.exp(-((r - r_mid) ** 2) / (2.0 * gap_width**2))
    return inner, outer, gap


def opposing_charge_density(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    cfg: DualChargeConfig | None = None,
) -> NDArray[np.floating]:
    """Surface-like charge density: +σ on outer shell, −σ on inner shell."""
    cfg = cfg or DualChargeConfig()
    inner, outer, _ = dual_shell_masks(x, y, cfg)
    return cfg.charge_density * (outer - inner)


def capacitive_field_magnitude(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    cfg: DualChargeConfig | None = None,
) -> NDArray[np.floating]:
    """Proxy |E| in the capacitive gap between dual charge envelopes.

    E ∝ σ / ε₀ * gap_envelope; we use unit ε and return a smooth gap field.
    """
    cfg = cfg or DualChargeConfig()
    _, _, gap = dual_shell_masks(x, y, cfg)
    gap_size = max(cfg.outer_radius - cfg.inner_radius, 1e-6)
    return cfg.charge_density * gap / gap_size


def charged_vibration_field(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    t: float = 0.0,
    cfg: DualChargeConfig | None = None,
    omega0: float = 1.0,
) -> dict[str, NDArray[np.floating]]:
    """Electro-vibrational drive on dual shells.

    Couples charge density to a time-harmonic mechanical/vibrational mode at
    frequency ``vibration_freq * omega0``. Returns charge, |E|, displacement
    proxy, and the electro-vibrational coupling density used as a control input.
    """
    cfg = cfg or DualChargeConfig()
    sigma = opposing_charge_density(x, y, cfg)
    e_field = capacitive_field_magnitude(x, y, cfg)
    omega = cfg.vibration_freq * omega0
    phase = omega * t + cfg.vibration_phase
    # Displacement proxy localized on shells (bipolar breathing)
    inner, outer, gap = dual_shell_masks(x, y, cfg)
    shell = inner + outer
    displacement = cfg.vibration_amp * shell * np.sin(phase)
    # Coupling density: |E| × vibration velocity proxy on the gap
    velocity = cfg.vibration_amp * omega * shell * np.cos(phase)
    coupling = e_field * (velocity + 1e-12 * displacement)

    return {
        "charge_density": sigma,
        "e_field": e_field,
        "displacement": displacement,
        "velocity": velocity,
        "coupling": coupling,
        "shell_envelope": shell,
        "gap_envelope": gap,
        "phase": np.full_like(x, phase, dtype=float),
    }


def electro_vibrational_defect_modulation(
    base_defect: NDArray[np.floating],
    coupling: NDArray[np.floating],
    strength: float = 0.35,
) -> NDArray[np.floating]:
    """Modulate a topological defect density λ by electro-vibrational coupling.

    λ_eff = λ₀ (1 + strength · ĉ) with ĉ a normalized coupling field.
    """
    c_norm = coupling / (np.max(np.abs(coupling)) + 1e-12)
    return base_defect * (1.0 + strength * c_norm)


def apply_active_precharge(
    charge_density: NDArray[np.floating],
    e_field: NDArray[np.floating],
    boost_field: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Apply transducer pre-charge boost to dual-shell σ and capacitive |E|.

    Thin wrapper used by the flux transducer motor path during store/ready.
    ``boost_field`` is a spatial multiplicative increment (0 = no boost).
    """
    factor = 1.0 + boost_field
    return charge_density * factor, e_field * factor
