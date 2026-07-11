"""Phase-alignment threshold and void-bubble nucleation control.

Introduces an order parameter between the boundary charged-vibration state and
the local resonant modes of the surrounding Hopf lattice / flux medium.
Crossing a threshold triggers a bifurcation that nucleates or enlarges a
controllable void bubble — dynamically tunable analog of ergoregions/horizons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PhaseAlignmentConfig:
    """Control parameters for resonant phase locking."""

    threshold: float = 0.72
    """Order-parameter threshold for void nucleation (0–1)."""
    medium_resonance: float = 1.0
    """ω₀ of the surrounding flux medium."""
    drive_frequency: float = 1.0
    """Boundary drive frequency (units of ω₀)."""
    drive_phase: float = 0.0
    medium_phase: float = 0.0
    coherence_length: float = 0.5
    """Spatial smoothing scale for local alignment."""
    nucleation_gain: float = 1.2
    """How strongly supercritical alignment enlarges the void."""
    hysteresis: float = 0.08
    """Asymmetric threshold band for hold vs release (storage lock-in)."""


def frequency_detuning(
    drive_frequency: float,
    medium_resonance: float = 1.0,
) -> float:
    """Normalized detuning δ = (ω_drive − ω₀) / ω₀."""
    w0 = medium_resonance if abs(medium_resonance) > 1e-12 else 1.0
    return (drive_frequency - medium_resonance) / w0


def phase_alignment_order(
    drive_phase: float,
    medium_phase: float,
    drive_frequency: float,
    medium_resonance: float = 1.0,
    detuning_penalty: float = 2.0,
) -> float:
    """Scalar order parameter ψ ∈ [0, 1] for global phase alignment.

    ψ = cos²(Δφ/2) · sech(α·δ) — peaks when phases lock and frequencies match.
    """
    dphi = drive_phase - medium_phase
    phase_factor = np.cos(0.5 * dphi) ** 2
    delta = frequency_detuning(drive_frequency, medium_resonance)
    freq_factor = 1.0 / np.cosh(detuning_penalty * delta)
    return float(np.clip(phase_factor * freq_factor, 0.0, 1.0))


def local_phase_alignment_field(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    shell_envelope: NDArray[np.floating],
    global_psi: float,
    cfg: PhaseAlignmentConfig | None = None,
) -> NDArray[np.floating]:
    """Spatially varying alignment concentrated on charged-vibration boundaries."""
    cfg = cfg or PhaseAlignmentConfig()
    r = np.sqrt(x**2 + y**2)
    medium = np.exp(-(r**2) / (2.0 * cfg.coherence_length**2))
    # Blend boundary drive coherence with bulk medium response
    return global_psi * (0.55 * shell_envelope + 0.45 * medium)


def nucleation_mask(
    alignment_field: NDArray[np.floating],
    cfg: PhaseAlignmentConfig | None = None,
    locked: bool = False,
) -> NDArray[np.bool_]:
    """Boolean region where alignment exceeds nucleation (or hold) threshold."""
    cfg = cfg or PhaseAlignmentConfig()
    thr = cfg.threshold - cfg.hysteresis if locked else cfg.threshold
    return alignment_field >= thr


def void_order_parameter(
    alignment_field: NDArray[np.floating],
    cfg: PhaseAlignmentConfig | None = None,
    locked: bool = False,
) -> NDArray[np.floating]:
    """Smooth void amplitude from supercritical bifurcation of ψ.

    Above threshold: η ∝ √(ψ − ψ_c) (mean-field pitchfork style).
    Below threshold: η → 0 (or residual if hysteresis-locked).
    """
    cfg = cfg or PhaseAlignmentConfig()
    thr = cfg.threshold - cfg.hysteresis if locked else cfg.threshold
    excess = np.maximum(alignment_field - thr, 0.0)
    eta = cfg.nucleation_gain * np.sqrt(excess)
    return eta


def phase_alignment_state(
    t: float,
    cfg: PhaseAlignmentConfig | None = None,
    observer_feedback: float = 0.0,
) -> dict[str, float]:
    """Time-dependent global alignment state with optional observer entrainment.

    ``observer_feedback`` ∈ [-1, 1] shifts the effective drive phase toward lock
    (positive) or unlock (negative) — see observer_sync.
    """
    cfg = cfg or PhaseAlignmentConfig()
    # Evolving phases at drive and medium frequencies
    drive_phase = cfg.drive_phase + cfg.drive_frequency * cfg.medium_resonance * t
    medium_phase = cfg.medium_phase + cfg.medium_resonance * t
    # Observer pulls drive phase toward medium phase
    if abs(observer_feedback) > 1e-12:
        dphi = medium_phase - drive_phase
        drive_phase = drive_phase + observer_feedback * 0.5 * np.sin(dphi)

    psi = phase_alignment_order(
        drive_phase,
        medium_phase,
        cfg.drive_frequency,
        cfg.medium_resonance,
    )
    # Soft boost from observer when already near lock
    psi = float(np.clip(psi + 0.15 * max(observer_feedback, 0.0) * psi, 0.0, 1.0))
    delta = frequency_detuning(cfg.drive_frequency, cfg.medium_resonance)
    supercritical = psi >= cfg.threshold
    return {
        "psi": psi,
        "drive_phase": float(drive_phase),
        "medium_phase": float(medium_phase),
        "detuning": float(delta),
        "supercritical": float(supercritical),
        "threshold": cfg.threshold,
    }
