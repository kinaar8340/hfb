"""Observer synchronization as resonant entrainment feedback.

Ties the insight from Observer_Synchronization-style entrainment into HFB:
an internal observer (or control system) can participate in stabilizing phase
alignment with the macro "hum" of the flux medium, extending physical
synchronization into active boundary control.

Modeled as a feedback term on the electro-vibrational control equations —
analog only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObserverSyncConfig:
    """Parameters for observer–medium resonant entrainment."""

    coupling: float = 0.4
    """Entrainment gain (0 = open-loop, 1 = strong lock)."""
    observer_frequency: float = 1.0
    """Observer / container internal clock (units of medium ω₀)."""
    observer_phase: float = 0.0
    hum_frequency: float = 1.0
    """Macro resonant hum of the flux lattice."""
    bandwidth: float = 0.25
    """Relative half-width for effective entrainment band."""
    delay: float = 0.0
    """Optional response lag (time units)."""
    noise: float = 0.0
    """Optional stochastic jitter amplitude on feedback."""


def entrainment_strength(
    observer_frequency: float,
    hum_frequency: float = 1.0,
    bandwidth: float = 0.25,
) -> float:
    """Lorentzian proximity of observer clock to macro hum frequency."""
    df = observer_frequency - hum_frequency
    bw = max(bandwidth, 1e-6)
    return float(1.0 / (1.0 + (df / bw) ** 2))


def phase_error(
    observer_phase: float,
    medium_phase: float,
) -> float:
    """Wrapped phase error Δφ ∈ (−π, π]."""
    err = (observer_phase - medium_phase + np.pi) % (2.0 * np.pi) - np.pi
    return float(err)


def observer_feedback(
    t: float,
    cfg: ObserverSyncConfig | None = None,
    medium_phase: float | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Compute scalar feedback for phase-alignment control.

    Returns:
        feedback ∈ [-1, 1] — positive favors locking to the hum;
        also returns entrainment, phase_error, and instantaneous observer phase.
    """
    cfg = cfg or ObserverSyncConfig()
    t_eff = t - cfg.delay
    obs_phase = cfg.observer_phase + cfg.observer_frequency * t_eff
    if medium_phase is None:
        medium_phase = cfg.hum_frequency * t_eff

    ent = entrainment_strength(
        cfg.observer_frequency, cfg.hum_frequency, cfg.bandwidth
    )
    err = phase_error(obs_phase, medium_phase)
    # Negative feedback on phase error, scaled by entrainment × coupling
    raw = -cfg.coupling * ent * np.sin(err)
    if cfg.noise > 0.0:
        rng = rng or np.random.default_rng(0)
        raw = raw + cfg.noise * float(rng.normal())
    feedback = float(np.clip(raw, -1.0, 1.0))
    return {
        "feedback": feedback,
        "entrainment": ent,
        "phase_error": err,
        "observer_phase": float(obs_phase),
        "medium_phase": float(medium_phase),
        "coupling": cfg.coupling,
    }


def apply_observer_to_alignment(
    psi: float,
    feedback: float,
    boost: float = 0.2,
) -> float:
    """Adjust global alignment order parameter with observer feedback."""
    # Positive feedback near lock raises ψ; opposing feedback lowers it
    adjusted = psi + boost * feedback * max(psi, 0.1)
    return float(np.clip(adjusted, 0.0, 1.0))
