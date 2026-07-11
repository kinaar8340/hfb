"""Storage / release dynamics for the rear-hemi flux flywheel slingshot.

Storage phase: resonant holding via phase-locked charged vibrations —
electrostatic energy in dual envelopes + flux/twist in stretched packing.
Release phase: controlled detuning contracts the rear boundary forward
(directional slingshot reconfiguration of the effective geometry).

Analog control layer on top of topological / flow-engineered metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from hfb.electro_vibrational.charge_envelopes import (
    DualChargeConfig,
    charged_vibration_field,
    electro_vibrational_defect_modulation,
)
from hfb.electro_vibrational.observer_sync import (
    ObserverSyncConfig,
    apply_observer_to_alignment,
    observer_feedback,
)
from hfb.electro_vibrational.phase_alignment import (
    PhaseAlignmentConfig,
    local_phase_alignment_field,
    phase_alignment_order,
    phase_alignment_state,
    void_order_parameter,
)


class SlingshotPhase(str, Enum):
    IDLE = "idle"
    NUCLEATE = "nucleate"
    STORE = "store"
    RELEASE = "release"
    COAST = "coast"


@dataclass
class SlingshotConfig:
    """Full resonant control configuration for hemi-void slingshot cycle."""

    charge: DualChargeConfig | None = None
    phase: PhaseAlignmentConfig | None = None
    observer: ObserverSyncConfig | None = None
    store_detuning: float = 0.0
    """Drive detuning while holding (near 0 = locked)."""
    release_detuning: float = 0.35
    """Detuning step that unlocks rear-hemi contraction."""
    store_duration: float = 2.0
    release_duration: float = 0.8
    nucleate_duration: float = 1.0
    flywheel_capacity: float = 1.0
    store_rate: float = 0.55
    release_rate: float = 1.8
    rear_bias: float = 0.85
    """How strongly stored energy weights the rear hemisphere."""
    modulation_strength: float = 0.35
    enable_observer: bool = True


@dataclass
class SlingshotState:
    """Instantaneous state of the resonant void-control engine."""

    t: float
    phase: SlingshotPhase
    psi: float
    stored_energy: float
    release_impulse: float
    detuning: float
    void_amplitude: float
    observer_feedback: float
    supercritical: bool


def cycle_phase_at_time(
    t: float,
    cfg: SlingshotConfig | None = None,
) -> SlingshotPhase:
    """Map absolute time onto a simple open-loop cycle schedule."""
    cfg = cfg or SlingshotConfig()
    t0 = 0.0
    t1 = t0 + cfg.nucleate_duration
    t2 = t1 + cfg.store_duration
    t3 = t2 + cfg.release_duration
    period = t3 + 1.0  # short coast
    tau = t % period
    if tau < t1:
        return SlingshotPhase.NUCLEATE
    if tau < t2:
        return SlingshotPhase.STORE
    if tau < t3:
        return SlingshotPhase.RELEASE
    return SlingshotPhase.COAST


def drive_frequency_for_phase(
    phase: SlingshotPhase,
    cfg: SlingshotConfig | None = None,
) -> float:
    """Select drive frequency (relative to ω₀) for the current cycle phase."""
    cfg = cfg or SlingshotConfig()
    phase_cfg = cfg.phase or PhaseAlignmentConfig()
    base = phase_cfg.medium_resonance
    if phase in (SlingshotPhase.NUCLEATE, SlingshotPhase.STORE):
        return base * (1.0 + cfg.store_detuning)
    if phase == SlingshotPhase.RELEASE:
        return base * (1.0 + cfg.release_detuning)
    return base * (1.0 + 0.5 * cfg.release_detuning)


def update_flywheel(
    stored: float,
    phase: SlingshotPhase,
    psi: float,
    dt: float,
    cfg: SlingshotConfig | None = None,
) -> tuple[float, float]:
    """Integrate flywheel energy; return (stored, release_impulse)."""
    cfg = cfg or SlingshotConfig()
    cap = max(cfg.flywheel_capacity, 1e-9)
    impulse = 0.0
    if phase == SlingshotPhase.STORE and psi > 0.5:
        # Resonant charging proportional to lock quality
        stored = min(cap, stored + cfg.store_rate * psi * dt)
    elif phase == SlingshotPhase.RELEASE and stored > 0.0:
        dump = min(stored, cfg.release_rate * stored * dt + cfg.release_rate * 0.05 * dt)
        stored = max(0.0, stored - dump)
        impulse = dump
    elif phase == SlingshotPhase.COAST:
        stored = max(0.0, stored - 0.02 * dt)  # slow leakage
    return stored, impulse


def rear_hemi_weight(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    axis: str = "x",
    softness: float = 0.25,
) -> NDArray[np.floating]:
    """Smooth weight ≈1 on the rear hemisphere (negative axis), ≈0 on front."""
    coord = x if axis == "x" else y
    # Logistic: rear (coord < 0) → 1
    return 1.0 / (1.0 + np.exp(coord / max(softness, 1e-6)))


def flywheel_shift_boost(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    stored_energy: float,
    release_impulse: float,
    cfg: SlingshotConfig | None = None,
) -> NDArray[np.floating]:
    """Directional shift contribution from rear-hemi storage and release.

    Storage stretches rear (negative axial bias); release contracts forward
    (positive impulse along +axis) — the rubber-band slingshot analog.
    """
    cfg = cfg or SlingshotConfig()
    charge = cfg.charge or DualChargeConfig()
    axis = charge.axis
    coord = x if axis == "x" else y
    rear = rear_hemi_weight(x, y, axis=axis)
    r = np.sqrt(x**2 + y**2)
    wall = np.exp(-((r - charge.outer_radius) ** 2) / (2.0 * charge.wall_width**2))

    # During storage: hold negative shift on rear (tension reservoir)
    store_term = -cfg.rear_bias * stored_energy * rear * wall * np.sign(coord + 1e-12)
    # On release: forward impulse (positive along +axis)
    release_term = 2.2 * release_impulse * wall * (1.0 - 0.5 * rear)
    if axis == "y":
        # same functional form; applied as scalar shift field along axis
        pass
    return store_term + release_term


def resonant_control_step(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    t: float,
    stored_energy: float = 0.0,
    cfg: SlingshotConfig | None = None,
    dt: float = 0.05,
    base_defect: NDArray[np.floating] | None = None,
) -> dict:
    """One control step: alignment → void → flywheel → field modulation.

    Returns dict of fields and scalar state for coupling into flux_bubble_metric.
    """
    cfg = cfg or SlingshotConfig()
    charge_cfg = cfg.charge or DualChargeConfig()
    phase_cfg = cfg.phase or PhaseAlignmentConfig()
    obs_cfg = cfg.observer or ObserverSyncConfig()

    phase = cycle_phase_at_time(t, cfg)
    drive_freq = drive_frequency_for_phase(phase, cfg)
    # Transient phase config with active drive frequency
    active_phase = replace(
        phase_cfg,
        drive_frequency=drive_freq / max(phase_cfg.medium_resonance, 1e-12),
    )

    obs_fb = 0.0
    if cfg.enable_observer:
        medium_phase = phase_cfg.medium_resonance * t
        obs = observer_feedback(t, obs_cfg, medium_phase=medium_phase)
        obs_fb = obs["feedback"]
    else:
        obs = {"feedback": 0.0, "entrainment": 0.0, "phase_error": 0.0}

    align = phase_alignment_state(t, active_phase, observer_feedback=obs_fb)
    psi = align["psi"]
    if cfg.enable_observer:
        psi = apply_observer_to_alignment(psi, obs_fb)

    locked = phase == SlingshotPhase.STORE
    vib = charged_vibration_field(
        x, y, t=t, cfg=charge_cfg, omega0=phase_cfg.medium_resonance
    )
    align_field = local_phase_alignment_field(
        x, y, vib["shell_envelope"], psi, active_phase
    )
    void_amp = void_order_parameter(align_field, active_phase, locked=locked)
    mean_void = float(np.mean(void_amp))

    stored, impulse = update_flywheel(stored_energy, phase, psi, dt, cfg)
    shift_boost = flywheel_shift_boost(x, y, stored, impulse, cfg)

    if base_defect is not None:
        lam_mod = electro_vibrational_defect_modulation(
            base_defect, vib["coupling"], strength=cfg.modulation_strength
        )
        # Void suppresses defect density inside nucleated region
        lam_mod = lam_mod * (1.0 - 0.65 * np.clip(void_amp, 0.0, 1.0))
    else:
        lam_mod = None

    state = SlingshotState(
        t=t,
        phase=phase,
        psi=psi,
        stored_energy=stored,
        release_impulse=impulse,
        detuning=float(align["detuning"]),
        void_amplitude=mean_void,
        observer_feedback=obs_fb,
        supercritical=bool(align["supercritical"] > 0.5),
    )

    return {
        "state": state,
        "alignment_field": align_field,
        "void_amplitude_field": void_amp,
        "shift_boost": shift_boost,
        "charge_density": vib["charge_density"],
        "e_field": vib["e_field"],
        "coupling": vib["coupling"],
        "displacement": vib["displacement"],
        "defect_modulated": lam_mod,
        "observer": obs,
        "psi": psi,
        "stored_energy": stored,
        "release_impulse": impulse,
        "phase": phase,
    }


def simulate_slingshot_cycle(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    t_max: float = 5.0,
    dt: float = 0.05,
    cfg: SlingshotConfig | None = None,
    base_defect: NDArray[np.floating] | None = None,
) -> dict:
    """Integrate a full nucleation → store → release → coast cycle.

    Returns time series of scalars plus final field snapshots.
    """
    cfg = cfg or SlingshotConfig()
    times = np.arange(0.0, t_max + 0.5 * dt, dt)
    series = {
        "t": [],
        "psi": [],
        "stored": [],
        "impulse": [],
        "void": [],
        "phase": [],
        "detuning": [],
        "observer_fb": [],
    }
    stored = 0.0
    last = None
    for t in times:
        step = resonant_control_step(
            x, y, float(t), stored_energy=stored, cfg=cfg, dt=dt, base_defect=base_defect
        )
        st: SlingshotState = step["state"]
        stored = st.stored_energy
        series["t"].append(st.t)
        series["psi"].append(st.psi)
        series["stored"].append(st.stored_energy)
        series["impulse"].append(st.release_impulse)
        series["void"].append(st.void_amplitude)
        series["phase"].append(st.phase.value)
        series["detuning"].append(st.detuning)
        series["observer_fb"].append(st.observer_feedback)
        last = step

    for key in ("t", "psi", "stored", "impulse", "void", "detuning", "observer_fb"):
        series[key] = np.asarray(series[key], dtype=float)

    return {"series": series, "final": last, "config": cfg}


def global_alignment_at(
    drive_frequency: float,
    drive_phase: float = 0.0,
    medium_phase: float = 0.0,
    medium_resonance: float = 1.0,
) -> float:
    """Convenience wrapper for scalar ψ without full dynamics."""
    return phase_alignment_order(
        drive_phase, medium_phase, drive_frequency, medium_resonance
    )
