"""Storage / release dynamics for the rear-hemi flux flywheel slingshot.

Storage phase: the craft **flux transducer** actively pre-charges dual
envelopes and pre-twists the rear-hemi flywheel while also collecting ambient
flux — loading the void to a commanded energy target (READY).

Release phase: the transducer reverts channels and meters intensity so
detuning contracts the rear boundary forward as a directional slingshot.

Energy is tracked in an explicit craft-local ledger (electrostatic + twist +
geometric tension), not as a term in the global curvature-flux bookkeeping.

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
from hfb.electro_vibrational.transducer import (
    FluxTransducer,
    TransducerConfig,
    TransducerReport,
    apply_precharge_to_fields,
    ledger_from_total,
    modulate_defect_by_channels,
    transducer_shift_contribution,
)


class SlingshotPhase(str, Enum):
    IDLE = "idle"
    NUCLEATE = "nucleate"
    STORE = "store"
    READY = "ready"
    RELEASE = "release"
    COAST = "coast"


@dataclass
class SlingshotConfig:
    """Full resonant control configuration for hemi-void slingshot cycle."""

    charge: DualChargeConfig | None = None
    phase: PhaseAlignmentConfig | None = None
    observer: ObserverSyncConfig | None = None
    transducer: TransducerConfig | None = None
    """Craft-local flux transducer (pump + ledger + channel reversion)."""
    store_detuning: float = 0.0
    """Drive detuning while holding (near 0 = locked)."""
    release_detuning: float = 0.35
    """Detuning step that unlocks rear-hemi contraction."""
    store_duration: float = 2.0
    ready_duration: float = 0.4
    """Hold window after store while rear hemi sits at target energy."""
    release_duration: float = 0.8
    nucleate_duration: float = 1.0
    flywheel_capacity: float = 1.0
    """Legacy capacity mirror; preferred source is transducer.capacity."""
    store_rate: float = 0.55
    """Legacy scalar store rate (used only if use_transducer=False)."""
    release_rate: float = 1.8
    """Legacy scalar release rate (used only if use_transducer=False)."""
    rear_bias: float = 0.85
    """How strongly stored energy weights the rear hemisphere."""
    modulation_strength: float = 0.35
    enable_observer: bool = True
    use_transducer: bool = True
    """If True, energy flows through FluxTransducer ledger + channel reversion."""
    release_intensity: float = 1.0
    """Throttle 0–1 for transducer dump rate and impulse strength."""
    pump_intensity: float = 1.0
    """Throttle 0–1 for active pre-charge / pre-twist during store."""


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
    # Transducer ledger breakdown (craft-local bookkeeping)
    e_electrostatic: float = 0.0
    e_twist: float = 0.0
    e_geometric: float = 0.0
    channel_direction: float = 0.0
    channel_intensity: float = 0.0
    channels_reversed: bool = False
    # Active pump status
    ready: bool = False
    pump_active: bool = False
    precharge_power: float = 0.0
    pretwist_power: float = 0.0
    pumped_total: float = 0.0


def cycle_phase_at_time(
    t: float,
    cfg: SlingshotConfig | None = None,
) -> SlingshotPhase:
    """Map absolute time onto open-loop cycle: nucleate→store→ready→release→coast."""
    cfg = cfg or SlingshotConfig()
    t0 = 0.0
    t1 = t0 + cfg.nucleate_duration
    t2 = t1 + cfg.store_duration
    t3 = t2 + cfg.ready_duration
    t4 = t3 + cfg.release_duration
    period = t4 + 1.0  # short coast
    tau = t % period
    if tau < t1:
        return SlingshotPhase.NUCLEATE
    if tau < t2:
        return SlingshotPhase.STORE
    if tau < t3:
        return SlingshotPhase.READY
    if tau < t4:
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
    if phase in (SlingshotPhase.NUCLEATE, SlingshotPhase.STORE, SlingshotPhase.READY):
        return base * (1.0 + cfg.store_detuning)
    if phase == SlingshotPhase.RELEASE:
        return base * (1.0 + cfg.release_detuning)
    return base * (1.0 + 0.5 * cfg.release_detuning)


def _default_transducer_config(cfg: SlingshotConfig) -> TransducerConfig:
    charge = cfg.charge or DualChargeConfig()
    base = cfg.transducer or TransducerConfig()
    return replace(
        base,
        capacity=cfg.transducer.capacity if cfg.transducer else cfg.flywheel_capacity,
        intensity=cfg.release_intensity if cfg.transducer is None else base.intensity,
        pump_intensity=cfg.pump_intensity if cfg.transducer is None else base.pump_intensity,
        axis=charge.axis,
    )


def update_flywheel(
    stored: float,
    phase: SlingshotPhase,
    psi: float,
    dt: float,
    cfg: SlingshotConfig | None = None,
) -> tuple[float, float]:
    """Legacy scalar flywheel integrate (no channel breakdown)."""
    cfg = cfg or SlingshotConfig()
    cap = max(cfg.flywheel_capacity, 1e-9)
    impulse = 0.0
    if phase in (SlingshotPhase.STORE, SlingshotPhase.READY) and psi > 0.5:
        stored = min(cap, stored + cfg.store_rate * psi * dt)
    elif phase == SlingshotPhase.RELEASE and stored > 0.0:
        dump = min(stored, cfg.release_rate * stored * dt + cfg.release_rate * 0.05 * dt)
        stored = max(0.0, stored - dump)
        impulse = dump
    elif phase == SlingshotPhase.COAST:
        stored = max(0.0, stored - 0.02 * dt)
    return stored, impulse


def rear_hemi_weight(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    axis: str = "x",
    softness: float = 0.25,
) -> NDArray[np.floating]:
    """Smooth weight ≈1 on the rear hemisphere (negative axis), ≈0 on front."""
    coord = x if axis == "x" else y
    return 1.0 / (1.0 + np.exp(coord / max(softness, 1e-6)))


def flywheel_shift_boost(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    stored_energy: float,
    release_impulse: float,
    cfg: SlingshotConfig | None = None,
) -> NDArray[np.floating]:
    """Directional shift contribution (legacy scalar path, no transducer)."""
    cfg = cfg or SlingshotConfig()
    charge = cfg.charge or DualChargeConfig()
    axis = charge.axis
    coord = x if axis == "x" else y
    rear = rear_hemi_weight(x, y, axis=axis)
    r = np.sqrt(x**2 + y**2)
    wall = np.exp(-((r - charge.outer_radius) ** 2) / (2.0 * charge.wall_width**2))

    store_term = -cfg.rear_bias * stored_energy * rear * wall * np.sign(coord + 1e-12)
    release_term = 2.2 * release_impulse * wall * (1.0 - 0.5 * rear)
    return store_term + release_term


def resonant_control_step(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    t: float,
    stored_energy: float = 0.0,
    cfg: SlingshotConfig | None = None,
    dt: float = 0.05,
    base_defect: NDArray[np.floating] | None = None,
    transducer: FluxTransducer | None = None,
    dx: float | None = None,
) -> dict:
    """One control step: alignment → void → active pump / dump → field modulation.

    Store/ready: transducer pre-charges dual shells and pre-twists the rear
    flywheel toward target energy. Release: channel reversion + intensity dump.
    """
    cfg = cfg or SlingshotConfig()
    charge_cfg = cfg.charge or DualChargeConfig()
    phase_cfg = cfg.phase or PhaseAlignmentConfig()
    obs_cfg = cfg.observer or ObserverSyncConfig()
    if dx is None:
        dx = float(x[0, 1] - x[0, 0]) if x.shape[1] > 1 else 0.1

    phase = cycle_phase_at_time(t, cfg)
    drive_freq = drive_frequency_for_phase(phase, cfg)
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

    locked = phase in (SlingshotPhase.STORE, SlingshotPhase.READY)
    vib = charged_vibration_field(
        x, y, t=t, cfg=charge_cfg, omega0=phase_cfg.medium_resonance
    )
    align_field = local_phase_alignment_field(
        x, y, vib["shell_envelope"], psi, active_phase
    )
    void_amp = void_order_parameter(align_field, active_phase, locked=locked)
    mean_void = float(np.mean(void_amp))

    tx_report: TransducerReport | None = None
    e_es = e_tw = e_geo = 0.0
    ch_dir = ch_int = 0.0
    ch_rev = False
    channel_field = None
    flow_field = None
    ready = False
    pump_active = False
    precharge_power = pretwist_power = pumped_total = 0.0
    charge_density = vib["charge_density"]
    e_field = vib["e_field"]
    velocity = vib.get("velocity")
    displacement = vib["displacement"]
    coupling = vib["coupling"]

    if cfg.use_transducer:
        tcfg = _default_transducer_config(cfg)
        if transducer is None:
            transducer = FluxTransducer(cfg=tcfg)
            transducer.ledger = ledger_from_total(stored_energy, tcfg)
        else:
            transducer.cfg = replace(
                transducer.cfg,
                intensity=cfg.release_intensity,
                pump_intensity=cfg.pump_intensity,
                axis=charge_cfg.axis,
                capacity=tcfg.capacity,
            )

        intensity = cfg.release_intensity
        pump_intensity = cfg.pump_intensity
        if cfg.enable_observer:
            if phase == SlingshotPhase.RELEASE:
                intensity = float(
                    np.clip(intensity * (0.85 + 0.3 * max(obs_fb, 0.0)), 0.0, 1.0)
                )
            if phase in (SlingshotPhase.STORE, SlingshotPhase.READY):
                # Observer can request harder pre-charge / pre-twist
                pump_intensity = float(
                    np.clip(pump_intensity * (0.85 + 0.3 * max(obs_fb, 0.0)), 0.0, 1.0)
                )

        tx_report = transducer.step(
            x,
            y,
            mode=phase.value,
            e_field=e_field,
            charge_density=charge_density,
            void_amplitude_field=void_amp,
            psi=psi,
            dt=dt,
            dx=dx,
            velocity=velocity,
            intensity=intensity,
            pump_intensity=pump_intensity,
        )

        # Apply pre-charge boost to dual-shell fields
        if tx_report.charge_boost_field is not None and tx_report.pump is not None:
            if tx_report.pump.charge_boost > 1e-12:
                charge_density, e_field = apply_precharge_to_fields(
                    charge_density, e_field, tx_report.charge_boost_field
                )
        # Apply pre-twist azimuthal velocity
        if tx_report.pretwist_velocity is not None and velocity is not None:
            pvx, pvy = tx_report.pretwist_velocity
            # velocity from charged_vibration is scalar field; build vector boost
            # and fold magnitude into coupling / displacement proxies
            pretwist_mag = np.sqrt(pvx**2 + pvy**2)
            if velocity is not None:
                velocity = velocity + pretwist_mag
            displacement = displacement + 0.35 * pretwist_mag * np.sign(
                displacement + 1e-12
            )
            coupling = e_field * (velocity + 1e-12 * displacement)

        # Phase assist from active pumping (help hold lock while loading)
        if tx_report.pump is not None and tx_report.pump.phase_assist > 0.0:
            psi = float(np.clip(psi + tx_report.pump.phase_assist * (1.0 - psi), 0.0, 1.0))

        stored = tx_report.ledger.total
        impulse = tx_report.directed_impulse
        e_es = tx_report.ledger.electrostatic
        e_tw = tx_report.ledger.twist
        e_geo = tx_report.ledger.geometric
        ch_dir = tx_report.channels.direction
        ch_int = tx_report.channels.intensity
        ch_rev = tx_report.channels.reversed
        channel_field = tx_report.channel_field
        flow_field = tx_report.flow_intensity_field
        if tx_report.pump is not None:
            ready = tx_report.pump.ready
            pump_active = tx_report.pump.pump_active
            precharge_power = tx_report.pump.precharge_power
            pretwist_power = tx_report.pump.pretwist_power
        pumped_total = transducer.total_pumped.total
        shift_boost = transducer_shift_contribution(
            x,
            y,
            tx_report.ledger,
            impulse,
            tx_report.channels,
            transducer.cfg,
            rear_bias=cfg.rear_bias,
            pump=tx_report.pump,
        )
        # Extra pretwist velocity components available for metric (vector form)
        pretwist_vx = pretwist_vy = None
        if tx_report.pretwist_velocity is not None:
            pretwist_vx, pretwist_vy = tx_report.pretwist_velocity
    else:
        stored, impulse = update_flywheel(stored_energy, phase, psi, dt, cfg)
        shift_boost = flywheel_shift_boost(x, y, stored, impulse, cfg)
        pretwist_vx = pretwist_vy = None

    if base_defect is not None:
        lam_mod = electro_vibrational_defect_modulation(
            base_defect, coupling, strength=cfg.modulation_strength
        )
        lam_mod = lam_mod * (1.0 - 0.65 * np.clip(void_amp, 0.0, 1.0))
        if channel_field is not None:
            lam_mod = modulate_defect_by_channels(lam_mod, channel_field, strength=0.22)
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
        e_electrostatic=e_es,
        e_twist=e_tw,
        e_geometric=e_geo,
        channel_direction=ch_dir,
        channel_intensity=ch_int,
        channels_reversed=ch_rev,
        ready=ready,
        pump_active=pump_active,
        precharge_power=precharge_power,
        pretwist_power=pretwist_power,
        pumped_total=pumped_total,
    )

    breakdown = None
    passive_total = 0.0
    pumped_efficiency = 0.0
    if cfg.use_transducer and transducer is not None:
        breakdown = transducer.get_storage_breakdown()
        passive_total = float(transducer.total_passive.total)
        pumped_efficiency = float(transducer.pumped_efficiency)

    out: dict = {
        "state": state,
        "alignment_field": align_field,
        "void_amplitude_field": void_amp,
        "shift_boost": shift_boost,
        "charge_density": charge_density,
        "e_field": e_field,
        "coupling": coupling,
        "displacement": displacement,
        "defect_modulated": lam_mod,
        "observer": obs,
        "psi": psi,
        "stored_energy": stored,
        "release_impulse": impulse,
        "phase": phase,
        "transducer": tx_report,
        "transducer_obj": transducer if cfg.use_transducer else None,
        "ledger": {
            "electrostatic": e_es,
            "twist": e_tw,
            "geometric": e_geo,
            "total": stored,
            "total_stored": stored,
        },
        "storage_breakdown": breakdown,
        "channel_field": channel_field,
        "flow_intensity_field": flow_field,
        "channel_direction": ch_dir,
        "channel_intensity": ch_int,
        "channels_reversed": ch_rev,
        "ready": ready,
        "pump_active": pump_active,
        "precharge_power": precharge_power,
        "pretwist_power": pretwist_power,
        "pumped_total": pumped_total,
        "passive_total": passive_total,
        "pumped_efficiency": pumped_efficiency,
        "pretwist_vx": pretwist_vx if cfg.use_transducer else None,
        "pretwist_vy": pretwist_vy if cfg.use_transducer else None,
    }
    return out


def simulate_slingshot_cycle(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    t_max: float = 5.0,
    dt: float = 0.05,
    cfg: SlingshotConfig | None = None,
    base_defect: NDArray[np.floating] | None = None,
) -> dict:
    """Integrate nucleation → store (active pump) → ready → release → coast.

    Returns time series including ledger channels, pump powers, and ready flag.
    """
    cfg = cfg or SlingshotConfig()
    times = np.arange(0.0, t_max + 0.5 * dt, dt)
    series: dict = {
        "t": [],
        "psi": [],
        "stored": [],
        "impulse": [],
        "void": [],
        "phase": [],
        "detuning": [],
        "observer_fb": [],
        "e_electrostatic": [],
        "e_twist": [],
        "e_geometric": [],
        "channel_direction": [],
        "channel_intensity": [],
        "ready": [],
        "pump_active": [],
        "precharge_power": [],
        "pretwist_power": [],
        "pumped_total": [],
        "passive_total": [],
        "pumped_efficiency": [],
    }
    stored = 0.0
    last = None
    transducer: FluxTransducer | None = None
    if cfg.use_transducer:
        transducer = FluxTransducer(cfg=_default_transducer_config(cfg))

    dx = float(x[0, 1] - x[0, 0]) if x.shape[1] > 1 else 0.1
    for t in times:
        step = resonant_control_step(
            x,
            y,
            float(t),
            stored_energy=stored,
            cfg=cfg,
            dt=dt,
            base_defect=base_defect,
            transducer=transducer,
            dx=dx,
        )
        if step.get("transducer_obj") is not None:
            transducer = step["transducer_obj"]
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
        series["e_electrostatic"].append(st.e_electrostatic)
        series["e_twist"].append(st.e_twist)
        series["e_geometric"].append(st.e_geometric)
        series["channel_direction"].append(st.channel_direction)
        series["channel_intensity"].append(st.channel_intensity)
        series["ready"].append(1.0 if st.ready else 0.0)
        series["pump_active"].append(1.0 if st.pump_active else 0.0)
        series["precharge_power"].append(st.precharge_power)
        series["pretwist_power"].append(st.pretwist_power)
        series["pumped_total"].append(st.pumped_total)
        series["passive_total"].append(float(step.get("passive_total", 0.0)))
        series["pumped_efficiency"].append(float(step.get("pumped_efficiency", 0.0)))
        last = step

    float_keys = (
        "t",
        "psi",
        "stored",
        "impulse",
        "void",
        "detuning",
        "observer_fb",
        "e_electrostatic",
        "e_twist",
        "e_geometric",
        "channel_direction",
        "channel_intensity",
        "ready",
        "pump_active",
        "precharge_power",
        "pretwist_power",
        "pumped_total",
        "passive_total",
        "pumped_efficiency",
    )
    for key in float_keys:
        series[key] = np.asarray(series[key], dtype=float)

    return {
        "series": series,
        "final": last,
        "config": cfg,
        "transducer": transducer,
    }


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
