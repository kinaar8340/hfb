"""Craft / payload dynamics driven by the flux-transducer engine.

The transducer remains the local motor–generator–gearbox (ledger + impulse).
This module is the next layer: it integrates **rigid-body / payload** motion
from:

  - ``get_storage_breakdown()`` — stored energy state and channel mix
  - ``directed_impulse`` — metered slingshot kick on release
  - optional shift-field summary (mean axial bias)

Analog only — effective-metric kick mapped to a 2D point-mass (or simple
rigid body) in the laboratory frame, not literal GR propulsion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CraftConfig:
    """Payload / hull parameters for analog craft motion."""

    mass: float = 1.0
    """Inertial mass of craft + payload (normalized units)."""
    axis: str = "x"
    """Primary boost axis ('x' or 'y')."""
    impulse_coupling: float = 1.0
    """Maps directed_impulse → Δv scale: Δv = coupling · J / m."""
    store_recoil: float = 0.04
    """Small rearward acceleration while storing (tension load reaction)."""
    ready_hold_gain: float = 0.01
    """Tiny residual tension while READY (should not walk the craft)."""
    drag: float = 0.05
    """Linear medium drag coefficient (a_drag = −drag · v)."""
    twist_lateral: float = 0.12
    """Fraction of twist-channel energy that couples to orthogonal motion."""
    geometric_spring: float = 0.08
    """Weak restoring spring from geometric packing (toward origin)."""
    es_boost: float = 0.10
    """Electrostatic store fraction slightly increases impulse coupling at dump."""
    max_speed: float = 8.0
    """Soft speed ceiling (tanh-style clip)."""
    x0: float = 0.0
    y0: float = 0.0
    vx0: float = 0.0
    vy0: float = 0.0


@dataclass
class CraftState:
    """Instantaneous craft kinematics + bookkeeping."""

    t: float = 0.0
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    speed: float = 0.0
    delta_v: float = 0.0
    """Speed change this step (signed along primary axis)."""
    integrated_impulse: float = 0.0
    """Cumulative directed impulse absorbed by the craft."""
    kinetic_energy: float = 0.0
    phase: str = "idle"
    total_stored: float = 0.0
    pumped_efficiency: float = 0.0

    def as_dict(self) -> dict[str, float | str]:
        return {
            "t": float(self.t),
            "x": float(self.x),
            "y": float(self.y),
            "vx": float(self.vx),
            "vy": float(self.vy),
            "ax": float(self.ax),
            "ay": float(self.ay),
            "speed": float(self.speed),
            "delta_v": float(self.delta_v),
            "integrated_impulse": float(self.integrated_impulse),
            "kinetic_energy": float(self.kinetic_energy),
            "phase": str(self.phase),
            "total_stored": float(self.total_stored),
            "pumped_efficiency": float(self.pumped_efficiency),
        }


def _axis_unit(axis: str) -> tuple[float, float]:
    if axis == "y":
        return 0.0, 1.0
    return 1.0, 0.0


def impulse_to_delta_v(
    directed_impulse: float,
    mass: float,
    coupling: float = 1.0,
    es_fraction: float = 0.0,
    es_boost: float = 0.0,
) -> float:
    """Convert transducer directed impulse to axial Δv.

    Optional electrostatic store fraction slightly raises coupling at dump
    (pre-charged envelopes give a cleaner kick).
    """
    m = max(mass, 1e-9)
    c = coupling * (1.0 + es_boost * float(np.clip(es_fraction, 0.0, 1.0)))
    return float(c * directed_impulse / m)


def craft_acceleration(
    state: CraftState,
    *,
    directed_impulse: float,
    dt: float,
    phase: str,
    breakdown: dict | None,
    cfg: CraftConfig,
) -> tuple[float, float, float]:
    """Compute (ax, ay, axial_delta_v) for one step.

    Release: primary kick from directed_impulse.
    Store: mild rear recoil proportional to total_stored.
    Ready: near-zero hold.
    Always: drag + optional geometric spring + twist lateral coupling.
    """
    cfg = cfg or CraftConfig()
    ux, uy = _axis_unit(cfg.axis)
    ox, oy = -uy, ux  # orthogonal

    bd = breakdown or {}
    total_stored = float(bd.get("total_stored", bd.get("total", 0.0)))
    es = float(bd.get("electrostatic", 0.0))
    twist = float(bd.get("twist", 0.0))
    geometric = float(bd.get("geometric", 0.0))
    es_frac = es / max(total_stored, 1e-12) if total_stored > 1e-12 else 0.0
    twist_frac = twist / max(total_stored, 1e-12) if total_stored > 1e-12 else 0.0
    geo_frac = geometric / max(total_stored, 1e-12) if total_stored > 1e-12 else 0.0

    phase_l = phase.lower()
    ax = ay = 0.0
    axial_dv = 0.0
    dt_safe = max(dt, 1e-12)

    if phase_l == "release" and directed_impulse > 0.0:
        axial_dv = impulse_to_delta_v(
            directed_impulse,
            cfg.mass,
            cfg.impulse_coupling,
            es_fraction=es_frac,
            es_boost=cfg.es_boost,
        )
        # Instantaneous acceleration proxy for bookkeeping
        a_mag = axial_dv / dt_safe
        ax += a_mag * ux
        ay += a_mag * uy
        # Twist channel bleeds a little energy into orthogonal motion
        lat = cfg.twist_lateral * twist_frac * a_mag
        ax += lat * ox
        ay += lat * oy
    elif phase_l == "store" and total_stored > 0.0:
        # Reaction while loading the rear flywheel (small reverse)
        a_mag = -cfg.store_recoil * total_stored / max(cfg.mass, 1e-9)
        ax += a_mag * ux
        ay += a_mag * uy
    elif phase_l == "ready" and total_stored > 0.0:
        a_mag = -cfg.ready_hold_gain * total_stored / max(cfg.mass, 1e-9)
        ax += a_mag * ux
        ay += a_mag * uy

    # Geometric packing: weak spring toward origin (rubber-band residual)
    if geometric > 0.0 or geo_frac > 0.0:
        k = cfg.geometric_spring * (0.25 + 0.75 * geo_frac)
        ax += -k * state.x / max(cfg.mass, 1e-9)
        ay += -k * state.y / max(cfg.mass, 1e-9)

    # Medium drag
    ax += -cfg.drag * state.vx
    ay += -cfg.drag * state.vy

    return float(ax), float(ay), float(axial_dv)


def craft_step(
    state: CraftState,
    *,
    directed_impulse: float = 0.0,
    dt: float = 0.05,
    phase: str = "coast",
    breakdown: dict | None = None,
    cfg: CraftConfig | None = None,
) -> CraftState:
    """Advance craft kinematics one timestep (semi-implicit Euler).

    Continuous forces (drag, spring, store recoil) integrate as a·dt.
    Release kick is applied as an authoritative impulsive Δv along the
    boost axis (plus small twist-lateral component).
    """
    cfg = cfg or CraftConfig()
    bd = breakdown or {}
    phase_l = phase.lower()
    ux, uy = _axis_unit(cfg.axis)
    ox, oy = -uy, ux

    # Continuous forces only (no impulse double-count)
    continuous_phase = "coast" if phase_l == "release" else phase_l
    ax_c, ay_c, _ = craft_acceleration(
        state,
        directed_impulse=0.0,
        dt=dt,
        phase=continuous_phase if continuous_phase != "release" else "coast",
        breakdown=breakdown,
        cfg=cfg,
    )
    # Store/ready recoil still applied when in those phases
    if phase_l in ("store", "ready", "nucleate"):
        ax_c, ay_c, _ = craft_acceleration(
            state,
            directed_impulse=0.0,
            dt=dt,
            phase=phase_l,
            breakdown=breakdown,
            cfg=cfg,
        )

    axial_dv = 0.0
    lat_dv = 0.0
    if phase_l == "release" and directed_impulse > 0.0:
        total_stored = float(bd.get("total_stored", bd.get("total", 0.0)))
        es = float(bd.get("electrostatic", 0.0))
        twist = float(bd.get("twist", 0.0))
        es_frac = es / max(total_stored, 1e-12) if total_stored > 1e-12 else 0.0
        twist_frac = twist / max(total_stored, 1e-12) if total_stored > 1e-12 else 0.0
        axial_dv = impulse_to_delta_v(
            directed_impulse,
            cfg.mass,
            cfg.impulse_coupling,
            es_fraction=es_frac,
            es_boost=cfg.es_boost,
        )
        lat_dv = cfg.twist_lateral * twist_frac * axial_dv

    vx = state.vx + ax_c * dt + axial_dv * ux + lat_dv * ox
    vy = state.vy + ay_c * dt + axial_dv * uy + lat_dv * oy

    speed = float(np.hypot(vx, vy))
    if speed > cfg.max_speed:
        scale = cfg.max_speed / speed
        vx *= scale
        vy *= scale
        speed = cfg.max_speed

    x = state.x + vx * dt
    y = state.y + vy * dt
    ke = 0.5 * cfg.mass * speed**2
    ax = (vx - state.vx) / max(dt, 1e-12)
    ay = (vy - state.vy) / max(dt, 1e-12)

    return CraftState(
        t=state.t + dt,
        x=float(x),
        y=float(y),
        vx=float(vx),
        vy=float(vy),
        ax=float(ax),
        ay=float(ay),
        speed=float(np.hypot(vx, vy)),
        delta_v=float(axial_dv),
        integrated_impulse=float(state.integrated_impulse + max(directed_impulse, 0.0)),
        kinetic_energy=float(ke),
        phase=phase,
        total_stored=float(bd.get("total_stored", 0.0)),
        pumped_efficiency=float(bd.get("pumped_efficiency", 0.0)),
    )


def initial_craft_state(cfg: CraftConfig | None = None, t0: float = 0.0) -> CraftState:
    cfg = cfg or CraftConfig()
    speed = float(np.hypot(cfg.vx0, cfg.vy0))
    return CraftState(
        t=t0,
        x=cfg.x0,
        y=cfg.y0,
        vx=cfg.vx0,
        vy=cfg.vy0,
        speed=speed,
        kinetic_energy=0.5 * cfg.mass * speed**2,
    )


def integrate_craft_from_series(
    series: dict,
    cfg: CraftConfig | None = None,
    breakdowns: list[dict] | None = None,
) -> dict[str, NDArray]:
    """Integrate craft motion from a slingshot time series.

    Expects series keys: t, impulse, phase (list of str), and preferably
    stored / pumped_efficiency. Optional per-step breakdowns list.
    """
    cfg = cfg or CraftConfig()
    t = np.asarray(series["t"], dtype=float)
    impulse = np.asarray(series.get("impulse", np.zeros_like(t)), dtype=float)
    phases = series.get("phase", ["coast"] * len(t))
    stored = np.asarray(series.get("stored", np.zeros_like(t)), dtype=float)
    peff = np.asarray(series.get("pumped_efficiency", np.zeros_like(t)), dtype=float)
    e_es = np.asarray(series.get("e_electrostatic", np.zeros_like(t)), dtype=float)
    e_tw = np.asarray(series.get("e_twist", np.zeros_like(t)), dtype=float)
    e_geo = np.asarray(series.get("e_geometric", np.zeros_like(t)), dtype=float)

    state = initial_craft_state(cfg, t0=float(t[0]) if len(t) else 0.0)
    out = {
        "t": [],
        "x": [],
        "y": [],
        "vx": [],
        "vy": [],
        "ax": [],
        "ay": [],
        "speed": [],
        "delta_v": [],
        "integrated_impulse": [],
        "kinetic_energy": [],
        "phase": [],
    }

    for i in range(len(t)):
        if i == 0:
            dt = float(t[1] - t[0]) if len(t) > 1 else 0.05
        else:
            dt = float(t[i] - t[i - 1])
            if dt <= 0:
                dt = 0.05
        if breakdowns is not None and i < len(breakdowns) and breakdowns[i]:
            bd = breakdowns[i]
        else:
            bd = {
                "total_stored": float(stored[i]),
                "electrostatic": float(e_es[i]),
                "twist": float(e_tw[i]),
                "geometric": float(e_geo[i]),
                "pumped_efficiency": float(peff[i]),
            }
        phase = phases[i] if i < len(phases) else "coast"
        if hasattr(phase, "value"):
            phase = phase.value
        state = craft_step(
            state,
            directed_impulse=float(impulse[i]),
            dt=dt,
            phase=str(phase),
            breakdown=bd,
            cfg=cfg,
        )
        # Align reported time with series
        state.t = float(t[i])
        for key in ("x", "y", "vx", "vy", "ax", "ay", "speed", "delta_v", "integrated_impulse", "kinetic_energy"):
            out[key].append(getattr(state, key))
        out["t"].append(state.t)
        out["phase"].append(state.phase)

    for key in out:
        if key == "phase":
            continue
        out[key] = np.asarray(out[key], dtype=float)
    return out
