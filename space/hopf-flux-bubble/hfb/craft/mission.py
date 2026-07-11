"""End-to-end mission: transducer control cycle + craft integration.

Couples ``electro_vibrational.simulate_slingshot_cycle`` (engine) to
``craft.dynamics`` (payload equations of motion).

Optional **craft→throttle feedback** (coupled mode only) closes a light
observer-style loop: craft speed/position gently biases pump and release
intensities without owning the transducer ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hfb.craft.dynamics import (
    CraftConfig,
    CraftState,
    craft_step,
    initial_craft_state,
    integrate_craft_from_series,
)
from hfb.electro_vibrational.dynamics import (
    SlingshotConfig,
    SlingshotPhase,
    cycle_phase_at_time,
    resonant_control_step,
    simulate_slingshot_cycle,
)
from hfb.electro_vibrational.transducer import FluxTransducer
from hfb.utils.grid import cartesian_grid


@dataclass
class MissionConfig:
    """Combined engine + craft mission parameters."""

    slingshot: SlingshotConfig | None = None
    craft: CraftConfig | None = None
    t_max: float = 5.5
    dt: float = 0.05
    grid_nx: int = 48
    grid_extent: float = 3.0

    # --- Craft → throttle feedback (coupled mode) ---
    enable_craft_feedback: bool = False
    """If True, craft speed/position slightly biases pump/release throttles."""
    feedback_target_speed: float = 0.8
    """Desired cruise speed (normalized); under-speed pumps/releases harder."""
    feedback_target_position: float = 1.0
    """Desired axial progress; lag boosts pump during store."""
    feedback_pump_gain: float = 0.35
    """How strongly speed/position error raises pump_intensity (store/ready)."""
    feedback_release_gain: float = 0.35
    """How strongly speed error raises release_intensity (release)."""
    feedback_position_gain: float = 0.15
    """Extra pump bias from axial position lag."""
    feedback_max_boost: float = 0.40
    """Clamp |throttle multiplier − 1|."""


def craft_throttle_feedback(
    craft_state: CraftState,
    phase: str | SlingshotPhase,
    cfg: MissionConfig,
    craft_cfg: CraftConfig | None = None,
) -> dict[str, float]:
    """Map craft kinematics → pump/release intensity multipliers.

    Returns multipliers near 1.0 (clipped). Under target speed / position
    lag → pump or release harder; overspeed → slightly softer release.
    Does not touch the energy ledger — only throttle knobs.
    """
    craft_cfg = craft_cfg or cfg.craft or CraftConfig()
    if not cfg.enable_craft_feedback:
        return {
            "pump_multiplier": 1.0,
            "release_multiplier": 1.0,
            "speed_error": 0.0,
            "position_error": 0.0,
        }

    phase_s = phase.value if hasattr(phase, "value") else str(phase)
    phase_l = phase_s.lower()
    axis = craft_cfg.axis
    pos = craft_state.x if axis == "x" else craft_state.y
    speed = craft_state.speed

    speed_err = cfg.feedback_target_speed - speed  # >0 if too slow
    pos_err = cfg.feedback_target_position - pos  # >0 if behind

    pump_mult = 1.0
    release_mult = 1.0

    if phase_l in ("store", "ready", "nucleate"):
        # Pump harder when slow or lagging target position
        bias = (
            cfg.feedback_pump_gain * np.tanh(speed_err)
            + cfg.feedback_position_gain * np.tanh(pos_err)
        )
        pump_mult = 1.0 + float(np.clip(bias, -cfg.feedback_max_boost, cfg.feedback_max_boost))
    elif phase_l == "release":
        # Release harder if still under target speed; ease if already fast
        bias = cfg.feedback_release_gain * np.tanh(speed_err)
        release_mult = 1.0 + float(
            np.clip(bias, -cfg.feedback_max_boost, cfg.feedback_max_boost)
        )

    pump_mult = float(np.clip(pump_mult, 1.0 - cfg.feedback_max_boost, 1.0 + cfg.feedback_max_boost))
    release_mult = float(
        np.clip(release_mult, 1.0 - cfg.feedback_max_boost, 1.0 + cfg.feedback_max_boost)
    )
    return {
        "pump_multiplier": pump_mult,
        "release_multiplier": release_mult,
        "speed_error": float(speed_err),
        "position_error": float(pos_err),
    }


def energy_flow_summary(result: dict[str, Any]) -> dict[str, float]:
    """Summarize engine→craft energy conversion for a mission result.

    Reports:
      - total_pumped (motor path from transducer)
      - total_passive (generator path)
      - net_impulse (∫ directed impulse delivered to craft)
      - final_ke (craft kinetic energy at end)
      - peak_ke
      - efficiency_ke_over_pumped = final_ke / total_pumped
      - efficiency_ke_over_intake = final_ke / (pumped + passive)
    """
    series = result.get("series") or {}
    craft = result.get("craft") or {}
    final = result.get("final_craft") or {}
    tx = result.get("transducer")

    if tx is not None and hasattr(tx, "total_pumped"):
        total_pumped = float(tx.total_pumped.total)
        total_passive = (
            float(tx.total_passive.total) if hasattr(tx, "total_passive") else 0.0
        )
    else:
        pumped_arr = np.asarray(series.get("pumped_total", [0.0]), dtype=float)
        passive_arr = np.asarray(series.get("passive_total", [0.0]), dtype=float)
        total_pumped = float(pumped_arr[-1]) if len(pumped_arr) else 0.0
        total_passive = float(passive_arr[-1]) if len(passive_arr) else 0.0

    imp_arr = np.asarray(craft.get("integrated_impulse", series.get("impulse", [0.0])), dtype=float)
    if "integrated_impulse" in craft and len(craft["integrated_impulse"]):
        net_impulse = float(np.asarray(craft["integrated_impulse"], dtype=float)[-1])
    else:
        net_impulse = float(np.sum(np.asarray(series.get("impulse", [0.0]), dtype=float)))

    ke_arr = np.asarray(craft.get("kinetic_energy", [0.0]), dtype=float)
    final_ke = float(final.get("kinetic_energy", ke_arr[-1] if len(ke_arr) else 0.0))
    peak_ke = float(np.max(ke_arr)) if len(ke_arr) else final_ke

    intake = total_pumped + total_passive
    eff_pumped = final_ke / total_pumped if total_pumped > 1e-12 else 0.0
    eff_intake = final_ke / intake if intake > 1e-12 else 0.0
    # Impulse-to-KE proxy (not dimensionally universal; still useful fidelity)
    eff_impulse = final_ke / net_impulse if net_impulse > 1e-12 else 0.0

    return {
        "total_pumped": total_pumped,
        "total_passive": total_passive,
        "total_intake": intake,
        "net_impulse": net_impulse,
        "final_ke": final_ke,
        "peak_ke": peak_ke,
        "efficiency_ke_over_pumped": float(eff_pumped),
        "efficiency_ke_over_intake": float(eff_intake),
        "efficiency_ke_over_impulse": float(eff_impulse),
        "pumped_fraction_of_intake": float(total_pumped / intake) if intake > 1e-12 else 0.0,
    }


def format_energy_flow_summary(summary: dict[str, float]) -> str:
    """Human-readable multi-line energy flow report."""
    return (
        "Energy flow summary (engine → craft)\n"
        f"  total pumped (motor)     : {summary['total_pumped']:.4f}\n"
        f"  total passive (generator): {summary['total_passive']:.4f}\n"
        f"  total intake             : {summary['total_intake']:.4f}\n"
        f"  net impulse delivered    : {summary['net_impulse']:.4f}\n"
        f"  final craft KE           : {summary['final_ke']:.4f}\n"
        f"  peak craft KE            : {summary['peak_ke']:.4f}\n"
        f"  efficiency KE/pumped     : {summary['efficiency_ke_over_pumped']:.4f}\n"
        f"  efficiency KE/intake     : {summary['efficiency_ke_over_intake']:.4f}\n"
        f"  motor share of intake    : {summary['pumped_fraction_of_intake']:.4f}"
    )


def _attach_summary(result: dict) -> dict:
    summary = energy_flow_summary(result)
    result["energy_flow"] = summary
    result["energy_flow_text"] = format_energy_flow_summary(summary)
    return result


def simulate_mission(
    x: NDArray[np.floating] | None = None,
    y: NDArray[np.floating] | None = None,
    cfg: MissionConfig | None = None,
    *,
    t_max: float | None = None,
    dt: float | None = None,
    slingshot: SlingshotConfig | None = None,
    craft: CraftConfig | None = None,
) -> dict:
    """Run nucleate→store→ready→release→coast and integrate craft motion.

    Returns dict with series, craft, final_craft, transducer, energy_flow, …
    """
    cfg = cfg or MissionConfig()
    slingshot = slingshot or cfg.slingshot or SlingshotConfig()
    craft_cfg = craft or cfg.craft or CraftConfig(
        axis=(slingshot.charge.axis if slingshot.charge else "x")
    )
    t_max = cfg.t_max if t_max is None else t_max
    dt = cfg.dt if dt is None else dt

    if x is None or y is None:
        x, y = cartesian_grid(cfg.grid_nx, cfg.grid_nx, extent=cfg.grid_extent)

    engine = simulate_slingshot_cycle(x, y, t_max=t_max, dt=dt, cfg=slingshot)
    series = engine["series"]
    craft_series = integrate_craft_from_series(series, cfg=craft_cfg)

    final_craft = CraftState(
        t=float(craft_series["t"][-1]) if len(craft_series["t"]) else 0.0,
        x=float(craft_series["x"][-1]) if len(craft_series["x"]) else 0.0,
        y=float(craft_series["y"][-1]) if len(craft_series["y"]) else 0.0,
        vx=float(craft_series["vx"][-1]) if len(craft_series["vx"]) else 0.0,
        vy=float(craft_series["vy"][-1]) if len(craft_series["vy"]) else 0.0,
        speed=float(craft_series["speed"][-1]) if len(craft_series["speed"]) else 0.0,
        integrated_impulse=float(craft_series["integrated_impulse"][-1])
        if len(craft_series["integrated_impulse"])
        else 0.0,
        kinetic_energy=float(craft_series["kinetic_energy"][-1])
        if len(craft_series["kinetic_energy"])
        else 0.0,
        phase=str(craft_series["phase"][-1]) if craft_series["phase"] else "idle",
    )

    result = {
        "series": series,
        "craft": craft_series,
        "final_control": engine.get("final"),
        "final_craft": final_craft.as_dict(),
        "transducer": engine.get("transducer"),
        "config": MissionConfig(
            slingshot=slingshot,
            craft=craft_cfg,
            t_max=t_max,
            dt=dt,
            enable_craft_feedback=cfg.enable_craft_feedback,
            feedback_target_speed=cfg.feedback_target_speed,
            feedback_target_position=cfg.feedback_target_position,
            feedback_pump_gain=cfg.feedback_pump_gain,
            feedback_release_gain=cfg.feedback_release_gain,
            feedback_position_gain=cfg.feedback_position_gain,
            feedback_max_boost=cfg.feedback_max_boost,
        ),
        "engine": engine,
        "coupled": False,
    }
    return _attach_summary(result)


def simulate_mission_coupled(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    cfg: MissionConfig | None = None,
) -> dict:
    """Step-coupled mission: craft state updated each control step live.

    When ``enable_craft_feedback`` is True, craft speed/position gently
    modulates pump_intensity (store/ready) and release_intensity (release),
    closing a light observer-style loop around the transducer throttles.
    """
    cfg = cfg or MissionConfig()
    base_slingshot = cfg.slingshot or SlingshotConfig()
    craft_cfg = cfg.craft or CraftConfig(
        axis=(base_slingshot.charge.axis if base_slingshot.charge else "x")
    )
    dt = cfg.dt
    times = np.arange(0.0, cfg.t_max + 0.5 * dt, dt)

    transducer: FluxTransducer | None = None
    if base_slingshot.use_transducer:
        from hfb.electro_vibrational.dynamics import _default_transducer_config

        transducer = FluxTransducer(cfg=_default_transducer_config(base_slingshot))

    craft_state = initial_craft_state(craft_cfg)
    series: dict = {
        "t": [],
        "psi": [],
        "stored": [],
        "impulse": [],
        "phase": [],
        "ready": [],
        "e_electrostatic": [],
        "e_twist": [],
        "e_geometric": [],
        "pumped_total": [],
        "passive_total": [],
        "pumped_efficiency": [],
        "precharge_power": [],
        "pretwist_power": [],
        "channel_direction": [],
        "channel_intensity": [],
        "void": [],
        "detuning": [],
        "observer_fb": [],
        "pump_active": [],
        "pump_multiplier": [],
        "release_multiplier": [],
        "speed_error": [],
        "position_error": [],
    }
    craft_series: dict = {
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

    dx = float(x[0, 1] - x[0, 0]) if x.shape[1] > 1 else 0.1
    stored = 0.0
    last = None
    base_pump = base_slingshot.pump_intensity
    base_release = base_slingshot.release_intensity

    for t in times:
        phase_preview = cycle_phase_at_time(float(t), base_slingshot)
        fb = craft_throttle_feedback(craft_state, phase_preview, cfg, craft_cfg)
        pump_i = float(np.clip(base_pump * fb["pump_multiplier"], 0.0, 1.0))
        rel_i = float(np.clip(base_release * fb["release_multiplier"], 0.0, 1.0))
        slingshot = replace(
            base_slingshot,
            pump_intensity=pump_i,
            release_intensity=rel_i,
        )

        step = resonant_control_step(
            x,
            y,
            float(t),
            stored_energy=stored,
            cfg=slingshot,
            dt=dt,
            transducer=transducer,
            dx=dx,
        )
        if step.get("transducer_obj") is not None:
            transducer = step["transducer_obj"]
        st = step["state"]
        stored = st.stored_energy
        phase = st.phase.value if hasattr(st.phase, "value") else str(st.phase)
        breakdown = step.get("storage_breakdown") or {
            "total_stored": st.stored_energy,
            "electrostatic": st.e_electrostatic,
            "twist": st.e_twist,
            "geometric": st.e_geometric,
            "pumped_efficiency": float(step.get("pumped_efficiency", 0.0)),
        }
        craft_state = craft_step(
            craft_state,
            directed_impulse=st.release_impulse,
            dt=dt,
            phase=phase,
            breakdown=breakdown,
            cfg=craft_cfg,
        )
        craft_state.t = float(t)

        series["t"].append(st.t)
        series["psi"].append(st.psi)
        series["stored"].append(st.stored_energy)
        series["impulse"].append(st.release_impulse)
        series["phase"].append(phase)
        series["ready"].append(1.0 if st.ready else 0.0)
        series["e_electrostatic"].append(st.e_electrostatic)
        series["e_twist"].append(st.e_twist)
        series["e_geometric"].append(st.e_geometric)
        series["pumped_total"].append(st.pumped_total)
        series["passive_total"].append(float(step.get("passive_total", 0.0)))
        series["pumped_efficiency"].append(float(step.get("pumped_efficiency", 0.0)))
        series["precharge_power"].append(st.precharge_power)
        series["pretwist_power"].append(st.pretwist_power)
        series["channel_direction"].append(st.channel_direction)
        series["channel_intensity"].append(st.channel_intensity)
        series["void"].append(st.void_amplitude)
        series["detuning"].append(st.detuning)
        series["observer_fb"].append(st.observer_feedback)
        series["pump_active"].append(1.0 if st.pump_active else 0.0)
        series["pump_multiplier"].append(fb["pump_multiplier"])
        series["release_multiplier"].append(fb["release_multiplier"])
        series["speed_error"].append(fb["speed_error"])
        series["position_error"].append(fb["position_error"])

        for key in (
            "x",
            "y",
            "vx",
            "vy",
            "ax",
            "ay",
            "speed",
            "delta_v",
            "integrated_impulse",
            "kinetic_energy",
        ):
            craft_series[key].append(getattr(craft_state, key))
        craft_series["t"].append(craft_state.t)
        craft_series["phase"].append(craft_state.phase)
        last = step

    for key, val in series.items():
        if key != "phase":
            series[key] = np.asarray(val, dtype=float)
    for key, val in craft_series.items():
        if key != "phase":
            craft_series[key] = np.asarray(val, dtype=float)

    result = {
        "series": series,
        "craft": craft_series,
        "final_control": last,
        "final_craft": craft_state.as_dict(),
        "transducer": transducer,
        "config": MissionConfig(
            slingshot=base_slingshot,
            craft=craft_cfg,
            t_max=cfg.t_max,
            dt=dt,
            enable_craft_feedback=cfg.enable_craft_feedback,
            feedback_target_speed=cfg.feedback_target_speed,
            feedback_target_position=cfg.feedback_target_position,
            feedback_pump_gain=cfg.feedback_pump_gain,
            feedback_release_gain=cfg.feedback_release_gain,
            feedback_position_gain=cfg.feedback_position_gain,
            feedback_max_boost=cfg.feedback_max_boost,
        ),
        "coupled": True,
    }
    return _attach_summary(result)
