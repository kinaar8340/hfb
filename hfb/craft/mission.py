"""End-to-end mission: transducer control cycle + craft integration.

Couples ``electro_vibrational.simulate_slingshot_cycle`` (engine) to
``craft.dynamics`` (payload equations of motion).
"""

from __future__ import annotations

from dataclasses import dataclass

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

    Returns
    -------
    dict with keys:
      series — engine time series (ψ, ledger, impulse, …)
      craft — craft kinematic series (x, y, v, speed, KE, …)
      final_control — last control step dict
      final_craft — final CraftState fields
      transducer — FluxTransducer instance (ledger / breakdown)
      config — MissionConfig used
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

    engine = simulate_slingshot_cycle(
        x, y, t_max=t_max, dt=dt, cfg=slingshot
    )
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

    return {
        "series": series,
        "craft": craft_series,
        "final_control": engine.get("final"),
        "final_craft": final_craft.as_dict(),
        "transducer": engine.get("transducer"),
        "config": MissionConfig(
            slingshot=slingshot, craft=craft_cfg, t_max=t_max, dt=dt
        ),
        "engine": engine,
    }


def simulate_mission_coupled(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    cfg: MissionConfig | None = None,
) -> dict:
    """Step-coupled mission: craft state updated each control step live.

    Same physics as ``simulate_mission`` but integrates craft inside the
    control loop (useful when craft state later feeds observer feedback).
    """
    cfg = cfg or MissionConfig()
    slingshot = cfg.slingshot or SlingshotConfig()
    craft_cfg = cfg.craft or CraftConfig(
        axis=(slingshot.charge.axis if slingshot.charge else "x")
    )
    dt = cfg.dt
    times = np.arange(0.0, cfg.t_max + 0.5 * dt, dt)

    transducer: FluxTransducer | None = None
    if slingshot.use_transducer:
        from hfb.electro_vibrational.dynamics import _default_transducer_config

        transducer = FluxTransducer(cfg=_default_transducer_config(slingshot))

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

    for t in times:
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

    return {
        "series": series,
        "craft": craft_series,
        "final_control": last,
        "final_craft": craft_state.as_dict(),
        "transducer": transducer,
        "config": MissionConfig(
            slingshot=slingshot, craft=craft_cfg, t_max=cfg.t_max, dt=dt
        ),
    }
