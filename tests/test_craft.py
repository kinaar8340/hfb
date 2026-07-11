"""Tests for craft / payload dynamics layer."""

from __future__ import annotations

import numpy as np

from hfb.craft import (
    CraftConfig,
    CraftState,
    craft_step,
    impulse_to_delta_v,
    integrate_craft_from_series,
    initial_craft_state,
    simulate_mission,
    simulate_mission_coupled,
    MissionConfig,
)
from hfb.electro_vibrational import (
    PhaseAlignmentConfig,
    SlingshotConfig,
    TransducerConfig,
)
from hfb.utils.grid import cartesian_grid


def test_impulse_to_delta_v_scales_with_mass():
    dv1 = impulse_to_delta_v(1.0, mass=1.0, coupling=1.0)
    dv2 = impulse_to_delta_v(1.0, mass=2.0, coupling=1.0)
    assert dv1 == 1.0
    assert dv2 == 0.5


def test_craft_step_release_moves_forward():
    cfg = CraftConfig(mass=1.0, axis="x", impulse_coupling=1.0, drag=0.0, geometric_spring=0.0)
    state = initial_craft_state(cfg)
    bd = {
        "total_stored": 0.9,
        "electrostatic": 0.4,
        "twist": 0.3,
        "geometric": 0.2,
        "pumped_efficiency": 0.5,
    }
    nxt = craft_step(
        state,
        directed_impulse=0.5,
        dt=0.1,
        phase="release",
        breakdown=bd,
        cfg=cfg,
    )
    assert nxt.vx > state.vx
    assert nxt.delta_v > 0.0
    assert nxt.integrated_impulse > 0.0
    assert nxt.x > state.x


def test_store_recoil_is_rearward():
    cfg = CraftConfig(mass=1.0, axis="x", store_recoil=0.2, drag=0.0, geometric_spring=0.0)
    state = initial_craft_state(cfg)
    bd = {"total_stored": 1.0, "electrostatic": 0.4, "twist": 0.3, "geometric": 0.3}
    nxt = craft_step(state, directed_impulse=0.0, dt=0.1, phase="store", breakdown=bd, cfg=cfg)
    assert nxt.vx < 0.0  # rearward reaction while loading


def test_integrate_from_synthetic_series():
    t = np.linspace(0, 2.0, 21)
    impulse = np.zeros_like(t)
    impulse[15:18] = 0.3  # release window
    series = {
        "t": t,
        "impulse": impulse,
        "phase": ["store"] * 15 + ["release"] * 3 + ["coast"] * 3,
        "stored": np.linspace(0, 1, 21),
        "e_electrostatic": np.linspace(0, 0.4, 21),
        "e_twist": np.linspace(0, 0.3, 21),
        "e_geometric": np.linspace(0, 0.3, 21),
        "pumped_efficiency": np.full(21, 0.4),
    }
    craft = integrate_craft_from_series(series, cfg=CraftConfig(drag=0.02))
    assert craft["speed"].max() > 0.0
    assert craft["integrated_impulse"][-1] > 0.0
    assert len(craft["x"]) == len(t)


def test_simulate_mission_end_to_end():
    x, y = cartesian_grid(32, 32, extent=2.5)
    mission = MissionConfig(
        slingshot=SlingshotConfig(
            nucleate_duration=0.3,
            store_duration=1.2,
            ready_duration=0.3,
            release_duration=0.6,
            use_transducer=True,
            pump_intensity=1.0,
            release_intensity=1.0,
            phase=PhaseAlignmentConfig(threshold=0.5, drive_frequency=1.0, medium_resonance=1.0),
            transducer=TransducerConfig(
                capacity=1.0,
                enable_precharge=True,
                enable_pretwist=True,
                precharge_rate=0.7,
                pretwist_rate=0.6,
                target_energy=0.9,
            ),
        ),
        craft=CraftConfig(mass=1.0, impulse_coupling=1.2, drag=0.03),
        t_max=3.5,
        dt=0.1,
        grid_nx=32,
        grid_extent=2.5,
    )
    result = simulate_mission(x, y, cfg=mission)
    assert "craft" in result and "series" in result
    assert result["craft"]["speed"].max() >= 0.0
    # With a successful pump/release, expect net axial progress or impulse integral
    assert result["craft"]["integrated_impulse"][-1] >= 0.0
    assert "x" in result["final_craft"]


def test_simulate_mission_coupled_matches_keys():
    x, y = cartesian_grid(24, 24, extent=2.0)
    mission = MissionConfig(
        slingshot=SlingshotConfig(
            nucleate_duration=0.2,
            store_duration=0.8,
            ready_duration=0.2,
            release_duration=0.5,
            phase=PhaseAlignmentConfig(threshold=0.5, drive_frequency=1.0, medium_resonance=1.0),
            transducer=TransducerConfig(enable_precharge=True, enable_pretwist=True, precharge_rate=0.6),
        ),
        craft=CraftConfig(mass=1.0),
        t_max=2.5,
        dt=0.1,
    )
    result = simulate_mission_coupled(x, y, cfg=mission)
    assert len(result["craft"]["t"]) == len(result["series"]["t"])
    assert result["final_craft"]["speed"] >= 0.0
    assert "energy_flow" in result
    assert result["energy_flow"]["final_ke"] >= 0.0


def test_energy_flow_summary_fields():
    from hfb.craft import energy_flow_summary, format_energy_flow_summary

    x, y = cartesian_grid(28, 28, extent=2.2)
    mission = MissionConfig(
        slingshot=SlingshotConfig(
            nucleate_duration=0.3,
            store_duration=1.0,
            ready_duration=0.2,
            release_duration=0.5,
            phase=PhaseAlignmentConfig(threshold=0.5, drive_frequency=1.0, medium_resonance=1.0),
            transducer=TransducerConfig(
                enable_precharge=True, enable_pretwist=True, precharge_rate=0.7, pretwist_rate=0.6
            ),
        ),
        craft=CraftConfig(mass=1.0, impulse_coupling=1.0),
        t_max=3.0,
        dt=0.1,
    )
    result = simulate_mission(x, y, cfg=mission)
    s = result["energy_flow"]
    for key in (
        "total_pumped",
        "total_passive",
        "net_impulse",
        "final_ke",
        "efficiency_ke_over_pumped",
        "efficiency_ke_over_intake",
    ):
        assert key in s
    text = format_energy_flow_summary(s)
    assert "total pumped" in text
    assert s["efficiency_ke_over_pumped"] >= 0.0


def test_craft_throttle_feedback_boosts_when_slow():
    from hfb.craft import CraftState, craft_throttle_feedback

    cfg = MissionConfig(
        enable_craft_feedback=True,
        feedback_target_speed=1.0,
        feedback_target_position=2.0,
        feedback_pump_gain=0.5,
        feedback_release_gain=0.5,
        feedback_position_gain=0.3,
        feedback_max_boost=0.4,
        craft=CraftConfig(axis="x"),
    )
    slow = CraftState(x=0.0, vx=0.0, speed=0.0)
    fb_store = craft_throttle_feedback(slow, "store", cfg)
    assert fb_store["pump_multiplier"] > 1.0
    fb_rel = craft_throttle_feedback(slow, "release", cfg)
    assert fb_rel["release_multiplier"] > 1.0

    fast = CraftState(x=3.0, vx=2.0, speed=2.0)
    fb_fast = craft_throttle_feedback(fast, "release", cfg)
    assert fb_fast["release_multiplier"] < fb_rel["release_multiplier"]


def test_coupled_feedback_records_multipliers():
    x, y = cartesian_grid(24, 24, extent=2.0)
    mission = MissionConfig(
        slingshot=SlingshotConfig(
            nucleate_duration=0.2,
            store_duration=0.8,
            ready_duration=0.2,
            release_duration=0.5,
            phase=PhaseAlignmentConfig(threshold=0.5, drive_frequency=1.0, medium_resonance=1.0),
            transducer=TransducerConfig(enable_precharge=True, enable_pretwist=True, precharge_rate=0.6),
        ),
        craft=CraftConfig(mass=1.0),
        t_max=2.5,
        dt=0.1,
        enable_craft_feedback=True,
        feedback_target_speed=1.5,
        feedback_pump_gain=0.4,
        feedback_release_gain=0.4,
    )
    result = simulate_mission_coupled(x, y, cfg=mission)
    assert result["coupled"] is True
    assert "pump_multiplier" in result["series"]
    assert result["series"]["pump_multiplier"].max() >= 1.0
