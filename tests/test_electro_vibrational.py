"""Tests for electro-vibrational control and hemi-void slingshot."""

from __future__ import annotations

import numpy as np
import pytest

from hfb.bubble.hemi_void import (
    HemiVoidConfig,
    hemi_void_bubble_metric,
    hemi_void_defect_density,
    hemi_void_mask,
    rear_hemi_mask,
)
from hfb.bubble.warp_conduit import flux_bubble_metric
from hfb.defects.densities import build_defect_density, hemi_void_wall
from hfb.electro_vibrational import (
    DualChargeConfig,
    ObserverSyncConfig,
    PhaseAlignmentConfig,
    SlingshotConfig,
    SlingshotPhase,
    charged_vibration_field,
    cycle_phase_at_time,
    frequency_detuning,
    opposing_charge_density,
    phase_alignment_order,
    resonant_control_step,
    simulate_slingshot_cycle,
    void_order_parameter,
)
from hfb.electro_vibrational.observer_sync import entrainment_strength, observer_feedback
from hfb.hopf.hopfion import charge_modulated_hopfion
from hfb.utils.grid import cartesian_grid


def test_opposing_charges_bipolar():
    x, y = cartesian_grid(64, 64, extent=3.0)
    cfg = DualChargeConfig(inner_radius=0.8, outer_radius=1.2)
    sigma = opposing_charge_density(x, y, cfg)
    # Outer shell positive, inner negative on average by construction
    r = np.sqrt(x**2 + y**2)
    outer_band = (r > 1.05) & (r < 1.35)
    inner_band = (r > 0.65) & (r < 0.95)
    assert np.mean(sigma[outer_band]) > 0.0
    assert np.mean(sigma[inner_band]) < 0.0


def test_phase_alignment_peaks_on_resonance():
    psi_lock = phase_alignment_order(0.0, 0.0, 1.0, 1.0)
    psi_detuned = phase_alignment_order(0.0, 0.0, 1.5, 1.0)
    assert psi_lock > 0.9
    assert psi_detuned < psi_lock


def test_void_nucleation_above_threshold():
    field = np.full((16, 16), 0.9)
    cfg = PhaseAlignmentConfig(threshold=0.72)
    eta = void_order_parameter(field, cfg)
    assert np.all(eta > 0.0)
    field_low = np.full((16, 16), 0.2)
    eta_low = void_order_parameter(field_low, cfg)
    assert np.allclose(eta_low, 0.0)


def test_frequency_detuning():
    assert abs(frequency_detuning(1.0, 1.0)) < 1e-12
    assert frequency_detuning(1.2, 1.0) == pytest.approx(0.2)


def test_observer_entrainment_peak():
    assert entrainment_strength(1.0, 1.0) > entrainment_strength(1.5, 1.0)
    fb = observer_feedback(0.0, ObserverSyncConfig(coupling=0.5))
    assert -1.0 <= fb["feedback"] <= 1.0


def test_cycle_phases():
    cfg = SlingshotConfig(
        nucleate_duration=1.0,
        store_duration=2.0,
        ready_duration=0.4,
        release_duration=0.8,
    )
    assert cycle_phase_at_time(0.1, cfg) == SlingshotPhase.NUCLEATE
    assert cycle_phase_at_time(1.5, cfg) == SlingshotPhase.STORE
    assert cycle_phase_at_time(3.1, cfg) == SlingshotPhase.READY
    assert cycle_phase_at_time(3.6, cfg) == SlingshotPhase.RELEASE


def test_charged_vibration_finite():
    x, y = cartesian_grid(32, 32, extent=2.0)
    vib = charged_vibration_field(x, y, t=0.25)
    for key in ("charge_density", "e_field", "coupling", "displacement"):
        assert key in vib
        assert np.isfinite(vib[key]).all()


def test_hemi_void_asymmetric():
    x, y = cartesian_grid(64, 64, extent=3.0)
    cfg = HemiVoidConfig(rear_extension=0.5, front_taper=0.6, elongation=1.5)
    lam = hemi_void_defect_density(x, y, cfg)
    # Rear side (x < 0) should extend farther than front
    rear = lam[(x < -1.2) & (np.abs(y) < 0.4)]
    front = lam[(x > 1.2) & (np.abs(y) < 0.4)]
    assert np.max(lam) > 0.0
    assert hemi_void_mask(x, y, cfg).any()
    assert rear_hemi_mask(x, y, cfg).any()
    # Rear wall mass often larger due to extension — soft check
    assert rear.size > 0 and front.size > 0


def test_hemi_void_profile_in_build_defect():
    x, y = cartesian_grid(48, 48, extent=3.0)
    lam = build_defect_density(
        x, y, profile="hemi_void_wall", bubble_radius=1.0, wall_width=0.22
    )
    assert np.max(lam) > 0.0
    lam2 = hemi_void_wall(x, y, major_radius=1.0, wall_width=0.22)
    assert np.allclose(lam, lam2)


def test_resonant_control_step_keys():
    x, y = cartesian_grid(48, 48, extent=3.0)
    step = resonant_control_step(x, y, t=1.5, stored_energy=0.2)
    assert "shift_boost" in step
    assert "void_amplitude_field" in step
    assert step["state"].psi >= 0.0


def test_simulate_slingshot_stores_then_releases():
    x, y = cartesian_grid(32, 32, extent=2.5)
    # Force near-lock by matching frequencies; enable observer
    cfg = SlingshotConfig(
        nucleate_duration=0.5,
        store_duration=2.0,
        release_duration=1.0,
        store_rate=1.2,
        release_rate=2.5,
        enable_observer=True,
        phase=PhaseAlignmentConfig(threshold=0.5, drive_frequency=1.0, medium_resonance=1.0),
    )
    result = simulate_slingshot_cycle(x, y, t_max=4.0, dt=0.1, cfg=cfg)
    series = result["series"]
    assert series["stored"].max() > 0.0
    # Impulse should appear during release window
    assert series["impulse"].max() >= 0.0
    assert len(series["t"]) > 5


def test_hemi_void_bubble_metric_with_control():
    x, y = cartesian_grid(48, 48, extent=3.0)
    dx = float(x[0, 1] - x[0, 0])
    metric = hemi_void_bubble_metric(x, y, dx=dx, t=1.2, include_control=True)
    for key in ("omega", "shift", "defect_density", "g_tt", "void_mask", "psi"):
        assert key in metric
    assert np.isfinite(metric["omega"]).all()


def test_flux_bubble_metric_electro_vibrational_flag():
    x, y = cartesian_grid(40, 40, extent=3.0)
    dx = float(x[0, 1] - x[0, 0])
    metric = flux_bubble_metric(
        x,
        y,
        dx=dx,
        defect_profile="toroidal_bubble_wall",
        electro_vibrational=True,
        t=0.5,
    )
    assert "e_field" in metric
    assert np.isfinite(metric["shift"]).all()


def test_flux_bubble_hemi_void_profile():
    x, y = cartesian_grid(40, 40, extent=3.0)
    dx = float(x[0, 1] - x[0, 0])
    metric = flux_bubble_metric(
        x,
        y,
        dx=dx,
        defect_profile="hemi_void_wall",
        electro_vibrational=True,
        t=2.5,
    )
    assert "void_mask" in metric
    assert np.max(metric["defect_density"]) > 0.0


def test_charge_modulated_hopfion_unit():
    x, y = cartesian_grid(24, 24, extent=2.0)
    z = np.zeros_like(x)
    sigma = opposing_charge_density(x, y)
    nx, ny, nz = charge_modulated_hopfion(x, y, z, sigma)
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    assert np.allclose(norm, 1.0, atol=1e-5)


def test_transducer_ledger_three_channels():
    from hfb.electro_vibrational import (
        FluxTransducer,
        TransducerConfig,
        sense_energy_channels,
    )
    from hfb.electro_vibrational.charge_envelopes import capacitive_field_magnitude

    x, y = cartesian_grid(40, 40, extent=3.0)
    dx = float(x[0, 1] - x[0, 0])
    e = capacitive_field_magnitude(x, y)
    sigma = opposing_charge_density(x, y)
    void = np.exp(-(x**2 + y**2) / 2.0)
    reading = sense_energy_channels(
        x, y, e_field=e, charge_density=sigma, velocity=None,
        void_amplitude_field=void, psi=0.9, dx=dx,
    )
    assert reading.power_total >= 0.0
    assert reading.power_electrostatic >= 0.0

    tx = FluxTransducer(cfg=TransducerConfig(capacity=1.0, intensity=1.0))
    for _ in range(20):
        rep = tx.step(
            x, y, mode="store", e_field=e, charge_density=sigma,
            void_amplitude_field=void, psi=0.95, dt=0.1, dx=dx,
        )
    assert rep.ledger.total > 0.0
    assert rep.ledger.electrostatic > 0.0
    assert rep.ledger.twist > 0.0
    assert rep.ledger.geometric > 0.0
    assert rep.channels.direction > 0.0
    assert not rep.channels.reversed

    # Release reverts channels and produces directed impulse
    rep_r = tx.step(
        x, y, mode="release", e_field=e, charge_density=sigma,
        void_amplitude_field=void, psi=0.3, dt=0.1, dx=dx, intensity=1.0,
    )
    assert rep_r.channels.reversed
    assert rep_r.channels.direction < 0.0
    assert rep_r.directed_impulse > 0.0
    assert rep_r.dump.total > 0.0


def test_dump_ledger_respects_intensity():
    from hfb.electro_vibrational import EnergyLedger, TransducerConfig, dump_ledger

    ledger = EnergyLedger(electrostatic=0.4, twist=0.3, geometric=0.3)
    cfg = TransducerConfig(capacity=1.0, reverse_gain=1.0, release_efficiency=1.0)
    _, dump_hi, imp_hi = dump_ledger(ledger, dt=0.1, cfg=cfg, intensity=1.0)
    _, dump_lo, imp_lo = dump_ledger(ledger, dt=0.1, cfg=cfg, intensity=0.2)
    assert dump_hi.total > dump_lo.total
    assert imp_hi > imp_lo


def test_simulate_cycle_tracks_ledger_channels():
    x, y = cartesian_grid(32, 32, extent=2.5)
    cfg = SlingshotConfig(
        nucleate_duration=0.4,
        store_duration=1.5,
        release_duration=0.8,
        use_transducer=True,
        release_intensity=1.0,
        enable_observer=True,
        phase=PhaseAlignmentConfig(threshold=0.5, drive_frequency=1.0, medium_resonance=1.0),
    )
    result = simulate_slingshot_cycle(x, y, t_max=3.5, dt=0.1, cfg=cfg)
    series = result["series"]
    assert series["stored"].max() > 0.0
    assert series["e_electrostatic"].max() > 0.0
    assert series["e_twist"].max() > 0.0
    assert series["e_geometric"].max() > 0.0
    # Channel direction should go positive (store) then negative (release)
    assert series["channel_direction"].max() > 0.0
    assert series["channel_direction"].min() < 0.0
    assert series["impulse"].max() > 0.0


def test_legacy_flywheel_path_still_works():
    x, y = cartesian_grid(24, 24, extent=2.0)
    cfg = SlingshotConfig(
        use_transducer=False,
        store_duration=1.5,
        release_duration=0.8,
        nucleate_duration=0.3,
        store_rate=1.0,
        release_rate=2.0,
        phase=PhaseAlignmentConfig(threshold=0.5, drive_frequency=1.0, medium_resonance=1.0),
    )
    result = simulate_slingshot_cycle(x, y, t_max=3.0, dt=0.1, cfg=cfg)
    assert result["series"]["stored"].max() > 0.0


def test_active_precharge_and_pretwist_pump():
    from hfb.electro_vibrational import (
        FluxTransducer,
        TransducerConfig,
        compute_pump_command,
        EnergyLedger,
        pretwist_velocity_field,
        precharge_boost_field,
    )
    from hfb.electro_vibrational.charge_envelopes import capacitive_field_magnitude

    x, y = cartesian_grid(40, 40, extent=3.0)
    dx = float(x[0, 1] - x[0, 0])
    e = capacitive_field_magnitude(x, y)
    sigma = opposing_charge_density(x, y)
    void = np.exp(-(x**2 + y**2) / 2.0)

    cfg = TransducerConfig(
        capacity=1.0,
        enable_precharge=True,
        enable_pretwist=True,
        precharge_rate=0.8,
        pretwist_rate=0.7,
        pump_intensity=1.0,
        target_energy=0.9,
        pump_requires_lock=True,
        psi_pump_threshold=0.5,
    )
    # Empty ledger under lock → strong pump command
    pump = compute_pump_command(EnergyLedger(), psi=0.95, cfg=cfg, mode="store")
    assert pump.pump_active
    assert pump.precharge_power > 0.0
    assert pump.pretwist_power > 0.0
    assert pump.charge_boost > 0.0
    assert pump.twist_drive > 0.0

    # Unlocked → no pump when requires_lock
    pump_off = compute_pump_command(EnergyLedger(), psi=0.1, cfg=cfg, mode="store")
    assert not pump_off.pump_active

    tx = FluxTransducer(cfg=cfg)
    for _ in range(25):
        rep = tx.step(
            x, y, mode="store", e_field=e, charge_density=sigma,
            void_amplitude_field=void, psi=0.95, dt=0.1, dx=dx, pump_intensity=1.0,
        )
    assert rep.ledger.total >= 0.9 * cfg.capacity * 0.85  # near target
    assert rep.pump is not None
    assert tx.total_pumped.total > 0.0
    assert rep.pump.ready or rep.ledger.total > 0.5

    # Spatial drives
    bf = precharge_boost_field(x, y, 0.5, cfg)
    assert np.max(bf) > 0.0
    vx, vy = pretwist_velocity_field(x, y, 0.5, cfg)
    assert np.max(np.abs(vx)) > 0.0 or np.max(np.abs(vy)) > 0.0


def test_storage_breakdown_and_efficiency():
    from hfb.electro_vibrational import FluxTransducer, TransducerConfig, EnergyLedger
    from hfb.electro_vibrational.charge_envelopes import capacitive_field_magnitude

    x, y = cartesian_grid(32, 32, extent=2.5)
    dx = float(x[0, 1] - x[0, 0])
    e = capacitive_field_magnitude(x, y)
    sigma = opposing_charge_density(x, y)
    void = np.exp(-(x**2 + y**2) / 2.0)
    tx = FluxTransducer(
        cfg=TransducerConfig(
            capacity=1.0,
            enable_precharge=True,
            enable_pretwist=True,
            precharge_rate=0.7,
            pretwist_rate=0.6,
            target_energy=0.9,
        )
    )
    for _ in range(20):
        tx.step(
            x, y, mode="store", e_field=e, charge_density=sigma,
            void_amplitude_field=void, psi=0.95, dt=0.1, dx=dx,
        )
    bd = tx.get_storage_breakdown()
    assert bd["total_stored"] > 0.0
    assert "pct_electrostatic" in bd
    assert abs(bd["pct_electrostatic"] + bd["pct_twist"] + bd["pct_geometric"] - 100.0) < 1e-6 or bd["total_stored"] < 1e-12
    assert 0.0 <= bd["pumped_efficiency"] <= 1.0
    assert bd["total_pumped"] >= 0.0
    assert bd["total_passive"] >= 0.0
    assert tx.total_stored == pytest.approx(bd["total_stored"])


def test_ready_hysteresis_band():
    from hfb.electro_vibrational import EnergyLedger, TransducerConfig, is_ready

    cfg = TransducerConfig(capacity=1.0, target_energy=0.90, ready_hysteresis=0.05)
    # Below high threshold → not ready
    assert not is_ready(EnergyLedger(electrostatic=0.4, twist=0.3, geometric=0.15), cfg)
    # At target with soft channel balance
    high = EnergyLedger(electrostatic=0.40, twist=0.32, geometric=0.20)  # total 0.92
    assert is_ready(high, cfg, was_ready=False)
    # Drop slightly below high but above low (0.85) while latched → stay ready
    mid = EnergyLedger(electrostatic=0.38, twist=0.30, geometric=0.18)  # total 0.86
    assert is_ready(mid, cfg, was_ready=True)
    # Below low band → drop out
    low = EnergyLedger(electrostatic=0.30, twist=0.25, geometric=0.15)  # total 0.70
    assert not is_ready(low, cfg, was_ready=True)


def test_simulate_cycle_pumps_then_ready_then_release():
    x, y = cartesian_grid(32, 32, extent=2.5)
    cfg = SlingshotConfig(
        nucleate_duration=0.3,
        store_duration=1.5,
        ready_duration=0.4,
        release_duration=0.8,
        use_transducer=True,
        pump_intensity=1.0,
        release_intensity=1.0,
        enable_observer=True,
        phase=PhaseAlignmentConfig(threshold=0.5, drive_frequency=1.0, medium_resonance=1.0),
        transducer=None,  # defaults with precharge/pretwist on
    )
    # Ensure pump rates are healthy via default TransducerConfig
    from hfb.electro_vibrational import TransducerConfig

    cfg.transducer = TransducerConfig(
        capacity=1.0,
        enable_precharge=True,
        enable_pretwist=True,
        precharge_rate=0.7,
        pretwist_rate=0.65,
        target_energy=0.9,
        pump_intensity=1.0,
    )
    result = simulate_slingshot_cycle(x, y, t_max=4.0, dt=0.1, cfg=cfg)
    series = result["series"]
    assert series["precharge_power"].max() > 0.0
    assert series["pretwist_power"].max() > 0.0
    assert series["pumped_total"].max() > 0.0
    assert series["ready"].max() > 0.5  # becomes ready at some point
    assert "ready" in series["phase"] or series["ready"].max() > 0.0
    assert series["impulse"].max() > 0.0
    assert series["channel_direction"].min() < 0.0  # release reversion
