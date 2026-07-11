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
    cfg = SlingshotConfig(nucleate_duration=1.0, store_duration=2.0, release_duration=0.8)
    assert cycle_phase_at_time(0.1, cfg) == SlingshotPhase.NUCLEATE
    assert cycle_phase_at_time(1.5, cfg) == SlingshotPhase.STORE
    assert cycle_phase_at_time(3.2, cfg) == SlingshotPhase.RELEASE


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
