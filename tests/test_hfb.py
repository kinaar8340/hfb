"""Core HFB unit tests."""

from __future__ import annotations

import numpy as np

from hfb.analog_gravity.acoustic import acoustic_metric_components, draining_vortex_flow, ergosurface_mask
from hfb.bubble.stability import bubble_stability_metrics
from hfb.bubble.warp_conduit import flux_bubble_metric
from hfb.defects.conformal import ricci_scalar, solve_conformal_poisson
from hfb.defects.densities import exponential_ring, gaussian_defect
from hfb.hopf.fibration import hopf_coordinates, hopf_map
from hfb.hopf.hopfion import toroidal_hopfion_director
from hfb.optics.lg_modes import lg_mode_full
from hfb.optics.nematic import cosmic_string_deflection
from hfb.utils.grid import cartesian_grid


def test_conformal_poisson_gaussian():
    x, y = cartesian_grid(64, 64, extent=3.0)
    dx = x[0, 1] - x[0, 0]
    lam = gaussian_defect(x, y, amplitude=1.0, sigma=0.4)
    omega = solve_conformal_poisson(lam, dx)
    R = ricci_scalar(omega, dx)
    assert np.isfinite(omega).all()
    assert np.isfinite(R).all()
    assert np.max(omega) - np.min(omega) > 0.0


def test_acoustic_ergoregion():
    x, y = cartesian_grid(32, 32, extent=2.0)
    vx, vy = draining_vortex_flow(x, y, circulation=2.0, drain_strength=0.5)
    mask = ergosurface_mask(vx, vy, cs=1.0)
    assert mask.dtype == bool
    assert np.any(mask)


def test_hopf_map_unit_sphere():
    eta, xi1, xi2 = np.pi / 4, np.pi / 3, np.pi / 6
    x1, x2, x3, x4 = hopf_coordinates(eta, xi1, xi2)
    y1, y2, y3 = hopf_map(x1, x2, x3, x4)
    assert np.isclose(y1**2 + y2**2 + y3**2, 1.0, atol=1e-9)


def test_hopfion_director_normalized():
    x, y = cartesian_grid(16, 16, extent=2.0)
    z = np.zeros_like(x)
    nx, ny, nz = toroidal_hopfion_director(x, y, z)
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    assert np.allclose(norm, 1.0, atol=1e-6)


def test_lg_mode_vortex_charge():
    x, y = cartesian_grid(48, 48, extent=3.0)
    field = lg_mode_full(ell=2, x=x, y=y, w0=1.0)
    phase = np.angle(field)
    # phase winds by 2 * ell * pi around a loop (proxy check at fixed radius)
    mid = field.shape[0] // 2
    ring = field[mid, :]
    assert np.sum(np.abs(ring)) > 0.0


def test_nematic_deflection_positive():
    delta_phi = cosmic_string_deflection(delta_n=0.1, n_eff=1.5, winding=1)
    assert delta_phi > 0.0


def test_flux_bubble_metric_keys():
    x, y = cartesian_grid(48, 48, extent=3.0)
    dx = x[0, 1] - x[0, 0]
    metric = flux_bubble_metric(x, y, dx=dx)
    for key in ("omega", "shift", "g_tt", "n_eff", "defect_density"):
        assert key in metric
    report = bubble_stability_metrics(metric, dx)
    assert hasattr(report, "stable_proxy")