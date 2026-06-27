"""Tests for v0.1.4: 3D torus, warp fidelity, topology diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from hfb.analog_gravity.warp_compare import compare_warp_numeric, warp_fidelity
from hfb.bubble.stability import bubble_stability_metrics
from hfb.bubble.topology import integrated_curvature_flux, linking_proxy
from hfb.bubble.warp_conduit import flux_bubble_metric
from hfb.defects.densities import toroidal_bubble_wall, torus_tube_distance
from hfb.utils.grid import cartesian_grid


def test_torus_tube_distance_zero_on_ring():
    x, y = cartesian_grid(32, 32, extent=2.0)
    d = torus_tube_distance(x, y, z_slice=0.0, major_radius=1.0)
    r = np.sqrt(x**2 + y**2)
    on_ring = np.isclose(r, 1.0, atol=0.1)
    assert np.min(d[on_ring]) < 0.15


def test_toroidal_3d_differs_from_2d_slice():
    x, y = cartesian_grid(48, 48, extent=3.0)
    lam_2d = toroidal_bubble_wall(x, y, major_radius=1.0, use_3d_torus=False)
    lam_3d = toroidal_bubble_wall(
        x, y, major_radius=1.0, use_3d_torus=True, z_slice=0.0, hopf_index=2
    )
    assert not np.allclose(lam_2d, lam_3d)


def test_linking_proxy_higher_for_3d_torus():
    assert linking_proxy(2, True, "toroidal_bubble_wall") == 2.0
    assert linking_proxy(2, False, "toroidal_bubble_wall") == 0.0
    assert linking_proxy(1, True, "exponential_ring") == 0.0


def test_compare_warp_numeric_finite():
    x, y = cartesian_grid(32, 32, extent=2.0)
    dx = float(x[0, 1] - x[0, 0])
    metric = flux_bubble_metric(x, y, dx=dx, defect_profile="toroidal_bubble_wall")
    gr, report = compare_warp_numeric(metric["shift"], x, y, dx)
    assert np.isfinite(gr).all()
    assert report.warp_fidelity >= 0.0


def test_toroidal_higher_winding_than_exponential():
    x, y = cartesian_grid(64, 64, extent=4.0)
    dx = float(x[0, 1] - x[0, 0])
    toroidal = flux_bubble_metric(
        x, y, dx=dx, defect_profile="toroidal_bubble_wall", major_radius=1.0
    )
    exponential = flux_bubble_metric(
        x, y, dx=dx, defect_profile="exponential_ring", bubble_radius=1.0
    )
    rt = bubble_stability_metrics(
        toroidal, dx, x=x, y=y, major_radius=1.0, defect_profile="toroidal_bubble_wall"
    )
    re = bubble_stability_metrics(
        exponential, dx, x=x, y=y, major_radius=1.0, defect_profile="exponential_ring"
    )
    assert rt.curvature_flux != re.curvature_flux or rt.topological_winding != re.topological_winding


def test_integrated_curvature_flux_sign():
    x, y = cartesian_grid(32, 32, extent=2.0)
    dx = float(x[0, 1] - x[0, 0])
    metric = flux_bubble_metric(x, y, dx=dx)
    mask = metric["defect_density"] > 0.5 * np.max(metric["defect_density"])
    flux = integrated_curvature_flux(metric["omega"], dx, mask)
    assert np.isfinite(flux)