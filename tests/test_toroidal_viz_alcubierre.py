"""Tests for toroidal defect, Alcubierre symbolic, and 3D viz extensions."""

from __future__ import annotations

import numpy as np
import pytest

from hfb.analog_gravity.symbolic import (
    alcubierre_line_element,
    compare_effective_warp,
    lambdify_alcubierre_shift,
)
from hfb.bubble.warp_conduit import flux_bubble_metric
from hfb.defects.densities import build_defect_density, toroidal_bubble_wall
from hfb.utils.grid import cartesian_grid
from hfb.utils.viz import plot_flux_bubble_3d


def test_toroidal_bubble_wall_peaks_on_ring():
    x, y = cartesian_grid(64, 64, extent=3.0)
    lam = toroidal_bubble_wall(x, y, major_radius=1.0, minor_radius=0.35)
    r = np.sqrt(x**2 + y**2)
    on_ring = np.abs(r - 1.0) < 0.15
    assert lam[on_ring].max() > lam[~on_ring].max()


def test_build_defect_density_profiles():
    x, y = cartesian_grid(32, 32, extent=2.0)
    for profile in ("exponential_ring", "toroidal_bubble_wall", "gaussian"):
        lam = build_defect_density(x, y, profile=profile, bubble_radius=1.0)
        assert np.isfinite(lam).all()
        assert lam.max() > 0


def test_flux_bubble_toroidal_profile():
    x, y = cartesian_grid(48, 48, extent=3.0)
    dx = float(x[0, 1] - x[0, 0])
    metric = flux_bubble_metric(
        x,
        y,
        dx=dx,
        defect_profile="toroidal_bubble_wall",
        major_radius=1.0,
        minor_radius=0.35,
    )
    assert "vx" in metric
    assert "vy" in metric
    assert metric["defect_density"].max() > 0


def test_alcubierre_line_element_returns_triple():
    ds2, shift, shape = alcubierre_line_element(vs=0.5, rs=1.0, sigma=0.5)
    assert ds2 is not None
    assert shift is not None
    assert shape is not None


def test_compare_effective_warp():
    import sympy as sp

    x = sp.Symbol("x")
    analog = sp.tanh(x)
    gr = sp.Symbol("v_s") * sp.tanh(x)
    diff, ratio = compare_effective_warp(analog, gr, subs_dict={"v_s": 1.0})
    assert diff is not None


def test_lambdify_alcubierre_shift_numeric():
    fn = lambdify_alcubierre_shift()
    val = fn(0.0, 0.0, 0.5, 1.0, 0.5)
    assert np.isfinite(float(val))


def test_plot_flux_bubble_3d_returns_figure():
    pytest.importorskip("matplotlib")
    x, y = cartesian_grid(24, 24, extent=2.0)
    dx = float(x[0, 1] - x[0, 0])
    metric = flux_bubble_metric(x, y, dx=dx, defect_profile="toroidal_bubble_wall")
    fig = plot_flux_bubble_3d(metric["omega"], metric["vx"], metric["vy"], dx=dx, extent=2.0)
    assert fig is not None