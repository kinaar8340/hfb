"""Tests for symbolic, BEC, and SLM extension modules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hfb.analog_gravity.symbolic import (
    acoustic_line_element,
    acoustic_metric_tensor,
    conformal_ricci_scalar,
    evaluate_metric_numeric,
    horizon_condition,
)
from hfb.bec.acoustic import bec_acoustic_metric, bec_sound_speed
from hfb.bec.bogoliubov import bogoliubov_dispersion
from hfb.bec.gpe import thomas_fermi_density
from hfb.bec.vortex import imprinted_vortex_phase, vortex_ring_velocity
from hfb.integration.vqc_proto import vqc_proto_available
from hfb.optics.slm_export import (
    SLMExportConfig,
    VortexConduitSpec,
    export_flux_bubble_hologram,
    field_to_slm_phase,
    flux_bubble_vortex_field,
)
from hfb.utils.grid import cartesian_grid


def test_symbolic_horizon_factorizes():
    sym = horizon_condition()
    factored = pytest.importorskip("sympy").factor(sym)
    assert factored is not None


def test_symbolic_metric_numeric_matches_acoustic():
    g = evaluate_metric_numeric(0.3, 0.4, 1.0)
    assert g[0][0] == pytest.approx(-(1.0 - 0.25))
    assert g[0][1] == pytest.approx(-0.3)


def test_conformal_ricci_gaussian():
    import sympy as sp

    x, y, Omega = sp.symbols("x y Omega")
    omega = -sp.exp(-(x**2 + y**2))
    R = conformal_ricci_scalar({"x": x, "y": y, "omega": omega})
    assert R != 0


def test_bec_sound_speed_positive():
    x, y = cartesian_grid(32, 32, extent=2.0)
    n = thomas_fermi_density(x, y, n0=1.0, radius=1.0)
    cs = bec_sound_speed(n)
    assert np.all(cs[n > 0] > 0)


def test_vortex_ring_velocity_finite():
    x, y = cartesian_grid(48, 48, extent=3.0)
    vx, vy, phase = vortex_ring_velocity(x, y, ring_radius=1.0, num_vortices=6)
    assert np.isfinite(vx).all()
    assert np.isfinite(vy).all()
    assert np.isfinite(phase).all()


def test_bec_acoustic_metric_keys():
    x, y = cartesian_grid(32, 32, extent=2.0)
    vx, vy, _ = vortex_ring_velocity(x, y, num_vortices=4)
    metric = bec_acoustic_metric(x, y, vx, vy)
    assert "density" in metric
    assert "ergoregion" in metric


def test_bogoliubov_dispersion_monotone():
    k = np.linspace(0.1, 5.0, 20)
    eps = bogoliubov_dispersion(k, c_s=1.0)
    assert np.all(np.diff(eps) > 0)


def test_flux_bubble_vortex_field_nonzero():
    cfg = SLMExportConfig(resolution_x=64, resolution_y=64)
    spec = VortexConduitSpec(num_vortices=8)
    field = flux_bubble_vortex_field(spec, cfg)
    assert np.max(np.abs(field)) > 0


def test_slm_export_writes_package(tmp_path: Path):
    summary = export_flux_bubble_hologram(
        tmp_path / "slm",
        spec=VortexConduitSpec(num_vortices=6),
        cfg=SLMExportConfig(resolution_x=64, resolution_y=64),
    )
    assert (tmp_path / "slm" / "manifest.json").exists()
    assert (tmp_path / "slm" / "frames" / "phase_0000.png").exists()
    meta = json.loads((tmp_path / "slm" / "manifest.json").read_text())
    assert meta["num_vortices"] == 6
    assert summary["frames"] == 1


def test_field_to_slm_phase_range():
    field = np.exp(1j * np.linspace(0, 4 * np.pi, 16).reshape(4, 4))
    phase = field_to_slm_phase(field)
    assert phase.min() >= 0.0
    assert phase.max() <= 2 * np.pi + 1e-9


def test_vqc_proto_availability_is_bool():
    assert isinstance(vqc_proto_available(), bool)