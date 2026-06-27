"""Tests for symbolic, BEC, and SLM extension modules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hfb.analog_gravity.symbolic import (
    acoustic_line_element,
    acoustic_metric_tensor,
    alcubierre_line_element,
    compare_effective_warp,
    conformal_ricci_scalar,
    evaluate_metric_numeric,
    horizon_condition,
    lambdify_alcubierre_shift,
)
from hfb.bec.acoustic import bec_acoustic_metric, bec_sound_speed
from hfb.bec.bogoliubov import bogoliubov_dispersion
from hfb.bec.gpe import thomas_fermi_density
from hfb.bec.vortex import imprinted_vortex_phase, vortex_ring_velocity
from hfb.analog_gravity.symbolic import symbolic_summary
from hfb.bec.demo import run_bec_acoustic_demo
from hfb.integration.vqc_proto import resolve_vqc_root, vqc_proto_available, vqc_proto_status
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


def test_symbolic_summary_keys():
    summary = symbolic_summary()
    assert "horizon" in summary
    assert "line_element" in summary
    assert "alcubierre_shift" in summary


def test_alcubierre_line_element_has_shift():
    ds2, shift, shape = alcubierre_line_element(vs=0.5, rs=1.0, sigma=0.5)
    assert ds2 is not None
    assert shift is not None
    assert shape is not None


def test_compare_effective_warp_diff_zero_at_match():
    import sympy as sp

    x = sp.Symbol("x")
    diff, ratio = compare_effective_warp(x, x)
    assert sp.simplify(diff) == 0
    assert sp.simplify(ratio) == 1


def test_lambdify_alcubierre_shift_finite():
    shift_fn = lambdify_alcubierre_shift()
    val = shift_fn(0.0, 0.0, 0.5, 1.0, 0.5)
    assert np.isfinite(val)


def test_vqc_proto_status_dataclass():
    status = vqc_proto_status()
    assert hasattr(status, "lg_modes")
    assert hasattr(status, "message")


def test_resolve_vqc_root_finds_local_clone():
    root = resolve_vqc_root()
    if (Path.home() / "Projects" / "vqc_proto").exists():
        assert root is not None


def test_bec_acoustic_demo_writes(tmp_path: Path):
    summary = run_bec_acoustic_demo(tmp_path, nx=32, extent=2.0, num_vortices=4)
    assert Path(summary["output"]).is_file()