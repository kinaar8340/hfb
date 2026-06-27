"""Shared helpers for the Hopf Flux Bubble Gradio demo and HF Space."""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from hfb.analog_gravity.warp_compare import compare_warp_numeric, plot_warp_comparison
from hfb.bubble.stability import StabilityReport, bubble_stability_metrics, parameter_sweep
from hfb.bubble.warp_conduit import flux_bubble_metric
from hfb.optics.raytrace import trace_rays_conformal
from hfb.utils.grid import cartesian_grid
from hfb.utils.viz import plot_flux_bubble_3d

GITHUB_URL = "https://github.com/kinaar8340/hfb"
VQC_URL = "https://github.com/kinaar8340/vqc_proto"
HF_SPACE_URL = "https://huggingface.co/spaces/kinaar111/hopf-flux-bubble"
HFB_WALLPAPER_URL = "https://raw.githubusercontent.com/kinaar8340/vqc_proto/main/hfb.png"

_WEB_DIR = Path(__file__).resolve().parent
_DEFAULT_CFG_CANDIDATES = (
    _WEB_DIR / "configs" / "default.yaml",
    _WEB_DIR.parent / "configs" / "default.yaml",
)
DEFAULT_CONFIG_PATH = next(
    (path for path in _DEFAULT_CFG_CANDIDATES if path.is_file()),
    _DEFAULT_CFG_CANDIDATES[-1],
)


def is_hf_space() -> bool:
    return bool(os.environ.get("SPACE_ID"))


def load_default_config() -> dict[str, Any]:
    with DEFAULT_CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _grid_nx(on_hf: bool | None = None) -> int:
    if on_hf is None:
        on_hf = is_hf_space()
    return 96 if on_hf else 128


def _metric_from_params(
    *,
    bubble_radius: float,
    wall_width: float,
    defect_amplitude: float,
    circulation: float,
    sound_speed: float,
    defect_profile: str,
    use_3d_torus: bool,
    hopf_index: int,
    major_radius: float,
    minor_radius: float,
    extent: float = 4.0,
    nx: int | None = None,
) -> tuple[dict[str, np.ndarray], float, np.ndarray, np.ndarray]:
    nx = nx or _grid_nx()
    x, y = cartesian_grid(nx, nx, extent=extent)
    dx = float(x[0, 1] - x[0, 0])
    metric = flux_bubble_metric(
        x,
        y,
        bubble_radius=bubble_radius,
        wall_width=wall_width,
        defect_amplitude=defect_amplitude,
        circulation=circulation,
        c0=sound_speed,
        dx=dx,
        defect_profile=defect_profile,
        major_radius=major_radius,
        minor_radius=minor_radius,
        use_3d_torus=use_3d_torus,
        hopf_index=hopf_index,
    )
    return metric, dx, x, y


def stability_report_text(report: StabilityReport) -> str:
    data = asdict(report)
    lines = [
        f"stable_proxy      : {data['stable_proxy']}",
        f"max |R|           : {data['max_ricci']:.4f}",
        f"mean R (wall)     : {data['mean_ricci_wall']:.4f}",
        f"ergo_fraction     : {data['ergo_fraction']:.4f}",
        f"shift_grad_max    : {data['shift_gradient_max']:.4f}",
        f"curvature_flux Φ_R: {data['curvature_flux']:.4f}",
        f"topo_winding      : {data['topological_winding']:.4f}",
        f"linking_proxy     : {data['linking_proxy']:.4f}",
    ]
    return "\n".join(lines)


def run_flux_bubble_demo(
    *,
    bubble_radius: float,
    wall_width: float,
    defect_amplitude: float,
    circulation: float,
    sound_speed: float,
    defect_profile: str,
    use_3d_torus: bool,
    hopf_index: int,
    major_radius: float,
    minor_radius: float,
    include_3d: bool = False,
) -> tuple[str, str, str | None]:
    """Return (metrics_text, panel_figure_path, optional_3d_path)."""
    extent = 4.0
    metric, dx, x, y = _metric_from_params(
        bubble_radius=bubble_radius,
        wall_width=wall_width,
        defect_amplitude=defect_amplitude,
        circulation=circulation,
        sound_speed=sound_speed,
        defect_profile=defect_profile,
        use_3d_torus=use_3d_torus,
        hopf_index=hopf_index,
        major_radius=major_radius,
        minor_radius=minor_radius,
        extent=extent,
    )
    report = bubble_stability_metrics(
        metric,
        dx,
        x=x,
        y=y,
        major_radius=major_radius,
        hopf_index=hopf_index,
        use_3d_torus=use_3d_torus,
        defect_profile=defect_profile,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    extent_lim = [-extent, extent]
    im0 = axes[0, 0].imshow(metric["omega"], origin="lower", extent=extent_lim * 2)
    axes[0, 0].set_title("Conformal factor Ω")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(metric["shift"], origin="lower", extent=extent_lim * 2)
    axes[0, 1].set_title("Effective shift (warp analog)")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(-metric["g_tt"], origin="lower", extent=extent_lim * 2)
    axes[1, 0].set_title("c_eff² − v² (acoustic)")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    for angle in np.linspace(-0.4, 0.4, 5):
        rx, ry = trace_rays_conformal(metric["omega"], -extent * 0.8, angle * 2.0, angle, dx)
        axes[1, 1].plot(rx, ry, lw=0.8)
    axes[1, 1].set_title("Conformal geodesics (rays)")
    axes[1, 1].set_xlim(-extent, extent)
    axes[1, 1].set_ylim(-extent, extent)
    axes[1, 1].set_aspect("equal")

    fig.suptitle(
        f"HFB flux bubble | stable={report.stable_proxy} | max|R|={report.max_ricci:.2f} | "
        f"Φ_R={report.curvature_flux:.3f}",
        fontsize=10,
    )
    fig.tight_layout()

    panel_path = tempfile.NamedTemporaryFile(suffix="_flux_bubble.png", delete=False).name
    fig.savefig(panel_path, dpi=140)
    plt.close(fig)

    path_3d = None
    if include_3d:
        fig3d = plot_flux_bubble_3d(metric["omega"], metric["vx"], metric["vy"], dx=dx, extent=extent)
        path_3d = tempfile.NamedTemporaryFile(suffix="_flux_bubble_3d.png", delete=False).name
        fig3d.savefig(path_3d, dpi=140)
        plt.close(fig3d)

    return stability_report_text(report), panel_path, path_3d


def run_warp_compare_figure(
    *,
    bubble_radius: float,
    wall_width: float,
    defect_amplitude: float,
    circulation: float,
    sound_speed: float,
    defect_profile: str,
    use_3d_torus: bool,
    hopf_index: int,
    major_radius: float,
    minor_radius: float,
    vs: float = 0.3,
    rs: float = 1.0,
    sigma: float = 0.25,
) -> tuple[str, str]:
    extent = 4.0
    metric, dx, x, y = _metric_from_params(
        bubble_radius=bubble_radius,
        wall_width=wall_width,
        defect_amplitude=defect_amplitude,
        circulation=circulation,
        sound_speed=sound_speed,
        defect_profile=defect_profile,
        use_3d_torus=use_3d_torus,
        hopf_index=hopf_index,
        major_radius=major_radius,
        minor_radius=minor_radius,
        extent=extent,
    )
    gr_shift, report = compare_warp_numeric(metric["shift"], x, y, dx, vs=vs, rs=rs, sigma=sigma)
    fig = plot_warp_comparison(
        x,
        y,
        metric["shift"],
        gr_shift,
        report,
        extent=extent,
    )
    out_path = tempfile.NamedTemporaryFile(suffix="_warp_compare.png", delete=False).name
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    summary = (
        f"warp_fidelity (L¹) : {report.warp_fidelity:.5f}\n"
        f"max |Δβ|           : {report.max_shift_diff:.5f}\n"
        f"mean |Δβ|          : {report.mean_abs_diff:.5f}\n"
        f"mean ratio         : {report.mean_ratio:.5f}\n"
        f"Alcubierre vs      : {vs}\n"
        f"rs / sigma         : {rs} / {sigma}"
    )
    return summary, out_path


def run_parameter_sweep_summary(
    *,
    defect_profile: str,
    use_3d_torus: bool,
    hopf_index: int,
    major_radius: float,
    minor_radius: float,
) -> str:
    cfg = load_default_config()
    sweep = cfg.get("sweep", {})
    radii = sweep.get("radii", [0.8, 1.0, 1.2])
    circs = sweep.get("circulations", [0.2, 0.4, 0.6])
    nx = _grid_nx()
    results = parameter_sweep(
        radii,
        circs,
        nx=nx,
        extent=cfg.get("grid", {}).get("extent", 4.0),
        defect_profile=defect_profile,
        major_radius=major_radius,
        minor_radius=minor_radius,
        use_3d_torus=use_3d_torus,
        hopf_index=hopf_index,
    )
    lines = ["radius  circulation  stable  max|R|  ergo_frac"]
    for radius, circ, report in results:
        lines.append(
            f"{radius:5.2f}  {circ:11.2f}  "
            f"{str(report.stable_proxy):5s}  {report.max_ricci:6.2f}  {report.ergo_fraction:.4f}"
        )
    return "\n".join(lines)


def get_build_label() -> str:
    try:
        from build_info import BUILD_COMMIT, BUILD_UPDATED_UTC  # noqa: WPS433

        return f"Last updated: {BUILD_UPDATED_UTC} UTC · commit `{BUILD_COMMIT}`"
    except ImportError:
        return "Last updated: local dev build"


def default_run_params() -> dict[str, Any]:
    cfg = load_default_config()
    bubble = cfg.get("bubble", {})
    hopf = cfg.get("hopf", {})
    return {
        "bubble_radius": float(bubble.get("radius", 1.0)),
        "wall_width": float(bubble.get("wall_width", 0.25)),
        "defect_amplitude": float(bubble.get("defect_amplitude", 1.0)),
        "circulation": float(bubble.get("circulation", 0.35)),
        "sound_speed": float(bubble.get("sound_speed", 1.0)),
        "defect_profile": str(bubble.get("defect_profile", "toroidal_bubble_wall")),
        "use_3d_torus": bool(hopf.get("use_3d_torus", bubble.get("use_3d_torus", False))),
        "hopf_index": int(hopf.get("hopf_index", 1)),
        "major_radius": float(hopf.get("major_radius", bubble.get("radius", 1.0))),
        "minor_radius": float(hopf.get("minor_radius", 0.35)),
    }