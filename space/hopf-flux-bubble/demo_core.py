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
QVPIC_URL = "https://github.com/kinaar8340/qvpic"
HF_SPACE_URL = "https://huggingface.co/spaces/kinaar111/hopf-flux-bubble"
HFB_RAW_URL = "https://raw.githubusercontent.com/kinaar8340/vqc_proto/main/hfb.png"
HFB_WALLPAPER_URL = HFB_RAW_URL
GALLERY_FLUX_URL = "https://raw.githubusercontent.com/kinaar8340/hfb/main/outputs/flux_bubble_demo.png"
GALLERY_3D_URL = "https://raw.githubusercontent.com/kinaar8340/hfb/main/outputs/flux_bubble_3d.png"
GALLERY_WARP_URL = "https://raw.githubusercontent.com/kinaar8340/hfb/main/outputs/warp_compare.png"

BOOT_QUOTE_STRING = "HOPF FLUX BUBBLE · ANALOG METRIC · VQC"

CAVEATS_MD = """
> **Effective analog only** — not literal GR curvature or superluminal transport.
> Stability metrics (max |R|, ergo fraction, curvature flux Φ_R) are heuristic proxies
> for tabletop exploration, not experimental claims.
"""

ONBOARDING_MD = """
### Think effective metric, not exotic matter
**Hopf Flux Bubble (HFB)** treats a "warp bubble" as an **emergent effective geometry** from
topologically protected flux — linked Hopfions, vortex conduits, and defect-induced curvature in
condensed-matter / metamaterial analogs. Not literal spacetime warping.

### Three steps (60 seconds)
1. **Run flux bubble** — conformal factor Ω, effective shift (warp analog), acoustic c_eff² − v²,
   and conformal geodesic rays in a 4-panel figure.
2. **Run warp compare** — L¹ fidelity between analog shift and symbolic Alcubierre β field.
3. **Run stability sweep** — grid over bubble radius × circulation; stable_proxy, max |R|, ergo_fraction.

### What the metrics mean
| Metric | Plain English |
|--------|----------------|
| **stable_proxy** | Heuristic pass/fail from max |R| and ergo fraction thresholds. |
| **max \|R\|** | Peak effective Ricci scalar on the defect wall. |
| **curvature_flux Φ_R** | Integrated curvature through the bubble ring — topology diagnostic. |
| **warp_fidelity** | L¹ agreement between analog shift and Alcubierre reference. |

**Tip:** Default `toroidal_bubble_wall` profile is smoother than legacy `exponential_ring`.
"""

VQC_CLAIMS_MD = """
| VQC / analog element | HFB demo shows… |
|----------------------|-----------------|
| **Helical / vortex geometry** | Vortex circulation + toroidal Hopf wall texture (`hopf_index`). |
| **Nested shielding** | Defect wall + flow composite isolates effective metric inside bubble. |
| **OAM / orbital carrier** | LG vortex optics module; SLM export via vqc_proto when available. |
| **Quaternion / topological mux** | Hopf linking proxy + winding on braided flux structures. |
| **Acoustic metric analog** | ds² from c_eff² − v² — analog gravity pillar (`analog_gravity/`). |
| **Warp bubble vision** | Effective shift profile compared to Alcubierre β (`warp_compare.py`). |
| **Optical stack link** | [orbital-braille-vqc](https://huggingface.co/spaces/kinaar111/orbital-braille-vqc) carrier layer. |
| **Identity memory link** | [qvpic](https://huggingface.co/spaces/kinaar111/qvpic) persistent identity conduit. |

Full optical prototype: [vqc_proto](https://github.com/kinaar8340/vqc_proto) · HFB analog layer: [hfb](https://github.com/kinaar8340/hfb)
"""

TERM_KEY_ACTIONS: dict[int, tuple[str, str]] = {
    1: ("home", "Return to selection menu"),
    2: ("status", "Live pipeline & environment"),
    3: ("hopf", "Hopfion → flux bubble analogy"),
    4: ("pipeline", "Defect wall + vortex flow loop"),
    5: ("metrics", "Stability metrics baseline"),
    6: ("build", "Build stamp & deploy info"),
    7: ("help", "D-pad / keypad navigation"),
    8: ("helix", "HFB flux helix — any key exits"),
    9: ("claims", "VQC claim ↔ HFB demo map"),
    10: ("defects", "Conformal Ω & winding"),
    11: ("warp", "Alcubierre compare + acoustic"),
    12: ("presets", "Default run params catalog"),
}

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


def terminal_keypad_map() -> str:
    lines = ["Assigned prog keys (01–12):", ""]
    for index in sorted(TERM_KEY_ACTIONS):
        _action, desc = TERM_KEY_ACTIONS[index]
        tag = "01 Home" if index == 1 else f"{index:02d}"
        lines.append(f"  [{tag}]  {desc}")
    lines.extend(
        [
            "",
            "D-pad: ▲▼◀▶ move menu · enter confirm · clear blank",
            "Keys 13–24: reserved (latch only)",
            "Menu items 01–08 mirror d-pad selection.",
            "08 / menu 08 → HFB flux helix screensaver (any key stops).",
        ]
    )
    return "\n".join(lines)


def terminal_hopf_analogy() -> str:
    defaults = default_run_params()
    return "\n".join(
        [
            "Classical GR warp  →  HFB effective analog:",
            "",
            "  exotic matter      →  defect wall + vortex flow composite",
            "  Alcubierre β       →  effective shift from flux_bubble_metric",
            "  spacetime curvature →  conformal Ω = exp(2λ) from ΔΩ = −λ",
            "  superluminal claim →  acoustic metric tabletop analog only",
            "  Hopf structure     →  toroidal flux wall + linking proxy",
            "",
            f"Default profile: {defaults['defect_profile']!r}",
            "toroidal_bubble_wall — smoother max |R| vs exponential_ring.",
            "",
            "Run flux bubble below → 4-panel figure + optional 3D torus.",
        ]
    )


def terminal_pipeline_scope() -> str:
    on_hf = is_hf_space()
    grid_note = "96×96 grid on HF" if on_hf else "128×128 grid locally"
    return "\n".join(
        [
            "THIS SPACE — browser simulation (you are here):",
            "  · flux_bubble_metric → Ω, shift, acoustic g_tt",
            "  · conformal geodesic ray trace (optics/raytrace.py)",
            "  · stability metrics + parameter sweep",
            "  · warp compare vs symbolic Alcubierre β",
            "",
            "GITHUB REPO — full HFB depth:",
            "  · hfb-demo · hfb-symbolic · hfb-bec-demo",
            "  · hfb/optics/slm_export.py (vqc_proto LG coupling)",
            "  · notebooks/01_flux_bubble_explorer.ipynb",
            "",
            f"Grid: {grid_note}",
            "Not literal GR — effective analog for geodesics & horizons.",
        ]
    )


def terminal_metrics_baseline() -> str:
    defaults = default_run_params()
    return "\n".join(
        [
            "Default config (configs/default.yaml):",
            "",
            f"  bubble_radius     {defaults['bubble_radius']}",
            f"  wall_width        {defaults['wall_width']}",
            f"  circulation       {defaults['circulation']}",
            f"  defect_profile    {defaults['defect_profile']!r}",
            f"  hopf_index        {defaults['hopf_index']}",
            "",
            "Metrics after Run flux bubble:",
            "  · stable_proxy — heuristic stability flag",
            "  · max |R| — peak effective Ricci on wall",
            "  · ergo_fraction — ergoregion-like acoustic fraction",
            "  · curvature_flux Φ_R — ring-integrated curvature",
            "  · linking_proxy — Hopf topological diagnostic",
            "",
            "toroidal_bubble_wall: max |R| ≈ 2.6 vs ~4.7 exponential_ring.",
        ]
    )


def terminal_claims_snapshot() -> str:
    lines = [
        "VQC / analog element  →  HFB demo output:",
        "",
        "  Helical vortex       →  circulation + toroidal wall",
        "  Nested shielding     →  defect wall isolates metric",
        "  OAM carrier link     →  orbital-braille-vqc Space",
        "  Hopf topology        →  linking_proxy + hopf_index texture",
        "  Acoustic metric      →  c_eff² − v² panel (bottom-left)",
        "  Warp analog          →  shift profile + Alcubierre compare",
        "  Conformal geodesics  →  ray trace panel (bottom-right)",
        "  SLM bench path       →  hfb-export-slm (local + vqc_proto)",
        "",
        "Expand Claims tab for full table · 09 Claims keypad shortcut.",
    ]
    return "\n".join(lines)


def terminal_defects_shards() -> str:
    return "\n".join(
        [
            "Defect curvature + conformal factor:",
            "",
            "  ΔΩ = −λ(r,θ)     conformal Poisson defect source",
            "  Ω = exp(2λ)      effective metric conformal factor",
            "  toroidal_bubble_wall — Hopfion-motivated smooth wall",
            "  exponential_ring   — legacy ring profile (sharper |R|)",
            "  topological_winding — quantized winding on defect loop",
            "",
            "Top-left panel: Ω heatmap after Run flux bubble.",
            "Tune defect_amplitude + wall_width — watch max |R| in metrics.",
        ]
    )


def terminal_warp_export() -> str:
    return "\n".join(
        [
            "Warp compare (Run warp compare button):",
            "",
            "  analog shift     →  metric['shift'] from flux bubble",
            "  Alcubierre β     →  symbolic reference (vs, rs, σ sliders)",
            "  warp_fidelity    →  L¹ agreement (lower = closer match)",
            "  max/mean |Δβ|    →  pointwise shift residual",
            "",
            "Also: Run stability sweep — radius × circulation grid.",
            f"Local CLI: {GITHUB_URL}#quick-start",
        ]
    )


def terminal_presets_catalog() -> str:
    defaults = default_run_params()
    lines = ["Default run params from configs/default.yaml:", ""]
    for key, value in defaults.items():
        lines.append(f"  {key:<18} {value!r}")
    lines.extend(
        [
            "",
            "Warp sliders (vs, rs, σ) apply to Run warp compare only.",
            "Enable 3D torus slice + surface plot for Hopf texture viz.",
        ]
    )
    return "\n".join(lines)