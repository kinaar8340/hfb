"""Command-line entry points for HFB demos."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from hfb.bubble.stability import bubble_stability_metrics, parameter_sweep
from hfb.bubble.warp_conduit import flux_bubble_metric
from hfb.optics.raytrace import trace_rays_conformal
from hfb.optics.slm_export import SLMExportConfig, VortexConduitSpec, export_flux_bubble_hologram
from hfb.utils.grid import cartesian_grid
from hfb.utils.viz import plot_flux_bubble_3d


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def run_bubble_demo(cfg: dict, output_dir: Path, viz3d: bool = False) -> None:
    import matplotlib.pyplot as plt

    grid = cfg["grid"]
    bubble = cfg["bubble"]
    hopf = cfg.get("hopf", {})
    x, y = cartesian_grid(grid["nx"], grid["ny"], extent=grid["extent"])
    dx = float(x[0, 1] - x[0, 0])

    metric = flux_bubble_metric(
        x,
        y,
        bubble_radius=bubble["radius"],
        wall_width=bubble["wall_width"],
        defect_amplitude=bubble["defect_amplitude"],
        circulation=bubble["circulation"],
        c0=bubble["sound_speed"],
        dx=dx,
        defect_profile=bubble.get("defect_profile", "exponential_ring"),
        major_radius=hopf.get("major_radius"),
        minor_radius=hopf.get("minor_radius"),
    )
    report = bubble_stability_metrics(metric, dx)

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    im0 = axes[0, 0].imshow(metric["omega"], origin="lower", extent=[-grid["extent"], grid["extent"]] * 2)
    axes[0, 0].set_title("Conformal factor Ω")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(metric["shift"], origin="lower", extent=[-grid["extent"], grid["extent"]] * 2)
    axes[0, 1].set_title("Effective shift (warp analog)")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(-metric["g_tt"], origin="lower", extent=[-grid["extent"], grid["extent"]] * 2)
    axes[1, 0].set_title("c_eff² - v² (acoustic)")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    for angle in np.linspace(-0.4, 0.4, 5):
        rx, ry = trace_rays_conformal(metric["omega"], -grid["extent"] * 0.8, angle * 2.0, angle, dx)
        axes[1, 1].plot(rx, ry, lw=0.8)
    axes[1, 1].set_title("Conformal geodesics (rays)")
    axes[1, 1].set_xlim(-grid["extent"], grid["extent"])
    axes[1, 1].set_ylim(-grid["extent"], grid["extent"])
    axes[1, 1].set_aspect("equal")

    fig.suptitle(
        f"HFB Flux Bubble | stable_proxy={report.stable_proxy} | "
        f"max|R|={report.max_ricci:.2f} | ergo={report.ergo_fraction:.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = output_dir / "flux_bubble_demo.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")

    if viz3d:
        fig3d = plot_flux_bubble_3d(metric["omega"], metric["vx"], metric["vy"], dx=dx)
        out_3d = output_dir / "flux_bubble_3d.png"
        fig3d.savefig(out_3d, dpi=150)
        plt.close(fig3d)
        print(f"Wrote {out_3d}")

    print(report)


def run_sweep_demo(cfg: dict) -> None:
    sweep = cfg.get("sweep", {})
    radii = sweep.get("radii", [0.8, 1.0, 1.2])
    circs = sweep.get("circulations", [0.2, 0.4, 0.6])
    bubble = cfg.get("bubble", {})
    hopf = cfg.get("hopf", {})
    results = parameter_sweep(
        radii,
        circs,
        nx=cfg["grid"]["nx"],
        extent=cfg["grid"]["extent"],
        defect_profile=bubble.get("defect_profile", "exponential_ring"),
        major_radius=hopf.get("major_radius"),
        minor_radius=hopf.get("minor_radius"),
    )
    print("radius  circulation  stable  max|R|  ergo_frac")
    for radius, circ, report in results:
        print(
            f"{radius:5.2f}  {circ:11.2f}  "
            f"{str(report.stable_proxy):5s}  {report.max_ricci:6.2f}  {report.ergo_fraction:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Hopf Flux Bubble demos")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="YAML config path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Output directory for figures",
    )
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep instead of plot demo")
    parser.add_argument("--viz3d", action="store_true", help="Also write flux_bubble_3d.png surface plot")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.sweep:
        run_sweep_demo(cfg)
    else:
        run_bubble_demo(cfg, args.output, viz3d=args.viz3d)


def export_slm_main() -> None:
    parser = argparse.ArgumentParser(description="Export HFB flux-bubble SLM hologram")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/slm_hologram"))
    parser.add_argument("--preset", type=str, default=None, help="vqc_proto SLM device preset")
    parser.add_argument("--frames", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    slm_cfg_yaml = cfg.get("slm", {})
    preset = args.preset or slm_cfg_yaml.get("device_preset", "generic_512")
    spec = VortexConduitSpec(
        ring_radius_mm=slm_cfg_yaml.get("ring_radius_mm", 1.2),
        num_vortices=slm_cfg_yaml.get("num_vortices", 12),
        winding=slm_cfg_yaml.get("winding", cfg.get("optics", {}).get("lg_winding", 1)),
        amplitude=slm_cfg_yaml.get("amplitude", 1.0),
        core_sigma_mm=slm_cfg_yaml.get("core_sigma_mm", 0.25),
    )
    export_cfg = SLMExportConfig.from_vqc_preset(preset, extent_mm=slm_cfg_yaml.get("extent_mm", 4.0))
    summary = export_flux_bubble_hologram(
        args.output,
        spec=spec,
        cfg=export_cfg,
        num_frames=args.frames,
        phase_twist_per_frame=slm_cfg_yaml.get("phase_twist_per_frame", 0.0),
    )
    print(summary)


def symbolic_main() -> None:
    """Print SymPy acoustic metric summaries (requires [symbolic] extra)."""
    try:
        from hfb.analog_gravity.symbolic import symbolic_summary
    except ImportError as exc:
        raise SystemExit('Install symbolic extra: pip install -e ".[symbolic]"') from exc

    print("=" * 60)
    print("HFB — Symbolic Acoustic Metrics")
    print("=" * 60)
    for key, value in symbolic_summary().items():
        print(f"\n[{key}]\n{value}")


def bec_main() -> None:
    parser = argparse.ArgumentParser(description="BEC acoustic backend demo")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    from hfb.bec.demo import run_bec_acoustic_demo

    cfg = load_config(args.config)
    bec = cfg.get("bec", {})
    grid = cfg.get("grid", {})
    summary = run_bec_acoustic_demo(
        args.output,
        nx=grid.get("nx", 96),
        extent=grid.get("extent", 3.0),
        ring_radius=bec.get("vortex_ring_radius", 1.0),
        num_vortices=bec.get("num_vortices", 8),
        n0=bec.get("n0", 1.0),
        interaction=bec.get("interaction", 1.0),
    )
    print(summary)


def check_ecosystem_main() -> None:
    """Report optional feature availability (symbolic, notebook, vqc_proto, bec)."""
    from hfb.integration.vqc_proto import vqc_proto_status

    print("HFB feature checklist")
    print("-" * 40)

    try:
        import sympy  # noqa: F401

        print("symbolic (SymPy):     OK")
    except ImportError:
        print("symbolic (SymPy):     missing — pip install -e \".[symbolic]\"")

    try:
        import ipywidgets  # noqa: F401

        print("notebook (widgets):   OK")
    except ImportError:
        print("notebook (widgets):   missing — pip install -e \".[notebook]\"")

    try:
        import PIL  # noqa: F401

        print("slm (Pillow):         OK")
    except ImportError:
        print("slm (Pillow):         missing — pip install -e \".[slm]\"")

    status = vqc_proto_status()
    print(f"vqc_proto root:       {status.root or 'not found'}")
    print(f"vqc_proto LG modes:   {status.lg_modes}")
    print(f"vqc_proto SLM export: {status.slm_typehead}")
    print(f"vqc_proto note:       {status.message}")
    print("bec acoustic backend: built-in (hfb-bec-demo)")


if __name__ == "__main__":
    main()