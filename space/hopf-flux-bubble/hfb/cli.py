"""Command-line entry points for HFB demos."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from hfb.analog_gravity.warp_compare import compare_warp_numeric, plot_warp_comparison
from hfb.bubble.stability import bubble_stability_metrics, parameter_sweep
from hfb.bubble.warp_conduit import flux_bubble_metric
from hfb.optics.raytrace import trace_rays_conformal
from hfb.optics.slm_export import SLMExportConfig, VortexConduitSpec, export_flux_bubble_hologram
from hfb.utils.grid import cartesian_grid
from hfb.utils.viz import plot_flux_bubble_3d


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _hopf_flags(cfg: dict) -> tuple[dict, dict, bool]:
    bubble = cfg.get("bubble", {})
    hopf = cfg.get("hopf", {})
    use_3d = bubble.get("use_3d_torus", hopf.get("use_3d_torus", False))
    return bubble, hopf, use_3d


def run_bubble_demo(cfg: dict, output_dir: Path, viz3d: bool = False) -> None:
    import matplotlib.pyplot as plt

    grid = cfg["grid"]
    bubble, hopf, use_3d = _hopf_flags(cfg)
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
        use_3d_torus=use_3d,
        z_slice=hopf.get("z_slice", 0.0),
        hopf_index=hopf.get("hopf_index", 1),
    )
    report = bubble_stability_metrics(
        metric,
        dx,
        x=x,
        y=y,
        major_radius=hopf.get("major_radius", bubble["radius"]),
        hopf_index=hopf.get("hopf_index", 1),
        use_3d_torus=use_3d,
        defect_profile=bubble.get("defect_profile", "toroidal_bubble_wall"),
    )

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
        f"HFB | stable={report.stable_proxy} | max|R|={report.max_ricci:.2f} | "
        f"Φ_R={report.curvature_flux:.3f} | link={report.linking_proxy:.1f}",
        fontsize=10,
    )
    fig.tight_layout()
    out_path = output_dir / "flux_bubble_demo.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")

    if viz3d:
        fig3d = plot_flux_bubble_3d(
            metric["omega"],
            metric["vx"],
            metric["vy"],
            dx=dx,
            extent=grid["extent"],
        )
        out_3d = output_dir / "flux_bubble_3d.png"
        fig3d.savefig(out_3d, dpi=150)
        plt.close(fig3d)
        print(f"Wrote {out_3d}")

    print(report)


def run_compare_warp(cfg: dict, output_dir: Path) -> None:
    """Compare toroidal vs exponential profiles + warp fidelity to Alcubierre."""
    import matplotlib.pyplot as plt

    grid = cfg["grid"]
    bubble, hopf, use_3d = _hopf_flags(cfg)
    warp = cfg.get("warp_compare", {})
    x, y = cartesian_grid(grid["nx"], grid["ny"], extent=grid["extent"])
    dx = float(x[0, 1] - x[0, 0])

    vs = warp.get("vs", 0.3)
    rs = warp.get("rs", bubble.get("radius", 1.0))
    sigma = warp.get("sigma", bubble.get("wall_width", 0.25))

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"{'profile':<22} {'stable':>6} {'max|R|':>8} {'curv_flux':>10} {'link':>5} {'fidelity':>10}")
    print("-" * 72)

    default_profile = bubble.get("defect_profile", "toroidal_bubble_wall")
    for profile in ("toroidal_bubble_wall", "exponential_ring"):
        profile_3d = use_3d and profile == "toroidal_bubble_wall"
        metric = flux_bubble_metric(
            x,
            y,
            bubble_radius=bubble["radius"],
            wall_width=bubble["wall_width"],
            defect_amplitude=bubble["defect_amplitude"],
            circulation=bubble["circulation"],
            c0=bubble["sound_speed"],
            dx=dx,
            defect_profile=profile,
            major_radius=hopf.get("major_radius"),
            minor_radius=hopf.get("minor_radius"),
            use_3d_torus=profile_3d,
            z_slice=hopf.get("z_slice", 0.0),
            hopf_index=hopf.get("hopf_index", 1),
        )
        report = bubble_stability_metrics(
            metric,
            dx,
            x=x,
            y=y,
            major_radius=hopf.get("major_radius", bubble["radius"]),
            hopf_index=hopf.get("hopf_index", 1),
            use_3d_torus=profile_3d,
            defect_profile=profile,
        )
        _, warp_report = compare_warp_numeric(metric["shift"], x, y, dx, vs=vs, rs=rs, sigma=sigma)
        print(
            f"{profile:<22} {str(report.stable_proxy):>6} {report.max_ricci:8.2f} "
            f"{report.curvature_flux:10.3f} {report.linking_proxy:5.1f} "
            f"{warp_report.warp_fidelity:10.4f}"
        )

        if profile == default_profile:
            gr_shift, wr = compare_warp_numeric(metric["shift"], x, y, dx, vs=vs, rs=rs, sigma=sigma)
            fig = plot_warp_comparison(
                x,
                y,
                metric["shift"],
                gr_shift,
                wr,
                extent=grid["extent"],
                output_path=str(output_dir / "warp_compare.png"),
            )
            plt.close(fig)
            print(f"Wrote {output_dir / 'warp_compare.png'}")


def run_sweep_demo(cfg: dict) -> None:
    sweep = cfg.get("sweep", {})
    radii = sweep.get("radii", [0.8, 1.0, 1.2])
    circs = sweep.get("circulations", [0.2, 0.4, 0.6])
    bubble, hopf, use_3d = _hopf_flags(cfg)
    results = parameter_sweep(
        radii,
        circs,
        nx=cfg["grid"]["nx"],
        extent=cfg["grid"]["extent"],
        defect_profile=bubble.get("defect_profile", "exponential_ring"),
        major_radius=hopf.get("major_radius"),
        minor_radius=hopf.get("minor_radius"),
        use_3d_torus=use_3d,
        hopf_index=hopf.get("hopf_index", 1),
    )
    print("radius  circulation  stable  max|R|  ergo_frac")
    for radius, circ, report in results:
        print(
            f"{radius:5.2f}  {circ:11.2f}  "
            f"{str(report.stable_proxy):5s}  {report.max_ricci:6.2f}  {report.ergo_fraction:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Hopf Flux Bubble demos")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--sweep", action="store_true", help="Parameter sweep")
    parser.add_argument("--viz3d", action="store_true", help="Write flux_bubble_3d.png")
    parser.add_argument("--compare-warp", action="store_true", help="Toroidal vs exponential + fidelity")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.compare_warp:
        run_compare_warp(cfg, args.output)
    elif args.sweep:
        run_sweep_demo(cfg)
    else:
        run_bubble_demo(cfg, args.output, viz3d=args.viz3d)


def export_slm_main() -> None:
    parser = argparse.ArgumentParser(description="Export HFB flux-bubble SLM hologram")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/slm_hologram"))
    parser.add_argument("--preset", type=str, default=None)
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
    from hfb.integration.vqc_proto import vqc_proto_status

    print("HFB feature checklist")
    print("-" * 40)
    for name, mod in [("symbolic", "sympy"), ("notebook", "ipywidgets"), ("slm", "PIL")]:
        try:
            __import__(mod)
            print(f"{name:20s} OK")
        except ImportError:
            print(f"{name:20s} missing")
    status = vqc_proto_status()
    print(f"vqc_proto LG:        {status.lg_modes}")
    print(f"vqc_proto SLM:       {status.slm_typehead}")


if __name__ == "__main__":
    main()