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


def run_slingshot_demo(cfg: dict, output_dir: Path) -> None:
    """Nucleation → store → release demo for hemi-void electro-vibrational control."""
    import matplotlib.pyplot as plt

    from hfb.bubble.hemi_void import HemiVoidConfig, hemi_void_bubble_metric
    from hfb.electro_vibrational import (
        DualChargeConfig,
        ObserverSyncConfig,
        PhaseAlignmentConfig,
        SlingshotConfig,
        simulate_slingshot_cycle,
    )

    grid = cfg["grid"]
    ev = cfg.get("electro_vibrational", {})
    hemi_cfg_yaml = cfg.get("hemi_void", {})
    bubble = cfg.get("bubble", {})

    nx = min(int(grid.get("nx", 96)), 96)
    extent = float(grid.get("extent", 4.0))
    x, y = cartesian_grid(nx, nx, extent=extent)
    dx = float(x[0, 1] - x[0, 0])

    hemi = HemiVoidConfig(
        major_radius=hemi_cfg_yaml.get("major_radius", bubble.get("radius", 1.0)),
        wall_width=hemi_cfg_yaml.get("wall_width", bubble.get("wall_width", 0.22)),
        elongation=hemi_cfg_yaml.get("elongation", 1.4),
        rear_extension=hemi_cfg_yaml.get("rear_extension", 0.45),
        front_taper=hemi_cfg_yaml.get("front_taper", 0.7),
        hopf_index=cfg.get("hopf", {}).get("hopf_index", 1),
        axis=hemi_cfg_yaml.get("axis", "x"),
    )
    from hfb.electro_vibrational import TransducerConfig

    tx_yaml = ev.get("transducer", {})
    slingshot = SlingshotConfig(
        charge=DualChargeConfig(
            inner_radius=ev.get("inner_radius", 0.85),
            outer_radius=ev.get("outer_radius", 1.15),
            charge_density=ev.get("charge_density", 1.0),
            vibration_amp=ev.get("vibration_amp", 0.15),
            elongation=hemi.elongation,
            axis=hemi.axis,
        ),
        phase=PhaseAlignmentConfig(
            threshold=ev.get("phase_threshold", 0.72),
            medium_resonance=ev.get("medium_resonance", 1.0),
            drive_frequency=ev.get("drive_frequency", 1.0),
        ),
        observer=ObserverSyncConfig(
            coupling=ev.get("observer_coupling", 0.4),
            observer_frequency=ev.get("observer_frequency", 1.0),
            hum_frequency=ev.get("hum_frequency", 1.0),
        ),
        transducer=TransducerConfig(
            capacity=tx_yaml.get("capacity", 1.0),
            w_electrostatic=tx_yaml.get("w_electrostatic", 0.40),
            w_twist=tx_yaml.get("w_twist", 0.35),
            w_geometric=tx_yaml.get("w_geometric", 0.25),
            intensity=tx_yaml.get("intensity", ev.get("release_intensity", 1.0)),
            pump_intensity=tx_yaml.get("pump_intensity", ev.get("pump_intensity", 1.0)),
            reverse_gain=tx_yaml.get("reverse_gain", 1.35),
            enable_precharge=tx_yaml.get("enable_precharge", True),
            enable_pretwist=tx_yaml.get("enable_pretwist", True),
            precharge_rate=tx_yaml.get("precharge_rate", 0.55),
            pretwist_rate=tx_yaml.get("pretwist_rate", 0.50),
            target_energy=tx_yaml.get("target_energy", 0.95),
            ready_hysteresis=tx_yaml.get("ready_hysteresis", 0.04),
            axis=hemi.axis,
        ),
        store_duration=ev.get("store_duration", 2.0),
        ready_duration=ev.get("ready_duration", 0.4),
        release_duration=ev.get("release_duration", 0.8),
        nucleate_duration=ev.get("nucleate_duration", 1.0),
        release_detuning=ev.get("release_detuning", 0.35),
        enable_observer=ev.get("enable_observer", True),
        use_transducer=ev.get("use_transducer", True),
        release_intensity=ev.get("release_intensity", 1.0),
        pump_intensity=ev.get("pump_intensity", 1.0),
    )

    t_max = float(ev.get("t_max", 5.0))
    dt = float(ev.get("dt", 0.05))
    cycle = simulate_slingshot_cycle(x, y, t_max=t_max, dt=dt, cfg=slingshot)
    series = cycle["series"]

    # Snapshots at store peak and release peak (account for ready hold)
    t_store = float(slingshot.nucleate_duration + 0.6 * slingshot.store_duration)
    t_release = float(
        slingshot.nucleate_duration
        + slingshot.store_duration
        + slingshot.ready_duration
        + 0.4 * slingshot.release_duration
    )
    stored_at = float(
        series["stored"][int(np.argmin(np.abs(series["t"] - t_store)))]
    )
    metric_store = hemi_void_bubble_metric(
        x, y, dx=dx, hemi=hemi, slingshot=slingshot, t=t_store, stored_energy=stored_at
    )
    stored_rel = float(
        series["stored"][int(np.argmin(np.abs(series["t"] - t_release)))]
    )
    metric_rel = hemi_void_bubble_metric(
        x, y, dx=dx, hemi=hemi, slingshot=slingshot, t=t_release, stored_energy=stored_rel
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    extent_box = [-extent, extent, -extent, extent]

    im0 = axes[0, 0].imshow(
        metric_store["defect_density"], origin="lower", extent=extent_box
    )
    axes[0, 0].set_title("Hemi-void λ (store)")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(metric_store["shift"], origin="lower", extent=extent_box)
    axes[0, 1].set_title("Shift + transducer store")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(
        metric_store.get("e_field", metric_store["omega"]),
        origin="lower",
        extent=extent_box,
    )
    axes[0, 2].set_title("|E| capacitive gap")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    im3 = axes[1, 0].imshow(metric_rel["shift"], origin="lower", extent=extent_box)
    axes[1, 0].set_title("Shift (channel-reverted release)")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    # Transducer ledger + active pump
    ax = axes[1, 1]
    if "e_electrostatic" in series:
        ax.plot(series["t"], series["e_electrostatic"], label="E electrostatic", lw=1.2)
        ax.plot(series["t"], series["e_twist"], label="E twist", lw=1.2)
        ax.plot(series["t"], series["e_geometric"], label="E geometric", lw=1.2)
    ax.plot(series["t"], series["stored"], label="ledger total", lw=1.6, color="k")
    ax.plot(series["t"], series["impulse"], label="directed impulse", lw=1.2, ls="--")
    if "precharge_power" in series:
        ax.plot(series["t"], series["precharge_power"], label="pre-charge P", lw=1.0, alpha=0.8)
        ax.plot(series["t"], series["pretwist_power"], label="pre-twist P", lw=1.0, alpha=0.8)
    ax.set_xlabel("t")
    ax.set_title("Ledger + active pump")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(series["t"], series["psi"], label="ψ alignment", lw=1.5)
    ax.plot(series["t"], series["void"], label="void amp", lw=1.0, alpha=0.8)
    if "channel_direction" in series:
        ax.plot(
            series["t"],
            series["channel_direction"],
            label="channel dir (+store/−rev)",
            lw=1.2,
        )
    if "ready" in series:
        ax.plot(series["t"], series["ready"], label="ready", lw=1.2, ls=":")
        ax.plot(series["t"], series["pump_active"], label="pump on", lw=1.0, alpha=0.85)
    if "pumped_total" in series:
        ax.plot(series["t"], series["pumped_total"], label="Σ pumped", lw=1.0, alpha=0.8)
    ax.set_xlabel("t")
    ax.set_title("Lock · pump · ready · channels")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "HFB hemi-void slingshot | transducer motor–generator (pre-charge/pre-twist + dump)",
        fontsize=10,
    )
    fig.tight_layout()
    out_path = output_dir / "hemi_void_slingshot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")

    # Second figure: full cycle diagnostics (channel · pump · impulse)
    fig2, axes2 = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    ax = axes2[0]
    if "channel_direction" in series:
        ax.plot(series["t"], series["channel_direction"], label="channel dir (+1 store / −1 rev)", lw=1.6, color="C0")
        ax.fill_between(
            series["t"],
            0.0,
            series["channel_direction"],
            where=np.asarray(series["channel_direction"]) >= 0,
            alpha=0.15,
            color="C0",
        )
        ax.fill_between(
            series["t"],
            0.0,
            series["channel_direction"],
            where=np.asarray(series["channel_direction"]) < 0,
            alpha=0.15,
            color="C3",
        )
    if "ready" in series:
        ax.plot(series["t"], series["ready"], label="ready", lw=1.2, ls="--", color="C2")
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    ax.set_ylabel("channel / ready")
    ax.set_title("Flux channel polarity over the cycle")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.3, 1.3)

    ax = axes2[1]
    if "precharge_power" in series:
        ax.plot(series["t"], series["precharge_power"], label="pre-charge power", lw=1.4)
        ax.plot(series["t"], series["pretwist_power"], label="pre-twist power", lw=1.4)
    if "pump_active" in series:
        ax.plot(series["t"], series["pump_active"], label="pump on", lw=1.0, ls=":", alpha=0.8)
    ax.set_ylabel("pump power")
    ax.set_title("Active motor path (pre-charge / pre-twist)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes2[2]
    ax.plot(series["t"], series["stored"], label="total stored", lw=1.5, color="k")
    ax.plot(series["t"], series["impulse"], label="release impulse", lw=1.5, color="C3")
    if "pumped_total" in series:
        ax.plot(series["t"], series["pumped_total"], label="Σ pumped (motor)", lw=1.2, alpha=0.9)
    if "passive_total" in series:
        ax.plot(series["t"], series["passive_total"], label="Σ passive (gen)", lw=1.2, alpha=0.9)
    if "pumped_efficiency" in series:
        ax.plot(
            series["t"],
            series["pumped_efficiency"],
            label="pumped efficiency (motor share)",
            lw=1.2,
            ls="--",
            color="C4",
        )
    ax.set_xlabel("t")
    ax.set_ylabel("energy / efficiency")
    ax.set_title("Storage, dump impulse, motor vs passive fidelity")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Phase boundary markers
    t_n = slingshot.nucleate_duration
    t_s = t_n + slingshot.store_duration
    t_r = t_s + slingshot.ready_duration
    t_rel = t_r + slingshot.release_duration
    for ax in axes2:
        for t_b, name in ((t_n, "store"), (t_s, "ready"), (t_r, "release"), (t_rel, "coast")):
            ax.axvline(t_b, color="gray", lw=0.7, ls=":", alpha=0.7)

    fig2.suptitle(
        "HFB cycle diagnostics | nucleate → store → ready → release → coast",
        fontsize=11,
    )
    fig2.tight_layout()
    out_diag = output_dir / "hemi_void_slingshot_cycle.png"
    fig2.savefig(out_diag, dpi=150)
    plt.close(fig2)
    print(f"Wrote {out_diag}")

    peak_es = float(series["e_electrostatic"].max()) if "e_electrostatic" in series else 0.0
    peak_tw = float(series["e_twist"].max()) if "e_twist" in series else 0.0
    peak_geo = float(series["e_geometric"].max()) if "e_geometric" in series else 0.0
    pumped = float(series["pumped_total"].max()) if "pumped_total" in series else 0.0
    passive = float(series["passive_total"].max()) if "passive_total" in series else 0.0
    peff = float(series["pumped_efficiency"].max()) if "pumped_efficiency" in series else 0.0
    ready_frac = float(series["ready"].mean()) if "ready" in series else 0.0
    # Final breakdown from transducer if present
    breakdown = None
    if cycle.get("transducer") is not None:
        breakdown = cycle["transducer"].get_storage_breakdown()
    print(
        f"max ψ={series['psi'].max():.3f}  max stored={series['stored'].max():.3f}  "
        f"max impulse={series['impulse'].max():.3f}"
    )
    print(
        f"ledger peaks  es={peak_es:.3f}  twist={peak_tw:.3f}  geometric={peak_geo:.3f}"
    )
    print(
        f"fidelity  Σ pumped={pumped:.3f}  Σ passive={passive:.3f}  "
        f"pumped_efficiency={peff:.3f}  ready duty={ready_frac:.2f}"
    )
    if breakdown is not None:
        print(
            f"breakdown  total={breakdown['total_stored']:.3f}  "
            f"ES={breakdown['pct_electrostatic']:.1f}%  "
            f"twist={breakdown['pct_twist']:.1f}%  "
            f"geo={breakdown['pct_geometric']:.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Hopf Flux Bubble demos")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--sweep", action="store_true", help="Parameter sweep")
    parser.add_argument("--viz3d", action="store_true", help="Write flux_bubble_3d.png")
    parser.add_argument("--compare-warp", action="store_true", help="Toroidal vs exponential + fidelity")
    parser.add_argument(
        "--slingshot",
        action="store_true",
        help="Hemi-void electro-vibrational slingshot demo",
    )
    parser.add_argument(
        "--mission",
        action="store_true",
        help="Transducer + craft/payload dynamics mission demo",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mission:
        run_mission_demo(cfg, args.output)
    elif args.slingshot:
        run_slingshot_demo(cfg, args.output)
    elif args.compare_warp:
        run_compare_warp(cfg, args.output)
    elif args.sweep:
        run_sweep_demo(cfg)
    else:
        run_bubble_demo(cfg, args.output, viz3d=args.viz3d)


def slingshot_main() -> None:
    parser = argparse.ArgumentParser(description="HFB hemi-void slingshot demo")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    run_slingshot_demo(load_config(args.config), args.output)


def run_mission_demo(cfg: dict, output_dir: Path, *, coupled: bool = False) -> None:
    """Transducer cycle + craft/payload trajectory integration demo."""
    import matplotlib.pyplot as plt

    from hfb.craft import (
        CraftConfig,
        MissionConfig,
        format_energy_flow_summary,
        simulate_mission,
        simulate_mission_coupled,
    )
    from hfb.electro_vibrational import (
        DualChargeConfig,
        ObserverSyncConfig,
        PhaseAlignmentConfig,
        SlingshotConfig,
        TransducerConfig,
    )

    grid = cfg["grid"]
    ev = cfg.get("electro_vibrational", {})
    craft_yaml = cfg.get("craft", {})
    hemi_cfg = cfg.get("hemi_void", {})
    fb_yaml = craft_yaml.get("feedback", {})

    nx = min(int(grid.get("nx", 64)), 64)
    extent = float(grid.get("extent", 4.0))
    x, y = cartesian_grid(nx, nx, extent=extent)

    axis = craft_yaml.get("axis", hemi_cfg.get("axis", "x"))
    tx_yaml = ev.get("transducer", {})
    slingshot = SlingshotConfig(
        charge=DualChargeConfig(
            inner_radius=ev.get("inner_radius", 0.85),
            outer_radius=ev.get("outer_radius", 1.15),
            charge_density=ev.get("charge_density", 1.0),
            vibration_amp=ev.get("vibration_amp", 0.15),
            elongation=hemi_cfg.get("elongation", 1.4),
            axis=axis,
        ),
        phase=PhaseAlignmentConfig(
            threshold=ev.get("phase_threshold", 0.72),
            medium_resonance=ev.get("medium_resonance", 1.0),
            drive_frequency=ev.get("drive_frequency", 1.0),
        ),
        observer=ObserverSyncConfig(
            coupling=ev.get("observer_coupling", 0.4),
            observer_frequency=ev.get("observer_frequency", 1.0),
            hum_frequency=ev.get("hum_frequency", 1.0),
        ),
        transducer=TransducerConfig(
            capacity=tx_yaml.get("capacity", 1.0),
            enable_precharge=tx_yaml.get("enable_precharge", True),
            enable_pretwist=tx_yaml.get("enable_pretwist", True),
            precharge_rate=tx_yaml.get("precharge_rate", 0.55),
            pretwist_rate=tx_yaml.get("pretwist_rate", 0.50),
            target_energy=tx_yaml.get("target_energy", 0.95),
            ready_hysteresis=tx_yaml.get("ready_hysteresis", 0.04),
            pump_intensity=tx_yaml.get("pump_intensity", ev.get("pump_intensity", 1.0)),
            intensity=tx_yaml.get("intensity", ev.get("release_intensity", 1.0)),
            axis=axis,
        ),
        store_duration=ev.get("store_duration", 2.0),
        ready_duration=ev.get("ready_duration", 0.4),
        release_duration=ev.get("release_duration", 0.8),
        nucleate_duration=ev.get("nucleate_duration", 1.0),
        release_detuning=ev.get("release_detuning", 0.35),
        enable_observer=ev.get("enable_observer", True),
        use_transducer=ev.get("use_transducer", True),
        release_intensity=ev.get("release_intensity", 1.0),
        pump_intensity=ev.get("pump_intensity", 1.0),
    )
    craft_cfg = CraftConfig(
        mass=craft_yaml.get("mass", 1.0),
        axis=axis,
        impulse_coupling=craft_yaml.get("impulse_coupling", 1.0),
        store_recoil=craft_yaml.get("store_recoil", 0.04),
        drag=craft_yaml.get("drag", 0.05),
        twist_lateral=craft_yaml.get("twist_lateral", 0.12),
        geometric_spring=craft_yaml.get("geometric_spring", 0.08),
        es_boost=craft_yaml.get("es_boost", 0.10),
        max_speed=craft_yaml.get("max_speed", 8.0),
    )
    use_coupled = coupled or bool(craft_yaml.get("coupled", False)) or bool(
        fb_yaml.get("enable", False)
    )
    mission = MissionConfig(
        slingshot=slingshot,
        craft=craft_cfg,
        t_max=float(ev.get("t_max", 5.5)),
        dt=float(ev.get("dt", 0.05)),
        grid_nx=nx,
        grid_extent=extent,
        enable_craft_feedback=bool(fb_yaml.get("enable", use_coupled and craft_yaml.get("feedback_enable", False)))
        or bool(craft_yaml.get("enable_craft_feedback", False)),
        feedback_target_speed=float(fb_yaml.get("target_speed", craft_yaml.get("feedback_target_speed", 0.8))),
        feedback_target_position=float(
            fb_yaml.get("target_position", craft_yaml.get("feedback_target_position", 1.0))
        ),
        feedback_pump_gain=float(fb_yaml.get("pump_gain", 0.35)),
        feedback_release_gain=float(fb_yaml.get("release_gain", 0.35)),
        feedback_position_gain=float(fb_yaml.get("position_gain", 0.15)),
        feedback_max_boost=float(fb_yaml.get("max_boost", 0.40)),
    )
    if use_coupled:
        result = simulate_mission_coupled(x, y, cfg=mission)
    else:
        result = simulate_mission(x, y, cfg=mission)
    series = result["series"]
    craft = result["craft"]
    final = result["final_craft"]
    energy = result.get("energy_flow", {})

    output_dir.mkdir(parents=True, exist_ok=True)

    # Main figure: causal chain — ledger channels + impulse | trajectory
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))

    ax = axes[0, 0]
    if "e_electrostatic" in series:
        ax.plot(series["t"], series["e_electrostatic"], label="ES", lw=1.3)
        ax.plot(series["t"], series["e_twist"], label="twist", lw=1.3)
        ax.plot(series["t"], series["e_geometric"], label="geometric", lw=1.3)
    ax.plot(series["t"], series["stored"], label="ledger total", lw=1.6, color="k")
    ax.plot(series["t"], series["impulse"], label="directed impulse", lw=1.4, ls="--", color="C3")
    ax.set_ylabel("energy / impulse")
    ax.set_title("Transducer ledger channels + impulse (engine)")
    ax.legend(fontsize=6.5, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(craft["x"], craft["y"], lw=1.6, color="C0")
    ax.scatter([craft["x"][0]], [craft["y"][0]], c="g", s=45, zorder=3, label="start")
    ax.scatter([craft["x"][-1]], [craft["y"][-1]], c="r", s=45, zorder=3, label="end")
    # Color trajectory by impulse epochs if available
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Craft trajectory (lab frame)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(craft["t"], craft["x"] if axis == "x" else craft["y"], label="position (axis)", lw=1.5)
    ax.plot(craft["t"], craft["speed"], label="speed", lw=1.4)
    ax.plot(craft["t"], craft["kinetic_energy"], label="KE", lw=1.3, color="C3")
    ax.plot(series["t"], series["impulse"], label="impulse", lw=1.2, ls="--")
    ax.set_xlabel("t")
    ax.set_title("Craft kinematics (response to impulse)")
    ax.legend(fontsize=6.5)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(craft["t"], craft["integrated_impulse"], label="∫ impulse → craft", lw=1.5)
    ax.plot(craft["t"], craft["kinetic_energy"], label="craft KE", lw=1.4, color="C3")
    if "pumped_total" in series:
        ax.plot(series["t"], series["pumped_total"], label="Σ pumped", lw=1.2, alpha=0.9)
    if "passive_total" in series:
        ax.plot(series["t"], series["passive_total"], label="Σ passive", lw=1.2, alpha=0.9)
    if energy:
        ax.axhline(
            energy.get("efficiency_ke_over_pumped", 0.0),
            color="C4",
            ls=":",
            lw=1.2,
            label=f"KE/pumped={energy.get('efficiency_ke_over_pumped', 0):.3f}",
        )
    ax.set_xlabel("t")
    ax.set_title("Energy flow: engine intake → craft KE")
    ax.legend(fontsize=6.5)
    ax.grid(True, alpha=0.3)

    t_n = slingshot.nucleate_duration
    t_s = t_n + slingshot.store_duration
    t_r = t_s + slingshot.ready_duration
    t_rel = t_r + slingshot.release_duration
    for ax in (axes[0, 0], axes[1, 0], axes[1, 1]):
        for t_b in (t_n, t_s, t_r, t_rel):
            ax.axvline(t_b, color="gray", lw=0.7, ls=":", alpha=0.6)

    mode = "coupled feedback" if result.get("coupled") else "open-loop"
    fig.suptitle(
        f"HFB mission | engine ledger → craft trajectory ({mode})",
        fontsize=11,
    )
    fig.tight_layout()
    out = output_dir / "craft_mission.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")

    # Causal-chain companion figure: stacked ledger | impulse | trajectory x(t)
    fig2, axes2 = plt.subplots(3, 1, figsize=(11, 8), sharex=False)
    ax = axes2[0]
    if "e_electrostatic" in series:
        ax.stackplot(
            series["t"],
            series["e_electrostatic"],
            series["e_twist"],
            series["e_geometric"],
            labels=["ES", "twist", "geometric"],
            alpha=0.75,
        )
    ax.plot(series["t"], series["stored"], color="k", lw=1.2, label="total")
    ax.set_ylabel("ledger")
    ax.set_title("1 · Transducer channel store")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes2[1]
    ax.plot(series["t"], series["impulse"], color="C3", lw=1.6, label="directed impulse")
    ax.fill_between(series["t"], 0.0, series["impulse"], alpha=0.25, color="C3")
    if "channel_direction" in series:
        ax.plot(
            series["t"],
            series["channel_direction"],
            color="C0",
            lw=1.2,
            label="channel dir",
            alpha=0.85,
        )
    ax.set_ylabel("impulse / dir")
    ax.set_title("2 · Channel reversion + metered dump")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes2[2]
    ax.plot(craft["t"], craft["x"] if axis == "x" else craft["y"], lw=1.6, label="craft position")
    ax.plot(craft["t"], craft["speed"], lw=1.3, label="speed")
    ax.plot(craft["t"], craft["kinetic_energy"], lw=1.3, label="KE", color="C3")
    ax.set_xlabel("t")
    ax.set_ylabel("craft")
    ax.set_title("3 · Craft response (trajectory / speed / KE)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    for ax in axes2:
        for t_b in (t_n, t_s, t_r, t_rel):
            ax.axvline(t_b, color="gray", lw=0.7, ls=":", alpha=0.6)

    fig2.suptitle(
        "Causal chain | store channels → impulse dump → craft motion",
        fontsize=11,
    )
    fig2.tight_layout()
    out2 = output_dir / "craft_mission_chain.png"
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    print(f"Wrote {out2}")

    print(
        f"final  x={final['x']:.4f}  y={final['y']:.4f}  "
        f"speed={final['speed']:.4f}  KE={final['kinetic_energy']:.4f}  "
        f"∫J={final['integrated_impulse']:.4f}"
    )
    if result.get("energy_flow_text"):
        print(result["energy_flow_text"])
    elif energy:
        print(format_energy_flow_summary(energy))


def mission_main() -> None:
    parser = argparse.ArgumentParser(description="HFB craft mission demo")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--coupled",
        action="store_true",
        help="Use simulate_mission_coupled (optional craft→throttle feedback)",
    )
    parser.add_argument(
        "--feedback",
        action="store_true",
        help="Enable craft speed/position throttle feedback (implies --coupled)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.feedback:
        cfg.setdefault("craft", {})["enable_craft_feedback"] = True
        cfg.setdefault("craft", {}).setdefault("feedback", {})["enable"] = True
        args.coupled = True
    run_mission_demo(cfg, args.output, coupled=args.coupled)


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