"""BEC acoustic backend demo plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hfb.bec.acoustic import bec_acoustic_metric
from hfb.bec.bogoliubov import bogoliubov_dispersion
from hfb.bec.vortex import vortex_ring_velocity
from hfb.utils.grid import cartesian_grid


def run_bec_acoustic_demo(
    output_dir: Path,
    *,
    nx: int = 96,
    extent: float = 3.0,
    ring_radius: float = 1.0,
    num_vortices: int = 8,
    n0: float = 1.0,
    interaction: float = 1.0,
) -> dict:
    """Plot BEC density, sound speed, ergoregion, and Bogoliubov dispersion."""
    output_dir.mkdir(parents=True, exist_ok=True)
    x, y = cartesian_grid(nx, nx, extent=extent)
    vx, vy, phase = vortex_ring_velocity(x, y, ring_radius=ring_radius, num_vortices=num_vortices)
    metric = bec_acoustic_metric(x, y, vx, vy, n0=n0, radius=ring_radius, interaction=interaction)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    lim = [-extent, extent]

    im0 = axes[0, 0].imshow(metric["density"], origin="lower", extent=lim * 2)
    axes[0, 0].set_title("Thomas-Fermi density")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(metric["c_s"], origin="lower", extent=lim * 2)
    axes[0, 1].set_title("Local sound speed c_s")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[1, 0].imshow(metric["ergoregion"].astype(float), origin="lower", extent=lim * 2)
    axes[1, 0].set_title("Ergoregion (v² ≥ c_s²)")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046)

    k = np.linspace(0.05, 4.0, 80)
    eps = bogoliubov_dispersion(k, c_s=float(np.median(metric["c_s"][metric["density"] > 0])))
    axes[1, 1].plot(k, eps, "b-")
    axes[1, 1].set_title("Bogoliubov dispersion (median c_s)")
    axes[1, 1].set_xlabel("k")
    axes[1, 1].set_ylabel("ε(k)")
    axes[1, 1].grid(True, alpha=0.3)

    ergo_frac = float(np.mean(metric["ergoregion"]))
    fig.suptitle(
        f"BEC Acoustic Backend | vortices={num_vortices} | ergo_fraction={ergo_frac:.4f}",
        fontsize=11,
    )
    fig.tight_layout()
    out_path = output_dir / "bec_acoustic_demo.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "output": str(out_path),
        "ergo_fraction": ergo_frac,
        "num_vortices": num_vortices,
        "ring_radius": ring_radius,
    }