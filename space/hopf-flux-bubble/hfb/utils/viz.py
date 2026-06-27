"""Visualization helpers for Hopf Flux Bubble."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def plot_flux_bubble_3d(
    omega: NDArray[np.floating],
    vx: NDArray[np.floating],
    vy: NDArray[np.floating],
    dx: float,
    extent: float | None = None,
    title: str = "Hopf Flux Bubble — Effective Geometry",
    cmap: str = "RdBu_r",
    quiver_step: int = 8,
) -> plt.Figure:
    """3D visualization of conformal factor Ω with overlaid flow vectors."""
    ny, nx = omega.shape
    half = extent if extent is not None else dx * (nx - 1) / 2.0
    x = np.linspace(-half, half, nx)
    y = np.linspace(-half, half, ny)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(X, Y, omega, cmap=cmap, alpha=0.7, linewidth=0)
    fig.colorbar(surf, ax=ax, shrink=0.5, label=r"$\Omega$ (conformal factor)")

    ax.quiver(
        X[::quiver_step, ::quiver_step],
        Y[::quiver_step, ::quiver_step],
        np.zeros_like(X[::quiver_step, ::quiver_step]),
        vx[::quiver_step, ::quiver_step],
        vy[::quiver_step, ::quiver_step],
        np.zeros_like(vx[::quiver_step, ::quiver_step]),
        length=0.3,
        normalize=True,
        color="black",
        alpha=0.6,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(r"$\Omega$")
    ax.set_title(title)
    plt.tight_layout()
    return fig