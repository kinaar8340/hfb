"""Imprinted vortex phases and superfluid velocity fields."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def imprinted_vortex_phase(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    centers: list[tuple[float, float]],
    windings: list[int],
    core_radius: float = 0.1,
) -> NDArray[np.floating]:
    """Superpose phase windings θ = Σ kᵢ arctan2(y-yᵢ, x-xᵢ) with core regularization."""
    theta = np.zeros_like(x, dtype=float)
    for (x0, y0), k in zip(centers, windings):
        dx = x - x0
        dy = y - y0
        r = np.sqrt(dx**2 + dy**2) + 1e-12
        envelope = np.exp(-(r / core_radius) ** 2)
        theta += k * np.arctan2(dy, dx) * envelope
    return theta


def vortex_ring_velocity(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    ring_radius: float = 1.0,
    num_vortices: int = 8,
    winding: int = 1,
    core_radius: float = 0.12,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Velocity v = ∇θ from a ring of imprinted vortices.

    Returns (vx, vy, phase).
    """
    centers = [
        (ring_radius * np.cos(2 * np.pi * i / num_vortices), ring_radius * np.sin(2 * np.pi * i / num_vortices))
        for i in range(num_vortices)
    ]
    windings = [winding] * num_vortices
    theta = imprinted_vortex_phase(x, y, centers, windings, core_radius=core_radius)
    dx = float(x[0, 1] - x[0, 0]) if x.ndim == 2 else float(x[1] - x[0])
    vy, vx = np.gradient(theta, dx, edge_order=2)
    return vx, vy, theta