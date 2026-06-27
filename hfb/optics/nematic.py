"""Nematic vortex lensing — cosmic-string analog in LCs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cosmic_string_deflection(
    delta_n: float,
    n_eff: float,
    winding: int = 1,
) -> float:
    """Angular deficit Δφ ≈ 2π Δn / n_e for nematic vortex (cf. 8πGμ/c² for strings)."""
    return 2.0 * np.pi * abs(winding) * delta_n / n_eff


def nematic_deflection_field(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    vortex_positions: list[tuple[float, float]],
    windings: list[int],
    delta_n: float = 0.1,
    n_eff: float = 1.5,
    core_radius: float = 0.15,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Piecewise deflection vector field from an array of nematic vortices."""
    dx_field = np.zeros_like(x)
    dy_field = np.zeros_like(y)
    for (x0, y0), k in zip(vortex_positions, windings):
        rx = x - x0
        ry = y - y0
        r = np.sqrt(rx**2 + ry**2) + 1e-12
        deficit = cosmic_string_deflection(delta_n, n_eff, winding=k)
        # Tangential kick ∝ deficit / r outside core
        envelope = np.exp(-(r / core_radius) ** 2)
        dx_field += deficit * (-ry / r) * envelope / r
        dy_field += deficit * (rx / r) * envelope / r
    return dx_field, dy_field