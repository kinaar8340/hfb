"""Bogoliubov excitation proxies for negative-energy analogs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def bogoliubov_dispersion(
    k: NDArray[np.floating],
    c_s: float | NDArray[np.floating],
    m: float = 1.0,
) -> NDArray[np.floating]:
    """ε(k) = k²/(2m) √[1 + (m c_s k / ℏ)²] with ℏ=1."""
    return (k**2 / (2.0 * m)) * np.sqrt(1.0 + (m * c_s * k) ** 2)


def excitation_energy_density(
    k_grid: NDArray[np.floating],
    c_s: NDArray[np.floating],
    temperature: float = 0.0,
    k_max: float = 10.0,
) -> NDArray[np.floating]:
    """
    Thermal Bogoliubov energy density proxy (analog negative-pressure tuning).

    At T=0 returns half the zero-point energy ∫ ε(k) dk/(2π) discretized.
    """
    dk = k_grid[1] - k_grid[0] if k_grid.ndim == 1 else k_grid[0, 1] - k_grid[0, 0]
    eps = bogoliubov_dispersion(np.abs(k_grid), c_s)
    zero_point = 0.5 * eps
    if temperature <= 0.0:
        return zero_point
    thermal = temperature * np.log(1.0 + np.exp(-eps / max(temperature, 1e-12)))
    return zero_point + thermal