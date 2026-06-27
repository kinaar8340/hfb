"""Defect density profiles λ(r, θ) for conformal Poisson sources."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def gaussian_defect(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    amplitude: float = 1.0,
    sigma: float = 0.5,
    x0: float = 0.0,
    y0: float = 0.0,
) -> NDArray[np.floating]:
    """Isotropic Gaussian defect density."""
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    return amplitude * np.exp(-r2 / (2.0 * sigma**2))


def exponential_ring(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    radius: float = 1.0,
    width: float = 0.2,
    amplitude: float = 1.0,
) -> NDArray[np.floating]:
    """Toroidal wall density peaked at |r| = radius."""
    r = np.sqrt(x**2 + y**2)
    return amplitude * np.exp(-((r - radius) ** 2) / (2.0 * width**2))


def vortex_core(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    winding: int = 1,
    core_radius: float = 0.15,
    amplitude: float = 1.0,
) -> NDArray[np.floating]:
    """Regularized vortex core density with quantized winding k."""
    r = np.sqrt(x**2 + y**2)
    return amplitude * (r / core_radius) ** (2 * abs(winding)) * np.exp(-(r / core_radius) ** 2)


def winding_density(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    winding: int = 1,
    core_radius: float = 0.15,
) -> NDArray[np.floating]:
    """Phase winding proxy used as a smooth defect source."""
    theta = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    phase = winding * theta
    envelope = np.exp(-(r / core_radius) ** 2)
    return np.sin(phase) ** 2 * envelope