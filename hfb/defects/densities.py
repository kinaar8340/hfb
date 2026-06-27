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


def toroidal_bubble_wall(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    major_radius: float = 1.0,
    minor_radius: float = 0.35,
    amplitude: float = 1.0,
    wall_width: float = 0.25,
) -> NDArray[np.floating]:
    """Toroidal defect density for Hopf Flux Bubble wall.

    Approximates a linked toroidal flux wall (2D slice of Hopfion-like structure).
    Use together with hopf config parameters.
    """
    r = np.sqrt(x**2 + y**2)
    radial_dist = np.abs(r - major_radius)
    wall = np.exp(-(radial_dist**2) / (2.0 * wall_width**2))
    envelope = np.exp(-((y / minor_radius) ** 2))
    return amplitude * wall * envelope


def build_defect_density(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    profile: str = "exponential_ring",
    bubble_radius: float = 1.0,
    wall_width: float = 0.25,
    defect_amplitude: float = 1.0,
    major_radius: float | None = None,
    minor_radius: float | None = None,
) -> NDArray[np.floating]:
    """Select defect density λ(x, y) by profile name."""
    major = major_radius if major_radius is not None else bubble_radius
    minor = minor_radius if minor_radius is not None else 0.35 * bubble_radius
    if profile == "toroidal_bubble_wall":
        return toroidal_bubble_wall(
            x,
            y,
            major_radius=major,
            minor_radius=minor,
            amplitude=defect_amplitude,
            wall_width=wall_width,
        )
    if profile == "gaussian":
        return gaussian_defect(x, y, amplitude=defect_amplitude, sigma=wall_width)
    if profile == "exponential_ring":
        lam = exponential_ring(
            x, y, radius=bubble_radius, width=wall_width, amplitude=defect_amplitude
        )
        lam += 0.25 * gaussian_defect(x, y, amplitude=defect_amplitude, sigma=wall_width)
        return lam
    raise ValueError(f"Unknown defect profile: {profile!r}")