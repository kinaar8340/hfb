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


def torus_tube_distance(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    z_slice: float,
    major_radius: float,
) -> NDArray[np.floating]:
    """Distance from (x, y, z_slice) to the major-radius centerline of a 3D torus."""
    rho = np.sqrt(x**2 + y**2)
    return np.sqrt((rho - major_radius) ** 2 + z_slice**2)


def toroidal_bubble_wall(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    major_radius: float = 1.0,
    minor_radius: float = 0.35,
    amplitude: float = 1.0,
    wall_width: float = 0.25,
    use_3d_torus: bool = False,
    z_slice: float = 0.0,
    hopf_index: int = 1,
) -> NDArray[np.floating]:
    """Toroidal defect density for Hopf Flux Bubble wall.

    With ``use_3d_torus=True``, uses tube distance to a 3D torus at ``z_slice``
    plus a Hopf-style θ + φ texture modulated by ``hopf_index``.
    Otherwise falls back to a 2D pseudo-toroidal projection.
    """
    if use_3d_torus:
        d = torus_tube_distance(x, y, z_slice, major_radius)
        tube = np.exp(-(d**2) / (2.0 * wall_width**2))
        theta = np.arctan2(y, x)
        rho = np.sqrt(x**2 + y**2)
        poloidal = np.arctan2(z_slice, rho - major_radius + 1e-12)
        hopf_texture = 0.5 * (1.0 + np.cos(hopf_index * theta + poloidal))
        return amplitude * tube * hopf_texture

    r = np.sqrt(x**2 + y**2)
    radial_dist = np.abs(r - major_radius)
    wall = np.exp(-(radial_dist**2) / (2.0 * wall_width**2))
    envelope = np.exp(-((y / minor_radius) ** 2))
    return amplitude * wall * envelope


def hemi_void_wall(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    major_radius: float = 1.0,
    wall_width: float = 0.22,
    amplitude: float = 1.0,
    elongation: float = 1.4,
    rear_extension: float = 0.45,
    front_taper: float = 0.7,
    hopf_index: int = 1,
    axis: str = "x",
) -> NDArray[np.floating]:
    """Hemi/gourd-shaped void wall (thin wrapper around bubble.hemi_void)."""
    from hfb.bubble.hemi_void import HemiVoidConfig, hemi_void_defect_density

    return hemi_void_defect_density(
        x,
        y,
        HemiVoidConfig(
            major_radius=major_radius,
            wall_width=wall_width,
            amplitude=amplitude,
            elongation=elongation,
            rear_extension=rear_extension,
            front_taper=front_taper,
            hopf_index=hopf_index,
            axis=axis,
        ),
    )


def build_defect_density(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    profile: str = "exponential_ring",
    bubble_radius: float = 1.0,
    wall_width: float = 0.25,
    defect_amplitude: float = 1.0,
    major_radius: float | None = None,
    minor_radius: float | None = None,
    use_3d_torus: bool = False,
    z_slice: float = 0.0,
    hopf_index: int = 1,
    elongation: float = 1.4,
    rear_extension: float = 0.45,
    axis: str = "x",
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
            use_3d_torus=use_3d_torus,
            z_slice=z_slice,
            hopf_index=hopf_index,
        )
    if profile == "hemi_void_wall":
        return hemi_void_wall(
            x,
            y,
            major_radius=major,
            wall_width=wall_width,
            amplitude=defect_amplitude,
            elongation=elongation,
            rear_extension=rear_extension,
            hopf_index=hopf_index,
            axis=axis,
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