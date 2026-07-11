"""Hopfion director textures for nematic / flux-bubble walls."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def toroidal_hopfion_director(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    z: NDArray[np.floating],
    major_radius: float = 1.0,
    minor_radius: float = 0.35,
    hopf_index: int = 1,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Smooth toroidal Hopfion-like director field (unit vector)."""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    theta = np.arctan2(z, rho - major_radius)

    nx = np.sin(theta) * np.cos(hopf_index * phi)
    ny = np.sin(theta) * np.sin(hopf_index * phi)
    nz = np.cos(theta)

    # Radial envelope localizes texture on the torus
    dist = np.sqrt((rho - major_radius) ** 2 + z**2)
    envelope = np.exp(-(dist / minor_radius) ** 2)
    nx *= envelope
    ny *= envelope
    nz = nz * envelope + (1.0 - envelope)

    norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-12
    return nx / norm, ny / norm, nz / norm


def hopf_charge_density(
    nx: NDArray[np.floating],
    ny: NDArray[np.floating],
    nz: NDArray[np.floating],
    dx: float,
) -> float:
    """Proxy Hopf charge Q_H ∝ ∫ A · B d³x from director gradients."""
    dnx_dy, dnx_dx = np.gradient(nx, dx, edge_order=2)
    dny_dy, dny_dx = np.gradient(ny, dx, edge_order=2)
    dnz_dy, dnz_dx = np.gradient(nz, dx, edge_order=2)

    # Pretend field A ~ n, B ~ ∇×n (2D slice proxy)
    bx = dnz_dy - dny_dx
    by = dnx_dx - dnz_dy
    ax, ay, az = nx, ny, nz
    integrand = ax * bx + ay * by + az * 0.0
    return float(np.sum(integrand) * dx**2)


def charge_modulated_hopfion(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    z: NDArray[np.floating],
    charge_density: NDArray[np.floating],
    major_radius: float = 1.0,
    minor_radius: float = 0.35,
    hopf_index: int = 1,
    polarization: float = 0.4,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Hopfion director polarized by a dual-shell electrostatic charge field.

    The electrostatic envelope tilts the local director toward the radial
    (capacitive gap) direction — a handle for charge-modulated topological
    texture without breaking unit normalization.
    """
    nx, ny, nz = toroidal_hopfion_director(
        x, y, z, major_radius=major_radius, minor_radius=minor_radius, hopf_index=hopf_index
    )
    # Radial polarization preference from charge gradient proxy
    rho = np.sqrt(x**2 + y**2) + 1e-12
    rx, ry = x / rho, y / rho
    c_norm = charge_density / (np.max(np.abs(charge_density)) + 1e-12)
    strength = polarization * c_norm
    nx = nx + strength * rx
    ny = ny + strength * ry
    # nz slightly suppressed where charge is strong (wall-localized tilt)
    nz = nz * (1.0 - 0.25 * np.abs(strength))
    norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-12
    return nx / norm, ny / norm, nz / norm