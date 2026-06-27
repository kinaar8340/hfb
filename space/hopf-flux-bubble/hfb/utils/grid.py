"""2D grid utilities for conformal and ray-tracing solvers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cartesian_grid(
    nx: int,
    ny: int,
    extent: float = 5.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Uniform Cartesian grid centered at origin."""
    x = np.linspace(-extent, extent, nx)
    y = np.linspace(-extent, extent, ny)
    return np.meshgrid(x, y, indexing="xy")


def polar_grid(
    nr: int,
    ntheta: int,
    r_max: float = 5.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Polar grid with r in [0, r_max]."""
    r = np.linspace(0.0, r_max, nr)
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    return np.meshgrid(r, theta, indexing="ij")


def laplacian_fft(field: NDArray[np.floating], dx: float) -> NDArray[np.floating]:
    """Periodic Laplacian via spectral differentiation."""
    kx = 2.0 * np.pi * np.fft.fftfreq(field.shape[1], d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(field.shape[0], d=dx)
    kx2, ky2 = np.meshgrid(kx, ky, indexing="xy")
    k2 = kx2**2 + ky2**2
    f_hat = np.fft.fft2(field)
    return np.real(np.fft.ifft2(-k2 * f_hat))


def gradient_2d(
    field: NDArray[np.floating],
    dx: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Central finite-difference gradient on a uniform grid."""
    gy, gx = np.gradient(field, dx, edge_order=2)
    return gx, gy