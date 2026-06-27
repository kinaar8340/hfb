"""Conformal defect geometry: ΔΩ = -λ, R = 2 e^{2Ω} ΔΩ."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hfb.utils.grid import laplacian_fft


def solve_conformal_poisson(
    defect_density: NDArray[np.floating],
    dx: float,
    gauge_zero_mean: bool = True,
) -> NDArray[np.floating]:
    """Solve ΔΩ = -λ on a periodic grid via FFT."""
    kx = 2.0 * np.pi * np.fft.fftfreq(defect_density.shape[1], d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(defect_density.shape[0], d=dx)
    kx2, ky2 = np.meshgrid(kx, ky, indexing="xy")
    k2 = kx2**2 + ky2**2
    k2[0, 0] = 1.0  # avoid division by zero; gauge fixes DC mode

    lam_hat = np.fft.fft2(defect_density)
    omega_hat = lam_hat / k2
    omega_hat[0, 0] = 0.0
    omega = np.real(np.fft.ifft2(omega_hat))
    if gauge_zero_mean:
        omega = omega - np.mean(omega)
    return omega


def conformal_factor(omega: NDArray[np.floating]) -> NDArray[np.floating]:
    """Metric conformal factor Ω(r, θ)."""
    return omega


def ricci_scalar(omega: NDArray[np.floating], dx: float) -> NDArray[np.floating]:
    """R = 2 e^{2Ω} ΔΩ for a 2D conformal metric ds² = e^{2Ω}(dx² + dy²)."""
    lap_omega = laplacian_fft(omega, dx)
    return 2.0 * np.exp(2.0 * omega) * lap_omega