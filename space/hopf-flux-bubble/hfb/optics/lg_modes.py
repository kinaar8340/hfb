"""Laguerre-Gaussian modes and OAM projection."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import factorial, genlaguerre


def lg_radial(p: int, ell: int, rho: NDArray[np.floating], w0: float) -> NDArray[np.floating]:
    """Radial LG_{p}^{|ell|} factor."""
    L = abs(ell)
    norm = np.sqrt(2 * factorial(L) / (np.pi * w0**2 * factorial(p)))
    rw = np.sqrt(2) * rho / w0
    lag = genlaguerre(p, L)(rw**2)
    return norm * (rw**L) * np.exp(-rw**2 / 2) * lag


def lg_mode(ell: int, rho: NDArray[np.floating], phi: NDArray[np.floating], w0: float = 1.0, p: int = 0) -> NDArray[np.complexfloating]:
    """Scalar LG mode with helical phase exp(i ell phi)."""
    return lg_radial(p, ell, rho, w0) * np.exp(1j * ell * phi)


def lg_mode_full(
    ell: int,
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    w0: float = 1.0,
    p: int = 0,
) -> NDArray[np.complexfloating]:
    """LG mode on a Cartesian grid."""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return lg_mode(ell, rho, phi, w0=w0, p=p)


def project_oam_spectrum(
    field: NDArray[np.complexfloating],
    rho: NDArray[np.floating],
    phi: NDArray[np.floating],
    ell_range: range | list[int],
    w0: float = 1.0,
) -> dict[int, complex]:
    """Project field onto LG basis (discrete inner product)."""
    weights: dict[int, complex] = {}
    dr = rho[1, 0] - rho[0, 0] if rho.ndim == 2 else rho[1] - rho[0]
    for ell in ell_range:
        basis = lg_mode(ell, rho, phi, w0=w0)
        integrand = field * np.conj(basis) * rho
        weights[ell] = np.sum(integrand) * dr**2
    return weights