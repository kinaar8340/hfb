"""Gross-Pitaevskii density and phase proxies for tabletop BEC analogs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def thomas_fermi_density(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    n0: float = 1.0,
    radius: float = 1.0,
    trap_power: float = 2.0,
) -> NDArray[np.floating]:
    """2D Thomas-Fermi disk: n = n₀ max(0, 1 - (r/R)^p)."""
    r = np.sqrt(x**2 + y**2)
    return n0 * np.maximum(0.0, 1.0 - (r / radius) ** trap_power)


def healing_length(
    density: NDArray[np.floating],
    interaction: float = 1.0,
    mass: float = 1.0,
) -> NDArray[np.floating]:
    """ξ = 1/√(8π a_s n) proxy (dimensionless units)."""
    return 1.0 / np.sqrt(8.0 * np.pi * interaction * np.maximum(density, 1e-12) / mass)


def gpe_phase(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    phase_offset: float = 0.0,
) -> NDArray[np.floating]:
    """Uniform-phase condensate (trivial ground state)."""
    return np.full_like(x, phase_offset, dtype=float)