"""Hopf map S³ → S² and linking diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def hopf_coordinates(
    eta: NDArray[np.floating],
    xi1: NDArray[np.floating],
    xi2: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """S³ parametrization (η, ξ₁, ξ₂) → (x₁, x₂, x₃, x₄) with Σ xᵢ² = 1."""
    c1 = np.cos(xi1)
    s1 = np.sin(xi1)
    c2 = np.cos(xi2)
    s2 = np.sin(xi2)
    ce = np.cos(eta)
    se = np.sin(eta)
    x1 = ce * c1
    x2 = ce * s1
    x3 = se * c2
    x4 = se * s2
    return x1, x2, x3, x4


def hopf_map(
    x1: NDArray[np.floating],
    x2: NDArray[np.floating],
    x3: NDArray[np.floating],
    x4: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Standard Hopf fibration to S²: (x₁² - x₂², 2x₁x₂, 2(x₃x₄ + x₁x₂))."""
    y1 = x1**2 - x2**2
    y2 = 2.0 * x1 * x2
    y3 = 2.0 * (x3 * x4 + x1 * x2)
    norm = np.sqrt(y1**2 + y2**2 + y3**2) + 1e-12
    return y1 / norm, y2 / norm, y3 / norm


def linking_number_pair(
    curve_a: NDArray[np.floating],
    curve_b: NDArray[np.floating],
) -> float:
    """Gauss linking integral for two closed 3D curves (discrete sum)."""
    n_a = curve_a.shape[0]
    n_b = curve_b.shape[0]
    total = 0.0
    for i in range(n_a):
        r1 = curve_a[i]
        r2 = curve_a[(i + 1) % n_a]
        dr1 = r2 - r1
        for j in range(n_b):
            s1 = curve_b[j]
            s2 = curve_b[(j + 1) % n_b]
            dr2 = s2 - s1
            r12 = s1 - r1
            denom = np.linalg.norm(r12) ** 3 + 1e-12
            total += np.dot(r12, np.cross(dr1, dr2)) / denom
    return total / (4.0 * np.pi)