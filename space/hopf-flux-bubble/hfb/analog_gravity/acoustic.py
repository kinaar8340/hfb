"""Acoustic metric ds² = -(c_s² - v²)dt² - 2v·dx dt + dx²."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def sound_speed_field(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    c0: float = 1.0,
    contrast: float = 0.0,
    sigma: float = 1.0,
) -> NDArray[np.floating]:
    """Spatially varying sound speed (e.g. defect-induced index change)."""
    r2 = x**2 + y**2
    return c0 * (1.0 + contrast * np.exp(-r2 / (2.0 * sigma**2)))


def draining_vortex_flow(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    circulation: float = 0.5,
    drain_strength: float = 0.1,
    core_radius: float = 0.2,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Azimuthal + radial draining vortex (acoustic black-hole analog)."""
    r = np.sqrt(x**2 + y**2) + 1e-12
    envelope = np.exp(-(r / core_radius) ** 2)
    vx = (-circulation * y / r - drain_strength * x / r) * envelope
    vy = (circulation * x / r - drain_strength * y / r) * envelope
    return vx, vy


def acoustic_metric_components(
    vx: NDArray[np.floating],
    vy: NDArray[np.floating],
    cs: NDArray[np.floating] | float,
) -> dict[str, NDArray[np.floating]]:
    """Return g_tt, g_tx, g_ty, g_xx, g_yy for the 2+1 acoustic line element."""
    v2 = vx**2 + vy**2
    if np.ndim(cs) == 0:
        cs_arr = np.full_like(vx, float(cs))
    else:
        cs_arr = cs
    g_tt = -(cs_arr**2 - v2)
    return {
        "g_tt": g_tt,
        "g_tx": -vx,
        "g_ty": -vy,
        "g_xx": np.ones_like(vx),
        "g_yy": np.ones_like(vx),
    }


def ergosurface_mask(
    vx: NDArray[np.floating],
    vy: NDArray[np.floating],
    cs: NDArray[np.floating] | float,
) -> NDArray[np.bool_]:
    """Regions where v² ≥ c_s² (acoustic ergoregion / horizon analog)."""
    v2 = vx**2 + vy**2
    if np.ndim(cs) == 0:
        cs2 = float(cs) ** 2
    else:
        cs2 = cs**2
    return v2 >= cs2