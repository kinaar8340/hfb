"""Topological diagnostics for flux-bubble defect walls."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hfb.defects.conformal import ricci_scalar


def torus_tube_distance(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    z_slice: float,
    major_radius: float,
) -> NDArray[np.floating]:
    """Re-export-friendly alias (canonical definition in defects.densities)."""
    from hfb.defects.densities import torus_tube_distance as _dist

    return _dist(x, y, z_slice, major_radius)


def integrated_curvature_flux(
    omega: NDArray[np.floating],
    dx: float,
    region_mask: NDArray[np.bool_],
) -> float:
    """Gauss-Bonnet style ∫ R dA over a masked bubble region (2D proxy)."""
    R = ricci_scalar(omega, dx)
    return float(np.sum(R[region_mask]) * dx**2)


def defect_winding_on_ring(
    lam: NDArray[np.floating],
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    major_radius: float,
    dx: float,
    n_samples: int = 256,
) -> float:
    """Approximate winding via phase circulation of ∇λ around the major ring."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    ring_x = major_radius * np.cos(theta)
    ring_y = major_radius * np.sin(theta)

    gx, gy = np.gradient(lam, dx, edge_order=2)
    x_max = float(x[0, -1])
    ix = ((ring_x + x_max) / dx).astype(int)
    iy = ((ring_y + x_max) / dx).astype(int)
    ix = np.clip(ix, 0, lam.shape[1] - 1)
    iy = np.clip(iy, 0, lam.shape[0] - 1)

    phase = np.arctan2(gy[iy, ix], gx[iy, ix] + 1e-12)
    dphase = np.unwrap(phase)
    return float((dphase[-1] - dphase[0]) / (2.0 * np.pi))


def linking_proxy(
    hopf_index: int,
    use_3d_torus: bool,
    defect_profile: str,
) -> float:
    """Hopf-linking proxy: hopf_index when 3D torus wall is active, else 0."""
    if use_3d_torus and defect_profile == "toroidal_bubble_wall":
        return float(hopf_index)
    return 0.0