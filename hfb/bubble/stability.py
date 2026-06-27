"""Bubble stability diagnostics and parameter sweeps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hfb.analog_gravity.acoustic import ergosurface_mask
from hfb.bubble.topology import (
    defect_winding_on_ring,
    integrated_curvature_flux,
    linking_proxy,
)
from hfb.bubble.warp_conduit import flux_bubble_metric
from hfb.defects.conformal import ricci_scalar
from hfb.utils.grid import cartesian_grid


@dataclass
class StabilityReport:
    max_ricci: float
    mean_ricci_wall: float
    ergo_fraction: float
    shift_gradient_max: float
    stable_proxy: bool
    curvature_flux: float = 0.0
    topological_winding: float = 0.0
    linking_proxy: float = 0.0


def bubble_stability_metrics(
    metric: dict[str, NDArray[np.floating]],
    dx: float,
    wall_mask: NDArray[np.bool_] | None = None,
    x: NDArray[np.floating] | None = None,
    y: NDArray[np.floating] | None = None,
    major_radius: float = 1.0,
    hopf_index: int = 1,
    use_3d_torus: bool = False,
    defect_profile: str = "toroidal_bubble_wall",
) -> StabilityReport:
    """Heuristic stability proxy from curvature, ergoregions, and topology."""
    R = ricci_scalar(metric["omega"], dx)
    vx = metric["g_tx"] * -1.0
    vy = metric["g_ty"] * -1.0
    cs = np.sqrt(np.maximum(-metric["g_tt"], 1e-9))
    ergo = ergosurface_mask(vx, vy, cs)

    if wall_mask is None:
        wall_mask = metric["defect_density"] > 0.5 * np.max(metric["defect_density"])

    grad_shift = np.gradient(metric["shift"], dx)
    shift_grad_max = float(max(np.max(np.abs(g)) for g in grad_shift))

    max_ricci = float(np.max(np.abs(R)))
    mean_wall = float(np.mean(R[wall_mask]))
    ergo_frac = float(np.mean(ergo))
    curv_flux = integrated_curvature_flux(metric["omega"], dx, wall_mask)

    topo_winding = 0.0
    if x is not None and y is not None:
        try:
            topo_winding = defect_winding_on_ring(
                metric["defect_density"], x, y, major_radius, dx
            )
        except (IndexError, ValueError):
            topo_winding = float(hopf_index) if use_3d_torus else 0.0

    link = linking_proxy(hopf_index, use_3d_torus, defect_profile)
    stable = max_ricci < 50.0 and ergo_frac < 0.15 and shift_grad_max < 5.0

    return StabilityReport(
        max_ricci=max_ricci,
        mean_ricci_wall=mean_wall,
        ergo_fraction=ergo_frac,
        shift_gradient_max=shift_grad_max,
        stable_proxy=stable,
        curvature_flux=curv_flux,
        topological_winding=topo_winding,
        linking_proxy=link,
    )


def parameter_sweep(
    radii: list[float],
    circulations: list[float],
    nx: int = 96,
    extent: float = 4.0,
    defect_profile: str = "exponential_ring",
    major_radius: float | None = None,
    minor_radius: float | None = None,
    use_3d_torus: bool = False,
    hopf_index: int = 1,
) -> list[tuple[float, float, StabilityReport]]:
    """Sweep bubble radius and vortex circulation; return stability reports."""
    x, y = cartesian_grid(nx, nx, extent=extent)
    dx = x[0, 1] - x[0, 0]
    results: list[tuple[float, float, StabilityReport]] = []
    for radius in radii:
        for circ in circulations:
            metric = flux_bubble_metric(
                x,
                y,
                bubble_radius=radius,
                circulation=circ,
                dx=dx,
                defect_profile=defect_profile,
                major_radius=major_radius,
                minor_radius=minor_radius,
                use_3d_torus=use_3d_torus,
                hopf_index=hopf_index,
            )
            report = bubble_stability_metrics(
                metric,
                dx,
                x=x,
                y=y,
                major_radius=major_radius or radius,
                hopf_index=hopf_index,
                use_3d_torus=use_3d_torus,
                defect_profile=defect_profile,
            )
            results.append((radius, circ, report))
    return results