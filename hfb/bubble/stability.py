"""Bubble stability diagnostics and parameter sweeps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hfb.analog_gravity.acoustic import ergosurface_mask
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


def bubble_stability_metrics(
    metric: dict[str, NDArray[np.floating]],
    dx: float,
    wall_mask: NDArray[np.bool_] | None = None,
) -> StabilityReport:
    """Heuristic stability proxy from curvature, ergoregions, and shift steepness."""
    R = ricci_scalar(metric["omega"], dx)
    vx = metric["g_tx"] * -1.0  # recover flow from g_tx = -vx
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

    # Proxy: avoid huge curvature spikes and large ergoregions
    stable = max_ricci < 50.0 and ergo_frac < 0.15 and shift_grad_max < 5.0

    return StabilityReport(
        max_ricci=max_ricci,
        mean_ricci_wall=mean_wall,
        ergo_fraction=ergo_frac,
        shift_gradient_max=shift_grad_max,
        stable_proxy=stable,
    )


def parameter_sweep(
    radii: list[float],
    circulations: list[float],
    nx: int = 96,
    extent: float = 4.0,
) -> list[tuple[float, float, StabilityReport]]:
    """Sweep bubble radius and vortex circulation; return stability reports."""
    x, y = cartesian_grid(nx, nx, extent=extent)
    dx = x[0, 1] - x[0, 0]
    results: list[tuple[float, float, StabilityReport]] = []
    for radius in radii:
        for circ in circulations:
            metric = flux_bubble_metric(x, y, bubble_radius=radius, circulation=circ, dx=dx)
            report = bubble_stability_metrics(metric, dx)
            results.append((radius, circ, report))
    return results