"""Combine defect curvature and flow/index gradients into an effective bubble metric."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hfb.analog_gravity.acoustic import acoustic_metric_components, draining_vortex_flow
from hfb.defects.conformal import solve_conformal_poisson
from hfb.defects.densities import build_defect_density


def index_from_omega(omega: NDArray[np.floating], n0: float = 1.0, alpha: float = 0.5) -> NDArray[np.floating]:
    """Map conformal factor to effective refractive index n = n₀ e^{αΩ}."""
    return n0 * np.exp(alpha * omega)


def effective_shift_profile(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    bubble_radius: float = 1.0,
    wall_width: float = 0.2,
    shift_amplitude: float = 0.3,
) -> NDArray[np.floating]:
    """
    Alcubierre-like shift analog: negative ahead, positive behind the bubble wall.
    Not literal GR — an effective propagation bias in the analog metric.
    """
    r = np.sqrt(x**2 + y**2)
    wall = np.exp(-((r - bubble_radius) ** 2) / (2.0 * wall_width**2))
    axial = np.tanh(x / bubble_radius)
    return shift_amplitude * wall * axial


def flux_bubble_metric(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    bubble_radius: float = 1.0,
    wall_width: float = 0.2,
    defect_amplitude: float = 1.0,
    circulation: float = 0.4,
    c0: float = 1.0,
    dx: float = 0.1,
    defect_profile: str = "exponential_ring",
    major_radius: float | None = None,
    minor_radius: float | None = None,
    use_3d_torus: bool = False,
    z_slice: float = 0.0,
    hopf_index: int = 1,
) -> dict[str, NDArray[np.floating]]:
    """
    Build a composite effective metric from:
    - toroidal defect wall (conformal Ω),
    - draining vortex flow (acoustic g_μν),
    - shift profile (warp-conduit analog).
    """
    lam = build_defect_density(
        x,
        y,
        profile=defect_profile,
        bubble_radius=bubble_radius,
        wall_width=wall_width,
        defect_amplitude=defect_amplitude,
        major_radius=major_radius,
        minor_radius=minor_radius,
        use_3d_torus=use_3d_torus,
        z_slice=z_slice,
        hopf_index=hopf_index,
    )
    omega = solve_conformal_poisson(lam, dx)

    vx, vy = draining_vortex_flow(x, y, circulation=circulation)
    shift = effective_shift_profile(x, y, bubble_radius=bubble_radius, wall_width=wall_width)
    vx = vx + shift
    n_eff = index_from_omega(omega)
    acoustic = acoustic_metric_components(vx, vy, c0 * n_eff)

    return {
        "omega": omega,
        "defect_density": lam,
        "shift": shift,
        "vx": vx,
        "vy": vy,
        "n_eff": n_eff,
        **acoustic,
    }