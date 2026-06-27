"""Ray propagation in effective conformal and index metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hfb.utils.grid import gradient_2d


def trace_rays_conformal(
    omega: NDArray[np.floating],
    x0: float,
    y0: float,
    angle: float,
    dx: float,
    n_steps: int = 500,
    step_size: float = 0.05,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Geodesics in ds² = e^{2Ω}(dx² + dy²) via Hamiltonian ray equations."""
    grad_ox, grad_oy = gradient_2d(omega, dx)

    def sample_grad(px: float, py: float) -> tuple[float, float]:
        ix = int(np.clip(np.round((px / dx) + omega.shape[1] / 2), 0, omega.shape[1] - 1))
        iy = int(np.clip(np.round((py / dx) + omega.shape[0] / 2), 0, omega.shape[0] - 1))
        return float(grad_ox[iy, ix]), float(grad_oy[iy, ix])

    px, py = x0, y0
    vx, vy = np.cos(angle), np.sin(angle)
    xs = np.empty(n_steps)
    ys = np.empty(n_steps)
    for i in range(n_steps):
        xs[i], ys[i] = px, py
        gx, gy = sample_grad(px, py)
        # d²x/ds² = -∇Ω (conformal geodesic in 2D)
        vx += -gx * step_size
        vy += -gy * step_size
        norm = np.hypot(vx, vy) + 1e-12
        vx /= norm
        vy /= norm
        px += vx * step_size
        py += vy * step_size
    return xs, ys


def trace_rays_eikonal(
    index_field: NDArray[np.floating],
    x0: float,
    y0: float,
    angle: float,
    dx: float,
    n_steps: int = 500,
    step_size: float = 0.05,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Eikonal rays in n(x): dr/ds ∝ ∇(optical path), bend ∝ ∇n."""
    grad_nx, grad_ny = gradient_2d(index_field, dx)

    def sample_grad(px: float, py: float) -> tuple[float, float]:
        ix = int(np.clip(np.round((px / dx) + index_field.shape[1] / 2), 0, index_field.shape[1] - 1))
        iy = int(np.clip(np.round((py / dx) + index_field.shape[0] / 2), 0, index_field.shape[0] - 1))
        return float(grad_nx[iy, ix]), float(grad_ny[iy, ix])

    px, py = x0, y0
    vx, vy = np.cos(angle), np.sin(angle)
    xs = np.empty(n_steps)
    ys = np.empty(n_steps)
    for i in range(n_steps):
        xs[i], ys[i] = px, py
        gx, gy = sample_grad(px, py)
        vx += gx * step_size
        vy += gy * step_size
        norm = np.hypot(vx, vy) + 1e-12
        vx /= norm
        vy /= norm
        px += vx * step_size
        py += vy * step_size
    return xs, ys