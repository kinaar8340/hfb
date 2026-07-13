"""2D grid utilities for conformal and ray-tracing solvers.

Thin re-export of the canonical implementation in flux_hopf_lib.utils.
"""

from __future__ import annotations

from flux_hopf_lib.utils.grid import (
    cartesian_grid,
    gradient_2d,
    laplacian_fft,
    polar_grid,
)

__all__ = [
    "cartesian_grid",
    "polar_grid",
    "laplacian_fft",
    "gradient_2d",
]
