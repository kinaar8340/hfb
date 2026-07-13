"""Defect density profiles λ(r, θ) for conformal Poisson sources.

Shared primitives (gaussian, ring, vortex, toroidal wall) re-export from
flux_hopf_lib.flux.defects. HFB-specific builders (hemi-void, profile select)
remain here.
"""

from __future__ import annotations

from numpy.typing import NDArray

from flux_hopf_lib.flux.defects import (
    exponential_ring,
    gaussian_defect,
    toroidal_bubble_wall,
    torus_tube_distance,
    vortex_core,
    winding_density,
)

__all__ = [
    "gaussian_defect",
    "exponential_ring",
    "vortex_core",
    "winding_density",
    "torus_tube_distance",
    "toroidal_bubble_wall",
    "hemi_void_wall",
    "build_defect_density",
]


def hemi_void_wall(
    x: NDArray,
    y: NDArray,
    major_radius: float = 1.0,
    wall_width: float = 0.22,
    amplitude: float = 1.0,
    elongation: float = 1.4,
    rear_extension: float = 0.45,
    front_taper: float = 0.7,
    hopf_index: int = 1,
    axis: str = "x",
) -> NDArray:
    """Hemi/gourd-shaped void wall (thin wrapper around bubble.hemi_void)."""
    from hfb.bubble.hemi_void import HemiVoidConfig, hemi_void_defect_density

    return hemi_void_defect_density(
        x,
        y,
        HemiVoidConfig(
            major_radius=major_radius,
            wall_width=wall_width,
            amplitude=amplitude,
            elongation=elongation,
            rear_extension=rear_extension,
            front_taper=front_taper,
            hopf_index=hopf_index,
            axis=axis,
        ),
    )


def build_defect_density(
    x: NDArray,
    y: NDArray,
    profile: str = "exponential_ring",
    bubble_radius: float = 1.0,
    wall_width: float = 0.25,
    defect_amplitude: float = 1.0,
    major_radius: float | None = None,
    minor_radius: float | None = None,
    use_3d_torus: bool = False,
    z_slice: float = 0.0,
    hopf_index: int = 1,
    elongation: float = 1.4,
    rear_extension: float = 0.45,
    axis: str = "x",
) -> NDArray:
    """Select defect density λ(x, y) by profile name."""
    major = major_radius if major_radius is not None else bubble_radius
    minor = minor_radius if minor_radius is not None else 0.35 * bubble_radius
    if profile == "toroidal_bubble_wall":
        return toroidal_bubble_wall(
            x,
            y,
            major_radius=major,
            minor_radius=minor,
            amplitude=defect_amplitude,
            wall_width=wall_width,
            use_3d_torus=use_3d_torus,
            z_slice=z_slice,
            hopf_index=hopf_index,
        )
    if profile == "hemi_void_wall":
        return hemi_void_wall(
            x,
            y,
            major_radius=major,
            wall_width=wall_width,
            amplitude=defect_amplitude,
            elongation=elongation,
            rear_extension=rear_extension,
            hopf_index=hopf_index,
            axis=axis,
        )
    if profile == "gaussian":
        return gaussian_defect(x, y, amplitude=defect_amplitude, sigma=wall_width)
    if profile == "exponential_ring":
        lam = exponential_ring(
            x, y, radius=bubble_radius, width=wall_width, amplitude=defect_amplitude
        )
        lam += 0.25 * gaussian_defect(x, y, amplitude=defect_amplitude, sigma=wall_width)
        return lam
    raise ValueError(f"Unknown defect profile: {profile!r}")
