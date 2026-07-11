"""Orthogonal hemi-shaped (gourd/melon) void bubble geometries.

Asymmetric void bubbles stabilized by electrostatic tension from opposing
charges + selective vibrational modes. The rear hemisphere acts as a hybrid
electrostatic + flux-flywheel reservoir; directional release via controlled
detuning implements the slingshot concept within analog gravity frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hfb.defects.conformal import solve_conformal_poisson
from hfb.defects.densities import exponential_ring, toroidal_bubble_wall
from hfb.electro_vibrational.charge_envelopes import DualChargeConfig
from hfb.electro_vibrational.dynamics import (
    SlingshotConfig,
    flywheel_shift_boost,
    resonant_control_step,
)
from hfb.analog_gravity.acoustic import acoustic_metric_components, draining_vortex_flow
from hfb.bubble.warp_conduit import effective_shift_profile, index_from_omega


@dataclass(frozen=True)
class HemiVoidConfig:
    """Geometry parameters for orthogonal hemi / gourd void bubbles."""

    major_radius: float = 1.0
    minor_radius: float = 0.35
    wall_width: float = 0.22
    amplitude: float = 1.0
    elongation: float = 1.4
    """Orthogonal stretch (gourd width vs length)."""
    rear_extension: float = 0.45
    """Extra axial stretch of the rear hemi (rubber-band packing)."""
    front_taper: float = 0.7
    """Front hemi compression factor (<1 → blunter nose)."""
    axis: str = "x"
    void_depth: float = 0.85
    """How strongly the interior density is suppressed (0–1)."""
    hopf_index: int = 1


def hemi_gourd_radius(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    cfg: HemiVoidConfig | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Direction-dependent wall radius for a gourd/melon silhouette.

    Rear (negative axis): elongated + wider packing.
    Front (positive axis): tapered.
    Orthogonal axis: elongated by ``elongation``.
    """
    cfg = cfg or HemiVoidConfig()
    if cfg.axis == "y":
        axial, ortho = y, x
    else:
        axial, ortho = x, y

    # Angular weight from nose (+1) to tail (-1)
    r_xy = np.sqrt(axial**2 + ortho**2) + 1e-12
    cos_a = axial / r_xy  # +1 front, -1 rear
    rear = 0.5 * (1.0 - cos_a)
    front = 0.5 * (1.0 + cos_a)

    r_wall = cfg.major_radius * (
        1.0
        + cfg.rear_extension * rear
        - (1.0 - cfg.front_taper) * front * 0.35
    )
    # Orthogonal melon flattening in effective polar radius
    r_eff = np.sqrt(axial**2 + (ortho / cfg.elongation) ** 2)
    return r_wall, r_eff


def hemi_void_defect_density(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    cfg: HemiVoidConfig | None = None,
) -> NDArray[np.floating]:
    """Asymmetric hemi-void wall density with suppressed interior."""
    cfg = cfg or HemiVoidConfig()
    r_wall, r_eff = hemi_gourd_radius(x, y, cfg)
    wall = np.exp(-((r_eff - r_wall) ** 2) / (2.0 * cfg.wall_width**2))

    # Mild toroidal texture along the ring for Hopf flavor
    if cfg.axis == "x":
        theta = np.arctan2(y, x)
    else:
        theta = np.arctan2(x, y)
    texture = 0.5 * (1.0 + 0.35 * np.cos(cfg.hopf_index * theta))
    lam = cfg.amplitude * wall * texture

    # Structured void: suppress density inside the gourd
    interior = r_eff < (r_wall - 0.5 * cfg.wall_width)
    lam = np.where(interior, lam * (1.0 - cfg.void_depth), lam)
    return lam


def hemi_void_mask(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    cfg: HemiVoidConfig | None = None,
) -> NDArray[np.bool_]:
    """Boolean mask of the structured void interior."""
    cfg = cfg or HemiVoidConfig()
    r_wall, r_eff = hemi_gourd_radius(x, y, cfg)
    return r_eff < (r_wall - 0.5 * cfg.wall_width)


def rear_hemi_mask(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    cfg: HemiVoidConfig | None = None,
) -> NDArray[np.bool_]:
    """Rear hemisphere of the void (flywheel reservoir region)."""
    cfg = cfg or HemiVoidConfig()
    interior = hemi_void_mask(x, y, cfg)
    axial = x if cfg.axis == "x" else y
    return interior & (axial < 0.0)


def hemi_void_bubble_metric(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    dx: float = 0.1,
    hemi: HemiVoidConfig | None = None,
    slingshot: SlingshotConfig | None = None,
    t: float = 0.0,
    stored_energy: float = 0.0,
    circulation: float = 0.35,
    c0: float = 1.0,
    shift_amplitude: float = 0.3,
    include_control: bool = True,
) -> dict[str, NDArray[np.floating] | object]:
    """Composite metric: hemi-void defect + vortex flow + resonant control.

    When ``include_control`` is True, applies electro-vibrational modulation,
    void nucleation from phase alignment, and rear-hemi flywheel shift boost.
    """
    hemi = hemi or HemiVoidConfig()
    slingshot = slingshot or SlingshotConfig(
        charge=DualChargeConfig(
            inner_radius=0.85 * hemi.major_radius,
            outer_radius=1.15 * hemi.major_radius,
            elongation=hemi.elongation,
            axis=hemi.axis,
        )
    )

    lam = hemi_void_defect_density(x, y, hemi)
    control = None
    if include_control:
        control = resonant_control_step(
            x,
            y,
            t=t,
            stored_energy=stored_energy,
            cfg=slingshot,
            base_defect=lam,
        )
        if control["defect_modulated"] is not None:
            lam = control["defect_modulated"]

    omega = solve_conformal_poisson(lam, dx)
    vx, vy = draining_vortex_flow(x, y, circulation=circulation)
    shift = effective_shift_profile(
        x,
        y,
        bubble_radius=hemi.major_radius,
        wall_width=hemi.wall_width,
        shift_amplitude=shift_amplitude,
    )
    if include_control and control is not None:
        shift = shift + control["shift_boost"]
        # Extra directional bias from void asymmetry during release
        impulse = float(control["release_impulse"])
        if impulse > 0.0:
            axial = x if hemi.axis == "x" else y
            shift = shift + impulse * np.exp(
                -((np.sqrt(x**2 + y**2) - hemi.major_radius) ** 2)
                / (2.0 * hemi.wall_width**2)
            ) * np.tanh(axial / hemi.major_radius)

    if hemi.axis == "x":
        vx = vx + shift
    else:
        vy = vy + shift

    n_eff = index_from_omega(omega)
    acoustic = acoustic_metric_components(vx, vy, c0 * n_eff)

    out: dict[str, NDArray[np.floating] | object] = {
        "omega": omega,
        "defect_density": lam,
        "shift": shift,
        "vx": vx,
        "vy": vy,
        "n_eff": n_eff,
        "void_mask": hemi_void_mask(x, y, hemi),
        "rear_mask": rear_hemi_mask(x, y, hemi),
        **acoustic,
    }
    if control is not None:
        out["control"] = control
        out["psi"] = control["psi"]
        out["stored_energy"] = control["stored_energy"]
        out["release_impulse"] = control["release_impulse"]
        out["void_amplitude_field"] = control["void_amplitude_field"]
        out["charge_density"] = control["charge_density"]
        out["e_field"] = control["e_field"]
        out["alignment_field"] = control["alignment_field"]
        out["slingshot_phase"] = control["phase"]
    return out


def hemi_void_from_profiles(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    blend: float = 0.5,
    major_radius: float = 1.0,
    wall_width: float = 0.22,
) -> NDArray[np.floating]:
    """Blend classic ring profiles into a mild hemi-void (legacy bridge)."""
    ring = exponential_ring(x, y, radius=major_radius, width=wall_width)
    torus = toroidal_bubble_wall(
        x, y, major_radius=major_radius, minor_radius=0.35 * major_radius, wall_width=wall_width
    )
    hemi = hemi_void_defect_density(
        x, y, HemiVoidConfig(major_radius=major_radius, wall_width=wall_width)
    )
    b = float(np.clip(blend, 0.0, 1.0))
    return (1.0 - b) * 0.5 * (ring + torus) + b * hemi
