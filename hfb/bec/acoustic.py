"""BEC acoustic metric from superfluid density and velocity."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hfb.analog_gravity.acoustic import acoustic_metric_components, ergosurface_mask
from hfb.bec.gpe import thomas_fermi_density


def bec_sound_speed(
    density: NDArray[np.floating],
    interaction: float = 1.0,
    mass: float = 1.0,
) -> NDArray[np.floating]:
    """Local phonon speed c_s = √(g n / m) in dimensionless GPE units."""
    return np.sqrt(interaction * np.maximum(density, 1e-12) / mass)


def bec_acoustic_metric(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    vx: NDArray[np.floating],
    vy: NDArray[np.floating],
    n0: float = 1.0,
    radius: float = 1.0,
    interaction: float = 1.0,
    mass: float = 1.0,
) -> dict[str, NDArray[np.floating]]:
    """Full BEC acoustic metric from Thomas-Fermi density + imprinted flow."""
    density = thomas_fermi_density(x, y, n0=n0, radius=radius)
    cs = bec_sound_speed(density, interaction=interaction, mass=mass)
    metric = acoustic_metric_components(vx, vy, cs)
    metric["density"] = density
    metric["c_s"] = cs
    metric["ergoregion"] = ergosurface_mask(vx, vy, cs)
    return metric