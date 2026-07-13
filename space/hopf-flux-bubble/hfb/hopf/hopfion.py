"""Hopfion director textures for nematic / flux-bubble walls.

Thin re-export of the canonical implementation in flux_hopf_lib.hopf.
"""

from __future__ import annotations

from flux_hopf_lib.hopf.hopfion import (
    charge_modulated_hopfion,
    hopf_charge_density,
    toroidal_hopfion_director,
)

__all__ = [
    "toroidal_hopfion_director",
    "hopf_charge_density",
    "charge_modulated_hopfion",
]
