"""Hopf fibration, linking, and Hopfion texture templates.

Implementations re-export from flux_hopf_lib.hopf (single source of truth).
"""

from .fibration import (
    hopf_coordinates,
    hopf_map,
    hopf_map_from_angles,
    linking_number_pair,
    s3_from_quaternion,
)
from .hopfion import (
    charge_modulated_hopfion,
    hopf_charge_density,
    toroidal_hopfion_director,
)

__all__ = [
    "hopf_coordinates",
    "hopf_map",
    "hopf_map_from_angles",
    "linking_number_pair",
    "s3_from_quaternion",
    "toroidal_hopfion_director",
    "hopf_charge_density",
    "charge_modulated_hopfion",
]