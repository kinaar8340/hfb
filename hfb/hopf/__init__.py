"""Hopf fibration, linking, and Hopfion texture templates."""

from .fibration import hopf_coordinates, hopf_map, linking_number_pair
from .hopfion import (
    charge_modulated_hopfion,
    hopf_charge_density,
    toroidal_hopfion_director,
)

__all__ = [
    "hopf_coordinates",
    "hopf_map",
    "linking_number_pair",
    "toroidal_hopfion_director",
    "hopf_charge_density",
    "charge_modulated_hopfion",
]