"""Hopf map S³ → S² and linking diagnostics.

Thin re-export of the canonical implementation in flux_hopf_lib.hopf.
Prefer::

    from flux_hopf_lib.hopf import hopf_map, hopf_coordinates, linking_number_pair
"""

from __future__ import annotations

from flux_hopf_lib.hopf.fibration import (
    hopf_coordinates,
    hopf_map,
    hopf_map_from_angles,
    linking_number_pair,
    s3_from_quaternion,
)

__all__ = [
    "hopf_coordinates",
    "hopf_map",
    "hopf_map_from_angles",
    "linking_number_pair",
    "s3_from_quaternion",
]
