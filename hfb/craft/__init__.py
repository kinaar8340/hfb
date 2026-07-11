"""Craft / payload dynamics on top of the flux-transducer engine.

Transducer = local motor–generator–gearbox (ledger + directed impulse).
Craft dynamics = rigid-body / payload integration of that kick.
"""

from .dynamics import (
    CraftConfig,
    CraftState,
    craft_acceleration,
    craft_step,
    initial_craft_state,
    impulse_to_delta_v,
    integrate_craft_from_series,
)
from .mission import MissionConfig, simulate_mission, simulate_mission_coupled

__all__ = [
    "CraftConfig",
    "CraftState",
    "MissionConfig",
    "craft_acceleration",
    "craft_step",
    "initial_craft_state",
    "impulse_to_delta_v",
    "integrate_craft_from_series",
    "simulate_mission",
    "simulate_mission_coupled",
]
