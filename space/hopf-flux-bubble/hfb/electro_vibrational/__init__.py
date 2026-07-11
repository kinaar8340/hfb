"""Resonant electro-vibrational control of Hopf flux bubbles.

Dual opposing charge envelopes, charged vibrations, phase-alignment thresholds,
observer synchronization, and rear-hemi flywheel storage/release (slingshot).

This is a practical control layer on top of topological defects and
flow-engineered effective metrics — analog only.
"""

from .charge_envelopes import (
    DualChargeConfig,
    capacitive_field_magnitude,
    charged_vibration_field,
    dual_shell_masks,
    electro_vibrational_defect_modulation,
    opposing_charge_density,
)
from .dynamics import (
    SlingshotConfig,
    SlingshotPhase,
    SlingshotState,
    cycle_phase_at_time,
    flywheel_shift_boost,
    global_alignment_at,
    resonant_control_step,
    simulate_slingshot_cycle,
    update_flywheel,
)
from .observer_sync import (
    ObserverSyncConfig,
    apply_observer_to_alignment,
    entrainment_strength,
    observer_feedback,
)
from .phase_alignment import (
    PhaseAlignmentConfig,
    frequency_detuning,
    local_phase_alignment_field,
    nucleation_mask,
    phase_alignment_order,
    phase_alignment_state,
    void_order_parameter,
)

__all__ = [
    "DualChargeConfig",
    "PhaseAlignmentConfig",
    "ObserverSyncConfig",
    "SlingshotConfig",
    "SlingshotPhase",
    "SlingshotState",
    "dual_shell_masks",
    "opposing_charge_density",
    "capacitive_field_magnitude",
    "charged_vibration_field",
    "electro_vibrational_defect_modulation",
    "frequency_detuning",
    "phase_alignment_order",
    "local_phase_alignment_field",
    "nucleation_mask",
    "void_order_parameter",
    "phase_alignment_state",
    "entrainment_strength",
    "observer_feedback",
    "apply_observer_to_alignment",
    "cycle_phase_at_time",
    "update_flywheel",
    "flywheel_shift_boost",
    "resonant_control_step",
    "simulate_slingshot_cycle",
    "global_alignment_at",
]
