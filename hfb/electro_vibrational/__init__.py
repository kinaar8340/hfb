"""Resonant electro-vibrational control of Hopf flux bubbles.

Dual opposing charge envelopes, charged vibrations, phase-alignment thresholds,
observer synchronization, rear-hemi flywheel storage/release (slingshot), and
a craft-local **flux transducer** that:

- actively **pre-charges** and **pre-twists** the rear hemi (motor path)
- quantifies a three-channel energy ledger
- **reverts flux channels** and meters release intensity (gearbox path)

Analog only.
"""

from .charge_envelopes import (
    DualChargeConfig,
    apply_active_precharge,
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
from .transducer import (
    ChannelState,
    EnergyLedger,
    FluxTransducer,
    PumpCommand,
    TransducerConfig,
    TransducerReading,
    TransducerReport,
    accumulate_ledger,
    apply_active_pump,
    apply_precharge_to_fields,
    channel_direction_for_mode,
    channel_targets,
    compute_pump_command,
    dump_ledger,
    flux_channel_polarity_field,
    is_ready,
    ledger_from_total,
    precharge_boost_field,
    pretwist_velocity_field,
    sense_energy_channels,
    transducer_shift_contribution,
)

# FluxTransducer.get_storage_breakdown / pumped_efficiency are methods on the class

__all__ = [
    "DualChargeConfig",
    "PhaseAlignmentConfig",
    "ObserverSyncConfig",
    "SlingshotConfig",
    "SlingshotPhase",
    "SlingshotState",
    "TransducerConfig",
    "EnergyLedger",
    "ChannelState",
    "PumpCommand",
    "FluxTransducer",
    "TransducerReading",
    "TransducerReport",
    "dual_shell_masks",
    "opposing_charge_density",
    "capacitive_field_magnitude",
    "charged_vibration_field",
    "electro_vibrational_defect_modulation",
    "apply_active_precharge",
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
    "sense_energy_channels",
    "channel_direction_for_mode",
    "accumulate_ledger",
    "dump_ledger",
    "flux_channel_polarity_field",
    "transducer_shift_contribution",
    "ledger_from_total",
    "compute_pump_command",
    "apply_active_pump",
    "apply_precharge_to_fields",
    "precharge_boost_field",
    "pretwist_velocity_field",
    "channel_targets",
    "is_ready",
]
