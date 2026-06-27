"""Acoustic and flow-based analog gravity metrics."""

from .acoustic import (
    acoustic_metric_components,
    draining_vortex_flow,
    ergosurface_mask,
    sound_speed_field,
)

__all__ = [
    "acoustic_metric_components",
    "draining_vortex_flow",
    "ergosurface_mask",
    "sound_speed_field",
]

try:
    from . import symbolic as symbolic
    from .symbolic import (
        acoustic_line_element,
        acoustic_metric_inverse,
        acoustic_metric_tensor,
        alcubierre_line_element,
        compare_effective_warp,
        conformal_ricci_scalar,
        horizon_condition,
        lambdify_alcubierre_shift,
    )

    __all__ += [
        "symbolic",
        "acoustic_line_element",
        "acoustic_metric_tensor",
        "acoustic_metric_inverse",
        "horizon_condition",
        "conformal_ricci_scalar",
        "alcubierre_line_element",
        "compare_effective_warp",
        "lambdify_alcubierre_shift",
    ]
except ImportError:
    pass