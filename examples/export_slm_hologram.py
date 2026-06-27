#!/usr/bin/env python3
"""Export flux-bubble vortex ring SLM hologram."""

from pathlib import Path

import yaml

from hfb.optics.slm_export import SLMExportConfig, VortexConduitSpec, export_flux_bubble_hologram

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    with (root / "configs" / "default.yaml").open() as f:
        cfg = yaml.safe_load(f)
    slm = cfg.get("slm", {})
    spec = VortexConduitSpec(
        ring_radius_mm=slm.get("ring_radius_mm", 1.2),
        num_vortices=slm.get("num_vortices", 12),
        winding=slm.get("winding", 1),
    )
    export_cfg = SLMExportConfig.from_vqc_preset(
        slm.get("device_preset", "generic_512"),
        extent_mm=slm.get("extent_mm", 4.0),
    )
    summary = export_flux_bubble_hologram(root / "outputs" / "slm_hologram", spec=spec, cfg=export_cfg)
    print(summary)