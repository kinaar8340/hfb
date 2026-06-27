"""Cross-repo integration helpers."""

from .vqc_proto import VQCProtoBridge, vqc_proto_available, vqc_slm_available

__all__ = ["VQCProtoBridge", "vqc_proto_available", "vqc_slm_available"]