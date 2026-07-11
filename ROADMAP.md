# HFB Roadmap

## v0.1 ✅

- Modular package: `analog_gravity`, `defects`, `hopf`, `optics`, `bubble`, `bec`, `integration`
- Conformal Poisson solver + flux bubble metric composite
- Ray tracing, stability sweeps, SLM vortex-ring export
- vqc_proto bridge (`hfb/integration/vqc_proto.py`)
- Jupyter notebooks + pytest suite

## v0.2 ✅ — Resonant control layer

| Task | Module |
|------|--------|
| Dual opposing charge envelopes + charged vibrations | `electro_vibrational/charge_envelopes.py` |
| Phase-alignment threshold & void nucleation | `electro_vibrational/phase_alignment.py` |
| Observer synchronization feedback | `electro_vibrational/observer_sync.py` |
| Rear-hemi flywheel storage / release (slingshot) | `electro_vibrational/dynamics.py` |
| Orthogonal hemi / gourd void geometry | `bubble/hemi_void.py` |
| Charge-modulated Hopfion texture | `hopf/hopfion.py` |
| CLI + demo | `hfb-slingshot`, `examples/hemi_void_slingshot_demo.py` |

## v0.2.1 ✅ — Flux transducer ledger

| Task | Module |
|------|--------|
| Craft-local energy ledger (ES + twist + geometric) | `electro_vibrational/transducer.py` |
| Flux channel reversion + intensity throttle | `transducer.py` / `dynamics.py` |
| Defect modulation by channel polarity field | `modulate_defect_by_channels` |
| Demo plots ledger channels + channel direction | `hfb-slingshot` |

## v0.2.2 ✅ (current) — Active pre-charge / pre-twist

| Task | Module |
|------|--------|
| Active pre-charge of dual envelopes (motor path) | `transducer.compute_pump_command` |
| Active pre-twist of rear-hemi flywheel | `pretwist_velocity_field` |
| READY hold at target energy + top-up | `SlingshotPhase.READY` |
| Σ pumped fidelity accounting | `FluxTransducer.total_pumped` |
| Independent pump vs release throttles | `pump_intensity` / `release_intensity` |

## v0.3 — Simulation depth

| Task | Module |
|------|--------|
| 3D Hopfion texture volume + isosurfaces | `hopf/hopfion.py` |
| Full acoustic horizon finder (draining vortex) | `analog_gravity/acoustic.py` |
| Closed-loop PID on ψ, ledger channels, pump/release intensity | `electro_vibrational/` |
| Multi-transducer array / distributed channels | `transducer.py` |
| Parameter sweep → Fisher-Rao landscape export | `integration/` + `string_optimizer` |
| Negative effective pressure proxy in defect cores | `defects/` |

## v0.4 — Experiment bridge

| Task | Module |
|------|--------|
| Nematic LC director vortex array presets | `optics/nematic.py` |
| BEC imprinted vortex ring GPE time-step | `bec/gpe.py` |
| SLM bench manifest (shared format with vqc_proto) | `optics/slm_export.py` |
| Electro-vibrational drive → SLM / piezo channel map | `optics/` + `electro_vibrational/` |
| Hugging Face / Gradio flux-bubble + slingshot explorer | `web/`, `space/` |

## v0.5 — Ecosystem

| Task | Repo link |
|------|-----------|
| Shared LG basis with vqc_proto `orbital_braille` | `vqc_proto` |
| Topological invariant checks | `pic` |
| Information geometry stability metrics | `Fisher_Rao` |
| Observer-sync paper cross-links | docs / external PDF |

## Caveats (always)

Effective analog only — not literal GR, exotic matter, or superluminal transport. See README disclaimer.
