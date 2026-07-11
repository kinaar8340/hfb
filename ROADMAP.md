# HFB Roadmap

## v0.1 ✅

- Modular package: `analog_gravity`, `defects`, `hopf`, `optics`, `bubble`, `bec`, `integration`
- Conformal Poisson solver + flux bubble metric composite
- Ray tracing, stability sweeps, SLM vortex-ring export
- vqc_proto bridge (`hfb/integration/vqc_proto.py`)
- Jupyter notebooks + pytest suite

## v0.2 ✅ (current) — Resonant control layer

| Task | Module |
|------|--------|
| Dual opposing charge envelopes + charged vibrations | `electro_vibrational/charge_envelopes.py` |
| Phase-alignment threshold & void nucleation | `electro_vibrational/phase_alignment.py` |
| Observer synchronization feedback | `electro_vibrational/observer_sync.py` |
| Rear-hemi flywheel storage / release (slingshot) | `electro_vibrational/dynamics.py` |
| Orthogonal hemi / gourd void geometry | `bubble/hemi_void.py` |
| Charge-modulated Hopfion texture | `hopf/hopfion.py` |
| CLI + demo | `hfb-slingshot`, `examples/hemi_void_slingshot_demo.py` |

## v0.3 — Simulation depth

| Task | Module |
|------|--------|
| 3D Hopfion texture volume + isosurfaces | `hopf/hopfion.py` |
| Full acoustic horizon finder (draining vortex) | `analog_gravity/acoustic.py` |
| Closed-loop slingshot PID on ψ and stored energy | `electro_vibrational/dynamics.py` |
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
