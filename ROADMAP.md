# HFB Roadmap

## v0.1 ✅ (current)

- Modular package: `analog_gravity`, `defects`, `hopf`, `optics`, `bubble`, `bec`, `integration`
- Conformal Poisson solver + flux bubble metric composite
- Ray tracing, stability sweeps, SLM vortex-ring export
- vqc_proto bridge (`hfb/integration/vqc_proto.py`)
- 4 Jupyter notebooks + 18 pytest cases

## v0.2 — Simulation depth

| Task | Module |
|------|--------|
| 3D Hopfion texture volume + isosurfaces | `hopf/hopfion.py` |
| Full acoustic horizon finder (draining vortex) | `analog_gravity/acoustic.py` |
| Parameter sweep → Fisher-Rao landscape export | `integration/` + `string_optimizer` |
| Negative effective pressure proxy in defect cores | `defects/` |

## v0.3 — Experiment bridge

| Task | Module |
|------|--------|
| Nematic LC director vortex array presets | `optics/nematic.py` |
| BEC imprinted vortex ring GPE time-step | `bec/gpe.py` |
| SLM bench manifest (shared format with vqc_proto) | `optics/slm_export.py` |
| Hugging Face / Gradio flux-bubble explorer | `examples/` |

## v0.4 — Ecosystem

| Task | Repo link |
|------|-----------|
| Shared LG basis with vqc_proto `orbital_braille` | `vqc_proto` |
| Topological invariant checks | `pic` |
| Information geometry stability metrics | `Fisher_Rao` |

## Caveats (always)

Effective analog only — not literal GR, exotic matter, or superluminal transport. See README disclaimer.