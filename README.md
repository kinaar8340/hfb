# Hopf Flux Bubble (HFB)

[![GitHub](https://img.shields.io/badge/GitHub-kinaar8340%2Fhfb-blue)](https://github.com/kinaar8340/hfb)
[![Ecosystem](https://img.shields.io/badge/stack-vqc__proto-lightgrey)](https://github.com/kinaar8340/vqc_proto)

**Analog effective geometries from topological defects, Hopf structures, and flow-engineered metrics.**

Local path: `~/Projects/hfb/` · Remote: **https://github.com/kinaar8340/hfb** (repo exists — clone or pull to sync)

Hopf Flux Bubble explores a creative synthesis: treating a "warp bubble" not as classical GR requiring exotic matter, but as an **emergent effective metric** engineered via topologically protected flux — linked Hopfions, vortex conduits, and defect-induced curvature in condensed-matter or metamaterial platforms.

> **Caveat:** These are powerful *analogs* for geodesics, horizons, and defect lensing in effective metrics. They do not produce literal spacetime curvature, negative vacuum energy, or superluminal travel. This is a speculative exploration playground — not a blueprint.

## Concept

Cross-referencing analog gravity and topological defects literature.

**Acoustic metric (analog gravity):** ds² = −(cₛ² − v²)dt² − 2**v**·d**x** dt + d**x**²

**Defect curvature (conformal):** ΔΩ = −λ(r,θ)

| Pillar | HFB module | Idea |
|--------|------------|------|
| Analog gravity | `analog_gravity/` | Acoustic metric from flows; draining vortices → ergoregions |
| Topological defects | `defects/` | ΔΩ = −λ conformal Poisson; quantized winding |
| Hopf / Hopfions | `hopf/` | Linked toroidal flux walls; topological protection |
| Optics / nematics | `optics/` | LG vortices, cosmic-string lensing, ray tracing |
| Flux bubble | `bubble/` | Shift profile + defect wall + vortex flow composite |
| BEC analog | `bec/` | Thomas-Fermi density, vortex rings, Bogoliubov dispersion |
| SLM export | `optics/slm_export.py` | Vortex ring holograms (vqc_proto-coupled) |

Broader framing: [concept thread on X](https://x.com/kinaar8340/status/2070662842496491671)

## Quick Start

```bash
cd ~/Projects/hfb
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
hfb-check                         # feature + vqc_proto availability
pytest
hfb-demo                          # → outputs/flux_bubble_demo.png
hfb-symbolic                      # SymPy acoustic metrics (needs [symbolic])
hfb-bec-demo                      # BEC acoustic backend figure
hfb-export-slm                    # vortex ring SLM (vqc_proto LG when found)
jupyter notebook notebooks/       # ipywidgets + feature tour
```

See **[FEATURES.md](FEATURES.md)** for the four headline capabilities.

### Optional extras

| Extra | Install | Enables |
|-------|---------|---------|
| `symbolic` | `pip install -e ".[symbolic]"` | SymPy acoustic metrics (`analog_gravity/symbolic.py`) |
| `notebook` | `pip install -e ".[notebook]"` | Jupyter notebooks under `notebooks/` |
| `slm` | `pip install -e ".[slm]"` | SLM PNG export (Pillow) |
| `all` | `pip install -e ".[all]"` | Everything above + dev tools |

## Configuration

Edit `configs/default.yaml` for grid size, bubble radius, vortex circulation, and sweep ranges.

## Docs

- [ECOSYSTEM.md](ECOSYSTEM.md) — kinaar8340 stack map
- [GLOSSARY.md](GLOSSARY.md) — terms (Hopfion, acoustic metric, …)
- [ROADMAP.md](ROADMAP.md) — v0.2–v0.4 plan
- [docs/REFERENCES.md](docs/REFERENCES.md) — literature pointers

## Ecosystem

Part of the [kinaar8340](https://github.com/kinaar8340) stack. See [ECOSYSTEM.md](ECOSYSTEM.md) for cross-repo links to `vqc_proto`, `string_optimizer`, `Fisher_Rao`, `pic`, and `toe`.

## License

MIT