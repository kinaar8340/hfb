# Hopf Flux Bubble (HFB)

**Analog effective geometries from topological defects, Hopf structures, and flow-engineered metrics.**

Hopf Flux Bubble explores a creative synthesis: treating a "warp bubble" not as classical GR requiring exotic matter, but as an **emergent effective metric** engineered via topologically protected flux — linked Hopfions, vortex conduits, and defect-induced curvature in condensed-matter or metamaterial platforms.

> **Caveat:** These are powerful *analogs* for geodesics, horizons, and defect lensing in effective metrics. They do not produce literal spacetime curvature, negative vacuum energy, or superluminal travel. This is a speculative exploration playground — not a blueprint.

## Concept

Cross-referencing analog gravity and topological defects literature:

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
pytest
hfb-demo                          # writes outputs/flux_bubble_demo.png
hfb-demo --sweep                  # stability parameter sweep
hfb-export-slm                      # flux-bubble vortex ring SLM hologram
jupyter notebook notebooks/       # interactive exploration
```

### Optional extras

| Extra | Install | Enables |
|-------|---------|---------|
| `symbolic` | `pip install -e ".[symbolic]"` | SymPy acoustic metrics (`analog_gravity/symbolic.py`) |
| `notebook` | `pip install -e ".[notebook]"` | Jupyter notebooks under `notebooks/` |
| `slm` | `pip install -e ".[slm]"` | SLM PNG export (Pillow) |
| `all` | `pip install -e ".[all]"` | Everything above + dev tools |

## Configuration

Edit `configs/default.yaml` for grid size, bubble radius, vortex circulation, and sweep ranges.

## Ecosystem

Part of the [kinaar8340](https://github.com/kinaar8340) stack. See [ECOSYSTEM.md](ECOSYSTEM.md) for cross-repo links to `vqc_proto`, `string_optimizer`, `Fisher_Rao`, `pic`, and `toe`.

## License

MIT