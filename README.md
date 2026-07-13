# Hopf Flux Bubble (HFB)

[![GitHub](https://img.shields.io/badge/GitHub-kinaar8340%2Fhfb-blue)](https://github.com/kinaar8340/hfb)
[![Ecosystem](https://img.shields.io/badge/stack-vqc__proto-lightgrey)](https://github.com/kinaar8340/vqc_proto)

**Analog effective geometries from topological defects, Hopf structures, flow-engineered metrics, and resonant electro-vibrational control.**

Local path: `~/Projects/hfb/` · Remote: **https://github.com/kinaar8340/hfb**

Hopf Flux Bubble explores a creative synthesis: treating a "warp bubble" not as classical GR requiring exotic matter, but as an **emergent effective metric** engineered via topologically protected flux — linked Hopfions, vortex conduits, and defect-induced curvature in condensed-matter or metamaterial platforms — with a **resonant control layer** that nucleates, shapes, and directionally releases structured void bubbles via charged vibrations and phase alignment.

> **Caveat:** These are powerful *analogs* for geodesics, horizons, and defect lensing in effective metrics. They do not produce literal spacetime curvature, negative vacuum energy, or superluminal travel. This is a speculative exploration playground — not a blueprint.

## Concept

Cross-referencing analog gravity and topological defects literature.

**Acoustic metric (analog gravity):** ds² = −(cₛ² − v²)dt² − 2**v**·d**x** dt + d**x**²

**Defect curvature (conformal):** ΔΩ = −λ(r,θ)

| Pillar | HFB module | Idea |
|--------|------------|------|
| Analog gravity | `analog_gravity/` | Acoustic metric from flows; draining vortices → ergoregions |
| Topological defects | `defects/` | ΔΩ = −λ conformal Poisson; quantized winding; hemi-void walls |
| Hopf / Hopfions | `hopf/` | Linked toroidal flux walls; charge-modulated textures |
| Optics / nematics | `optics/` | LG vortices, cosmic-string lensing, ray tracing |
| Flux bubble | `bubble/` | Shift profile + defect wall + hemi-void + vortex flow |
| Electro-vibrational | `electro_vibrational/` | Dual charge envelopes, phase lock, flywheel slingshot |
| BEC analog | `bec/` | Thomas-Fermi density, vortex rings, Bogoliubov dispersion |
| SLM export | `optics/slm_export.py` | Vortex ring holograms (vqc_proto-coupled) |

Broader framing: [concept thread on X](https://x.com/kinaar8340/status/2070662842496491671)

## What's new in v0.3.1 — Mission energy flow + craft→throttle feedback

- `energy_flow_summary()` — total pumped, net impulse, final KE, KE/pumped efficiency
- Demo: `craft_mission.png` + `craft_mission_chain.png` (ledger → impulse → trajectory)
- Coupled feedback: craft speed/position gently biases pump/release throttles
- CLI: `hfb-mission --coupled --feedback`

```python
from hfb.craft import MissionConfig, simulate_mission_coupled, energy_flow_summary

result = simulate_mission_coupled(x, y, cfg=MissionConfig(enable_craft_feedback=True))
print(result["energy_flow_text"])
```

### v0.3.0 — Craft / payload dynamics

Higher layer above the transducer engine:

- `hfb/craft/` — rigid-body / payload integration of directed impulse + storage breakdown
- `simulate_mission()` — full nucleate→store→ready→release→coast + craft trajectory
- CLI: `hfb-mission` → `outputs/craft_mission.png`

```python
from hfb.craft import CraftConfig, MissionConfig, simulate_mission

result = simulate_mission(cfg=MissionConfig(
    craft=CraftConfig(mass=1.0, impulse_coupling=1.0),
))
print(result["final_craft"])  # x, y, speed, KE, ∫impulse
```

Transducer stays the local engine; craft dynamics only consumes `get_storage_breakdown()` + directed impulse.

### v0.2.3 — Transducer diagnostics polish

- `get_storage_breakdown()` — channel %, `total_stored`, motor vs passive
- `pumped_efficiency` = Σ pumped / (Σ pumped + Σ passive)
- READY **hysteresis** band (`ready_hysteresis`) to stop chatter near target
- Second demo figure: `outputs/hemi_void_slingshot_cycle.png` (channel · pump · impulse)

### v0.2.2 — Active pre-charge / pre-twist

The flux transducer is now a full **bidirectional controller** (motor–generator + gearbox):

- **Pre-charge** dual opposing envelopes during store (active electrostatic load)
- **Pre-twist** rear-hemi flux flywheel via parametric azimuthal drive
- **Ready** hold at commanded `target_energy` before release
- Release still **reverts channels** and meters **intensity**
- Observer can bias both pump and release throttles

```python
from hfb.electro_vibrational import FluxTransducer, TransducerConfig

tx = FluxTransducer(cfg=TransducerConfig(
    enable_precharge=True,
    enable_pretwist=True,
    pump_intensity=1.0,
    target_energy=0.95,
))
tx.set_pump_intensity(0.8)
tx.set_targets(target_energy=0.9)
```

### v0.2.1 — Flux transducer ledger

Craft-local **three-channel ledger** + channel reversion + release throttle
(`hfb/electro_vibrational/transducer.py`).

### v0.2.0 — Resonant control layer

Practical engineering handle on top of the topological foundations:

1. **Dual opposing charge envelopes** — concentric shells with ±σ; capacitive gap couples **E** to Hopf flux and vibrational modes
2. **Phase-alignment threshold** — order parameter ψ; supercritical nucleation of void bubbles
3. **Orthogonal hemi / gourd void** — rear-hemi flywheel reservoir (`bubble/hemi_void.py`)
4. **Storage → release slingshot** — phase lock → store → detuning + transducer dump
5. **Observer synchronization** — entrainment with the macro “hum”

```bash
hfb-slingshot                 # → outputs/hemi_void_slingshot.png
hfb-demo --slingshot          # same
```

### Earlier releases

- **v0.1.4** — 3D torus slice, warp fidelity, topology diagnostics
- **v0.1.3** — toroidal bubble wall, Alcubierre symbolic, 3D viz

## Quick Start

```bash
cd ~/Projects/hfb
python3 -m venv .venv && source .venv/bin/activate
# Shared core (editable if developing flux_hopf_lib alongside):
#   pip install -e ../flux_hopf_lib
pip install -e ".[all]"
hfb-check                         # feature + vqc_proto availability
pytest
hfb-demo                          # → outputs/flux_bubble_demo.png
hfb-demo --viz3d                  # + flux_bubble_3d.png
hfb-demo --compare-warp           # toroidal vs exponential + warp fidelity
hfb-slingshot                     # hemi-void nucleation → store → release
hfb-symbolic                      # SymPy acoustic metrics (needs [symbolic])
hfb-bec-demo                      # BEC acoustic backend figure
hfb-export-slm                    # vortex ring SLM (vqc_proto LG when found)
jupyter notebook notebooks/       # ipywidgets + feature tour
```

Hopf maps, hopfions, shared defect densities, and grids re-export from
**[flux_hopf_lib](https://github.com/kinaar8340/flux_hopf_lib)** (`from hfb.hopf import hopf_map` unchanged).

See **[FEATURES.md](FEATURES.md)** for headline capabilities.

### Optional extras

| Extra | Install | Enables |
|-------|---------|---------|
| `symbolic` | `pip install -e ".[symbolic]"` | SymPy acoustic metrics (`analog_gravity/symbolic.py`) |
| `notebook` | `pip install -e ".[notebook]"` | Jupyter notebooks under `notebooks/` |
| `slm` | `pip install -e ".[slm]"` | SLM PNG export (Pillow) |
| `all` | `pip install -e ".[all]"` | Everything above + dev tools |

## Resonant control API (sketch)

```python
from hfb.bubble import hemi_void_bubble_metric, HemiVoidConfig
from hfb.electro_vibrational import SlingshotConfig, simulate_slingshot_cycle
from hfb.utils.grid import cartesian_grid

x, y = cartesian_grid(96, 96, extent=4.0)
dx = float(x[0, 1] - x[0, 0])

# Full cycle time series (ψ, flywheel energy, release impulse)
cycle = simulate_slingshot_cycle(x, y, t_max=5.0, dt=0.05)

# Snapshot metric with active control
metric = hemi_void_bubble_metric(
    x, y, dx=dx, hemi=HemiVoidConfig(), t=2.5, include_control=True
)
```

Or via the existing composite builder:

```python
from hfb.bubble import flux_bubble_metric

metric = flux_bubble_metric(
    x, y, dx=dx,
    defect_profile="hemi_void_wall",
    electro_vibrational=True,
    t=2.5,
)
```

## Configuration

Edit `configs/default.yaml` for grid size, bubble radius, vortex circulation,
`hemi_void`, and `electro_vibrational` control parameters.

## Docs

- [ECOSYSTEM.md](ECOSYSTEM.md) — kinaar8340 stack map
- [GLOSSARY.md](GLOSSARY.md) — terms (Hopfion, phase alignment, slingshot, …)
- [ROADMAP.md](ROADMAP.md) — v0.2–v0.4 plan
- [docs/REFERENCES.md](docs/REFERENCES.md) — literature pointers
- [FEATURES.md](FEATURES.md) — how to run each capability

## Ecosystem

Part of the [kinaar8340](https://github.com/kinaar8340) stack. See [ECOSYSTEM.md](ECOSYSTEM.md) for cross-repo links to `vqc_proto`, `string_optimizer`, `Fisher_Rao`, `pic`, and `toe`.

## License

MIT
