# HFB Feature Guide

Headline capabilities and how to run them.

## 0b. Craft / payload mission (v0.3)

**CLI:**
```bash
hfb-mission
# → outputs/craft_mission.png
```

**API:**
```python
from hfb.craft import CraftConfig, MissionConfig, simulate_mission

result = simulate_mission(cfg=MissionConfig(craft=CraftConfig(mass=1.0)))
print(result["final_craft"]["speed"], result["craft"]["x"][-1])
```

Engine (transducer) stays decoupled; craft only consumes storage breakdown + directed impulse.

---

## 0. Resonant electro-vibrational slingshot (v0.2)

**Built-in** (core `numpy`/`scipy`).

**CLI:**
```bash
hfb-slingshot
# or: hfb-demo --slingshot
# → outputs/hemi_void_slingshot.png
```

**API:**
```python
from hfb.electro_vibrational import simulate_slingshot_cycle, SlingshotConfig
from hfb.bubble import hemi_void_bubble_metric, HemiVoidConfig
from hfb.utils.grid import cartesian_grid

x, y = cartesian_grid(64, 64, extent=3.0)
cycle = simulate_slingshot_cycle(x, y, t_max=5.0)
metric = hemi_void_bubble_metric(x, y, dx=0.1, include_control=True, t=2.0)
```

**Modules:**
- `hfb/electro_vibrational/charge_envelopes.py` — dual ±σ shells, capacitive |E|, charged vibration
- `hfb/electro_vibrational/phase_alignment.py` — order parameter ψ, nucleation threshold
- `hfb/electro_vibrational/observer_sync.py` — entrainment feedback with macro hum
- `hfb/electro_vibrational/transducer.py` — craft flux transducer: ledger, channel reversion, throttle
- `hfb/electro_vibrational/dynamics.py` — store / release cycle wired through the transducer
- `hfb/bubble/hemi_void.py` — gourd/melon void geometry + composite metric

**Flux transducer (v0.2.2):** craft-local **motor–generator + gearbox** — not a global curvature-flux term.

| Mode | Behavior |
|------|----------|
| **Store** | Active **pre-charge** (dual shells) + **pre-twist** (flywheel) toward targets, plus passive collection |
| **Ready** | Hold / top-up at `target_energy`; channels stay inbound |
| **Release** | **Revert channels** + **intensity** throttle → directed impulse |

| Channel | Sense / pump proxy |
|---------|-------------------|
| Electrostatic | ∫ ½ E² rear + active pre-charge rate |
| Twist | ∫ \|v\|·\|σ\|·ψ + active pretwist rate |
| Geometric | ∫ η² rear + small co-pump from packing |

**Config keys:** `electro_vibrational:`, `electro_vibrational.transducer:` (`enable_precharge`, `enable_pretwist`, `pump_intensity`, `target_energy`), `hemi_void:`.

---

## 1. SymPy symbolic acoustic metrics

**Install:** `pip install -e ".[symbolic]"`

**CLI:**
```bash
hfb-symbolic
```

**Notebook:** `notebooks/02_acoustic_metrics_symbolic.ipynb`

**API:**
```python
from hfb.analog_gravity.symbolic import (
    acoustic_line_element,
    horizon_condition,
    draining_vortex_velocity,
    symbolic_summary,
)
print(symbolic_summary()["horizon"])
```

Covers Barceló–Visser–Liberati acoustic line elements, horizon condition v² = cₛ², draining vortex flows, and conformal Poisson ΔΩ = −λ.

---

## 2. Jupyter interactive exploration

**Install:** `pip install -e ".[notebook]"`

```bash
jupyter notebook notebooks/
```

| Notebook | Widgets / interactivity |
|----------|-------------------------|
| `01_flux_bubble_explorer.ipynb` | `ipywidgets` — radius, wall, circulation, defect_amp |
| `02_acoustic_metrics_symbolic.ipynb` | SymPy `display()` |
| `03_bec_acoustic_backend.ipynb` | BEC density + ergoregion plots |
| `04_slm_vortex_export.ipynb` | SLM phase preview |
| `05_feature_tour.ipynb` | All four features in one tour |

---

## 3. vqc_proto LG / SLM coupling

**Requires:** clone [vqc_proto](https://github.com/kinaar8340/vqc_proto) (auto-detected under `~/Projects/vqc_proto/`)

**Check:**
```bash
hfb-check
export VQC_PROTO_PATH=~/Projects/vqc_proto/space/orbital-braille  # optional override
```

**Export flux-bubble vortex ring hologram:**
```bash
pip install -e ".[slm]"
hfb-export-slm --preset holoeye_pluto_2 --frames 16
```

Uses `orbital_braille.lg_modes` and `slm_typehead` when available; falls back to `hfb.optics.lg_modes` locally.

`manifest.json` includes `"vqc_proto_coupled": true|false`.

---

## 4. BEC acoustic backends

**Built-in** (no extra install beyond core `numpy`/`scipy`).

**CLI:**
```bash
hfb-bec-demo
# → outputs/bec_acoustic_demo.png
```

**Modules:**
- `hfb.bec.gpe` — Thomas-Fermi density
- `hfb.bec.vortex` — imprinted vortex ring velocity
- `hfb.bec.acoustic` — BEC sound speed + acoustic metric + ergoregion mask
- `hfb.bec.bogoliubov` — dispersion ε(k)

**Notebook:** `notebooks/03_bec_acoustic_backend.ipynb`