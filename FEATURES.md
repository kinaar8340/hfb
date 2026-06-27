# HFB Feature Guide

Four headline capabilities and how to run them.

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