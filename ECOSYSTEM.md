# HFB Ecosystem Stack

Hopf Flux Bubble (`hfb`) is a modular node in the [kinaar8340](https://github.com/kinaar8340) research stack. Cross-repo integration points:

| Repo | Relationship |
|------|----------------|
| [flux_hopf_lib](https://github.com/kinaar8340/flux_hopf_lib) | **Shared core** — Hopf maps, hopfions, defect densities, grids (re-exported by `hfb.hopf`, `hfb.defects`, `hfb.utils.grid`) |
| [vqc_proto](https://github.com/kinaar8340/vqc_proto) | LG/OAM optics + SLM export (`orbital_braille/slm_typehead`) via `hfb/integration/vqc_proto.py` |
| [string_optimizer](https://github.com/kinaar8340/string_optimizer) | Riemannian geometry / Fisher-Rao metrics for parameter sweeps |
| [Fisher_Rao](https://github.com/kinaar8340/Fisher_Rao) | Information geometry for stability landscape analysis |
| [pic](https://github.com/kinaar8340/pic) | Topological invariants, winding, Platonic conduit metaphors |
| [toe](https://github.com/kinaar8340/toe) | Broader TOE / emergent geometry context |
| [mystery](https://github.com/kinaar8340/mystery) | φ-e-π survival probes (uses flux_hopf_lib.simulation) |

## Module Map

```
hfb/
├── analog_gravity/       # Acoustic metrics, draining vortices, ergoregions
│   └── symbolic.py       # SymPy line elements, horizons, conformal Ricci
├── bec/                  # GPE density, vortex imprinting, Bogoliubov proxies
├── defects/              # Conformal Poisson ΔΩ = -λ, defect / hemi-void densities
├── electro_vibrational/  # Dual charges, phase lock, observer sync, transducer, slingshot
├── craft/                # Payload/hull dynamics from transducer impulse + ledger
├── hopf/                 # Hopf fibration, Hopfion textures (→ flux_hopf_lib.hopf)
├── integration/          # vqc_proto bridge (VQC_PROTO_PATH)
├── optics/               # LG modes, nematic lensing, ray tracing, SLM export
├── bubble/               # Flux bubble + hemi-void synthesis + stability
└── utils/                # Grids, FFT Laplacian (→ flux_hopf_lib.utils)
```

Shared math is installed via `flux-hopf-lib`; local modules re-export so
`from hfb.hopf import hopf_map` keeps working.

## Notebooks

| Notebook | Topic |
|----------|-------|
| `01_flux_bubble_explorer.ipynb` | Interactive bubble radius, circulation, ray geodesics |
| `02_acoustic_metrics_symbolic.ipynb` | SymPy acoustic metric and horizon algebra |
| `03_bec_acoustic_backend.ipynb` | BEC density, vortex ring, Bogoliubov dispersion |
| `04_slm_vortex_export.ipynb` | Vortex ring SLM hologram preview + export |

## Install Extras

```bash
pip install -e ".[symbolic]"   # SymPy metrics
pip install -e ".[notebook]"   # Jupyter + ipywidgets
pip install -e ".[slm]"         # Pillow for SLM PNG export
pip install -e ".[all]"         # everything
```

## vqc_proto Coupling

Set `VQC_PROTO_PATH` to the `orbital-braille` root (default: `~/Projects/vqc_proto/space/orbital-braille`). When present, `hfb-export-slm` uses vqc_proto device presets and phase encoding.

## Concept Thread

- **Analog gravity**: effective curved propagation from flows (`analog_gravity/acoustic.py`)
- **Topological defects**: quantized winding, cosmic-string lensing (`defects/`, `optics/nematic.py`)
- **Hopf structures**: linked toroidal flux walls (`hopf/`)
- **Warp conduit analog**: shift + contraction/expansion via Ω and n_eff (`bubble/warp_conduit.py`)
- **Resonant control**: dual charge envelopes, phase alignment, hemi-void slingshot (`electro_vibrational/`, `bubble/hemi_void.py`)
- **Observer sync**: entrainment feedback with medium hum (`electro_vibrational/observer_sync.py`)
- **BEC tabletop analog**: Thomas-Fermi + imprinted vortices (`bec/`)

## External References

- Concept thread: [X post](https://x.com/kinaar8340/status/2070662842496491671)
- Analog gravity horizons: [Barceló et al., Living Rev. Relativity](https://arxiv.org)
- Defect-induced 2D curvature: conformal Poisson framework
- Nematic cosmic strings: [Pismen, Phys. Rev. E](https://colorado.edu)
- Hopfions in nematics: [Nature Physics](https://nature.com), [PRL](https://link.aps.org)