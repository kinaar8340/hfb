# HFB Ecosystem Stack

Hopf Flux Bubble (`hfb`) is a modular node in the [kinaar8340](https://github.com/kinaar8340) research stack. Cross-repo integration points:

| Repo | Relationship |
|------|----------------|
| [vqc_proto](https://github.com/kinaar8340/vqc_proto) | LG/OAM optics (`orbital_braille/lg_modes`), quaternion topology |
| [string_optimizer](https://github.com/kinaar8340/string_optimizer) | Riemannian geometry / Fisher-Rao metrics for parameter sweeps |
| [Fisher_Rao](https://github.com/kinaar8340/Fisher_Rao) | Information geometry for stability landscape analysis |
| [pic](https://github.com/kinaar8340/pic) | Topological invariants, winding, Platonic conduit metaphors |
| [toe](https://github.com/kinaar8340/toe) | Broader TOE / emergent geometry context |

## Module Map

```
hfb/
├── analog_gravity/   # Acoustic metrics, draining vortices, ergoregions
├── defects/          # Conformal Poisson ΔΩ = -λ, defect densities
├── hopf/             # Hopf fibration, Hopfion textures, linking
├── optics/           # LG modes, nematic lensing, ray tracing
├── bubble/           # Flux bubble synthesis + stability sweeps
└── utils/            # Grids, FFT Laplacian
```

## Concept Thread

- **Analog gravity**: effective curved propagation from flows (`analog_gravity/acoustic.py`)
- **Topological defects**: quantized winding, cosmic-string lensing (`defects/`, `optics/nematic.py`)
- **Hopf structures**: linked toroidal flux walls (`hopf/`)
- **Warp conduit analog**: shift + contraction/expansion via Ω and n_eff (`bubble/warp_conduit.py`)

## External References

- Concept thread: [X post](https://x.com/kinaar8340/status/2070662842496491671)
- Analog gravity horizons: [Barceló et al., Living Rev. Relativity](https://arxiv.org)
- Defect-induced 2D curvature: conformal Poisson framework
- Nematic cosmic strings: [Pismen, Phys. Rev. E](https://colorado.edu)
- Hopfions in nematics: [Nature Physics](https://nature.com), [PRL](https://link.aps.org)

## Planned Extensions

- SymPy symbolic acoustic metrics (`pip install hopf-flux-bubble[symbolic]`)
- Jupyter notebooks under `notebooks/`
- Coupling to `vqc_proto` LG mode codecs for experimental SLM patterns
- BEC acoustic analog backends