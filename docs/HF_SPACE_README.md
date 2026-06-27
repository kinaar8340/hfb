# Hopf Flux Bubble — Live Demo

<p align="center">
  <img src="https://raw.githubusercontent.com/kinaar8340/vqc_proto/main/hfb.png" alt="Hopf Flux Bubble — flux metric background" width="100%" style="max-width: 720px; border-radius: 12px;" />
</p>

**Browser demo** of the [Hopf Flux Bubble](https://github.com/kinaar8340/hfb) toolkit: analog effective geometries from topological defects, Hopf structures, and flow-engineered metrics.

No install required — use the **App** tab above.

> **Analog only** — these are effective-metric explorations for tabletop platforms. They do not produce literal spacetime curvature, negative vacuum energy, or superluminal transport.

---

## What you can run

| Action | Output |
|--------|--------|
| **Run flux bubble** | 4-panel figure: conformal factor Ω, shift β, acoustic c_eff² − v², conformal geodesics |
| **Run warp compare** | Analog shift vs Alcubierre symbolic profile + L¹ fidelity metrics |
| **Run stability sweep** | Radius × circulation grid with stability proxies (max \|R\|, ergo fraction) |
| **Include 3D surface** | Optional pseudo-3D torus slice (slower on HF) |

Tune bubble radius, wall width, circulation, Hopf torus parameters, and defect profile from `configs/default.yaml`.

---

## Matrix control panel

The **Flux Metric Control Panel** mirrors the Orbital Braille optics UI:

- **Selection menu** — D-pad navigable TUI in the matrix status display
- **Prog keys 02–05** — jump to Status, Flux Bubble, Warp & Stability, Help
- **01 Home** — return to the selection menu

---

## Links

- **Source:** [github.com/kinaar8340/hfb](https://github.com/kinaar8340/hfb)
- **Ecosystem:** [vqc_proto](https://github.com/kinaar8340/vqc_proto) (Orbital Braille, SLM export)
- **CLI:** `hfb-demo`, `hfb-demo --viz3d`, `hfb-demo --compare-warp`

---

## License

MIT — see the [hfb repository](https://github.com/kinaar8340/hfb).