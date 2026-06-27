"""Numeric and symbolic warp-shift comparison (analog vs Alcubierre)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class WarpComparisonReport:
    warp_fidelity: float
    max_shift_diff: float
    mean_abs_diff: float
    mean_ratio: float
    vs: float
    rs: float
    sigma: float


def warp_fidelity(
    analog_shift: NDArray[np.floating],
    gr_shift: NDArray[np.floating],
    dx: float,
) -> float:
    """L¹ mismatch ∫ |β_analog − β_GR| dA (lower = closer analog)."""
    return float(np.sum(np.abs(analog_shift - gr_shift)) * dx**2)


def evaluate_gr_shift(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    vs: float = 0.3,
    rs: float = 1.0,
    sigma: float = 0.25,
) -> NDArray[np.floating]:
    """Evaluate Alcubierre shift β(x, y) numerically."""
    from hfb.analog_gravity.symbolic import lambdify_alcubierre_shift

    gr_fn = lambdify_alcubierre_shift()
    return np.asarray(gr_fn(x, y, vs, rs, sigma), dtype=float)


def compare_warp_numeric(
    analog_shift: NDArray[np.floating],
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    dx: float,
    vs: float = 0.3,
    rs: float = 1.0,
    sigma: float = 0.25,
) -> tuple[NDArray[np.floating], WarpComparisonReport]:
    """Compare analog shift field to Alcubierre β; return GR field + report."""
    gr_shift = evaluate_gr_shift(x, y, vs=vs, rs=rs, sigma=sigma)
    diff = analog_shift - gr_shift
    fidelity = warp_fidelity(analog_shift, gr_shift, dx)
    denom = np.maximum(np.abs(gr_shift), 1e-9)
    ratio = analog_shift / denom

    report = WarpComparisonReport(
        warp_fidelity=fidelity,
        max_shift_diff=float(np.max(np.abs(diff))),
        mean_abs_diff=float(np.mean(np.abs(diff))),
        mean_ratio=float(np.mean(ratio)),
        vs=vs,
        rs=rs,
        sigma=sigma,
    )
    return gr_shift, report


def plot_warp_comparison(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    analog_shift: NDArray[np.floating],
    gr_shift: NDArray[np.floating],
    report: WarpComparisonReport,
    extent: float,
    output_path: str | None = None,
):
    """Side-by-side analog vs GR shift with diff panel."""
    import matplotlib.pyplot as plt

    diff = analog_shift - gr_shift
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ext = [-extent, extent]

    for ax, data, title in zip(
        axes,
        [analog_shift, gr_shift, diff],
        ["β analog (HFB)", "β GR (Alcubierre)", "β analog − β GR"],
    ):
        im = ax.imshow(data, origin="lower", extent=ext * 2, cmap="RdBu_r")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f"Warp fidelity L¹={report.warp_fidelity:.4f} | max|Δβ|={report.max_shift_diff:.4f}",
        fontsize=11,
    )
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig