"""SymPy symbolic acoustic and conformal metrics."""

from __future__ import annotations

from typing import Any

import sympy as sp
from sympy import Matrix, symbols


def acoustic_symbols() -> dict[str, sp.Symbol]:
    """Coordinates and flow parameters for 2+1 acoustic metrics."""
    t, x, y = symbols("t x y", real=True)
    vx, vy, cs = symbols("v_x v_y c_s", real=True)
    return {"t": t, "x": x, "y": y, "vx": vx, "vy": vy, "cs": cs}


def acoustic_line_element(sym: dict[str, sp.Symbol] | None = None) -> sp.Expr:
    """
    ds² = -(c_s² - v²)dt² - 2(v·dx)dt + dx² + dy².

    Barceló–Visser–Liberati acoustic metric (2+1 slice).
    """
    sym = sym or acoustic_symbols()
    dt, dx, dy = symbols("dt dx dy", real=True)
    v2 = sym["vx"] ** 2 + sym["vy"] ** 2
    return (
        -(sym["cs"] ** 2 - v2) * dt**2
        - 2 * (sym["vx"] * dx + sym["vy"] * dy) * dt
        + dx**2
        + dy**2
    )


def acoustic_metric_tensor(sym: dict[str, sp.Symbol] | None = None) -> Matrix:
    """4×4 metric g_{μν} in (t, x, y) coordinates (2 spatial dims)."""
    sym = sym or acoustic_symbols()
    v2 = sym["vx"] ** 2 + sym["vy"] ** 2
    return Matrix(
        [
            [-(sym["cs"] ** 2 - v2), -sym["vx"], -sym["vy"]],
            [-sym["vx"], 1, 0],
            [-sym["vy"], 0, 1],
        ]
    )


def acoustic_metric_inverse(sym: dict[str, sp.Symbol] | None = None) -> Matrix:
    """Closed-form g^{μν} for the acoustic line element."""
    sym = sym or acoustic_symbols()
    v2 = sym["vx"] ** 2 + sym["vy"] ** 2
    denom = sym["cs"] ** 2 - v2
    return Matrix(
        [
            [-1 / denom, -sym["vx"] / denom, -sym["vy"] / denom],
            [-sym["vx"] / denom, 1 + v2 / denom, sym["vx"] * sym["vy"] / denom],
            [-sym["vy"] / denom, sym["vx"] * sym["vy"] / denom, 1 + v2 / denom],
        ]
    )


def horizon_condition(sym: dict[str, sp.Symbol] | None = None) -> sp.Expr:
    """Acoustic horizon / ergosurface: v² = c_s²."""
    sym = sym or acoustic_symbols()
    return sym["vx"] ** 2 + sym["vy"] ** 2 - sym["cs"] ** 2


def draining_vortex_velocity(
    x: sp.Expr,
    y: sp.Expr,
    circulation: sp.Expr | float = 1.0,
    drain: sp.Expr | float = 0.0,
    core: sp.Expr | float = 0.2,
) -> tuple[sp.Expr, sp.Expr]:
    """Symbolic draining vortex (vx, vy)."""
    r = sp.sqrt(x**2 + y**2)
    envelope = sp.exp(-(r / core) ** 2)
    vx = (-circulation * y / r - drain * x / r) * envelope
    vy = (circulation * x / r - drain * y / r) * envelope
    return sp.simplify(vx), sp.simplify(vy)


def conformal_metric_symbols() -> dict[str, sp.Symbol]:
    """2D conformal metric ds² = e^{2Ω}(dx² + dy²)."""
    x, y, omega = symbols("x y Omega", real=True)
    return {"x": x, "y": y, "omega": omega}


def conformal_line_element(sym: dict[str, sp.Symbol] | None = None) -> sp.Expr:
    sym = sym or conformal_metric_symbols()
    dx, dy = symbols("dx dy", real=True)
    return sp.exp(2 * sym["omega"]) * (dx**2 + dy**2)


def conformal_ricci_scalar(sym: dict[str, sp.Symbol] | None = None) -> sp.Expr:
    """R = 2 e^{2Ω} ΔΩ for 2D conformal metrics."""
    sym = sym or conformal_metric_symbols()
    lap = sp.diff(sym["omega"], sym["x"], 2) + sp.diff(sym["omega"], sym["y"], 2)
    return sp.simplify(2 * sp.exp(2 * sym["omega"]) * lap)


def defect_poisson_equation(
    lam: sp.Expr,
    sym: dict[str, sp.Symbol] | None = None,
) -> sp.Eq:
    """ΔΩ = -λ conformal Poisson source equation."""
    sym = sym or conformal_metric_symbols()
    lap_omega = sp.diff(sym["omega"], sym["x"], 2) + sp.diff(sym["omega"], sym["y"], 2)
    return sp.Eq(lap_omega, -lam)


def christoffel_symbols(metric: Matrix, coords: list[sp.Symbol]) -> Matrix:
    """Christoffel symbols Γ^σ_{μν} (3×3×3 tensor as Matrix of Matrices)."""
    inv = metric.inv()
    n = len(coords)
    gamma = [[[sp.S.Zero for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for sigma in range(n):
        for mu in range(n):
            for nu in range(n):
                term = sp.S.Zero
                for lam in range(n):
                    d_mu = sp.diff(metric[lam, nu], coords[mu])
                    d_nu = sp.diff(metric[lam, mu], coords[nu])
                    d_lam = sp.diff(metric[mu, nu], coords[lam])
                    term += inv[sigma, lam] * (d_mu + d_nu - d_lam)
                gamma[sigma][mu][nu] = sp.Rational(1, 2) * sp.simplify(term)
    return gamma


def lambdify_acoustic_metric(
    sym: dict[str, sp.Symbol] | None = None,
) -> Any:
    """NumPy lambdify (vx, vy, cs) → 3×3 metric array."""
    import numpy as np

    sym = sym or acoustic_symbols()
    g = acoustic_metric_tensor(sym)
    fn = sp.lambdify((sym["vx"], sym["vy"], sym["cs"]), g, modules="numpy")
    return fn


def evaluate_metric_numeric(
    vx: float,
    vy: float,
    cs: float,
) -> list[list[float]]:
    """Evaluate symbolic acoustic metric at numeric flow parameters."""
    g_fn = lambdify_acoustic_metric()
    arr = g_fn(vx, vy, cs)
    return [[float(arr[i, j]) for j in range(3)] for i in range(3)]


def symbolic_summary() -> dict[str, str]:
    """Human-readable SymPy summaries for CLI and notebooks."""
    sym = acoustic_symbols()
    x, y = sym["x"], sym["y"]
    vx, vy = draining_vortex_velocity(x, y, circulation=1.0, drain=0.1)
    return {
        "line_element": str(acoustic_line_element(sym)),
        "horizon": str(sp.factor(horizon_condition(sym))),
        "draining_vx": str(vx),
        "draining_vy": str(vy),
        "conformal_ricci": str(conformal_ricci_scalar()),
        "defect_poisson": str(defect_poisson_equation(sp.Symbol("lambda"))),
        "metric_g00_at_v03_cs1": str(acoustic_metric_tensor(sym)[0, 0].subs(
            {sym["vx"]: 0.3, sym["vy"]: 0.4, sym["cs"]: 1.0}
        )),
    }