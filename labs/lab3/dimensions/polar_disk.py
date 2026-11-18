from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from config import PolarConfig
from solvers import solve_pentadiagonal

PolarBoundaryType = Literal["clamped", "hinged", "guided"]
Array1D = NDArray[np.float64]


@dataclass(frozen=True)
class PolarBoundary:
    """Radial boundary condition specification at an annular plate edge.

    Parameter meanings by kind:
        clamped: val1 = phi, val2 = d(phi)/dr
        hinged: val1 = phi, val2 = d2(phi)/dr2
        guided: val1 = d(phi)/dr, val2 = d2(phi)/dr2
    """

    kind: PolarBoundaryType
    val1: float = 0.0
    val2: float = 0.0


@dataclass(frozen=True)
class PolarScenario:
    """Definition of an axisymmetric polar plate bending test case."""

    name: str
    title: str
    inner_bc: PolarBoundary
    outer_bc: PolarBoundary
    load_fn: Callable[[Array1D], Array1D] | None = None


@dataclass(frozen=True)
class PolarResult:
    """Numerical and optional analytical results for a polar plate scenario."""

    name: str
    title: str
    r: Array1D
    phi_num: Array1D
    phi_exact: Array1D | None
    max_abs_error: float | None


def _apply_inner_bc(
    a: Array1D,
    b: Array1D,
    c: Array1D,
    f: Array1D,
    h: float,
    bc: PolarBoundary,
) -> None:
    if bc.kind == "clamped":
        y0, dy0 = bc.val1, bc.val2
        y1 = y0 + h * dy0
        f[0] -= a[0] * y0 + b[0] * y1
        a[0] = 0.0
        b[0] = 0.0
        f[1] -= a[1] * y1
        a[1] = 0.0
        return

    if bc.kind == "hinged":
        y0, d2y0 = bc.val1, bc.val2
        const_term = 0.5 * (y0 - (h**2) * d2y0)
        f[0] -= a[0] * y0 + b[0] * const_term
        c[0] -= 0.5 * b[0]
        a[0] = 0.0
        b[0] = 0.0
        f[1] -= a[1] * const_term
        b[1] += 0.5 * a[1]
        a[1] = 0.0
        return

    if bc.kind == "guided":
        dy0, d2y0 = bc.val1, bc.val2
        shift = h * dy0 + (h**2) * d2y0
        f[0] += a[0] * h * dy0
        b[0] += a[0]
        a[0] = 0.0
        f[0] += b[0] * shift
        c[0] -= b[0]
        b[0] = 0.0
        f[1] += a[1] * shift
        b[1] += a[1]
        a[1] = 0.0
        return

    raise ValueError(f"Unsupported inner boundary condition kind: '{bc.kind}'.")


def _apply_outer_bc(
    b: Array1D,
    c: Array1D,
    d: Array1D,
    e: Array1D,
    f: Array1D,
    h: float,
    bc: PolarBoundary,
) -> None:
    n = len(f)
    if bc.kind == "clamped":
        yn3, dyn3 = bc.val1, bc.val2
        yn2 = yn3 - h * dyn3
        f[n - 1] -= e[n - 1] * yn3 + d[n - 1] * yn2
        e[n - 1] = 0.0
        d[n - 1] = 0.0
        f[n - 2] -= e[n - 2] * yn2
        e[n - 2] = 0.0
        return

    if bc.kind == "hinged":
        yn3, d2yn3 = bc.val1, bc.val2
        const_term = 0.5 * (yn3 - (h**2) * d2yn3)
        f[n - 1] -= e[n - 1] * yn3 + d[n - 1] * const_term
        c[n - 1] -= 0.5 * d[n - 1]
        e[n - 1] = 0.0
        d[n - 1] = 0.0
        f[n - 2] -= e[n - 2] * const_term
        d[n - 2] += 0.5 * e[n - 2]
        e[n - 2] = 0.0
        return

    if bc.kind == "guided":
        dyn3, d2yn3 = bc.val1, bc.val2
        shift = (h**2) * d2yn3 + h * dyn3
        f[n - 1] -= e[n - 1] * h * dyn3
        d[n - 1] += e[n - 1]
        e[n - 1] = 0.0
        f[n - 1] -= d[n - 1] * shift
        c[n - 1] -= d[n - 1]
        d[n - 1] = 0.0
        f[n - 2] -= e[n - 2] * shift
        d[n - 2] += e[n - 2]
        e[n - 2] = 0.0
        return

    raise ValueError(f"Unsupported outer boundary condition kind: '{bc.kind}'.")


def _reconstruct_inner(phi_inner: Array1D, h: float, bc: PolarBoundary) -> Array1D:
    if bc.kind == "clamped":
        y0 = bc.val1
        y1 = y0 + h * bc.val2
        return np.array([y0, y1], dtype=np.float64)

    if bc.kind == "hinged":
        y0 = bc.val1
        y1 = 0.5 * (y0 + phi_inner[0] - (h**2) * bc.val2)
        return np.array([y0, y1], dtype=np.float64)

    if bc.kind == "guided":
        y1 = phi_inner[0] - h * bc.val1 - (h**2) * bc.val2
        y0 = y1 - h * bc.val1
        return np.array([y0, y1], dtype=np.float64)

    raise ValueError(f"Unsupported inner boundary condition kind: '{bc.kind}'.")


def _reconstruct_outer(phi_inner: Array1D, h: float, bc: PolarBoundary) -> Array1D:
    if bc.kind == "clamped":
        yn3 = bc.val1
        yn2 = yn3 - h * bc.val2
        return np.array([yn2, yn3], dtype=np.float64)

    if bc.kind == "hinged":
        yn3 = bc.val1
        yn2 = 0.5 * (yn3 + phi_inner[-1] - (h**2) * bc.val2)
        return np.array([yn2, yn3], dtype=np.float64)

    if bc.kind == "guided":
        yn2 = (h**2) * bc.val2 + h * bc.val1 + phi_inner[-1]
        yn3 = h * bc.val1 + yn2
        return np.array([yn2, yn3], dtype=np.float64)

    raise ValueError(f"Unsupported outer boundary condition kind: '{bc.kind}'.")


def _boundary_rows(radius: float, bc: PolarBoundary) -> tuple[Array1D, Array1D]:
    ln_r = np.log(radius)
    r2 = radius**2

    row_val = np.array([1.0, ln_r, r2, r2 * ln_r], dtype=np.float64)
    row_d1 = np.array(
        [0.0, 1.0 / radius, 2.0 * radius, radius * (2.0 * ln_r + 1.0)],
        dtype=np.float64,
    )
    row_d2 = np.array(
        [0.0, -1.0 / r2, 2.0, 2.0 * ln_r + 3.0],
        dtype=np.float64,
    )

    if bc.kind == "clamped":
        return row_val, row_d1
    if bc.kind == "hinged":
        return row_val, row_d2
    if bc.kind == "guided":
        return row_d1, row_d2

    raise ValueError(f"Unsupported boundary condition kind: '{bc.kind}'.")


def _solve_homogeneous_analytical(
    r: Array1D,
    r_a: float,
    r_b: float,
    inner_bc: PolarBoundary,
    outer_bc: PolarBoundary,
) -> Array1D:
    """Compute exact phi(r) = C1 + C2*ln(r) + C3*r^2 + C4*r^2*ln(r) for q = 0."""
    row0, row1 = _boundary_rows(r_a, inner_bc)
    row2, row3 = _boundary_rows(r_b, outer_bc)

    matrix = np.vstack([row0, row1, row2, row3])
    rhs = np.array(
        [inner_bc.val1, inner_bc.val2, outer_bc.val1, outer_bc.val2],
        dtype=np.float64,
    )
    coeffs = np.linalg.solve(matrix, rhs)

    ln_r = np.log(r)
    r2 = r**2
    return coeffs[0] + coeffs[1] * ln_r + coeffs[2] * r2 + coeffs[3] * r2 * ln_r


def solve_polar(config: PolarConfig, scenario: PolarScenario) -> PolarResult:
    """Solve an axisymmetric annular plate bending boundary value problem."""
    if config.r_a <= 0.0:
        raise ValueError(f"Inner radius r_a={config.r_a} must be strictly positive.")
    if config.r_b <= config.r_a:
        raise ValueError(f"Outer radius r_b={config.r_b} must exceed r_a={config.r_a}.")
    n = config.N_r
    if n < 4:
        raise ValueError(f"Radial internal node count N_r={n} must be at least 4.")

    h = (config.r_b - config.r_a) / (n + 3)
    r = np.linspace(config.r_a, config.r_b, n + 4, dtype=np.float64)
    r_in = r[2:-2]

    inv_h = 1.0 / h
    inv_h2 = inv_h**2
    inv_h3 = inv_h**3
    inv_h4 = inv_h**4

    inv_r = 1.0 / r_in
    inv_r2 = inv_r**2
    inv_r3 = inv_r**3

    a = inv_h4 - inv_r * inv_h3
    b = -4.0 * inv_h4 + 2.0 * inv_r * inv_h3 - inv_r2 * inv_h2 - 0.5 * inv_r3 * inv_h
    c = -6.0 * inv_h4 - 2.0 * inv_r2 * inv_h2
    d = -4.0 * inv_h4 - 2.0 * inv_r * inv_h3 - inv_r2 * inv_h2 + 0.5 * inv_r3 * inv_h
    e = inv_h4 + inv_r * inv_h3

    if scenario.load_fn is not None:
        f = scenario.load_fn(r_in) / config.D
    else:
        f = np.full(n, config.q / config.D, dtype=np.float64)

    _apply_inner_bc(a, b, c, f, h, scenario.inner_bc)
    _apply_outer_bc(b, c, d, e, f, h, scenario.outer_bc)

    phi_inner = solve_pentadiagonal(a, b, c, d, e, f)
    phi_left = _reconstruct_inner(phi_inner, h, scenario.inner_bc)
    phi_right = _reconstruct_outer(phi_inner, h, scenario.outer_bc)
    phi_num = np.concatenate([phi_left, phi_inner, phi_right])

    is_homogeneous = scenario.load_fn is None and np.isclose(config.q, 0.0)
    phi_exact: Array1D | None = None
    max_abs_error: float | None = None
    if is_homogeneous:
        phi_exact = _solve_homogeneous_analytical(
            r, config.r_a, config.r_b, scenario.inner_bc, scenario.outer_bc
        )
        max_abs_error = float(np.max(np.abs(phi_num - phi_exact)))

    return PolarResult(
        name=scenario.name,
        title=scenario.title,
        r=r,
        phi_num=phi_num,
        phi_exact=phi_exact,
        max_abs_error=max_abs_error,
    )


def get_polar_scenarios() -> dict[str, PolarScenario]:
    """Return all axisymmetric polar plate scenarios from Section 4.4."""
    return {
        "task_4_4": PolarScenario(
            name="task_4_4",
            title="Task 4.4: Clamped inner and outer edges",
            inner_bc=PolarBoundary("clamped", 1.0, 0.0),
            outer_bc=PolarBoundary("clamped", 0.0, 0.0),
        ),
        "task_4_5": PolarScenario(
            name="task_4_5",
            title="Task 4.5: Hinged inner edge, clamped outer edge",
            inner_bc=PolarBoundary("hinged", 1.0, 0.0),
            outer_bc=PolarBoundary("clamped", 0.0, 0.0),
        ),
        "task_4_6": PolarScenario(
            name="task_4_6",
            title="Task 4.6: Guided inner edge, sloped clamped outer edge",
            inner_bc=PolarBoundary("guided", 0.0, 0.0),
            outer_bc=PolarBoundary("clamped", 1.0, 1.0),
        ),
    }


def run_all_polar_scenarios(config: PolarConfig) -> list[PolarResult]:
    """Execute all polar plate scenarios and return their results."""
    return [
        solve_polar(config, scenario) for scenario in get_polar_scenarios().values()
    ]
