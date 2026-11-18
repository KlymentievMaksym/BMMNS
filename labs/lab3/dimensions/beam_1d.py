from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from labs.lab3.config import BeamConfig
from labs.lab3.solvers import solve_pentadiagonal

BoundaryType = Literal["clamped", "guided", "free"]
Array1D = NDArray[np.float64]


@dataclass(frozen=True)
class BoundaryCondition:
    """Boundary condition specification at a beam endpoint.

    Parameter meanings by kind:
        clamped: val1 = y, val2 = y'
        guided: val1 = y', val2 = y''
        free: val1 = y'', val2 = y'''
    """

    kind: BoundaryType
    val1: float = 0.0
    val2: float = 0.0


@dataclass(frozen=True)
class BeamScenario:
    """Definition of a 1D beam bending test case."""

    name: str
    title: str
    load_fn: Callable[[Array1D], Array1D]
    left_bc: BoundaryCondition
    right_bc: BoundaryCondition
    supports: tuple[tuple[float, float], ...] = ()
    use_config_supports: bool = False
    analytical_fn: Callable[[Array1D], Array1D] | None = None


@dataclass(frozen=True)
class BeamResult:
    """Numerical and optional analytical results for a 1D beam scenario."""

    name: str
    title: str
    x: Array1D
    y_num: Array1D
    y_exact: Array1D | None
    supports: list[tuple[float, float]]
    max_abs_error: float | None


def _apply_left_bc(
    a: Array1D,
    b: Array1D,
    c: Array1D,
    f: Array1D,
    h: float,
    bc: BoundaryCondition,
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

    raise ValueError(f"Unsupported left boundary condition kind: '{bc.kind}'.")


def _apply_right_bc(
    b: Array1D,
    c: Array1D,
    d: Array1D,
    e: Array1D,
    f: Array1D,
    h: float,
    bc: BoundaryCondition,
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

    if bc.kind == "free":
        d2yn3, d3yn3 = bc.val1, bc.val2
        shift = (h**2) * d2yn3 - (h**3) * d3yn3
        f[n - 1] -= e[n - 1] * (h**2) * d2yn3
        c[n - 1] += e[n - 1]
        d[n - 1] += 2.0 * e[n - 1]
        e[n - 1] = 0.0
        f[n - 1] -= d[n - 1] * shift
        b[n - 1] -= d[n - 1]
        c[n - 1] -= 2.0 * d[n - 1]
        d[n - 1] = 0.0
        f[n - 2] -= e[n - 2] * shift
        c[n - 2] += e[n - 2]
        d[n - 2] += 2.0 * e[n - 2]
        e[n - 2] = 0.0
        return

    raise ValueError(f"Unsupported right boundary condition kind: '{bc.kind}'.")


def _reconstruct_left(y_inner: Array1D, h: float, bc: BoundaryCondition) -> Array1D:
    if bc.kind == "clamped":
        y0 = bc.val1
        y1 = y0 + h * bc.val2
        return np.array([y0, y1], dtype=np.float64)

    if bc.kind == "guided":
        y1 = y_inner[0] - h * bc.val1 - (h**2) * bc.val2
        y0 = y1 - h * bc.val1
        return np.array([y0, y1], dtype=np.float64)

    raise ValueError(f"Unsupported left boundary condition kind: '{bc.kind}'.")


def _reconstruct_right(y_inner: Array1D, h: float, bc: BoundaryCondition) -> Array1D:
    if bc.kind == "clamped":
        yn3 = bc.val1
        yn2 = yn3 - h * bc.val2
        return np.array([yn2, yn3], dtype=np.float64)

    if bc.kind == "guided":
        yn2 = (h**2) * bc.val2 + h * bc.val1 + y_inner[-1]
        yn3 = h * bc.val1 + yn2
        return np.array([yn2, yn3], dtype=np.float64)

    if bc.kind == "free":
        yn2 = (h**2) * bc.val1 + 2.0 * y_inner[-1] - y_inner[-2] - (h**3) * bc.val2
        yn3 = (h**2) * bc.val1 + 2.0 * yn2 - y_inner[-1]
        return np.array([yn2, yn3], dtype=np.float64)

    raise ValueError(f"Unsupported right boundary condition kind: '{bc.kind}'.")


def solve_beam(config: BeamConfig, scenario: BeamScenario) -> BeamResult:
    """Solve a 1D continuous beam boundary value problem."""
    n = config.N
    if n < 4:
        raise ValueError(f"Beam internal node count N={n} must be at least 4.")

    h = config.L / (n + 3)
    x = np.linspace(0.0, config.L, n + 4, dtype=np.float64)
    inv_h4 = 1.0 / (h**4)

    a = np.full(n, inv_h4, dtype=np.float64)
    b = np.full(n, -4.0 * inv_h4, dtype=np.float64)
    c = np.full(n, -6.0 * inv_h4, dtype=np.float64)
    d = np.full(n, -4.0 * inv_h4, dtype=np.float64)
    e = np.full(n, inv_h4, dtype=np.float64)
    f = scenario.load_fn(x[2:-2]) * (config.q / config.EI)

    raw_supports = (
        list(config.supports)
        if scenario.use_config_supports
        else list(scenario.supports)
    )
    snapped_supports: list[tuple[float, float]] = []
    for pos, deflection in raw_supports:
        full_idx = int(round(pos / h))
        inner_idx = full_idx - 2
        if 0 <= inner_idx < n:
            a[inner_idx] = 0.0
            b[inner_idx] = 0.0
            c[inner_idx] = -1.0
            d[inner_idx] = 0.0
            e[inner_idx] = 0.0
            f[inner_idx] = deflection
            snapped_supports.append((float(x[full_idx]), deflection))

    _apply_left_bc(a, b, c, f, h, scenario.left_bc)
    _apply_right_bc(b, c, d, e, f, h, scenario.right_bc)

    y_inner = solve_pentadiagonal(a, b, c, d, e, f)
    y_left = _reconstruct_left(y_inner, h, scenario.left_bc)
    y_right = _reconstruct_right(y_inner, h, scenario.right_bc)
    y_num = np.concatenate([y_left, y_inner, y_right])

    y_exact: Array1D | None = None
    max_abs_error: float | None = None
    if scenario.analytical_fn is not None and np.isclose(config.L, 1.0):
        scale = config.q / config.EI
        y_exact = scenario.analytical_fn(x) * scale
        max_abs_error = float(np.max(np.abs(y_num - y_exact)))

    return BeamResult(
        name=scenario.name,
        title=scenario.title,
        x=x,
        y_num=y_num,
        y_exact=y_exact,
        supports=snapped_supports,
        max_abs_error=max_abs_error,
    )


def _analytical_2_3(x: Array1D) -> Array1D:
    c2 = (20.0 * np.cos(10.0) - 6.0 * np.sin(10.0) + 40.0) / 2.0
    c3 = (60.0 * np.cos(10.0) - 12.0 * np.sin(10.0) + 60.0) / 6.0
    return np.sin(10.0 * x) - 10.0 * x + c2 * (x**2) - c3 * (x**3) + 1.0


def _analytical_2_5(x: Array1D) -> Array1D:
    return (
        -(5.0 * (x**6)) / 18.0
        + (5.0 * (x**5)) / 3.0
        - (25.0 * (x**4)) / 6.0
        + 5.0 * (x**3)
        - (5.0 * (x**2)) / 2.0
    )


def _analytical_2_8(x: Array1D) -> Array1D:
    return (
        (0.025 * (x**6)) / 9.0
        - (0.025 * (x**5)) / 3.0
        + (0.25 * (x**3)) / 9.0
        - (0.125 * (x**2)) / 3.0
    )


def get_beam_scenarios() -> dict[str, BeamScenario]:
    """Return all 1D beam benchmark scenarios from Section 2.3."""
    return {
        "task_2_3": BeamScenario(
            name="task_2_3",
            title="Task 2.3: Clamped ends, sinusoidal load",
            load_fn=lambda x: 1e4 * np.sin(10.0 * x),
            left_bc=BoundaryCondition("clamped", 1.0, 0.0),
            right_bc=BoundaryCondition("clamped", 1.0, 0.0),
            analytical_fn=_analytical_2_3,
        ),
        "task_2_4": BeamScenario(
            name="task_2_4",
            title="Task 2.4: Clamped ends, point displacement at x=1/3",
            load_fn=lambda x: np.zeros_like(x),
            left_bc=BoundaryCondition("clamped", 0.0, 0.0),
            right_bc=BoundaryCondition("clamped", 0.0, 0.0),
            supports=((1.0 / 3.0, -1.0),),
        ),
        "task_2_5": BeamScenario(
            name="task_2_5",
            title="Task 2.5: Clamped left, guided right end",
            load_fn=lambda x: -100.0 * ((1.0 - x) ** 2),
            left_bc=BoundaryCondition("clamped", 0.0, 0.0),
            right_bc=BoundaryCondition("guided", 0.0, 0.0),
            analytical_fn=_analytical_2_5,
        ),
        "task_2_6": BeamScenario(
            name="task_2_6",
            title="Task 2.6: Three-span beam, clamped ends",
            load_fn=lambda x: -1e4 * x,
            left_bc=BoundaryCondition("clamped", 0.0, 0.0),
            right_bc=BoundaryCondition("clamped", 0.0, 0.0),
            use_config_supports=True,
        ),
        "task_2_7": BeamScenario(
            name="task_2_7",
            title="Task 2.7: Three-span beam, guided ends",
            load_fn=lambda x: -1e4 * x * (1.0 - x),
            left_bc=BoundaryCondition("guided", 0.0, 0.0),
            right_bc=BoundaryCondition("guided", 0.0, 0.0),
            use_config_supports=True,
        ),
        "task_2_8": BeamScenario(
            name="task_2_8",
            title="Task 2.8: Cantilever beam, clamped left and free right",
            load_fn=lambda x: -x * (1.0 - x),
            left_bc=BoundaryCondition("clamped", 0.0, 0.0),
            right_bc=BoundaryCondition("free", 0.0, 0.0),
            analytical_fn=_analytical_2_8,
        ),
    }


def run_all_beam_scenarios(config: BeamConfig) -> list[BeamResult]:
    """Execute all 1D beam scenarios and return their results."""
    return [solve_beam(config, scenario) for scenario in get_beam_scenarios().values()]
