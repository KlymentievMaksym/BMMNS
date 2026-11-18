from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from config import PlateConfig
from solvers import BiharmonicStencil, solve_sor

Array1D = NDArray[np.float64]
Array2D = NDArray[np.float64]


@dataclass(frozen=True)
class PlateBoundary:
    """Clamped boundary values and normal derivatives along the four plate edges."""

    left_val: float = 0.0
    right_val: float = 0.0
    bottom_val: float = 0.0
    top_val: float = 0.0
    left_deriv: float = 0.0
    right_deriv: float = 0.0
    bottom_deriv: float = 0.0
    top_deriv: float = 0.0


@dataclass(frozen=True)
class PointConstraint:
    """Pinned deflection at normalized plate coordinates in [0, 1] x [0, 1]."""

    rel_x: float
    rel_y: float
    value: float


@dataclass(frozen=True)
class LineConstraint:
    """Pinned deflection along a normalized grid line perpendicular to an axis."""

    axis: Literal["x", "y"]
    rel_pos: float
    value: float


@dataclass(frozen=True)
class PlateScenario:
    """Definition of a 2D rectangular plate bending test case."""

    name: str
    title: str
    load_fn: Callable[[Array2D, Array2D], Array2D]
    boundary: PlateBoundary = PlateBoundary()
    point_constraints: tuple[PointConstraint, ...] = ()
    line_constraints: tuple[LineConstraint, ...] = ()
    use_config_q: bool = False


@dataclass(frozen=True)
class PlateResult:
    """Numerical solution and solver metadata for a 2D plate scenario.

    The deflection array `u` has shape `(N_x, N_y)` where axis 0 aligns with `x`
    and axis 1 aligns with `y`.
    """

    name: str
    title: str
    x: Array1D
    y: Array1D
    u: Array2D
    iterations: int
    residual: float
    converged: bool
    pinned_points: list[tuple[float, float, float]]


def build_biharmonic_stencil(hx: float, hy: float) -> BiharmonicStencil:
    """Compute the 13-point finite-difference weights for the biharmonic operator."""
    inv_hx4 = 1.0 / (hx**4)
    inv_hy4 = 1.0 / (hy**4)
    inv_hx2_hy2 = 1.0 / ((hx * hy) ** 2)

    bx = -4.0 * inv_hx4 - 4.0 * inv_hx2_hy2
    by = -4.0 * inv_hy4 - 4.0 * inv_hx2_hy2
    cross = 2.0 * inv_hx2_hy2
    c = 6.0 * inv_hx4 + 8.0 * inv_hx2_hy2 + 6.0 * inv_hy4

    return BiharmonicStencil(
        ax=inv_hx4,
        bx=bx,
        dx=bx,
        ex=inv_hx4,
        ay=inv_hy4,
        by=by,
        dy=by,
        ey=inv_hy4,
        a00=cross,
        a01=cross,
        a10=cross,
        a11=cross,
        c=c,
    )


def _apply_clamped_boundaries(
    u: Array2D,
    hx: float,
    hy: float,
    bc: PlateBoundary,
) -> None:
    u[0, :] = bc.left_val
    u[1, :] = u[0, :] + bc.left_deriv * hx

    u[-1, :] = bc.right_val
    u[-2, :] = u[-1, :] - bc.right_deriv * hx

    u[1:-1, 0] = bc.bottom_val
    u[1:-1, 1] = u[1:-1, 0] + bc.bottom_deriv * hy

    u[1:-1, -1] = bc.top_val
    u[1:-1, -2] = u[1:-1, -1] - bc.top_deriv * hy


def solve_plate(config: PlateConfig, scenario: PlateScenario) -> PlateResult:
    """Solve a 2D rectangular plate biharmonic boundary value problem."""
    nx, ny = config.N_x, config.N_y
    if nx < 5 or ny < 5:
        raise ValueError(f"Plate grid ({nx}, {ny}) must be at least 5x5.")

    hx = config.L_x / (nx - 1)
    hy = config.L_y / (ny - 1)
    x = np.linspace(0.0, config.L_x, nx, dtype=np.float64)
    y = np.linspace(0.0, config.L_y, ny, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")

    stencil = build_biharmonic_stencil(hx, hy)
    if scenario.use_config_q:
        rhs = np.full((nx, ny), config.q / config.D, dtype=np.float64)
    else:
        rhs = scenario.load_fn(grid_x, grid_y) / config.D

    u_init = np.zeros((nx, ny), dtype=np.float64)
    _apply_clamped_boundaries(u_init, hx, hy, scenario.boundary)

    fixed_mask = np.zeros((nx, ny), dtype=np.bool_)
    for line in scenario.line_constraints:
        if line.axis == "x":
            idx = int(round(line.rel_pos * (nx - 1)))
            if 2 <= idx <= nx - 3:
                u_init[idx, 2:-2] = line.value
                fixed_mask[idx, 2:-2] = True
        else:
            idx = int(round(line.rel_pos * (ny - 1)))
            if 2 <= idx <= ny - 3:
                u_init[2:-2, idx] = line.value
                fixed_mask[2:-2, idx] = True

    pinned_points: list[tuple[float, float, float]] = []
    for pt in scenario.point_constraints:
        ix = int(round(pt.rel_x * (nx - 1)))
        iy = int(round(pt.rel_y * (ny - 1)))
        if 2 <= ix <= nx - 3 and 2 <= iy <= ny - 3:
            u_init[ix, iy] = pt.value
            fixed_mask[ix, iy] = True
            pinned_points.append((float(x[ix]), float(y[iy]), pt.value))

    sor_res = solve_sor(
        stencil=stencil,
        rhs=rhs,
        u_init=u_init,
        omega=config.omega,
        epsilon=config.epsilon,
        max_iter=config.max_iter,
        fixed_mask=fixed_mask,
    )

    return PlateResult(
        name=scenario.name,
        title=scenario.title,
        x=x,
        y=y,
        u=sor_res.u,
        iterations=sor_res.iterations,
        residual=sor_res.residual,
        converged=sor_res.converged,
        pinned_points=pinned_points,
    )


def get_plate_scenarios() -> dict[str, PlateScenario]:
    """Return all 2D rectangular plate benchmark scenarios from Section 3.4."""
    return {
        "task_3_1": PlateScenario(
            name="task_3_1",
            title="Task 3.1: Two-span clamped plate, uniform load",
            load_fn=lambda gx, gy: np.full_like(gx, -100.0),
            line_constraints=(LineConstraint("x", 0.5, 0.0),),
            use_config_q=True,
        ),
        "task_3_2": PlateScenario(
            name="task_3_2",
            title="Task 3.2: Clamped plate with central point support",
            load_fn=lambda gx, gy: np.full_like(gx, -100.0),
            point_constraints=(PointConstraint(0.5, 0.5, 0.0),),
            use_config_q=True,
        ),
        "task_3_3": PlateScenario(
            name="task_3_3",
            title="Task 3.3: Clamped plate with four point displacements",
            load_fn=lambda gx, gy: np.zeros_like(gx),
            point_constraints=(
                PointConstraint(1.0 / 3.0, 1.0 / 3.0, -1.0),
                PointConstraint(1.0 / 3.0, 2.0 / 3.0, -1.0),
                PointConstraint(2.0 / 3.0, 1.0 / 3.0, 2.0),
                PointConstraint(2.0 / 3.0, 2.0 / 3.0, -1.0),
            ),
        ),
    }


def run_all_plate_scenarios(config: PlateConfig) -> list[PlateResult]:
    """Execute all 2D plate scenarios and return their results."""
    return [
        solve_plate(config, scenario) for scenario in get_plate_scenarios().values()
    ]
