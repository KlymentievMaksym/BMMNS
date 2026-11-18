from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BiharmonicStencil:
    """13-point finite-difference stencil weights for the 2D biharmonic operator."""

    ax: float
    bx: float
    dx: float
    ex: float
    ay: float
    by: float
    dy: float
    ey: float
    a00: float
    a01: float
    a10: float
    a11: float
    c: float


@dataclass(frozen=True)
class SORResult:
    """Container for successive over-relaxation solver outputs."""

    u: NDArray[np.float64]
    iterations: int
    residual: float
    converged: bool


def solve_pentadiagonal(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    c: NDArray[np.float64],
    d: NDArray[np.float64],
    e: NDArray[np.float64],
    f: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Solve a pentadiagonal linear system using the sweep method.

    Follows the lab manual sign convention:
        a[i]*y[i-2] + b[i]*y[i-1] - c[i]*y[i] + d[i]*y[i+1] + e[i]*y[i+2] = f[i]

    With this convention, standard fourth-derivative coefficients use c[i] = -6/h^4
    and rigid internal supports set c[i] = -1. An alternative convention writes
    +c[i]*y[i] on the main diagonal, which flips the signs of c[i] and f[i] in the
    sweep recurrence coefficients below.
    """
    n = len(f)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    for name, arr in (("a", a), ("b", b), ("c", c), ("d", d), ("e", e)):
        if len(arr) != n:
            raise ValueError(f"Diagonal '{name}' length {len(arr)} != {n}.")

    if np.isclose(c[0], 0.0):
        raise ValueError("Zero pivot encountered at index 0.")

    if n == 1:
        return np.array([-f[0] / c[0]], dtype=np.float64)

    k = np.zeros(n, dtype=np.float64)
    l_coeff = np.zeros(n, dtype=np.float64)
    m = np.zeros(n, dtype=np.float64)

    k[0] = d[0] / c[0]
    l_coeff[0] = e[0] / c[0]
    m[0] = -f[0] / c[0]

    zn = float(c[1] - b[1] * k[0])
    if np.isclose(zn, 0.0):
        raise ValueError("Zero pivot encountered at index 1.")

    k[1] = (d[1] + b[1] * l_coeff[0]) / zn
    l_coeff[1] = e[1] / zn
    m[1] = (b[1] * m[0] - f[1]) / zn

    for i in range(2, n):
        alpha = a[i] * k[i - 2] + b[i]
        zn = float(c[i] - a[i] * l_coeff[i - 2] - alpha * k[i - 1])
        if np.isclose(zn, 0.0):
            raise ValueError(f"Zero pivot encountered at index {i}.")

        k[i] = (d[i] + alpha * l_coeff[i - 1]) / zn
        l_coeff[i] = e[i] / zn
        m[i] = (alpha * m[i - 1] + a[i] * m[i - 2] - f[i]) / zn

    y = np.zeros(n, dtype=np.float64)
    y[n - 1] = m[n - 1]
    y[n - 2] = k[n - 2] * y[n - 1] + m[n - 2]

    for i in range(n - 3, -1, -1):
        y[i] = k[i] * y[i + 1] + l_coeff[i] * y[i + 2] + m[i]

    return y


def solve_sor(
    stencil: BiharmonicStencil,
    rhs: NDArray[np.float64],
    u_init: NDArray[np.float64],
    omega: float = 1.85,
    epsilon: float = 1e-9,
    max_iter: int = 50000,
    fixed_mask: NDArray[np.bool_] | None = None,
) -> SORResult:
    """Solve the 2D biharmonic grid system via successive over-relaxation.

    Uses positive central weight c > 0 so the Gauss-Seidel target value is
    (rhs[i, j] - neighbors) / c. The lab manual negates c = -c and writes
    (neighbors - rhs[i, j]) / c, which is algebraically identical.
    """
    if not 0.0 < omega < 2.0:
        raise ValueError(f"Relaxation factor omega={omega} must be in (0, 2).")
    if np.isclose(stencil.c, 0.0):
        raise ValueError("Central stencil coefficient 'c' must be non-zero.")

    nx, ny = u_init.shape
    if rhs.shape != (nx, ny):
        raise ValueError(f"RHS shape {rhs.shape} does not match grid {(nx, ny)}.")
    if fixed_mask is not None and fixed_mask.shape != (nx, ny):
        raise ValueError(
            f"Mask shape {fixed_mask.shape} does not match grid {(nx, ny)}."
        )

    u = u_init.astype(np.float64, copy=True)
    ax, bx, dx, ex = stencil.ax, stencil.bx, stencil.dx, stencil.ex
    ay, by, dy, ey = stencil.ay, stencil.by, stencil.dy, stencil.ey
    a00, a01, a10, a11 = stencil.a00, stencil.a01, stencil.a10, stencil.a11
    inv_c = 1.0 / stencil.c

    residual = float("inf")
    iterations = 0

    for iteration in range(1, max_iter + 1):
        iterations = iteration
        max_diff = 0.0

        for i in range(2, nx - 2):
            for j in range(2, ny - 2):
                if fixed_mask is not None and fixed_mask[i, j]:
                    continue

                neighbors = (
                    ax * u[i - 2, j]
                    + bx * u[i - 1, j]
                    + dx * u[i + 1, j]
                    + ex * u[i + 2, j]
                    + ay * u[i, j - 2]
                    + by * u[i, j - 1]
                    + dy * u[i, j + 1]
                    + ey * u[i, j + 2]
                    + a00 * u[i - 1, j - 1]
                    + a01 * u[i - 1, j + 1]
                    + a10 * u[i + 1, j - 1]
                    + a11 * u[i + 1, j + 1]
                )
                u_gs = (rhs[i, j] - neighbors) * inv_c
                u_new = u[i, j] + omega * (u_gs - u[i, j])

                diff = abs(u_new - u[i, j])
                if diff > max_diff:
                    max_diff = diff
                u[i, j] = u_new

        residual = max_diff
        if residual <= epsilon:
            return SORResult(
                u=u,
                iterations=iterations,
                residual=residual,
                converged=True,
            )

    return SORResult(
        u=u,
        iterations=iterations,
        residual=residual,
        converged=False,
    )
