from dataclasses import dataclass

import numpy as np

from labs.lab9.entities import RichardsonParams


@dataclass(frozen=True)
class RichardsonStability:
    det: float  # m*n - a*b
    eigenvalues: np.ndarray
    equilibrium: tuple[float, float] | None
    stable: bool
    regime: str


@dataclass(frozen=True)
class WWIFit:
    mean_z: np.ndarray
    growth: np.ndarray
    k: float  # k = a - m
    f: float  # f = r + s


def analyze_richardson(p: RichardsonParams) -> RichardsonStability:
    """Equilibrium and stability from the Jacobian [[-m, a], [b, -n]]."""
    jac = np.array([[-p.m, p.a], [p.b, -p.n]])
    det = p.m * p.n - p.a * p.b
    eigenvalues = np.linalg.eigvals(jac)
    stable = bool(np.all(eigenvalues.real < 0.0))

    equilibrium = None
    if abs(det) > 1e-12:
        x_eq, y_eq = np.linalg.solve(jac, -np.array([p.r, p.s]))
        equilibrium = (float(x_eq), float(y_eq))

    if det > 0 and p.r > 0 and p.s > 0:
        regime = "equilibrium"
    elif det < 0 and p.r > 0 and p.s > 0:
        regime = "unbounded escalation"
    elif det > 0 and p.r < 0 and p.s < 0:
        regime = "mutual disarmament"
    elif det < 0 and p.r < 0 and p.s < 0:
        regime = "depends on initial state"
    else:
        regime = "mixed signs"
    return RichardsonStability(det, eigenvalues, equilibrium, stable, regime)


def fit_wwi_data() -> WWIFit:
    """Fit dz/dt = k*z + f on the 1909-1913 budgets (France, Russia, Germany, Austria-Hungary)."""
    z = (
        np.array([48.6, 50.9, 57.1, 63.2, 74.7])
        + np.array([66.7, 68.5, 70.7, 81.8, 92.7])
        + np.array([63.1, 62.0, 62.0, 68.2, 95.4])
        + np.array([20.8, 23.4, 23.4, 25.5, 26.9])
    )
    mean_z = 0.5 * (z[:-1] + z[1:])
    growth = np.diff(z)
    k, f = np.polyfit(mean_z, growth, deg=1)
    return WWIFit(mean_z, growth, float(k), float(f))
