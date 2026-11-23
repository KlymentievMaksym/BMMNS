import numpy as np


def exponential_growth_discrete(y0: float, a: float, steps: int) -> np.ndarray:
    """Discrete exponential population growth: y_t = (1 + a) * y_{t-1}."""
    y = np.zeros(steps + 1, dtype=np.float64)
    y[0] = y0
    for t in range(1, steps + 1):
        y[t] = (1.0 + a) * y[t - 1]
    return y


def exponential_growth_continuous(
    y0: float, a: float, dt: float, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Continuous exponential population growth: y(t) = y0 * exp(a * t)."""
    t_vals = np.linspace(0.0, steps * dt, steps + 1)
    y_vals = y0 * np.exp(a * t_vals)
    return t_vals, y_vals


def logistic_evolution(
    n0: float, r: float, k: float, m: float, dt: float, steps: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve dN/dt = r*N*(K - N) - m*N via RK4 and return stationary point Nc."""
    t_vals = np.linspace(0.0, steps * dt, steps + 1)
    n_vals = np.zeros(steps + 1, dtype=np.float64)
    n_vals[0] = n0

    def rhs(n_val: float) -> float:
        return r * n_val * (k - n_val) - m * n_val

    for i in range(steps):
        n = n_vals[i]
        k1 = rhs(n)
        k2 = rhs(n + 0.5 * dt * k1)
        k3 = rhs(n + 0.5 * dt * k2)
        k4 = rhs(n + dt * k3)
        n_vals[i + 1] = max(0.0, n + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

    equilibrium = (k - m / r) if (r * k - m) > 0.0 else 0.0
    return t_vals, n_vals, equilibrium
