import numpy as np


def exponential_growth(y0: float, a: float, steps: int) -> np.ndarray:
    """Model exponential population growth: y_t = (1 + a) * y_{t-1}."""
    y = np.zeros(steps + 1, dtype=np.float64)
    y[0] = y0
    for t in range(1, steps + 1):
        y[t] = (1.0 + a) * y[t - 1]
    return y


def logistic_evolution(
    n0: float, r: float, k: float, m: float, dt: float, steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Model logistic population evolution: dN/dt = r*N*(K - N) - m*N."""
    t_vals = np.linspace(0, steps * dt, steps + 1)
    n_vals = np.zeros(steps + 1, dtype=np.float64)
    n_vals[0] = n0

    for i in range(steps):
        n = n_vals[i]
        dn = r * n * (k - n) - m * n
        n_vals[i + 1] = max(0.0, n + dn * dt)

    return t_vals, n_vals