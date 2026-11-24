import numpy as np


def exponential_discrete(y0: float, rate: float, steps: int) -> np.ndarray:
    """y_t = (1 + a) * y_{t-1}  =>  y_t = y0 * (1 + a)^t."""
    return y0 * (1.0 + rate) ** np.arange(steps + 1)


def exponential_system(t: float, state: np.ndarray, rate: float) -> np.ndarray:
    """Continuous counterpart of the discrete model: dy/dt = ln(1 + a) * y."""
    return np.log1p(rate) * state


def logistic_system(
    t: float, state: np.ndarray, r: float, k: float, m: float
) -> np.ndarray:
    """dN/dt = r*N*(K - N) - m*N, stable stationary state Nc = K - m/r."""
    return r * state * (k - state) - m * state


def richardson_system(
    t: float,
    state: np.ndarray,
    a: float,
    m: float,
    r: float,
    b: float,
    n: float,
    s: float,
) -> np.ndarray:
    """dx/dt = a*y - m*x + r,  dy/dt = b*x - n*y + s (spending cannot go below 0)."""
    x, y = state
    dx = a * y - m * x + r
    dy = b * x - n * y + s
    if x <= 0.0 and dx < 0.0:
        dx = 0.0
    if y <= 0.0 and dy < 0.0:
        dy = 0.0
    return np.array([dx, dy], dtype=np.float64)
