import numpy as np


def richardson_arms_race(
    x0: float,
    y0: float,
    a: float,
    m: float,
    r: float,
    b: float,
    n: float,
    s: float,
    dt: float = 0.01,
    steps: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float] | None, bool]:
    """Simulate Richardson arms race model:

    dx/dt = a*y - m*x + r
    dy/dt = b*x - n*y + s
    """
    det = m * n - a * b
    is_stable = det > 0

    equilibrium = None
    if abs(det) > 1e-9:
        eq_x = (a * s + r * n) / det
        eq_y = (b * r + s * m) / det
        equilibrium = (eq_x, eq_y)

    t_vals = np.linspace(0, steps * dt, steps + 1)
    x_vals = np.zeros(steps + 1, dtype=np.float64)
    y_vals = np.zeros(steps + 1, dtype=np.float64)
    x_vals[0] = x0
    y_vals[0] = y0

    for i in range(steps):
        x, y = x_vals[i], y_vals[i]
        dx = a * y - m * x + r
        dy = b * x - n * y + s
        x_vals[i + 1] = max(0.0, x + dx * dt)
        y_vals[i + 1] = max(0.0, y + dy * dt)

    return t_vals, x_vals, y_vals, equilibrium, is_stable