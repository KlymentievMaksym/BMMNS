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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float] | None, bool, str]:
    """Simulate Richardson arms race ODEs with full stability regime classification."""
    det = m * n - a * b
    is_stable = bool(det > 0.0 and (m + n) > 0.0)

    equilibrium: tuple[float, float] | None = None
    if abs(det) > 1e-9:
        eq_x = float((a * s + r * n) / det)
        eq_y = float((b * r + s * m) / det)
        if eq_x >= 0.0 and eq_y >= 0.0:
            equilibrium = (eq_x, eq_y)

    if det > 0.0 and r > 0.0 and s > 0.0:
        regime = "Stable balance of power"
    elif det < 0.0 and r > 0.0 and s > 0.0:
        regime = "Runaway arms escalation"
    elif det > 0.0 and r < 0.0 and s < 0.0:
        regime = "Mutual disarmament"
    elif det < 0.0 and r < 0.0 and s < 0.0:
        regime = "Saddle regime (depends on initial arms)"
    else:
        regime = "Boundary / asymmetric regime"

    t_vals = np.linspace(0.0, steps * dt, steps + 1)
    x_vals = np.zeros(steps + 1, dtype=np.float64)
    y_vals = np.zeros(steps + 1, dtype=np.float64)
    x_vals[0] = x0
    y_vals[0] = y0

    def derivatives(x_cur: float, y_cur: float) -> tuple[float, float]:
        dx = a * y_cur - m * x_cur + r
        dy = b * x_cur - n * y_cur + s
        if x_cur <= 0.0 and dx < 0.0:
            dx = 0.0
        if y_cur <= 0.0 and dy < 0.0:
            dy = 0.0
        return dx, dy

    for i in range(steps):
        x, y = x_vals[i], y_vals[i]
        dx1, dy1 = derivatives(x, y)
        dx2, dy2 = derivatives(x + 0.5 * dt * dx1, y + 0.5 * dt * dy1)
        dx3, dy3 = derivatives(x + 0.5 * dt * dx2, y + 0.5 * dt * dy2)
        dx4, dy4 = derivatives(x + dt * dx3, y + dt * dy3)
        x_vals[i + 1] = max(0.0, x + (dt / 6.0) * (dx1 + 2.0 * dx2 + 2.0 * dx3 + dx4))
        y_vals[i + 1] = max(0.0, y + (dt / 6.0) * (dy1 + 2.0 * dy2 + 2.0 * dy3 + dy4))

    return t_vals, x_vals, y_vals, equilibrium, is_stable, regime


def fit_wwi_historical_data() -> dict[str, np.ndarray | float]:
    """Validate aggregate Richardson equation on 1909-1913 defense expenditures."""
    france = np.array([48.6, 50.9, 57.1, 63.2, 74.7], dtype=np.float64)
    russia = np.array([66.7, 68.5, 70.7, 81.8, 92.7], dtype=np.float64)
    germany = np.array([63.1, 62.0, 62.0, 68.2, 95.4], dtype=np.float64)
    austria = np.array([20.8, 23.4, 23.4, 25.5, 26.9], dtype=np.float64)

    total_z = france + russia + germany + austria
    mean_z = 0.5 * (total_z[:-1] + total_z[1:])
    growth_dz = np.diff(total_z)
    slope_k, intercept_f = np.polyfit(mean_z, growth_dz, deg=1)

    return {
        "mean_z": mean_z,
        "growth_dz": growth_dz,
        "k": float(slope_k),
        "f": float(intercept_f),
    }
