from collections.abc import Callable

import numpy as np


def euler_step(
    system_fn: Callable[..., np.ndarray],
    state: np.ndarray,
    t: float,
    dt: float,
    **params,
) -> np.ndarray:
    """Perform a single integration step using Euler's method."""
    return state + dt * system_fn(t, state, **params)


def rk4_step(
    system_fn: Callable[..., np.ndarray],
    state: np.ndarray,
    t: float,
    dt: float,
    **params,
) -> np.ndarray:
    """Perform a single integration step using 4th order Runge-Kutta method."""
    k1 = system_fn(t, state, **params)
    k2 = system_fn(t + 0.5 * dt, state + 0.5 * dt * k1, **params)
    k3 = system_fn(t + 0.5 * dt, state + 0.5 * dt * k2, **params)
    k4 = system_fn(t + dt, state + dt * k3, **params)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def solve_ode(
    system_fn: Callable[..., np.ndarray],
    initial_state: np.ndarray,
    t_max: float,
    dt: float,
    method: str = "rk4",
    **params,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate system of ODEs from t=0 to t_max."""
    step_fn = rk4_step if method.lower() == "rk4" else euler_step
    t_values = np.arange(0.0, t_max + dt, dt)
    history = np.zeros((len(t_values), len(initial_state)), dtype=np.float64)
    history[0] = initial_state

    state = np.array(initial_state, dtype=np.float64)
    for i in range(1, len(t_values)):
        state = step_fn(system_fn, state, t_values[i - 1], dt, **params)
        history[i] = state

    return t_values, history
