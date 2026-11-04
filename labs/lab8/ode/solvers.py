from collections.abc import Callable
from dataclasses import asdict
from enum import Enum

import numpy as np

SystemFn = Callable[..., np.ndarray]


class Method(str, Enum):
    EULER = "euler"
    RK4 = "rk4"


def euler_step(system_fn: SystemFn, state, t, dt, **params) -> np.ndarray:
    return state + dt * system_fn(t, state, **params)


def rk4_step(system_fn: SystemFn, state, t, dt, **params) -> np.ndarray:
    k1 = system_fn(t, state, **params)
    k2 = system_fn(t + 0.5 * dt, state + 0.5 * dt * k1, **params)
    k3 = system_fn(t + 0.5 * dt, state + 0.5 * dt * k2, **params)
    k4 = system_fn(t + dt, state + dt * k3, **params)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


STEPPERS: dict[Method, Callable[..., np.ndarray]] = {
    Method.EULER: euler_step,
    Method.RK4: rk4_step,
}


def solve_ode(
    system_fn: SystemFn,
    initial_state: tuple[float, ...],
    params,
    t_max: float,
    dt: float,
    method: Method = Method.RK4,
) -> tuple[np.ndarray, np.ndarray]:

    step_fn = STEPPERS[method]
    kwargs = asdict(params)

    n_steps = int(round(t_max / dt))
    t_values = np.arange(n_steps + 1) * dt

    state = np.asarray(initial_state, dtype=np.float64)
    history = np.empty((n_steps + 1, state.size), dtype=np.float64)
    history[0] = state

    for i in range(n_steps):
        state = step_fn(system_fn, state, t_values[i], dt, **kwargs)
        history[i + 1] = state

    return t_values, history
