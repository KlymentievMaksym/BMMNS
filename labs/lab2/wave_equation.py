from collections.abc import Sequence

import numpy as np
from boundaries import BOUNDARY_FUNCTIONS
from entities import WaveSimConfig
from interfaces import WaveSource


def step_wave_equation(
        hx_coeff: float,
        hy_coeff: float,
        U_past: np.ndarray,
        U_curr: np.ndarray,
        U_next: np.ndarray
    ) -> np.ndarray:

    U_next[1:-1, 1:-1] = (
        2 * U_curr[1:-1, 1:-1] - U_past[1:-1, 1:-1]
        + hx_coeff * (U_curr[2:, 1:-1] - 2 * U_curr[1:-1, 1:-1] + U_curr[0:-2, 1:-1])
        + hy_coeff * (U_curr[1:-1, 2:] - 2 * U_curr[1:-1, 1:-1] + U_curr[1:-1, 0:-2])
    )

    return U_next

def modify_U_next(
        U_next: np.ndarray,
        i: int | list[int],
        j: int | list[int],
        value: float | list[float]
    ) -> None:

    U_next[i, j] = value

def run_simulation(
    sources: WaveSource | Sequence[WaveSource],
    config: WaveSimConfig,
) -> np.ndarray:
    shape = (config.Nx, config.Ny)
    U_past = np.zeros(shape)
    U_curr = np.zeros(shape)
    U_next = np.zeros(shape)
    A_max = np.zeros(shape)

    record_threshold = int(config.K * 0.98)
    i_coords = [pos[0] for pos in config.source_positions]
    j_coords = [pos[1] for pos in config.source_positions]

    sources_list = sources if isinstance(sources, Sequence) else [sources]

    if len(sources_list) != len(config.source_positions):
        raise ValueError("Кількість джерел не відповідає кількості позицій u config.source_positions")

    for k in range(config.K):
        step_wave_equation(
            config.hx_coeff, config.hy_coeff, U_past, U_curr, U_next
        )

        t = (k + 1) * config.delta_t
        values = [src(t) for src in sources_list]

        modify_U_next(
            U_next=U_next,
            i=i_coords,
            j=j_coords,
            value=values
        )

        BOUNDARY_FUNCTIONS[config.boundary_type](U_next)

        if k >= record_threshold:
            np.maximum(A_max, np.abs(U_next), out=A_max)

        U_past, U_curr, U_next = U_curr, U_next, U_past

    return A_max