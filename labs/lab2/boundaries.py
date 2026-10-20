
from enum import Enum

import numpy as np
from interfaces import BoundaryApply


def apply_boundary_conditions_free(U: np.ndarray) -> None:
    U[0, :] = U[1, :]
    U[-1, :] = U[-2, :]
    U[:, 0] = U[:, 1]
    U[:, -1] = U[:, -2]

    U[0, 0] = 0.5 * (U[1, 0] + U[0, 1])
    U[-1, -1] = 0.5 * (U[-2, -1] + U[-1, -2])
    U[0, -1] = 0.5 * (U[0, -2] + U[1, -1])
    U[-1, 0] = 0.5 * (U[-1, 1] + U[-2, 0])

def apply_boundary_conditions_fixed(U: np.ndarray) -> None:
    U[0, :] = 0
    U[-1, :] = 0
    U[:, 0] = 0
    U[:, -1] = 0

class BoundaryType(str, Enum):
    FREE = "free"
    FIXED = "fixed"

BOUNDARY_FUNCTIONS: dict[BoundaryType, BoundaryApply] = {
    BoundaryType.FREE: apply_boundary_conditions_free,
    BoundaryType.FIXED: apply_boundary_conditions_fixed,
}