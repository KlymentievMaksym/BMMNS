from enum import Enum

import numpy as np

from labs.lab8.interfaces import BoundaryApply


def periodic_boundary(
    grid: np.ndarray,
    coords: tuple[int, ...],
    offsets: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    coords_arr = np.array(coords)

    neighbor_coords = coords_arr + offsets

    return grid[tuple(neighbor_coords.T)]


def fixed_boundary(
    grid: np.ndarray,
    coords: tuple[int, ...],
    offsets: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    coords_arr = np.array(coords)

    neighbor_coords = coords_arr + offsets

    shape_arr = np.array(shape)
    neighbor_coords = neighbor_coords % shape_arr

    return grid[tuple(neighbor_coords.T)]


class BoundaryType(str, Enum):
    PERIODIC = "periodic"
    FIXED = "fixed"


BOUNDARY_TO_FUNC: dict[BoundaryType, BoundaryApply] = {
    BoundaryType.PERIODIC: periodic_boundary,
    BoundaryType.FIXED: fixed_boundary,
}
