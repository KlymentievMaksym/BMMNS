from enum import Enum

import numpy as np

from .interfaces import BoundaryApply


def periodic_boundary(
    grid: np.ndarray,
    coords: tuple[int, ...],
    offsets: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    coords_arr = np.array(coords)
    shape_arr = np.array(shape)

    neighbor_coords = (coords_arr + offsets) % shape_arr

    return grid[tuple(neighbor_coords.T)]


def fixed_boundary(grid, coords, offsets, shape, fill_value=0) -> np.ndarray:
    neighbor_coords = np.asarray(coords) + offsets

    inside = np.all(
        (neighbor_coords >= 0) & (neighbor_coords < np.asarray(shape)), axis=1
    )
    values = np.full(len(offsets), fill_value, dtype=grid.dtype)
    values[inside] = grid[tuple(neighbor_coords[inside].T)]

    return values


class BoundaryType(str, Enum):
    PERIODIC = "periodic"
    FIXED = "fixed"


BOUNDARY_TO_FUNC: dict[BoundaryType, BoundaryApply] = {
    BoundaryType.PERIODIC: periodic_boundary,
    BoundaryType.FIXED: fixed_boundary,
}
