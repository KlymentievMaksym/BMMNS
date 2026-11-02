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


def fixed_boundary(
    grid: np.ndarray,
    coords: tuple[int, ...],
    offsets: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    coords_arr = np.array(coords)
    shape_arr = np.array(shape)

    neighbor_coords = coords_arr + offsets

    # Clip coordinates to remain within grid boundaries
    neighbor_coords = np.clip(neighbor_coords, 0, shape_arr - 1)

    return grid[tuple(neighbor_coords.T)]


class BoundaryType(str, Enum):
    PERIODIC = "periodic"
    FIXED = "fixed"


BOUNDARY_TO_FUNC: dict[BoundaryType, BoundaryApply] = {
    BoundaryType.PERIODIC: periodic_boundary,
    BoundaryType.FIXED: fixed_boundary,
}
