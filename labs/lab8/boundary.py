from enum import Enum

import numpy as np

from labs.lab8.interfaces import BoundaryApply

class BoundaryFunc(BoundaryApply):

def periodic_boundary(self, coords: tuple[int, ...]) -> np.ndarray:
        coords_arr = np.array(coords, dtype=np.int32)
        neighbor_coords = coords_arr + self.offsets

        if self.boundary == "periodic":
            shape_arr = np.array(self.shape, dtype=np.int32)
            neighbor_coords = neighbor_coords % shape_arr

        return neighbor_coords

class BoundaryType(str, Enum):
    PERIODIC = "periodic"
    FIXED = "fixed"

BOUNDARY_TO_FUNC: dict[BoundaryType, BoundaryApply] = {
    BoundaryType.PERIODIC: periodic_boundary,
    BoundaryType.FIXED: periodic_boundary,
}
