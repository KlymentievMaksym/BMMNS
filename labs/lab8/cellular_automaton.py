from abc import ABC, abstractmethod

import numpy as np

from labs.lab8.boundary import BoundaryType
from labs.lab8.neighborhood import NeighborhoodType, NEIGHBORHOOD_TO_FUNC


class CellularAutomaton(ABC):
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype,
        grid: np.ndarray | None = None,
        neighborhood: str = NeighborhoodType.MOORE,
        boundary: str = BoundaryType.PERIODIC,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.boundary = boundary
        self.neighborhood = NeighborhoodType(neighborhood)

        if grid is not None:
            self.grid = np.array(grid, dtype=dtype)
        else:
            self.grid = np.zeros(shape=shape, dtype=dtype)

        self.offsets = NEIGHBORHOOD_TO_FUNC[self.neighborhood](self.shape)

    def get_neighbor_coords(self, coords: tuple[int, ...]) -> np.ndarray:
        coords_arr = np.array(coords, dtype=np.int32)
        neighbor_coords = coords_arr + self.offsets

        if self.boundary == "periodic":
            shape_arr = np.array(self.shape, dtype=np.int32)
            neighbor_coords = neighbor_coords % shape_arr

        return neighbor_coords

    def get_neighbor_values(self, coords: tuple[int, ...]) -> np.ndarray:
        neighbor_coords = self.get_neighbor_coords(coords)
        return self.grid[tuple(neighbor_coords.T)]

    @abstractmethod
    def step(self) -> None:
        pass


class SynchronousCA(CellularAutomaton):
    def step(self) -> None:
        new_grid = np.empty_like(self.grid)
        for coords in np.ndindex(self.shape):
            new_grid[coords] = self.apply_rule(coords)
        self.grid = new_grid

    @abstractmethod
    def apply_rule(self, coords: tuple[int, ...]) -> int | float:
        pass


class AsynchronousCA(CellularAutomaton):
    def step(self) -> None:
        random_coords = tuple(np.random.randint(0, dim_size) for dim_size in self.shape)
        self.apply_rule_at(random_coords)

    @abstractmethod
    def apply_rule_at(self, coords: tuple[int, ...]) -> None:
        pass
