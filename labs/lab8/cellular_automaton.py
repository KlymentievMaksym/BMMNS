from abc import ABC, abstractmethod

import numpy as np

from labs.lab8.boundary import BOUNDARY_TO_FUNC, BoundaryType
from labs.lab8.neighborhood import NEIGHBORHOOD_TO_FUNC, NeighborhoodType


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
        self.neighborhood = NeighborhoodType(neighborhood)
        self.boundary = BoundaryType(boundary)

        if grid is not None:
            self.grid = np.array(grid, dtype=dtype)
        else:
            self.grid = np.zeros(shape=shape, dtype=dtype)

        self.offsets = np.array(
            NEIGHBORHOOD_TO_FUNC[self.neighborhood](self.shape), dtype=dtype
        )

    def get_neighbor_values(self, coords: tuple[int, ...]) -> np.ndarray:
        return BOUNDARY_TO_FUNC[self.boundary](
            grid=self.grid, coords=coords, offsets=self.offsets, shape=self.shape
        )

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
