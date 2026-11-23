import numpy as np

from labs.lab8.common.boundary import BoundaryType
from labs.lab8.common.cellular_automaton import AsynchronousCA, CellularAutomaton


class SeededAsynchronousCA(AsynchronousCA):
    """Asynchronous CA with explicit RNG state and cached boundary-aware neighbors."""

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype,
        grid: np.ndarray | None = None,
        neighborhood: str = "moore",
        boundary: str = "periodic",
        seed: int | None = None,
    ) -> None:
        super().__init__(
            shape=shape,
            dtype=dtype,
            grid=grid,
            neighborhood=neighborhood,
            boundary=boundary,
        )
        self.rng = np.random.default_rng(seed)
        self._shape_arr = np.asarray(self.shape, dtype=np.int32)
        self._neighbor_cache: dict[tuple[int, ...], list[tuple[int, ...]]] = {
            coords: self._compute_valid_neighbors(coords)
            for coords in np.ndindex(self.shape)
        }

    def _compute_valid_neighbors(
        self, coords: tuple[int, ...]
    ) -> list[tuple[int, ...]]:
        coords_arr = np.asarray(coords, dtype=np.int32)
        raw_neighbors = coords_arr + self.offsets
        if self.boundary == BoundaryType.PERIODIC:
            wrapped = raw_neighbors % self._shape_arr
            return [tuple(int(x) for x in row) for row in wrapped]

        inside = np.all(
            (raw_neighbors >= 0) & (raw_neighbors < self._shape_arr), axis=1
        )
        return [tuple(int(x) for x in row) for row in raw_neighbors[inside]]

    def step(self) -> None:
        random_coords = tuple(
            int(self.rng.integers(0, dim_size)) for dim_size in self.shape
        )
        self.apply_rule_at(random_coords)

    def get_valid_neighbor_coords(
        self, coords: tuple[int, ...]
    ) -> list[tuple[int, ...]]:
        return self._neighbor_cache[coords]


__all__ = ["CellularAutomaton", "SeededAsynchronousCA"]
