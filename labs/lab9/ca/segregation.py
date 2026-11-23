import numpy as np

from labs.lab8.common.boundary import BOUNDARY_TO_FUNC, BoundaryType
from labs.lab8.common.cellular_automaton import CellularAutomaton


class SchellingSegregationCA(CellularAutomaton):
    """Schelling's model of racial segregation."""

    def __init__(
        self,
        shape: tuple[int, int] = (50, 50),
        empty_ratio: float = 0.28,
        similarity_threshold: float = 0.5,
        boundary: str = "periodic",
        seed: int | None = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.rng = np.random.default_rng(seed)

        total_cells = shape[0] * shape[1]
        num_empty = int(round(total_cells * empty_ratio))
        num_group_a = (total_cells - num_empty) // 2
        num_group_b = total_cells - num_empty - num_group_a

        flat_grid = np.array(
            [0] * num_empty + [1] * num_group_a + [2] * num_group_b, dtype=np.int32
        )
        self.rng.shuffle(flat_grid)
        grid = flat_grid.reshape(shape)

        super().__init__(
            shape=shape,
            dtype=np.dtype(np.int32),
            grid=grid,
            neighborhood="moore",
            boundary=boundary,
        )
        self._search_offsets = self._build_sorted_offsets()

    def _build_sorted_offsets(self) -> list[tuple[int, int]]:
        rows, cols = self.shape
        candidates: list[tuple[int, int, int]] = []
        for dr in range(-rows + 1, rows):
            for dc in range(-cols + 1, cols):
                if dr != 0 or dc != 0:
                    candidates.append((dr * dr + dc * dc, dr, dc))
        candidates.sort(key=lambda item: item[0])
        return [(dr, dc) for _, dr, dc in candidates]

    def _is_satisfied(
        self,
        coords: tuple[int, int],
        group: int,
        ignore_coords: tuple[int, int] | None = None,
    ) -> bool:
        neighbors = BOUNDARY_TO_FUNC[self.boundary](
            grid=self.grid, coords=coords, offsets=self.offsets, shape=self.shape
        )
        if ignore_coords is not None:
            dr = ignore_coords[0] - coords[0]
            dc = ignore_coords[1] - coords[1]
            if self.boundary == BoundaryType.PERIODIC:
                dr = (dr + self.shape[0] // 2) % self.shape[0] - self.shape[0] // 2
                dc = (dc + self.shape[1] // 2) % self.shape[1] - self.shape[1] // 2
            if abs(dr) <= 1 and abs(dc) <= 1 and (dr != 0 or dc != 0):
                idx = np.where((self.offsets[:, 0] == dr) & (self.offsets[:, 1] == dc))[
                    0
                ]
                if len(idx) > 0:
                    neighbors = np.delete(neighbors, idx[0])

        occupied = neighbors[neighbors != 0]
        if len(occupied) == 0:
            return True
        same_ratio = float(np.sum(occupied == group)) / len(occupied)
        return same_ratio >= (self.similarity_threshold - 1e-12)

    def step(self) -> int:
        unhappy_coords: list[tuple[int, int]] = []
        for r in range(self.shape[0]):
            for c in range(self.shape[1]):
                val = int(self.grid[r, c])
                if val != 0 and not self._is_satisfied((r, c), val):
                    unhappy_coords.append((r, c))

        self.rng.shuffle(unhappy_coords)
        moves_made = 0
        rows, cols = self.shape

        for r, c in unhappy_coords:
            group = int(self.grid[r, c])
            if group == 0 or self._is_satisfied((r, c), group):
                continue

            for dr, dc in self._search_offsets:
                if self.boundary == BoundaryType.PERIODIC:
                    nr = (r + dr) % rows
                    nc = (c + dc) % cols
                else:
                    nr = r + dr
                    nc = c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        continue

                if self.grid[nr, nc] == 0 and self._is_satisfied(
                    (nr, nc), group, ignore_coords=(r, c)
                ):
                    self.grid[nr, nc] = group
                    self.grid[r, c] = 0
                    moves_made += 1
                    break

        return moves_made

    def segregation_index(self) -> float:
        ratios: list[float] = []
        for r in range(self.shape[0]):
            for c in range(self.shape[1]):
                val = int(self.grid[r, c])
                if val != 0:
                    neighbors = self.get_neighbor_values((r, c))
                    occupied = neighbors[neighbors != 0]
                    if len(occupied) > 0:
                        ratios.append(float(np.sum(occupied == val)) / len(occupied))
        return float(np.mean(ratios)) if ratios else 0.0
