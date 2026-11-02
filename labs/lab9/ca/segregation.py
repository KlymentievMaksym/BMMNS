import numpy as np
from labs.lab8.common.cellular_automaton import SynchronousCA


class SchellingSegregationCA(SynchronousCA):
    """Schelling's model of racial segregation."""

    def __init__(
        self,
        shape: tuple[int, int] = (50, 50),
        empty_ratio: float = 0.25,
        similarity_threshold: float = 0.5,
        boundary: str = "periodic",
    ) -> None:
        self.similarity_threshold = similarity_threshold
        total_cells = shape[0] * shape[1]
        num_empty = int(total_cells * empty_ratio)
        num_group_a = (total_cells - num_empty) // 2
        num_group_b = total_cells - num_empty - num_group_a

        flat_grid = np.array(
            [0] * num_empty + [1] * num_group_a + [2] * num_group_b, dtype=np.int32
        )
        np.random.shuffle(flat_grid)
        grid = flat_grid.reshape(shape)

        super().__init__(
            shape=shape,
            dtype=np.dtype(np.int32),
            grid=grid,
            neighborhood="moore",
            boundary=boundary,
        )

    def step(self) -> None:
        unhappy_coords = []
        empty_coords = []

        for coords in np.ndindex(self.shape):
            val = self.grid[coords]
            if val == 0:
                empty_coords.append(coords)
            else:
                neighbors = self.get_neighbor_values(coords)
                occupied = neighbors[neighbors != 0]
                if len(occupied) > 0:
                    same_ratio = np.sum(occupied == val) / len(occupied)
                    if same_ratio < self.similarity_threshold:
                        unhappy_coords.append(coords)

        np.random.shuffle(empty_coords)
        for u_coords in unhappy_coords:
            if not empty_coords:
                break
            e_coords = empty_coords.pop(0)
            self.grid[e_coords] = self.grid[u_coords]
            self.grid[u_coords] = 0
            empty_coords.append(u_coords)

    def apply_rule(self, coords: tuple[int, ...]) -> int:
        return int(self.grid[coords])