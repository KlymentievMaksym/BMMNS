import numpy as np
from labs.lab8.common.cellular_automaton import AsynchronousCA


class SznajdCA(AsynchronousCA):
    """Sznajd opinion model: United we stand, divided we fall."""

    def __init__(
        self,
        shape: tuple[int, int] = (50, 50),
        bias_ratio: float = 0.51,
        boundary: str = "periodic",
    ) -> None:
        grid = np.where(np.random.rand(*shape) < bias_ratio, 1, -1).astype(np.int32)
        super().__init__(
            shape=shape,
            dtype=np.dtype(np.int32),
            grid=grid,
            neighborhood="moore",
            boundary=boundary,
        )

    def apply_rule_at(self, coords: tuple[int, ...]) -> None:
        offsets = self.offsets
        rand_offset = offsets[np.random.choice(len(offsets))]
        neighbor_coords = tuple((np.array(coords) + rand_offset) % np.array(self.shape))

        val1 = self.grid[coords]
        val2 = self.grid[neighbor_coords]

        if val1 == val2:
            for offset in offsets:
                n_coords = tuple((np.array(coords) + offset) % np.array(self.shape))
                if n_coords != neighbor_coords:
                    self.grid[n_coords] = val1

    def is_consensus(self) -> bool:
        first_val = self.grid.ravel()[0]
        return bool(np.all(self.grid == first_val))