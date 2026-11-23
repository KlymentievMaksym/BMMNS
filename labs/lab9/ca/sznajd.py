import numpy as np

from labs.lab9.ca import SeededAsynchronousCA


class SznajdCA(SeededAsynchronousCA):
    """Sznajd opinion model: United we stand, divided we fall."""

    def __init__(
        self,
        shape: tuple[int, ...] = (50, 50),
        advantage: float = 0.01,
        boundary: str = "periodic",
        seed: int | None = None,
    ) -> None:
        super().__init__(
            shape=shape,
            dtype=np.dtype(np.int32),
            grid=None,
            neighborhood="moore",
            boundary=boundary,
            seed=seed,
        )
        total_cells = int(np.prod(shape))
        n_plus = int(round(total_cells * (0.5 + 0.5 * advantage)))
        n_plus = min(max(n_plus, 1), total_cells - 1)

        flat = np.full(total_cells, -1, dtype=np.int32)
        flat[:n_plus] = 1
        self.rng.shuffle(flat)
        self.grid = flat.reshape(shape)

    def apply_rule_at(self, coords: tuple[int, ...]) -> None:
        neighbors = self.get_valid_neighbor_coords(coords)
        if not neighbors:
            return
        neighbor_coords = neighbors[int(self.rng.integers(0, len(neighbors)))]

        val1 = int(self.grid[coords])
        val2 = int(self.grid[neighbor_coords])

        if val1 == val2:
            targets = set(neighbors) | set(
                self.get_valid_neighbor_coords(neighbor_coords)
            )
            targets.discard(coords)
            targets.discard(neighbor_coords)
            for target in targets:
                self.grid[target] = val1

    def is_consensus(self) -> bool:
        first_val = self.grid.ravel()[0]
        return bool(np.all(self.grid == first_val))

    def run_until_consensus(self, max_steps: int = 2_000_000) -> tuple[int, int]:
        steps = 0
        while not self.is_consensus() and steps < max_steps:
            self.step()
            steps += 1
        winner = int(self.grid.ravel()[0]) if self.is_consensus() else 0
        return winner, steps
