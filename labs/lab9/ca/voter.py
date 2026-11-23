import numpy as np

from labs.lab9.ca import SeededAsynchronousCA


class VoterModel(SeededAsynchronousCA):
    """Voter Model for opinion dynamics using asynchronous cellular automaton."""

    def __init__(
        self,
        shape: tuple[int, ...],
        rho: float = 0.5,
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
        self.grid = np.where(self.rng.random(shape) < rho, 1, -1).astype(np.int32)

    def apply_rule_at(self, coords: tuple[int, ...]) -> None:
        neighbors = self.get_valid_neighbor_coords(coords)
        if neighbors:
            chosen = neighbors[int(self.rng.integers(0, len(neighbors)))]
            self.grid[coords] = self.grid[chosen]

    def is_consensus(self) -> bool:
        first_val = self.grid.ravel()[0]
        return bool(np.all(self.grid == first_val))

    def run_until_consensus(self, max_steps: int = 500_000) -> tuple[int, int]:
        steps = 0
        while not self.is_consensus() and steps < max_steps:
            self.step()
            steps += 1
        winner = int(self.grid.ravel()[0]) if self.is_consensus() else 0
        return winner, steps
