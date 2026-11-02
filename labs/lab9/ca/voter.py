import numpy as np
from labs.lab8.common.cellular_automaton import AsynchronousCA


class VoterModel(AsynchronousCA):
    """Voter Model for opinion dynamics using asynchronous cellular automaton."""

    def __init__(
        self,
        shape: tuple[int, ...],
        rho: float = 0.5,
        boundary: str = "periodic",
    ) -> None:
        grid = np.where(np.random.rand(*shape) < rho, 1, -1).astype(np.int32)
        super().__init__(
            shape=shape,
            dtype=np.dtype(np.int32),
            grid=grid,
            neighborhood="moore",
            boundary=boundary,
        )

    def apply_rule_at(self, coords: tuple[int, ...]) -> None:
        neighbors = self.get_neighbor_values(coords)
        chosen_neighbor = np.random.choice(neighbors)
        self.grid[coords] = chosen_neighbor

    def is_consensus(self) -> bool:
        first_val = self.grid.ravel()[0]
        return bool(np.all(self.grid == first_val))

    def run_until_consensus(self, max_steps: int = 500000) -> int:
        steps = 0
        while not self.is_consensus() and steps < max_steps:
            self.step()
            steps += 1
        return steps