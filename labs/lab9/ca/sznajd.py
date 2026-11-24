import numpy as np

from labs.lab9.ca import SeededAsynchronousCA


class SznajdCA(SeededAsynchronousCA):
    """Sznajd model: if a pair agrees, all their neighbours adopt that opinion."""

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
        total = int(np.prod(shape))
        n_plus = min(max(int(round(total * (0.5 + 0.5 * advantage))), 1), total - 1)
        flat = np.full(total, -1, dtype=np.int32)
        flat[:n_plus] = 1
        self.rng.shuffle(flat)
        self.grid = flat.reshape(shape)
        self._plus = n_plus

    def apply_rule_at(self, coords: tuple[int, ...]) -> None:
        neighbors = self.get_valid_neighbor_coords(coords)
        if not neighbors:
            return
        partner = neighbors[int(self.rng.integers(0, len(neighbors)))]
        value = int(self.grid[coords])
        if value != int(self.grid[partner]):
            return

        targets = set(neighbors) | set(self.get_valid_neighbor_coords(partner))
        targets.discard(coords)
        targets.discard(partner)
        for target in targets:
            if int(self.grid[target]) != value:
                self._plus += value
                self.grid[target] = value

    def is_consensus(self) -> bool:
        return self._plus in (0, self.grid.size)

    def run_until_consensus(self, max_steps: int = 2_000_000) -> tuple[int, int]:
        steps = 0
        while not self.is_consensus() and steps < max_steps:
            self.step()
            steps += 1
        winner = (1 if self._plus else -1) if self.is_consensus() else 0
        return winner, steps
