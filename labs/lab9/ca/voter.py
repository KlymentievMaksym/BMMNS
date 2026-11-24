import numpy as np

from labs.lab9.ca import SeededAsynchronousCA


class VoterModel(SeededAsynchronousCA):
    """Voter model: a random voter copies the opinion of a random neighbour."""

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
        total = int(np.prod(shape))
        n_plus = int(round(rho * total))  # exact density, no Bernoulli jitter
        flat = np.full(total, -1, dtype=np.int32)
        flat[:n_plus] = 1
        self.rng.shuffle(flat)
        self.grid = flat.reshape(shape)
        self._plus = n_plus

    def apply_rule_at(self, coords: tuple[int, ...]) -> None:
        neighbors = self.get_valid_neighbor_coords(coords)
        if not neighbors:
            return
        chosen = neighbors[int(self.rng.integers(0, len(neighbors)))]
        new, old = int(self.grid[chosen]), int(self.grid[coords])
        if new != old:
            self._plus += new  # +1 -> one more "+", -1 -> one less
            self.grid[coords] = new

    def is_consensus(self) -> bool:
        return self._plus in (0, self.grid.size)

    def run_until_consensus(self, max_steps: int = 500_000) -> tuple[int, int]:
        steps = 0
        while not self.is_consensus() and steps < max_steps:
            self.step()
            steps += 1
        winner = (1 if self._plus else -1) if self.is_consensus() else 0
        return winner, steps
