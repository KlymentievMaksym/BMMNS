import numpy as np

from labs.lab8.common.cellular_automaton import SynchronousCA


class SIRCellularAutomaton(SynchronousCA):
    """Stochastic SIR epidemic model based on synchronous cellular automaton."""

    def __init__(
        self,
        shape: tuple[int, int] = (50, 50),
        recovery_threshold: int = 5,
        initial_infected_ratio: float = 0.02,
        boundary: str = "periodic",
    ) -> None:
        self.recovery_threshold = recovery_threshold
        self.recovered_state = 10

        grid = np.zeros(shape, dtype=np.int32)
        total_cells = shape[0] * shape[1]
        num_infected = max(1, int(total_cells * initial_infected_ratio))

        infected_indices = np.random.choice(
            total_cells, size=num_infected, replace=False
        )
        flat_grid = grid.ravel()
        flat_grid[infected_indices] = 1

        super().__init__(
            shape=shape,
            dtype=np.dtype(np.int32),
            grid=grid,
            neighborhood="moore",
            boundary=boundary,
        )

    def apply_rule(self, coords: tuple[int, ...]) -> int:
        current_state = int(self.grid[coords])

        if current_state == self.recovered_state:
            return self.recovered_state

        if 1 <= current_state < self.recovery_threshold:
            next_state = current_state + 1
            if next_state >= self.recovery_threshold:
                return self.recovered_state
            return next_state

        if current_state == 0:
            neighbor_values = self.get_neighbor_values(coords)
            infected_count = np.sum(
                (neighbor_values >= 1) & (neighbor_values < self.recovery_threshold)
            )
            if infected_count >= 3:
                return 1
            return 0

        return current_state

    def get_counts(self) -> tuple[int, int, int]:
        """Return current count of Susceptible, Infected, and Recovered cells."""
        susceptible = int(np.sum(self.grid == 0))
        infected = int(np.sum((self.grid >= 1) & (self.grid < self.recovery_threshold)))
        recovered = int(np.sum(self.grid == self.recovered_state))
        return susceptible, infected, recovered
