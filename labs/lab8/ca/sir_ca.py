import numpy as np

from labs.lab8.common.boundary import BoundaryType
from labs.lab8.common.cellular_automaton import SynchronousCA
from labs.lab8.common.neighborhood import NeighborhoodType


class SIRCellularAutomaton(SynchronousCA):
    SUSCEPTIBLE = 0
    RECOVERED_STATE = 10
    RECOVERY_TARGET = RECOVERED_STATE

    def __init__(
        self,
        shape: tuple[int, ...] = (50, 50),
        recovery_threshold: int = 5,
        initial_infected_ratio: float = 0.02,
        infection_threshold: int = 3,
        boundary: BoundaryType = BoundaryType.PERIODIC,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not 2 <= recovery_threshold < self.RECOVERED_STATE:
            raise ValueError(
                f"recovery_threshold must be in [2, {self.RECOVERED_STATE})"
            )
        self.recovery_threshold = recovery_threshold
        self.infection_threshold = infection_threshold

        self.rng = rng or np.random.default_rng()
        grid = np.zeros(shape, dtype=np.int32)
        num_infected = max(1, int(grid.size * initial_infected_ratio))
        grid.flat[self.rng.choice(grid.size, size=num_infected, replace=False)] = 1

        super().__init__(
            shape=shape,
            dtype=np.dtype(np.int32),
            grid=grid,
            neighborhood=NeighborhoodType.MOORE,
            boundary=boundary,
        )

    def _is_infected(self, values: np.ndarray) -> np.ndarray:
        return (values >= 1) & (values < self.recovery_threshold)

    def _exposed(self, coords: tuple[int, ...]) -> np.bool_:
        """True if enough neighbours are infected."""
        n = np.count_nonzero(self._is_infected(self.get_neighbor_values(coords)))
        return n >= self.infection_threshold

    def apply_rule(self, coords: tuple[int, ...]) -> int:
        state = int(self.grid[coords])

        if state == self.RECOVERED_STATE:
            return state

        if state == self.SUSCEPTIBLE:
            return 1 if self._exposed(coords) else self.SUSCEPTIBLE

        state += 1
        return self.RECOVERY_TARGET if state >= self.recovery_threshold else state

    def get_counts(self) -> tuple[int, int, int]:
        """Return (susceptible, infected, recovered) cell counts."""
        return (
            int(np.count_nonzero(self.grid == self.SUSCEPTIBLE)),
            int(np.count_nonzero(self._is_infected(self.grid))),
            int(np.count_nonzero(self.grid == self.RECOVERED_STATE)),
        )
