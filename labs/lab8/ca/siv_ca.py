import numpy as np

from labs.lab8.ca.sir_ca import SIRCellularAutomaton
from labs.lab8.common.boundary import BoundaryType


class SIVCellularAutomaton(SIRCellularAutomaton):
    """SIV model: SIR rules plus stochastic vaccination.

    States: 0 = S, 1..recovery_threshold-1 = I (infection age), 11 = V.
    Recovered cells become immune and are counted in V.
    """

    VACCINATED_STATE = 11
    RECOVERY_TARGET = VACCINATED_STATE

    def __init__(
        self,
        shape: tuple[int, ...] = (50, 50),
        recovery_threshold: int = 5,
        initial_infected_ratio: float = 0.02,
        infection_threshold: int = 3,
        vaccination_rate: float = 0.02,
        breakthrough_prob: float = 0.05,
        boundary: BoundaryType = BoundaryType.PERIODIC,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not 0.0 <= vaccination_rate <= 1.0:
            raise ValueError("vaccination_rate must be in [0, 1]")
        if not 0.0 <= breakthrough_prob <= 1.0:
            raise ValueError("breakthrough_prob must be in [0, 1]")
        self.vaccination_rate = vaccination_rate
        self.breakthrough_prob = breakthrough_prob
        super().__init__(
            shape=shape,
            recovery_threshold=recovery_threshold,
            initial_infected_ratio=initial_infected_ratio,
            infection_threshold=infection_threshold,
            boundary=boundary,
            rng=rng,
        )

    def apply_rule(self, coords: tuple[int, ...]) -> int:
        state = int(self.grid[coords])

        if state == self.VACCINATED_STATE:
            # leaky vaccine: roll the cheap random check first, then look at neighbours
            if self.rng.random() < self.breakthrough_prob and self._exposed(coords):
                return 1
            return state

        if state == self.SUSCEPTIBLE:
            if self._exposed(coords):
                return 1
            if self.rng.random() < self.vaccination_rate:
                return self.VACCINATED_STATE
            return self.SUSCEPTIBLE

        # infected cells: reuse the SIR ageing/recovery logic
        return super().apply_rule(coords)

    def get_counts(self) -> tuple[int, int, int]:
        """Return (susceptible, infected, vaccinated/immune) cell counts."""
        return (
            int(np.count_nonzero(self.grid == self.SUSCEPTIBLE)),
            int(np.count_nonzero(self._is_infected(self.grid))),
            int(np.count_nonzero(self.grid == self.VACCINATED_STATE)),
        )
