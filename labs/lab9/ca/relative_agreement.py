import numpy as np

from labs.lab9.ca import SeededAsynchronousCA


class RelativeAgreementCA(SeededAsynchronousCA):
    """Relative Agreement Interaction Model with continuous opinion values."""

    def __init__(
        self,
        shape: tuple[int, int] = (50, 50),
        e_threshold: float = 50.0,
        m_influence: float = 0.3,
        opinion_bounds: tuple[float, float] = (1.0, 256.0),
        boundary: str = "periodic",
        seed: int | None = None,
    ) -> None:
        self.e_threshold = e_threshold
        self.m_influence = m_influence
        self.opinion_bounds = opinion_bounds
        super().__init__(
            shape=shape,
            dtype=np.dtype(np.float64),
            grid=None,
            neighborhood="moore",
            boundary=boundary,
            seed=seed,
        )
        low, high = opinion_bounds
        self.grid = self.rng.uniform(low, high, size=shape).astype(np.float64)
        self.initial_grid = self.grid.copy()

    def apply_rule_at(self, coords: tuple[int, ...]) -> None:
        neighbors = self.get_valid_neighbor_coords(coords)
        if not neighbors:
            return
        neighbor_coords = neighbors[int(self.rng.integers(0, len(neighbors)))]

        oi = float(self.grid[coords])
        oj = float(self.grid[neighbor_coords])

        if abs(oi - oj) < self.e_threshold:
            shift = (self.m_influence / 2.0) * (oi - oj)
            low, high = self.opinion_bounds
            self.grid[coords] = np.clip(oi - shift, low, high)
            self.grid[neighbor_coords] = np.clip(oj + shift, low, high)

    def count_clusters(self, gap_tol: float = 2.0) -> int:
        sorted_vals = np.sort(self.grid.ravel())
        return int(1 + np.sum(np.diff(sorted_vals) > gap_tol))

    def mean_opinion_shift(self) -> float:
        return float(np.mean(np.abs(self.grid - self.initial_grid)))
