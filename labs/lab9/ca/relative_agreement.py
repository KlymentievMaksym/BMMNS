import numpy as np


class RelativeAgreementCA:
    """Bounded-confidence opinion model on N agents (well-mixed pairing).

    Two random agents interact only if |Oi - Oj| < e; then both move towards
    each other by m * |Oi - Oj| / 2. Optionally one "zealot" never changes.
    Pairing is global, so the grid is only a container (no spatial structure).
    """

    def __init__(
        self,
        shape: tuple[int, ...] = (50, 50),
        e_threshold: float = 50.0,
        m_influence: float = 0.3,
        opinion_bounds: tuple[float, float] = (1.0, 256.0),
        zealot: tuple[int, ...] | None = None,
        zealot_opinion: float = 256.0,
        seed: int | None = None,
    ) -> None:
        self.shape = shape
        self.e_threshold = e_threshold
        self.m_influence = m_influence
        self.opinion_bounds = opinion_bounds
        self.rng = np.random.default_rng(seed)

        low, high = opinion_bounds
        self.grid = self.rng.uniform(low, high, size=shape)
        self.zealot_idx: int | None = None
        if zealot is not None:
            self.zealot_idx = int(np.ravel_multi_index(zealot, shape))
            self.grid.flat[self.zealot_idx] = zealot_opinion
        self.initial_grid = self.grid.copy()
        self._n = self.grid.size

    def step(self) -> None:
        i = int(self.rng.integers(0, self._n))
        j = int(self.rng.integers(0, self._n))
        if i == j:
            return
        flat = self.grid.reshape(-1)
        oi, oj = flat[i], flat[j]
        if abs(oi - oj) >= self.e_threshold:
            return
        shift = 0.5 * self.m_influence * (oi - oj)
        low, high = self.opinion_bounds
        if i != self.zealot_idx:
            flat[i] = min(max(oi - shift, low), high)
        if j != self.zealot_idx:
            flat[j] = min(max(oj + shift, low), high)

    def count_clusters(self, gap_tol: float = 1.0, min_size_frac: float = 0.01) -> int:
        """Groups of sorted opinions split by gaps > gap_tol; tiny groups are ignored."""
        values = np.sort(self.grid.ravel())
        cuts = np.flatnonzero(np.diff(values) > gap_tol) + 1
        sizes = np.diff(np.concatenate(([0], cuts, [values.size])))
        return int(np.sum(sizes >= max(1, min_size_frac * values.size)))

    def mean_opinion(self) -> float:
        return float(np.mean(self.grid))

    def mean_opinion_shift(self) -> float:
        return float(np.mean(np.abs(self.grid - self.initial_grid)))
