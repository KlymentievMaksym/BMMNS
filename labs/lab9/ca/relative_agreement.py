import numpy as np
from labs.lab8.common.cellular_automaton import AsynchronousCA


class RelativeAgreementCA(AsynchronousCA):
    """Relative Agreement Interaction Model with continuous opinion values."""

    def __init__(
        self,
        shape: tuple[int, int] = (50, 50),
        e_threshold: float = 0.2,
        m_influence: float = 0.5,
        boundary: str = "periodic",
    ) -> None:
        self.e_threshold = e_threshold
        self.m_influence = m_influence
        grid = np.random.rand(*shape).astype(np.float64)
        super().__init__(
            shape=shape,
            dtype=np.dtype(np.float64),
            grid=grid,
            neighborhood="moore",
            boundary=boundary,
        )

    def apply_rule_at(self, coords: tuple[int, ...]) -> None:
        neighbor_offsets = self.offsets
        random_offset = neighbor_offsets[np.random.choice(len(neighbor_offsets))]
        neighbor_coords = tuple((np.array(coords) + random_offset) % np.array(self.shape))

        oi = float(self.grid[coords])
        oj = float(self.grid[neighbor_coords])

        diff = abs(oi - oj)
        if diff < self.e_threshold:
            shift = (self.m_influence / 2.0) * (oi - oj)
            self.grid[coords] = np.clip(oi - shift, 0.0, 1.0)
            self.grid[neighbor_coords] = np.clip(oj + shift, 0.0, 1.0)