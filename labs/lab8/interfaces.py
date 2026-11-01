from typing import Protocol

import numpy as np


class NeighborhoodOffsetsReceive(Protocol):
    def __call__(self, shape: tuple[int, ...]) -> list: ...


class BoundaryApply(Protocol):
    def __call__(
        self,
        grid: np.ndarray,
        coords: tuple[int, ...],
        offsets: np.ndarray,
        shape: tuple[int, ...],
    ) -> np.ndarray: ...
