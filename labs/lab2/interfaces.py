from typing import Protocol

import numpy as np


class WaveSource(Protocol):
    def __call__(self, t: float) -> float:
        ...

class BoundaryApply(Protocol):
    def __call__(self, U: np.ndarray) -> None:
        ...

class PlotWave(Protocol):
    def __call__(self, U_next: np.ndarray, k: int, K: int) -> None:
        ...
