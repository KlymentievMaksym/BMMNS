from typing import Protocol

import numpy as np


class WaveSource(Protocol):
    def __call__(self, t: float) -> float:
        ...

class BoundaryApply(Protocol):
    def __call__(self, U: np.ndarray) -> None:
        ...