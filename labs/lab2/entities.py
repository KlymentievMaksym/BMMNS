import argparse
from dataclasses import dataclass, field, fields

import numpy as np
from boundaries import BoundaryType
from interfaces import WaveSource


@dataclass
class SinusoidalSource(WaveSource):
    amplitude: float = 1e-4
    omega: float = 10 * np.pi

    def __call__(self, t: float) -> float:
        return self.amplitude * np.sin(self.omega * t)

@dataclass
class WaveSimConfig:
    Lx: float = 1.0
    Ly: float = 1.0
    Nx: int = 201
    Ny: int = 201
    c: float = 1.0
    delta_t: float = 5e-4
    K: int = 5000
    cmap: str = "viridis"
    amplitude: float = 1e-4
    omega: float = 10 * np.pi
    boundary_type: BoundaryType = BoundaryType.FREE
    source_positions: list[tuple[int, int]] = field(default_factory=list)
    show_percentage: float = 0.01
    pause_time: float = 0.001

    def __post_init__(self) -> None:
        self.hx = self.Lx / (self.Nx - 1)
        self.hy = self.Ly / (self.Ny - 1)
        
        stability_limit = min(self.hx, self.hy) / (self.c * np.sqrt(2))
        if self.delta_t > stability_limit:
            raise ValueError(
                f"Stability condition violated: delta_t ({self.delta_t}) "
                f"must be <= {stability_limit}"
            )
        
        self.hx_coeff = (self.delta_t / self.hx) ** 2
        self.hy_coeff = (self.delta_t / self.hy) ** 2
        self.i_center = (self.Nx - 1) // 2
        self.j_center = (self.Ny - 1) // 2

        if not self.source_positions:
            self.source_positions = [(self.i_center, self.j_center)]

        if isinstance(self.boundary_type, str):
            try:
                self.boundary_type = BoundaryType(self.boundary_type)
            except ValueError:
                valid_options = [b.value for b in BoundaryType]
                raise ValueError(
                    f"Invalid boundary_type '{self.boundary_type}'. "
                    f"Expected one of: {valid_options}"
                )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'WaveSimConfig':
        config_kwargs = {
            field.name: getattr(args, field.name)
            for field in fields(cls)
            if hasattr(args, field.name)
        }
        return cls(**config_kwargs)