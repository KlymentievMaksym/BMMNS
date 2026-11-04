from dataclasses import dataclass, field

from matplotlib.colors import ListedColormap

from labs.lab8.common.boundary import BoundaryType


@dataclass(frozen=True)
class SIRParams:
    beta: float = 0.00001
    gamma: float = 0.1


@dataclass(frozen=True)
class SIVParams(SIRParams):
    alpha: float = 0.02
    eps: float = 0.05
    a_birth: float = 100.0
    q: float = 0.7
    p: float = 0.1
    d_death: float = 0.01


@dataclass(frozen=True)
class ODEConfig:
    t_max: float = 30.0
    dt: float = 0.1
    sir_initial: tuple[int, int, int] = (45400, 2100, 2500)  # S, I, R
    siv_initial: tuple[int, int, int] = (45400, 2100, 2500)  # S, I, V
    sir: SIRParams = field(default_factory=SIRParams)
    siv: SIVParams = field(default_factory=SIVParams)


@dataclass(frozen=True)
class CAConfig:
    shape: tuple[int, ...] = (50, 50)
    boundary: BoundaryType = BoundaryType.PERIODIC
    recovery_threshold: int = 5
    infection_threshold: int = 3
    initial_infected_ratio: float = 0.03
    vaccination_rate: float = 0.02
    breakthrough_prob: float = 0.05
    steps: int = 100
    interval: int = 100
    figsize: tuple[int, int] = (8, 4)
    cmap: ListedColormap | tuple[str, ...] | str = ("#ffffff", "#e74c3c", "#2ecc71")
    save_path: str | None = None

    def __post_init__(self):
        if isinstance(self.cmap, str):
            if "," in self.cmap:
                colors = tuple(color.strip() for color in self.cmap.split(","))
                object.__setattr__(self, "cmap", ListedColormap(colors))
            else:
                object.__setattr__(self, "cmap", self.cmap)

        elif not isinstance(self.cmap, ListedColormap):
            object.__setattr__(self, "cmap", ListedColormap(self.cmap))


@dataclass(frozen=True)
class Lab8Config:
    ode: ODEConfig = field(default_factory=ODEConfig)
    ca: CAConfig = field(default_factory=CAConfig)
