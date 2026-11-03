from dataclasses import dataclass, field


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
    sir_initial: tuple[float, float, float] = (45400, 2100, 2500)  # S, I, R
    siv_initial: tuple[float, float, float] = (45400, 2100, 2500)  # S, I, V
    sir: SIRParams = field(default_factory=SIRParams)
    siv: SIVParams = field(default_factory=SIVParams)


@dataclass(frozen=True)
class CAConfig:
    shape: tuple[int, ...] = (50, 50)
    recovery_threshold: int = 5
    infection_threshold: int = 3
    initial_infected_ratio: float = 0.03
    vaccination_rate: float = 0.02
    breakthrough_prob: float = 0.05
    steps: int = 100
    interval: int = 50


@dataclass(frozen=True)
class Lab8Config:
    ode: ODEConfig = field(default_factory=ODEConfig)
    ca: CAConfig = field(default_factory=CAConfig)
