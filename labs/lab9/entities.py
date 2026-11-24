from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExponentialParams:
    rate: float = 0.12


@dataclass(frozen=True)
class LogisticParams:
    r: float = 0.001
    k: float = 1000.0
    m: float = 0.2


@dataclass(frozen=True)
class RichardsonParams:
    a: float = 0.45
    m: float = 0.8
    r: float = 10.0
    b: float = 0.5
    n: float = 0.9
    s: float = 12.0


@dataclass(frozen=True)
class RichardsonCase:
    title: str
    initial: tuple[float, float]
    params: RichardsonParams


RICHARDSON_CASES: tuple[RichardsonCase, ...] = (
    RichardsonCase(
        "1: equilibrium", (20, 15), RichardsonParams(0.45, 0.8, 10, 0.5, 0.9, 12)
    ),
    RichardsonCase(
        "2: escalation", (10, 8), RichardsonParams(0.85, 0.5, 4, 0.8, 0.55, 5)
    ),
    RichardsonCase(
        "3: disarmament", (35, 30), RichardsonParams(0.4, 0.8, -8, 0.45, 0.85, -10)
    ),
    RichardsonCase(
        "4a: saddle (down)", (25, 25), RichardsonParams(0.9, 0.6, -12, 0.8, 0.6, -10)
    ),
    RichardsonCase(
        "4b: saddle (up)", (60, 55), RichardsonParams(0.9, 0.6, -12, 0.8, 0.6, -10)
    ),
)


@dataclass(frozen=True)
class ODEConfig:
    t_max: float = 20.0
    dt: float = 0.05
    exp_y0: float = 100.0
    exp_rates: tuple[float, ...] = (0.05, 0.12, 0.20)
    logistic: LogisticParams = field(default_factory=LogisticParams)
    logistic_initial: tuple[float, ...] = (50.0, 1200.0)
    logistic_t_max: float = 15.0
    richardson_t_max: float = 10.0
    richardson_dt: float = 0.01


@dataclass(frozen=True)
class VoterConfig:
    shape: tuple[int, ...] = (10, 10)
    rhos: tuple[float, ...] = (0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
    trials: int = 200
    scaling_rho: float = 0.75
    scaling_trials: int = 30
    sizes_1d: tuple[int, ...] = (10, 20, 40, 80)
    sides_2d: tuple[int, ...] = (6, 8, 10, 14)
    max_steps: int = 5_000_000
    seed: int = 1000


@dataclass(frozen=True)
class RelativeAgreementConfig:
    shape: tuple[int, ...] = (50, 50)
    e_values: tuple[float, ...] = (10.0, 50.0, 100.0)
    m_values: tuple[float, ...] = (0.3, 0.6)
    steps: int = 1_000_000
    zealot_opinion: float = 256.0
    seed: int = 42


@dataclass(frozen=True)
class SznajdConfig:
    shape: tuple[int, ...] = (50, 50)
    advantages: tuple[float, ...] = (0.01, 0.05, 0.10)
    trials: int = 10
    max_steps: int = 2_000_000
    seed: int = 2000


@dataclass(frozen=True)
class SegregationConfig:
    shape: tuple[int, ...] = (50, 50)
    empty_ratio: float = 0.28
    thresholds: tuple[float, ...] = (0.5, 1.0 / 3.0)
    max_sweeps: int = 100
    seed: int = 42


@dataclass(frozen=True)
class CAConfig:
    voter: VoterConfig = field(default_factory=VoterConfig)
    relative_agreement: RelativeAgreementConfig = field(
        default_factory=RelativeAgreementConfig
    )
    sznajd: SznajdConfig = field(default_factory=SznajdConfig)
    segregation: SegregationConfig = field(default_factory=SegregationConfig)


@dataclass(frozen=True)
class Lab9Config:
    ode: ODEConfig = field(default_factory=ODEConfig)
    ca: CAConfig = field(default_factory=CAConfig)
