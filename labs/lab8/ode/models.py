import numpy as np


def sir_system(t: float, state: np.ndarray, beta: float, gamma: float) -> np.ndarray:
    """Right-hand side of classic SIR epidemic ODE system.

    ds/dt = -beta * S * I
    di/dt = beta * S * I - gamma * I
    dr/dt = gamma * I
    """
    s, i, r = state
    ds = -beta * s * i
    di = beta * s * i - gamma * i
    dr = gamma * i
    return np.array([ds, di, dr], dtype=np.float64)


def siv_system(
    t: float,
    state: np.ndarray,
    a_birth: float,
    q: float,
    p: float,
    beta: float,
    d_death: float,
    gamma: float,
    eps: float,
    alpha: float,
) -> np.ndarray:
    """Right-hand side of SIV epidemic model with vaccination ODE system.

    ds/dt = (1-q)A - beta*S*I - (p+d)S + gamma*I + eps*V
    di/dt = beta*S*I - (d+alpha+gamma)I
    dv/dt = q*A + p*S - (eps+d)V
    """
    s, i, v = state
    ds = (1.0 - q) * a_birth - beta * s * i - (p + d_death) * s + gamma * i + eps * v
    di = beta * s * i - (d_death + alpha + gamma) * i
    dv = q * a_birth + p * s - (eps + d_death) * v
    return np.array([ds, di, dv], dtype=np.float64)
