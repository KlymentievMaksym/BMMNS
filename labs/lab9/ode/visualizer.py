import matplotlib.pyplot as plt
import numpy as np

from labs.lab9.ode.analysis import RichardsonStability, WWIFit


def plot_exponential(
    t: np.ndarray,
    series: dict[float, dict[str, np.ndarray]],
    discrete: dict[float, np.ndarray],
) -> None:
    fig, axes = plt.subplots(
        1, len(series), figsize=(4.5 * len(series), 3.6), squeeze=False
    )
    for ax, (rate, histories) in zip(axes[0], series.items(), strict=True):
        for (method, y), ls in zip(histories.items(), ("--", "-"), strict=False):
            ax.plot(t, y, linestyle=ls, label=method)
        steps = discrete[rate]
        ax.scatter(
            np.arange(len(steps)),
            steps,
            color="black",
            s=14,
            label="discrete",
            zorder=3,
        )
        ax.set_title(f"Exponential growth, a={rate}")
        ax.set_xlabel("t")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    plt.show()


def plot_logistic(
    t: np.ndarray, histories: dict[float, np.ndarray], equilibrium: float
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for n0, y in histories.items():
        ax.plot(t, y, label=f"N0={n0:.0f}")
    ax.axhline(equilibrium, color="black", ls="--", lw=1, label=f"Nc={equilibrium:.0f}")
    ax.set_title("Logistic evolution dN/dt = rN(K-N) - mN")
    ax.set_xlabel("t")
    ax.set_ylabel("N")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_richardson(
    t: np.ndarray,
    runs: list[tuple[str, np.ndarray, RichardsonStability]],
) -> None:
    fig, axes = plt.subplots(2, len(runs), figsize=(4 * len(runs), 6.5), squeeze=False)
    for col, (title, hist, st) in enumerate(runs):
        hist = np.maximum(hist, 0.0)
        ax = axes[0, col]
        ax.plot(t, hist[:, 0], color="goldenrod", label="X")
        ax.plot(t, hist[:, 1], color="seagreen", ls="--", label="Y")
        ax.set_title(f"{title}\n{st.regime}", fontsize=9)
        ax.set_xlabel("t")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        ax = axes[1, col]
        ax.plot(hist[:, 0], hist[:, 1], color="tab:blue")
        ax.scatter(*hist[0], color="green", zorder=3, label="start")
        if st.equilibrium is not None:
            ax.scatter(
                *st.equilibrium, color="red", marker="x", zorder=3, label="equilibrium"
            )
        ax.set_title(f"stable={st.stable}, mn-ab={st.det:.3f}", fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def plot_wwi(fit: WWIFit) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(fit.mean_z, fit.growth, color="crimson", label="data")
    z = np.linspace(fit.mean_z.min() - 5, fit.mean_z.max() + 5, 50)
    ax.plot(z, fit.k * z + fit.f, color="black", label=f"k={fit.k:.3f}, f={fit.f:.1f}")
    ax.set_title("Richardson model vs 1909-1913 military budgets")
    ax.set_xlabel("mean total budget z")
    ax.set_ylabel("dz/dt")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()
