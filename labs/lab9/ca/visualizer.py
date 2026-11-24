import matplotlib.pyplot as plt
import numpy as np


def plot_voter(rhos, win_probs, mean_steps) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(rhos, rhos, "k--", label="theory P(+1)=rho")
    ax1.plot(rhos, win_probs, "o-", label="simulation")
    ax1.set_title("Voter model: probability of +1 consensus")
    ax1.set_xlabel("initial density rho")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(rhos, mean_steps, "s-")
    ax2.set_title("Voter model: mean steps to consensus")
    ax2.set_xlabel("initial density rho")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_voter_scaling(n_1d, steps_1d, n_2d, steps_2d) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    n_1d, n_2d = np.asarray(n_1d, float), np.asarray(n_2d, float)
    ax.loglog(n_1d, steps_1d, "o-", label="d=1")
    ax.loglog(n_2d, steps_2d, "s-", label="d=2")
    ref1 = n_1d**3
    ref2 = n_2d**2 * np.log(n_2d)
    ax.loglog(n_1d, ref1 * steps_1d[0] / ref1[0], "k:", label=r"$\propto N^3$")
    ax.loglog(n_2d, ref2 * steps_2d[0] / ref2[0], "k--", label=r"$\propto N^2 \ln N$")
    ax.set_title(
        "Voter model: single-site steps to consensus vs N\n(d=1: N^3, d=2: N^2 ln N; = N^2 and N ln N sweeps)"
    )
    ax.set_xlabel("N (number of sites)")
    ax.set_ylabel("mean steps")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_relative_agreement(results) -> None:
    """results: list of (title, final_grid)."""
    cols = 3
    rows = int(np.ceil(len(results) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, grid) in zip(axes.ravel(), results, strict=False):
        ax.axis("on")
        im = ax.imshow(grid, cmap="viridis", vmin=1, vmax=256, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="opinion")
    plt.show()


def plot_sznajd(advantages, win_rates, mean_steps) -> None:
    labels = [f"{a * 100:.0f}%" for a in advantages]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(labels, win_rates, color="tab:green")
    ax1.set_ylim(0, 100)
    ax1.set_title("Sznajd: majority wins (%)")
    ax1.set_xlabel("initial advantage")
    ax2.bar(labels, mean_steps, color="tab:red")
    ax2.set_title("Sznajd: mean steps to consensus")
    ax2.set_xlabel("initial advantage")
    fig.tight_layout()
    plt.show()


def plot_segregation(snapshots) -> None:
    """snapshots: list of (title, initial_grid, final_grid)."""
    fig, axes = plt.subplots(
        2, len(snapshots), figsize=(4.5 * len(snapshots), 8.5), squeeze=False
    )
    for col, (title, initial, final) in enumerate(snapshots):
        for row, (name, grid) in enumerate((("initial", initial), ("final", final))):
            ax = axes[row, col]
            ax.imshow(grid, cmap="Accent", vmin=0, vmax=2, interpolation="nearest")
            ax.set_title(f"{title}\n{name}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    plt.show()
