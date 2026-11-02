import matplotlib.pyplot as plt
import numpy as np


def plot_epidemic_dynamics(
    t_values: np.ndarray,
    history: np.ndarray,
    labels: list[str],
    title: str = "Epidemic Dynamics",
) -> None:
    """Plot time-series for Susceptible, Infected, Recovered/Vaccinated compartments."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = ["black", "blue", "red"]

    for idx, (label, color) in enumerate(zip(labels, colors)):
        axes[idx].plot(t_values, history[:, idx], color=color, linewidth=2)
        axes[idx].set_title(label)
        axes[idx].set_xlabel("t")
        axes[idx].set_ylabel(label[0])
        axes[idx].grid(True)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
