import matplotlib.pyplot as plt
import numpy as np

COLORS = ["black", "blue", "red"]
LINESTYLES = ["--", "-", ":", "-."]


def plot_epidemic_dynamics(
    t_values: np.ndarray,
    results: dict[str, dict[str, np.ndarray]],
    labels: dict[str, list[str]],
    title: str = "Epidemic Dynamics",
) -> None:
    """Plot every model in one figure: one row per model, one column per compartment.

    results: {model_name: {method_name: history}}, history shape (len(t_values), n_compartments)
    labels:  {model_name: [compartment labels]}
    """
    n_rows = len(results)
    n_cols = max(len(v) for v in labels.values())
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 3.8 * n_rows),
        sharex=True,
        squeeze=False,
    )

    for row, (model, histories) in zip(axes, results.items()):
        for idx, (label, color) in enumerate(zip(labels[model], COLORS)):
            ax = row[idx]
            for (method, history), ls in zip(histories.items(), LINESTYLES):
                ax.plot(
                    t_values, history[:, idx],
                    color=color, linestyle=ls, linewidth=2, label=method,
                )
            ax.set_title(f"{model}: {label}")
            ax.set_ylabel(label[0])
            ax.grid(True)
            ax.legend()

    for ax in axes[-1]:
        ax.set_xlabel("t")

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
