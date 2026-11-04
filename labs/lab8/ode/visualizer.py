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

    n_rows = len(results)
    n_cols = max(len(value) for value in labels.values())
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        sharex=True,
        squeeze=False,
    )

    for ax_row, (model, histories) in zip(axes, results.items(), strict=True):
        for index, (label, color) in enumerate(zip(labels[model], COLORS, strict=True)):
            ax = ax_row[index]
            for (method, history), ls in zip(
                histories.items(), LINESTYLES, strict=False
            ):
                ax.plot(
                    t_values,
                    history[:, index],
                    color=color,
                    linestyle=ls,
                    linewidth=2,
                    label=method,
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
