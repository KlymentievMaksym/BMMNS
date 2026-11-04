import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap

from labs.lab8.ca.sir_ca import SIRCellularAutomaton
from labs.lab8.ca.siv_ca import SIVCellularAutomaton


def animate_ca(
    models: tuple[SIRCellularAutomaton, SIVCellularAutomaton],
    figsize: tuple[int, int],
    cmap: ListedColormap | Colormap | str,
    steps: int,
    interval: int,
    save_path: str | None = None,
) -> None:
    fig, axes = plt.subplots(2, 2, width_ratios=[1, 1], figsize=figsize)

    s_history, i_history, r_history = [[], []], [[], []], [[], []]
    t_history = [[], []]

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    def update(frame: int):
        for index, (model, (ax_grid, ax_chart)) in enumerate(
            zip(models, axes, strict=True)
        ):
            if frame > 0:
                model.step()

            s_count, i_count, r_count = model.get_counts()
            s_history[index].append(s_count)
            i_history[index].append(i_count)
            r_history[index].append(r_count)
            t_history[index].append(frame)

            display_grid = np.zeros_like(model.grid)

            infected = model._is_infected(model.grid)
            display_grid[infected] = 1

            if isinstance(model, SIVCellularAutomaton):
                display_grid[model.grid == model.VACCINATED_STATE] = 2
            else:
                display_grid[model.grid == model.RECOVERED_STATE] = 2

            ax_grid.clear()
            ax_grid.imshow(
                display_grid,
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
            )
            ax_grid.set_title(f"Grid - Step {frame}")
            ax_grid.set_xticks([])
            ax_grid.set_yticks([])

            label = (
                "Vaccinated" if isinstance(model, SIVCellularAutomaton) else "Recovered"
            )
            ax_chart.clear()
            ax_chart.plot(
                t_history[index], s_history[index], color="blue", label="Susceptible"
            )
            ax_chart.plot(
                t_history[index], i_history[index], color="red", label="Infected"
            )
            ax_chart.plot(
                t_history[index], r_history[index], color="green", label=label
            )
            ax_chart.set_xlim(0, max(steps, frame + 1))
            ax_chart.set_ylim(0, model.shape[0] * model.shape[1])
            ax_chart.set_xlabel("Step")
            ax_chart.set_ylabel("Cells Count")
            ax_chart.grid(True)
            ax_chart.legend(loc="upper right")

    anim = FuncAnimation(fig, update, frames=steps, interval=interval, repeat=False)

    plt.tight_layout()
    if save_path:
        anim.save(
            save_path, writer="pillow" if save_path.endswith(".gif") else "ffmpeg"
        )
    else:
        plt.show()
