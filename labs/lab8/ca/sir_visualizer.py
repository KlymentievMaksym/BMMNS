import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from labs.lab8.ca.sir_ca import SIRCellularAutomaton


def animate_sir_ca(
    ca: SIRCellularAutomaton,
    steps: int = 100,
    interval: int = 100,
    save_path: str | None = None,
) -> None:
    """Animate 2D cellular automaton grid alongside line chart of SIR dynamics."""
    fig, (ax_grid, ax_chart) = plt.subplots(1, 2, figsize=(12, 5))

    cmap = ListedColormap(["#ffffff", "#e74c3c", "#2ecc71"])

    s_history, i_history, r_history = [], [], []
    t_history = []

    def update(frame: int):
        if frame > 0:
            ca.step()

        s_count, i_count, r_count = ca.get_counts()
        s_history.append(s_count)
        i_history.append(i_count)
        r_history.append(r_count)
        t_history.append(frame)

        display_grid = ca.grid.copy()
        display_grid[(display_grid >= 1) & (display_grid < ca.recovery_threshold)] = 1
        display_grid[display_grid == ca.RECOVERED_STATE] = 2

        ax_grid.clear()
        ax_grid.imshow(display_grid, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        ax_grid.set_title(f"SIR Grid - Step {frame}")
        ax_grid.set_xticks([])
        ax_grid.set_yticks([])

        ax_chart.clear()
        ax_chart.plot(t_history, s_history, color="blue", label="Susceptible")
        ax_chart.plot(t_history, i_history, color="red", label="Infected")
        ax_chart.plot(t_history, r_history, color="green", label="Recovered")
        ax_chart.set_xlim(0, max(steps, frame + 1))
        ax_chart.set_ylim(0, ca.shape[0] * ca.shape[1])
        ax_chart.set_xlabel("Step")
        ax_chart.set_ylabel("Cells Count")
        ax_chart.set_title("Epidemic Statistics")
        ax_chart.grid(True)
        ax_chart.legend(loc="upper right")

    anim = FuncAnimation(fig, update, frames=steps, interval=interval, repeat=False)

    if save_path:
        anim.save(
            save_path, writer="pillow" if save_path.endswith(".gif") else "ffmpeg"
        )
    else:
        plt.tight_layout()
        plt.show()
