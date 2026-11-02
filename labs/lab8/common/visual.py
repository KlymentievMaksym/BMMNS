from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def create_discrete_cmap(colors: Sequence[str]) -> ListedColormap:
    """Create a discrete ListedColormap from a list of color codes/names."""
    return ListedColormap(colors)


def draw_grid(
    grid: np.ndarray,
    cmap: ListedColormap | str = "viridis",
    title: str = "Cellular Automaton",
    ax: plt.Axes | None = None,
    vmin: float | int | None = None,
    vmax: float | int | None = None,
) -> plt.Axes:
    """Draw a 2D cellular automaton grid using Matplotlib."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax.clear()
    ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax
