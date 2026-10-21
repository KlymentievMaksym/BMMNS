import matplotlib.pyplot as plt
import numpy as np
from interfaces import PlotWave
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]


class Plotter(PlotWave):
    def __init__(
        self,
        show_percentage: float = 0.01,
        pause_time: float = 0.001,
        cmap: str = "viridis",
    ) -> None:
        self.show_percentage = show_percentage
        self.pause_time = pause_time
        self.cmap = cmap

        self._fig: Figure | None = None
        self._ax2d: Axes | None = None
        self._ax3d: Axes3D | None = None
        self._im: AxesImage | None = None
        self._X: np.ndarray | None = None
        self._Y: np.ndarray | None = None
        self._max_limit: float = 0.0

    def __call__(self, U_next: np.ndarray, k: int, K: int) -> None:
        step_interval = max(1, int(K * self.show_percentage))
        if k % step_interval != 0 and k != K - 1:
            return

        current_max = float(np.max(np.abs(U_next)))
        self._max_limit = max(self._max_limit, current_max)
        limit = self._max_limit if self._max_limit > 0 else 1.0

        if self._im is None:
            plt.ion()
            Nx, Ny = U_next.shape
            self._fig = plt.figure(figsize=(12, 5))

            self._ax2d = self._fig.add_subplot(1, 2, 1)
            self._im = self._ax2d.imshow(
                U_next,
                cmap=self.cmap,
                vmin=-limit,
                vmax=limit,
            )
            self._fig.colorbar(self._im, ax=self._ax2d, fraction=0.046, pad=0.04)

            self._ax3d = self._fig.add_subplot(1, 2, 2, projection="3d")
            x = np.arange(Ny)
            y = np.arange(Nx)
            self._X, self._Y = np.meshgrid(x, y)

            self._fig.tight_layout()
        else:
            self._im.set_data(U_next)
            self._im.set_clim(-limit, limit)

        assert self._fig is not None
        assert self._ax2d is not None
        assert self._ax3d is not None
        assert self._X is not None
        assert self._Y is not None

        self._ax2d.set_title(f"2D (Step {k+1}/{K})")

        elev, azim = self._ax3d.elev, self._ax3d.azim
        self._ax3d.clear()
        self._ax3d.plot_surface(
            self._X,
            self._Y,
            U_next,
            cmap=self.cmap,
            vmin=-limit,
            vmax=limit,
            rcount=50,
            ccount=50,
        )
        self._ax3d.set_zlim(-limit, limit)
        self._ax3d.set_title("3D")
        self._ax3d.view_init(elev=elev, azim=azim)

        self._fig.canvas.draw_idle()
        plt.pause(self.pause_time)


def plot_amplitude_map(A_max: np.ndarray, cmap: str, amplitude: float, omega: float | str) -> None:
    plt.ioff()
    plt.figure()
    plt.imshow(A_max, cmap=cmap)
    plt.colorbar()
    plt.title(f"(A = {amplitude}; Omega = {omega})\nMax amplitude map")
    plt.show()