import math
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from labs.lab3.dimensions.beam_1d import BeamResult
from labs.lab3.dimensions.plate_2d import PlateResult
from labs.lab3.dimensions.polar_disk import PolarResult


def _finalize_figure(
    fig: Figure,
    filename: str,
    save_dir: Path | None,
    show: bool,
) -> Figure:
    fig.tight_layout()
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / filename, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_beam_results(
    results: list[BeamResult],
    save_dir: Path | None = None,
    show: bool = True,
) -> Figure:
    """Plot numerical and analytical deflection curves for 1D beam scenarios."""
    if not results:
        raise ValueError("Cannot plot empty beam results list.")

    ncols = 2
    nrows = math.ceil(len(results) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.2 * nrows), squeeze=False)
    flat_axes = axes.ravel()

    for idx, res in enumerate(results):
        ax = flat_axes[idx]
        ax.plot(res.x, res.y_num, label="Numerical", linewidth=2.0)

        if res.y_exact is not None:
            ax.plot(
                res.x,
                res.y_exact,
                "--",
                label="Analytical",
                linewidth=1.5,
            )

        if res.supports:
            sx, sy = zip(*res.supports, strict=True)
            ax.scatter(
                sx,
                sy,
                color="crimson",
                marker="*",
                s=90,
                zorder=5,
                label="Supports",
            )

        title = res.title
        if res.max_abs_error is not None:
            title = f"{title}\nmax |err| = {res.max_abs_error:.2e}"

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y(x)")
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.legend(fontsize=8)

    for idx in range(len(results), len(flat_axes)):
        flat_axes[idx].set_visible(False)

    return _finalize_figure(fig, "beam_1d_deflection.png", save_dir, show)


def plot_beam_errors(
    results: list[BeamResult],
    save_dir: Path | None = None,
    show: bool = True,
) -> Figure | None:
    """Plot pointwise absolute errors for 1D beam scenarios with exact solutions."""
    exact_results = [res for res in results if res.y_exact is not None]
    if not exact_results:
        return None

    fig, axes = plt.subplots(
        1,
        len(exact_results),
        figsize=(5.2 * len(exact_results), 3.8),
        squeeze=False,
    )

    for idx, res in enumerate(exact_results):
        ax = axes[0, idx]
        assert res.y_exact is not None
        err = np.abs(res.y_num - res.y_exact)
        ax.plot(res.x, err, color="darkorange", linewidth=1.6)
        ax.set_title(f"{res.name}: |y_num - y_exact|", fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("Absolute error")
        ax.grid(True, linestyle=":", alpha=0.7)

    return _finalize_figure(fig, "beam_1d_errors.png", save_dir, show)


def plot_plate_results(
    results: list[PlateResult],
    save_dir: Path | None = None,
    show: bool = True,
) -> Figure:
    """Plot 3D deflection surfaces for 2D rectangular plate scenarios."""
    if not results:
        raise ValueError("Cannot plot empty plate results list.")

    n_plots = len(results)
    fig = plt.figure(figsize=(5.8 * n_plots, 5.0))

    for idx, res in enumerate(results, start=1):
        ax = cast(Axes3D, fig.add_subplot(1, n_plots, idx, projection="3d"))
        grid_x, grid_y = np.meshgrid(res.x, res.y, indexing="ij")
        ax.plot_surface(
            grid_x,
            grid_y,
            res.u,
            cmap="viridis",
            edgecolor="none",
            alpha=0.92,
        )

        if res.pinned_points:
            px, py, pz = zip(*res.pinned_points, strict=True)
            ax.scatter(
                px,
                py,
                pz,
                color="crimson",
                s=45,
                depthshade=False,
                label="Pinned",
            )

        status = "converged" if res.converged else "max_iter"
        ax.set_title(
            f"{res.title}\n{status} in {res.iterations} iters (res={res.residual:.1e})",
            fontsize=9.5,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("U(x, y)")

    return _finalize_figure(fig, "plate_2d_surfaces.png", save_dir, show)


def plot_polar_results(
    results: list[PolarResult],
    theta_steps: int = 72,
    save_dir: Path | None = None,
    show: bool = True,
) -> Figure:
    """Plot radial profiles and 3D annular surfaces for polar disk scenarios."""
    if not results:
        raise ValueError("Cannot plot empty polar results list.")

    nrows = len(results)
    fig = plt.figure(figsize=(12.0, 4.3 * nrows))
    theta = np.linspace(0.0, 2.0 * np.pi, theta_steps, dtype=np.float64)

    for row, res in enumerate(results):
        ax_2d = fig.add_subplot(nrows, 2, 2 * row + 1)
        ax_2d.plot(res.r, res.phi_num, label="Numerical", linewidth=2.0)
        if res.phi_exact is not None:
            ax_2d.plot(
                res.r,
                res.phi_exact,
                "--",
                label="Analytical",
                linewidth=1.5,
            )

        title = res.title
        if res.max_abs_error is not None:
            title = f"{title}\nmax |err| = {res.max_abs_error:.2e}"

        ax_2d.set_title(title, fontsize=10)
        ax_2d.set_xlabel("r")
        ax_2d.set_ylabel("phi(r)")
        ax_2d.grid(True, linestyle=":", alpha=0.7)
        ax_2d.legend(fontsize=8)

        ax_3d = cast(Axes3D, fig.add_subplot(nrows, 2, 2 * row + 2, projection="3d"))
        rad_grid, theta_grid = np.meshgrid(res.r, theta, indexing="ij")
        x_surf = rad_grid * np.cos(theta_grid)
        y_surf = rad_grid * np.sin(theta_grid)
        z_surf = np.broadcast_to(res.phi_num[:, None], rad_grid.shape)

        ax_3d.plot_surface(
            x_surf,
            y_surf,
            z_surf,
            cmap="plasma",
            edgecolor="none",
            alpha=0.9,
        )
        ax_3d.set_title(f"{res.name}: 3D annular surface", fontsize=10)
        ax_3d.set_xlabel("x")
        ax_3d.set_ylabel("y")
        ax_3d.set_zlabel("phi(r)")

    return _finalize_figure(fig, "polar_disk_results.png", save_dir, show)
