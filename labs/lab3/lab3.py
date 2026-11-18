from pathlib import Path

from labs.lab3.config import Config, parse_args
from labs.lab3.dimensions.beam_1d import run_all_beam_scenarios
from labs.lab3.dimensions.plate_2d import run_all_plate_scenarios
from labs.lab3.dimensions.polar_disk import run_all_polar_scenarios
from labs.lab3.visualization import (
    plot_beam_errors,
    plot_beam_results,
    plot_plate_results,
    plot_polar_results,
)


def _run_beam(config: Config, output_dir: Path) -> None:
    print("\n=== 1D Continuous Beam Models ===")
    results = run_all_beam_scenarios(config.beam)
    for res in results:
        y_min = float(res.y_num.min())
        y_max = float(res.y_num.max())
        err_info = (
            f", max |err| = {res.max_abs_error:.2e}"
            if res.max_abs_error is not None
            else ""
        )
        print(f"{res.title}: y in [{y_min:.4e}, {y_max:.4e}]{err_info}")

    plot_beam_results(results, save_dir=output_dir, show=True)
    plot_beam_errors(results, save_dir=output_dir, show=True)


def _run_plate(config: Config, output_dir: Path) -> None:
    print("\n=== 2D Rectangular Plate Models ===")
    results = run_all_plate_scenarios(config.plate)
    for res in results:
        u_min = float(res.u.min())
        u_max = float(res.u.max())
        status = "converged" if res.converged else "max_iter reached"
        print(
            f"{res.title}: {status} in {res.iterations} iters "
            f"[res={res.residual:.2e}], U in [{u_min:.4e}, {u_max:.4e}]"
        )

    plot_plate_results(results, save_dir=output_dir, show=True)


def _run_polar(config: Config, output_dir: Path) -> None:
    print("\n=== Axisymmetric Polar Plate Models ===")
    results = run_all_polar_scenarios(config.polar)
    for res in results:
        phi_min = float(res.phi_num.min())
        phi_max = float(res.phi_num.max())
        err_info = (
            f", max |err| = {res.max_abs_error:.2e}"
            if res.max_abs_error is not None
            else ""
        )
        print(f"{res.title}: phi in [{phi_min:.4e}, {phi_max:.4e}]{err_info}")

    plot_polar_results(results, save_dir=output_dir, show=True)


def main() -> None:
    config = parse_args()
    output_dir = Path(__file__).resolve().parent / "output"

    if config.task in ("beam", "all"):
        _run_beam(config, output_dir)
    if config.task in ("plate", "all"):
        _run_plate(config, output_dir)
    if config.task in ("polar", "all"):
        _run_polar(config, output_dir)


if __name__ == "__main__":
    main()
