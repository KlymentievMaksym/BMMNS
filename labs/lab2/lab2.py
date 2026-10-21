
from args_parser import parse_cli_args
from entities import SinusoidalSource, WaveSimConfig
from visual import Plotter, plot_amplitude_map
from wave_equation import run_simulation

if __name__ == "__main__":
    args = parse_cli_args()

    config = WaveSimConfig.from_args(args)
    print(config.source_positions)
    source = SinusoidalSource(
        amplitude=config.amplitude,
        omega=config.omega
    )
    plotter = Plotter(
        show_percentage=config.show_percentage,
        pause_time=config.pause_time
    )

    A_max = run_simulation(source, config, plotter)
    plot_amplitude_map(
        A_max=A_max,
        cmap=config.cmap,
        amplitude=config.amplitude,
        omega=config.omega
    )
