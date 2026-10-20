
from labs.lab2.args_parser import parse_cli_args
from labs.lab2.entities import SinusoidalSource, WaveSimConfig
from labs.lab2.visual import plot_amplitude_map
from labs.lab2.wave_equation import run_simulation

if __name__ == "__main__":
    args = parse_cli_args()
    config = WaveSimConfig.from_args(args)
    source = SinusoidalSource(
        amplitude=config.amplitude,
        omega=config.omega
    )
    A_max = run_simulation(source, config)
    plot_amplitude_map(
        A_max=A_max,
        cmap=config.cmap,
        amplitude=config.amplitude,
        omega=config.omega
    )
