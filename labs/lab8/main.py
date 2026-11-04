import numpy as np

from labs.lab8.ca.sir_ca import SIRCellularAutomaton
from labs.lab8.ca.siv_ca import SIVCellularAutomaton
from labs.lab8.ca.visualizer import animate_ca
from labs.lab8.cli import parse_config
from labs.lab8.entities import CAConfig, ODEConfig
from labs.lab8.ode.models import sir_system, siv_system
from labs.lab8.ode.solvers import Method, solve_ode
from labs.lab8.ode.visualizer import plot_epidemic_dynamics

SIR_LABELS = ["Susceptible", "Infected", "Recovered"]
SIV_LABELS = [*SIR_LABELS[:2], "Vaccinated"]

GENERATOR = np.random.default_rng(seed=42)


def run_ode_experiments(config: ODEConfig) -> None:
    models = [
        ("SIR", sir_system, config.sir_initial, config.sir, SIR_LABELS),
        ("SIV", siv_system, config.siv_initial, config.siv, SIV_LABELS),
    ]

    results: dict[str, dict[str, np.ndarray]] = {}
    labels: dict[str, list[str]] = {}
    t = None

    for name, system_fn, initial_state, params, model_labels in models:
        results[name] = {}
        labels[name] = model_labels
        for method in Method:
            t, results[name][method.name] = solve_ode(
                system_fn, initial_state, params, config.t_max, config.dt, method
            )

    if t is not None:
        plot_epidemic_dynamics(t, results, labels)


def run_ca_experiment(config: CAConfig) -> None:
    common = {
        "shape": config.shape,
        "boundary": config.boundary,
        "recovery_threshold": config.recovery_threshold,
        "initial_infected_ratio": config.initial_infected_ratio,
        "infection_threshold": config.infection_threshold,
        "rng": GENERATOR,
    }

    models: tuple[SIRCellularAutomaton, SIVCellularAutomaton] = (
        SIRCellularAutomaton(**common),
        SIVCellularAutomaton(
            **common,
            vaccination_rate=config.vaccination_rate,
            breakthrough_prob=config.breakthrough_prob,
        ),
    )

    # for model in models:
    animate_ca(
        models,
        figsize=config.figsize,
        cmap=config.cmap,  # pyright: ignore[reportArgumentType]
        steps=config.steps,
        interval=config.interval,
        save_path=config.save_path,
    )


if __name__ == "__main__":
    config = parse_config()
    run_ode_experiments(config.ode)
    run_ca_experiment(config.ca)
