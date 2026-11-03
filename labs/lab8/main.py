import numpy as np

from labs.lab8.ca.sir_ca import SIRCellularAutomaton
from labs.lab8.ca.sir_visualizer import animate_sir_ca
from labs.lab8.ca.siv_ca import SIVCellularAutomaton
from labs.lab8.cli import parse_config
from labs.lab8.entities import CAConfig, ODEConfig
from labs.lab8.ode.models import sir_system, siv_system
from labs.lab8.ode.solvers import Method, solve_ode
from labs.lab8.ode.visualizer import plot_epidemic_dynamics

SIR_LABELS = ["Susceptible", "Infected", "Recovered"]
SIV_LABELS = [*SIR_LABELS[:2], "Vaccinated"]


def run_ode_experiments(config: ODEConfig) -> None:
    """Compare Euler and RK4 for the SIR and SIV models."""
    print("Running ODE experiments...")
    models = [
        ("SIR", sir_system, config.sir_initial, config.sir, SIR_LABELS),
        ("SIV", siv_system, config.siv_initial, config.siv, SIV_LABELS),
    ]

    results: dict[str, dict[str, np.ndarray]] = {}
    labels: dict[str, list[str]] = {}
    t = None

    for name, system, y0, params, model_labels in models:
        results[name] = {}
        labels[name] = model_labels
        for method in Method:
            t, results[name][method.name] = solve_ode(
                system, y0, params, config.t_max, config.dt, method
            )

    if t is not None:
        plot_epidemic_dynamics(t, results, labels, "ODE Models: Euler vs RK4")


def run_ca_experiment(config: CAConfig) -> None:
    """Run the cellular-automaton SIR and SIV simulations."""
    print("Running Cellular Automaton simulation...")
    common = {
        "shape": config.shape,
        "recovery_threshold": config.recovery_threshold,
        "initial_infected_ratio": config.initial_infected_ratio,
        "infection_threshold": config.infection_threshold,
    }
    for ca in (
        SIRCellularAutomaton(**common),
        SIVCellularAutomaton(
            **common,
            vaccination_rate=config.vaccination_rate,
            breakthrough_prob=config.breakthrough_prob,
        ),
    ):
        animate_sir_ca(ca, steps=config.steps, interval=config.interval)


if __name__ == "__main__":
    config = parse_config()
    # run_ode_experiments(config.ode)
    run_ca_experiment(config.ca)
