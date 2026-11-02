import numpy as np

from labs.lab8.ca.sir_ca import SIRCellularAutomaton
from labs.lab8.ca.sir_visualizer import animate_sir_ca
from labs.lab8.ode.models import sir_system, siv_system
from labs.lab8.ode.solvers import solve_ode
from labs.lab8.ode.visualizer import plot_epidemic_dynamics


def run_ode_experiments():
    """Run ODE numerical solvers comparison for SIR and SIV models."""
    print("Running ODE experiments...")

    sir_init = np.array([45400.0, 2100.0, 2500.0])
    sir_params = {"beta": 0.00001, "gamma": 0.1}

    t_euler, h_euler = solve_ode(
        sir_system, sir_init, t_max=30.0, dt=0.1, method="euler", **sir_params
    )

    plot_epidemic_dynamics(
        t_euler, h_euler, ["Susceptible", "Infected", "Recovered"], "SIR ODE Model (EULER)"
    )

    t_rk4, h_rk4 = solve_ode(
        sir_system, sir_init, t_max=30.0, dt=0.1, method="rk4", **sir_params
    )

    plot_epidemic_dynamics(
        t_rk4, h_rk4, ["Susceptible", "Infected", "Recovered"], "SIR ODE Model (RK4)"
    )

    siv_init = np.array([45400.0, 2100.0, 2500.0])
    siv_params = {
        "a_birth": 100.0,
        "q": 0.7,
        "p": 0.1,
        "beta": 0.00001,
        "d_death": 0.01,
        "gamma": 0.1,
        "eps": 0.05,
        "alpha": 0.02,
    }

    t_siv, h_siv = solve_ode(
        siv_system, siv_init, t_max=30.0, dt=0.1, method="euler", **siv_params
    )
    plot_epidemic_dynamics(
        t_siv, h_siv, ["Susceptible", "Infected", "Vaccinated"], "SIV ODE Model (EULER)"
    )

    t_siv, h_siv = solve_ode(
        siv_system, siv_init, t_max=30.0, dt=0.1, method="rk4", **siv_params
    )
    plot_epidemic_dynamics(
        t_siv, h_siv, ["Susceptible", "Infected", "Vaccinated"], "SIV ODE Model (RK4)"
    )


def run_ca_experiment():
    """Run Cellular Automaton SIR simulation."""
    print("Running Cellular Automaton simulation...")
    ca = SIRCellularAutomaton(
        shape=(50, 50), recovery_threshold=5, initial_infected_ratio=0.03
    )
    animate_sir_ca(ca, steps=100, interval=50)


if __name__ == "__main__":
    # run_ode_experiments()
    run_ca_experiment()
