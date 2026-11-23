import matplotlib.pyplot as plt
import numpy as np

from labs.lab9.ca.relative_agreement import RelativeAgreementCA
from labs.lab9.ca.segregation import SchellingSegregationCA
from labs.lab9.ca.sznajd import SznajdCA
from labs.lab9.ca.voter import VoterModel
from labs.lab9.ode.population import (
    exponential_growth_continuous,
    exponential_growth_discrete,
    logistic_evolution,
)
from labs.lab9.ode.richardson import fit_wwi_historical_data, richardson_arms_race


def run_population_experiments() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for rate in (0.05, 0.12, 0.20):
        y_disc = exponential_growth_discrete(y0=100.0, a=rate, steps=20)
        t_cont, y_cont = exponential_growth_continuous(
            y0=100.0, a=rate, dt=0.1, steps=200
        )
        line = ax1.plot(t_cont, y_cont, label=f"Continuous a={rate}")[0]
        ax1.scatter(range(0, 21, 2), y_disc[::2], color=line.get_color(), s=20)

    ax1.set_title("Exponential Growth: Discrete vs Continuous")
    ax1.set_xlabel("Time t")
    ax1.set_ylabel("Population")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    for n0 in (50.0, 1200.0):
        t_log, n_log, eq = logistic_evolution(
            n0=n0, r=0.001, k=1000.0, m=0.2, dt=0.05, steps=300
        )
        ax2.plot(t_log, n_log, label=f"N0={n0:.0f} -> Nc={eq:.0f}")
    ax2.axhline(800.0, color="black", ls="--", lw=1.0, label="Equilibrium Nc=800")
    ax2.set_title("Logistic Evolution dN/dt = rN(K - N) - mN")
    ax2.set_xlabel("Time t")
    ax2.set_ylabel("Population N")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def run_richardson_experiments() -> None:
    cases = [
        (
            "Case 1: Stable balance",
            (20.0, 15.0, 0.45, 0.80, 10.0, 0.50, 0.90, 12.0, 1500),
        ),
        ("Case 2: Escalation", (10.0, 8.0, 0.85, 0.50, 4.0, 0.80, 0.55, 5.0, 600)),
        (
            "Case 3: Disarmament",
            (35.0, 30.0, 0.40, 0.80, -8.0, 0.45, 0.85, -10.0, 1000),
        ),
        (
            "Case 4a: Saddle disarm",
            (25.0, 25.0, 0.90, 0.60, -12.0, 0.80, 0.60, -10.0, 800),
        ),
        (
            "Case 4b: Saddle escalate",
            (60.0, 55.0, 0.90, 0.60, -12.0, 0.80, 0.60, -10.0, 600),
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    ax_list = axes.ravel()

    for idx, (label, params) in enumerate(cases):
        x0, y0, a, m, r, b, n, s, steps = params
        t_vals, x_vals, y_vals, eq, is_stable, regime = richardson_arms_race(
            x0, y0, a, m, r, b, n, s, dt=0.01, steps=steps
        )
        ax = ax_list[idx]
        ax.plot(t_vals, x_vals, label="X", color="goldenrod")
        ax.plot(t_vals, y_vals, label="Y", color="seagreen", ls="--")
        ax.set_title(f"{label}\n({regime})", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        print(
            f"{label}: stable={is_stable}, eq={eq}, final=({x_vals[-1]:.1f}, {y_vals[-1]:.1f})"
        )

    wwi = fit_wwi_historical_data()
    ax_wwi = ax_list[5]
    ax_wwi.scatter(wwi["mean_z"], wwi["growth_dz"], color="crimson")
    z_grid = np.linspace(195.0, 275.0, 50)
    ax_wwi.plot(
        z_grid,
        wwi["k"] * z_grid + wwi["f"],
        color="black",
        label=f"k={wwi['k']:.3f}, f={wwi['f']:.1f}",
    )
    ax_wwi.set_title("WWI 1909-1913 Historical Fit", fontsize=9.5)
    ax_wwi.set_xlabel("Mean total budget z")
    ax_wwi.set_ylabel("Annual growth dz/dt")
    ax_wwi.grid(True, alpha=0.3)
    ax_wwi.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def run_voter_experiments(trials: int = 20) -> None:
    rhos = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    win_probs: list[float] = []
    mean_steps: list[float] = []

    print("\n--- Voter Model (10x10, 20 trials per rho) ---")
    for rho in rhos:
        wins = 0
        steps_list: list[int] = []
        for trial in range(trials):
            model = VoterModel(shape=(10, 10), rho=rho, seed=1000 + trial)
            winner, steps = model.run_until_consensus()
            if winner == 1:
                wins += 1
            steps_list.append(steps)
        prob = wins / trials
        avg_s = float(np.mean(steps_list))
        win_probs.append(prob)
        mean_steps.append(avg_s)
        print(f"rho={rho:.2f} | P(+1)={prob:.2f} | mean steps={avg_s:.1f}")

    print("\n--- Voter Model: 1D vs 2D Scaling (rho=0.75, 20 trials) ---")
    scaling_configs = [
        ("1D N=20", (20,)),
        ("1D N=40", (40,)),
        ("2D 10x10 (N=100)", (10, 10)),
        ("2D 20x20 (N=400)", (20, 20)),
    ]
    for label, grid_shape in scaling_configs:
        steps_list = []
        for trial in range(trials):
            model = VoterModel(shape=grid_shape, rho=0.75, seed=3000 + trial)
            _, steps = model.run_until_consensus(max_steps=1_000_000)
            steps_list.append(steps)
        print(f"{label:18s} | mean steps={np.mean(steps_list):.1f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(rhos, rhos, "k--", label="Theoretical P(+1)=rho")
    ax1.plot(rhos, win_probs, "o-", label="Empirical 2D (10x10)")
    ax1.set_title("Voter Model: Consensus Probability")
    ax1.set_xlabel("Initial density rho")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(rhos, mean_steps, "s-", color="tab:blue")
    ax2.set_title("Voter Model: Mean Steps to Consensus")
    ax2.set_xlabel("Initial density rho")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def run_opinion_experiments() -> None:
    print("\n--- Relative Agreement Model (50x50 = 2500 agents) ---")
    for e_val in (10.0, 50.0, 100.0):
        for m_val in (0.3, 0.6):
            ra = RelativeAgreementCA(
                shape=(50, 50), e_threshold=e_val, m_influence=m_val, seed=42
            )
            for _ in range(200_000):
                ra.step()
            print(
                f"e={e_val:5.1f}, m={m_val:.1f} | clusters={ra.count_clusters():2d} | "
                f"mean shift={ra.mean_opinion_shift():.2f}"
            )

    print("\n--- Sznajd Model (50x50, periodic boundaries, 10 trials) ---")
    for adv in (0.01, 0.05, 0.10):
        wins = 0
        steps_list: list[int] = []
        for trial in range(10):
            sz = SznajdCA(shape=(50, 50), advantage=adv, seed=2000 + trial)
            winner, steps = sz.run_until_consensus(max_steps=2_000_000)
            if winner == 1:
                wins += 1
            steps_list.append(steps)
        print(
            f"Advantage {adv * 100:4.0f}% | majority win={wins * 10:3d}% | "
            f"mean steps={np.mean(steps_list):.1f}"
        )


def run_segregation_experiments() -> None:
    thresholds = [
        (0.5, "Rule 1: >= 1/2 same group"),
        (1.0 / 3.0, "Rule 2: >= 1/3 same group"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.5))

    print("\n--- Schelling Segregation Model (50x50, 28% empty) ---")
    for idx, (tau, title) in enumerate(thresholds):
        model = SchellingSegregationCA(
            shape=(50, 50), empty_ratio=0.28, similarity_threshold=tau, seed=42
        )
        init_idx = model.segregation_index()
        axes[0, idx].imshow(
            model.grid, cmap="Accent", vmin=0, vmax=2, interpolation="nearest"
        )
        axes[0, idx].set_title(
            f"{title}\nInitial seg. index: {init_idx:.3f}", fontsize=9.5
        )
        axes[0, idx].set_xticks([])
        axes[0, idx].set_yticks([])

        sweeps = 0
        for step in range(1, 101):
            sweeps = step
            if model.step() == 0:
                break

        final_idx = model.segregation_index()
        axes[1, idx].imshow(
            model.grid, cmap="Accent", vmin=0, vmax=2, interpolation="nearest"
        )
        axes[1, idx].set_title(
            f"Final ({sweeps} sweeps)\nFinal seg. index: {final_idx:.3f}", fontsize=9.5
        )
        axes[1, idx].set_xticks([])
        axes[1, idx].set_yticks([])
        print(f"{title}: {sweeps} sweeps | seg_index {init_idx:.3f} -> {final_idx:.3f}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_population_experiments()
    run_richardson_experiments()
    run_voter_experiments()
    run_opinion_experiments()
    run_segregation_experiments()
