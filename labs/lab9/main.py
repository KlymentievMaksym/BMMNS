import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from labs.lab9.ca.relative_agreement import RelativeAgreementCA
from labs.lab9.ca.segregation import SchellingSegregationCA
from labs.lab9.ca.sznajd import SznajdCA
from labs.lab9.ca.voter import VoterModel
from labs.lab9.ode.population import exponential_growth, logistic_evolution
from labs.lab9.ode.richardson import richardson_arms_race


def run_population_experiments() -> None:
    print("Running Population Growth Experiments...")
    y_exp = exponential_growth(y0=10.0, a=0.15, steps=20)
    t_log, n_log = logistic_evolution(
        n0=10.0, r=0.01, k=100.0, m=0.2, dt=0.1, steps=200
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(y_exp, marker="o", color="blue")
    ax1.set_title("Exponential Growth")
    ax1.grid(True)

    ax2.plot(t_log, n_log, color="green")
    ax2.set_title("Logistic Evolution")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def run_richardson_experiments() -> None:
    print("Running Richardson Arms Race Experiments...")
    t_vals, x_vals, y_vals, eq, is_stable = richardson_arms_race(
        x0=10.0, y0=5.0, a=0.2, m=0.4, r=1.0, b=0.3, n=0.5, s=2.0
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(t_vals, x_vals, label="Country X", color="gold")
    ax1.plot(t_vals, y_vals, label="Country Y", color="green")
    ax1.set_title(f"Arms Race (Stable: {is_stable})")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(x_vals, y_vals, color="purple")
    if eq:
        ax2.plot(eq[0], eq[1], "ro", label=f"Equilibrium {eq}")
        ax2.legend()
    ax2.set_title("Phase Portrait")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def run_all_ca_animations() -> None:
    print("Running Unified 2x2 Animation for All Social CA Models...")

    shape = (30, 30)
    voter = VoterModel(shape=shape, rho=0.55)
    sznajd = SznajdCA(shape=shape, bias_ratio=0.55)
    ra = RelativeAgreementCA(shape=shape, e_threshold=0.25, m_influence=0.4)
    schelling = SchellingSegregationCA(
        shape=shape, empty_ratio=0.25, similarity_threshold=0.5
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    ax_voter, ax_sznajd = axes[0, 0], axes[0, 1]
    ax_ra, ax_schelling = axes[1, 0], axes[1, 1]

    im_voter = ax_voter.imshow(
        voter.grid, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest"
    )
    ax_voter.set_title("Voter Model")

    im_sznajd = ax_sznajd.imshow(
        sznajd.grid, cmap="bwr", vmin=-1, vmax=1, interpolation="nearest"
    )
    ax_sznajd.set_title("Sznajd Model")

    im_ra = ax_ra.imshow(
        ra.grid, cmap="viridis", vmin=0, vmax=1, interpolation="nearest"
    )
    ax_ra.set_title("Relative Agreement Model")

    im_schelling = ax_schelling.imshow(
        schelling.grid, cmap="Accent", vmin=0, vmax=2, interpolation="nearest"
    )
    ax_schelling.set_title("Schelling Segregation Model")

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    def update(frame: int):
        for _ in range(200):
            if not voter.is_consensus():
                voter.step()

            if not sznajd.is_consensus():
                sznajd.step()

            ra.step()

        schelling.step()

        im_voter.set_array(voter.grid)
        im_sznajd.set_array(sznajd.grid)
        im_ra.set_array(ra.grid)
        im_schelling.set_array(schelling.grid)

        ax_voter.set_title(
            f"Voter Model (Consensus: {voter.is_consensus()})"
        )
        ax_sznajd.set_title(
            f"Sznajd Model (Consensus: {sznajd.is_consensus()})"
        )
        ax_ra.set_title(f"Relative Agreement (Step {frame * 200})")
        ax_schelling.set_title(f"Schelling Segregation (Step {frame})")

        return im_voter, im_sznajd, im_ra, im_schelling

    _ = FuncAnimation(
        fig, update, frames=200, interval=50, blit=False, repeat=False
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_population_experiments()
    run_richardson_experiments()
    run_all_ca_animations()