import numpy as np

from labs.lab8.ode.solvers import Method, solve_ode
from labs.lab9.ca.relative_agreement import RelativeAgreementCA
from labs.lab9.ca.segregation import SchellingSegregationCA
from labs.lab9.ca.sznajd import SznajdCA
from labs.lab9.ca.visualizer import (
    plot_relative_agreement,
    plot_segregation,
    plot_sznajd,
    plot_voter,
    plot_voter_scaling,
)
from labs.lab9.ca.voter import VoterModel
from labs.lab9.cli import parse_config
from labs.lab9.entities import (
    RICHARDSON_CASES,
    CAConfig,
    ExponentialParams,
    ODEConfig,
    RelativeAgreementConfig,
    SegregationConfig,
    SznajdConfig,
    VoterConfig,
)
from labs.lab9.ode.analysis import analyze_richardson, fit_wwi_data
from labs.lab9.ode.models import (
    exponential_discrete,
    exponential_system,
    logistic_system,
    richardson_system,
)
from labs.lab9.ode.visualizer import (
    plot_exponential,
    plot_logistic,
    plot_richardson,
    plot_wwi,
)


# ----------------------------------------------------------------------- ODE
def run_exponential_experiment(config: ODEConfig) -> None:
    steps = int(config.t_max)
    stride = int(round(1.0 / config.dt))
    t = np.array([])
    series: dict[float, dict[str, np.ndarray]] = {}
    discrete: dict[float, np.ndarray] = {}

    for rate in config.exp_rates:
        series[rate] = {}
        for method in Method:
            t, hist = solve_ode(
                exponential_system,
                (config.exp_y0,),
                ExponentialParams(rate),
                config.t_max,
                config.dt,
                method,
            )
            series[rate][method.name] = hist[:, 0]
        discrete[rate] = exponential_discrete(config.exp_y0, rate, steps)
        rk4 = series[rate][Method.RK4.name][::stride][: steps + 1]
        rel_err = np.max(np.abs(rk4 - discrete[rate]) / discrete[rate])
        print(
            f"a={rate:.2f}: y({steps})={discrete[rate][-1]:.1f}, max rel. err RK4 vs discrete={rel_err:.2e}"
        )

    plot_exponential(t, series, discrete)


def run_logistic_experiment(config: ODEConfig) -> None:
    p = config.logistic
    equilibrium = p.k - p.m / p.r
    histories: dict[float, np.ndarray] = {}
    t = np.array([])
    for n0 in config.logistic_initial:
        t, hist = solve_ode(
            logistic_system, (n0,), p, config.logistic_t_max, config.dt, Method.RK4
        )
        histories[n0] = hist[:, 0]
        print(f"N0={n0:.0f}: N(T)={hist[-1, 0]:.1f}, Nc={equilibrium:.1f}")
    plot_logistic(t, histories, equilibrium)


def run_richardson_experiment(config: ODEConfig) -> None:
    runs = []
    t = np.array([])
    for case in RICHARDSON_CASES:
        stability = analyze_richardson(case.params)
        t, hist = solve_ode(
            richardson_system,
            case.initial,
            case.params,
            config.richardson_t_max,
            config.richardson_dt,
            Method.RK4,
        )
        runs.append((case.title, hist, stability))
        print(
            f"Case {case.title}: mn-ab={stability.det:.3f}, "
            f"eig={np.round(stability.eigenvalues, 3)}, stable={stability.stable}, "
            f"eq={stability.equilibrium}, final=({hist[-1, 0]:.1f}, {hist[-1, 1]:.1f})"
        )
    plot_richardson(t, runs)

    fit = fit_wwi_data()
    print(f"WWI fit: k=a-m={fit.k:.3f}, f=r+s={fit.f:.2f}")
    plot_wwi(fit)


def run_ode_experiments(config: ODEConfig) -> None:
    run_exponential_experiment(config)
    run_logistic_experiment(config)
    run_richardson_experiment(config)


# ------------------------------------------------------------------------ CA
def run_voter_experiment(config: VoterConfig) -> None:
    win_probs: list[float] = []
    mean_steps: list[float] = []
    for idx, rho in enumerate(config.rhos):
        wins, steps_list = 0, []
        for trial in range(config.trials):
            seed = config.seed + 1000 * idx + trial
            model = VoterModel(shape=config.shape, rho=rho, seed=seed)
            winner, steps = model.run_until_consensus(config.max_steps)
            wins += winner == 1
            steps_list.append(steps)
        win_probs.append(wins / config.trials)
        mean_steps.append(float(np.mean(steps_list)))
        print(
            f"rho={rho:.2f} | P(+1)={win_probs[-1]:.2f} | mean steps={mean_steps[-1]:.1f}"
        )
    plot_voter(config.rhos, win_probs, mean_steps)

    def mean_consensus_time(shape: tuple[int, ...]) -> float:
        times = [
            VoterModel(
                shape, config.scaling_rho, seed=config.seed + 500 + i
            ).run_until_consensus(config.max_steps)[1]
            for i in range(config.scaling_trials)
        ]
        return float(np.mean(times))

    n_1d = list(config.sizes_1d)
    n_2d = [side * side for side in config.sides_2d]
    steps_1d = np.array([mean_consensus_time((n,)) for n in n_1d])
    steps_2d = np.array([mean_consensus_time((s, s)) for s in config.sides_2d])
    for n, st in zip(n_1d, steps_1d, strict=True):
        print(f"d=1 N={n:4d} | mean steps={st:.1f}")
    for n, st in zip(n_2d, steps_2d, strict=True):
        print(f"d=2 N={n:4d} | mean steps={st:.1f}")
    plot_voter_scaling(n_1d, steps_1d, n_2d, steps_2d)


def run_relative_agreement_experiment(config: RelativeAgreementConfig) -> None:
    results = []
    centre = tuple(dim // 2 for dim in config.shape)
    for e in config.e_values:
        for m in config.m_values:
            model = RelativeAgreementCA(config.shape, e, m, seed=config.seed)
            for _ in range(config.steps):
                model.step()
            print(
                f"e={e:5.1f}, m={m:.1f} | clusters={model.count_clusters():3d} | "
                f"mean shift={model.mean_opinion_shift():.2f}"
            )
            results.append(
                (
                    f"e={e:g}, m={m:g}\nclusters={model.count_clusters()}",
                    model.grid.copy(),
                )
            )
    plot_relative_agreement(results)

    # influence of a single stubborn individual on the rest
    e, m = config.e_values[len(config.e_values) // 2], config.m_values[0]
    free = RelativeAgreementCA(config.shape, e, m, seed=config.seed)
    biased = RelativeAgreementCA(
        config.shape,
        e,
        m,
        zealot=centre,
        zealot_opinion=config.zealot_opinion,
        seed=config.seed,
    )
    for _ in range(config.steps):
        free.step()
        biased.step()
    print(
        f"single zealot (e={e:g}, m={m:g}): mean opinion {free.mean_opinion():.1f} -> "
        f"{biased.mean_opinion():.1f}, clusters {free.count_clusters()} -> {biased.count_clusters()}"
    )
    plot_relative_agreement(
        [("no zealot", free.grid), ("one zealot (opinion 256)", biased.grid)]
    )


def run_sznajd_experiment(config: SznajdConfig) -> None:
    win_rates: list[float] = []
    mean_steps: list[float] = []
    for adv in config.advantages:
        wins, steps_list = 0, []
        for trial in range(config.trials):
            model = SznajdCA(config.shape, adv, seed=config.seed + trial)
            winner, steps = model.run_until_consensus(config.max_steps)
            wins += winner == 1
            steps_list.append(steps)
        win_rates.append(100.0 * wins / config.trials)
        mean_steps.append(float(np.mean(steps_list)))
        print(
            f"advantage {adv * 100:4.0f}% | majority wins={win_rates[-1]:5.1f}% | mean steps={mean_steps[-1]:.1f}"
        )
    plot_sznajd(config.advantages, win_rates, mean_steps)


def run_segregation_experiment(config: SegregationConfig) -> None:
    snapshots = []
    for tau in config.thresholds:
        model = SchellingSegregationCA(
            config.shape, config.empty_ratio, tau, seed=config.seed
        )
        initial_grid, initial_idx = model.grid.copy(), model.segregation_index()
        sweeps = 0
        for sweeps in range(1, config.max_sweeps + 1):
            if model.step() == 0:
                break
        final_idx = model.segregation_index()
        print(
            f"tau={tau:.2f}: {sweeps} sweeps | segregation index {initial_idx:.3f} -> {final_idx:.3f}"
        )
        snapshots.append((f"tau={tau:.2f}", initial_grid, model.grid.copy()))
    plot_segregation(snapshots)


def run_ca_experiment(config: CAConfig) -> None:
    run_voter_experiment(config.voter)
    run_relative_agreement_experiment(config.relative_agreement)
    run_sznajd_experiment(config.sznajd)
    run_segregation_experiment(config.segregation)


if __name__ == "__main__":
    config = parse_config()
    run_ode_experiments(config.ode)
    run_ca_experiment(config.ca)
