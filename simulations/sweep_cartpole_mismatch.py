"""Sweep pole length mismatch at fixed MPC settings for Figure 2A."""

from configs import DT, SEED
import numpy as np
from agents.dynamics import make_perturbation_cartpole
from agents.mpc_python import make_cartpole_mpc
from simulations.simulation import run_simulation, run_pool

# Sweep defaults
MODEL_LENGTH_FACTORS = list(np.arange(0.2, 4.2, 0.2))
HORIZON = 50          # 1.0s at DT=0.02
RECOMPUTE_INTERVAL = 1  # every step (0.02s)


def _mismatch_worker(args):
    """Run one episode at given length factor, return cost rate and duration."""
    length_factor, initial_theta, perturbation = args

    agent, env = make_cartpole_mpc(
        model_args={'length': 0.5 * length_factor},
        agent_args={
            'proposal_args': {'tsteps': HORIZON},
            'recompute_interval': RECOMPUTE_INTERVAL,
        },
    )
    env.reset(np.array([0.0, 0.0, initial_theta, 0.0]))

    _, _, history = run_simulation(
        agent, env, n_steps=1000, perturbation=perturbation, interval=None,
    )
    states, _, costs = history.get_state_action_cost()

    # Cost rate over full episode
    duration_sec = 1000 * DT
    cost_rate = float(np.sum(costs)) / duration_sec

    return length_factor, cost_rate, duration_sec


def run_mismatch_sweep(n_episodes=60, length_factors=None, perturbation_kwargs=None):
    """Sweep length factors at fixed H/R, return grouped results dict.

    perturbation_kwargs: if provided, dict passed to make_perturbation_cartpole
        to build a shared perturbation schedule for all episodes.
    """
    if length_factors is None:
        length_factors = MODEL_LENGTH_FACTORS

    rng = np.random.RandomState(SEED)
    init_thetas = rng.uniform(-0.1, 0.1, size=n_episodes)

    # Build perturbation schedule (shared across all episodes, or None)
    perturbation = None
    if perturbation_kwargs is not None:
        perturbation = make_perturbation_cartpole(1000, **perturbation_kwargs)

    # One job per (length_factor, episode)
    args_list = [
        (lf, theta, perturbation)
        for lf in length_factors
        for theta in init_thetas
    ]

    print(f"Mismatch sweep: {len(args_list)} episodes "
          f"({len(length_factors)} factors x {n_episodes} eps)")

    raw_results = run_pool(_mismatch_worker, args_list)

    # Group by length factor
    cost_rates = {lf: [] for lf in length_factors}
    durations = {lf: [] for lf in length_factors}
    for lf, cr, dur in raw_results:
        cost_rates[lf].append(cr)
        durations[lf].append(dur)

    # Compute summary statistics
    summary = []
    for lf in length_factors:
        cr = np.array(cost_rates[lf])
        dur = np.array(durations[lf])
        n = len(cr)
        summary.append({
            'length_factor': lf,
            'mean_cost_rate': float(np.mean(cr)),
            'sem_cost_rate': float(np.std(cr) / np.sqrt(n)),
            'mean_duration_sec': float(np.mean(dur)),
            'sem_duration_sec': float(np.std(dur) / np.sqrt(n)),
        })

    return summary
