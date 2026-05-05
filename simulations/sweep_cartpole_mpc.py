"""Parallel MPC grid sweep over horizon, recompute interval, and model mismatch."""

from configs import SEED, RESULTS_DIR
import numpy as np
from datetime import datetime
from agents.mpc_python import make_cartpole_mpc
from simulations.dataio import pack_mpc_results, save_mpc_results
from simulations.simulation import run_simulation, run_pool

# Sweep grid defaults
HORIZONS = list(range(20, 220, 10))
RECOMPUTE_INTERVALS = list(range(1, 21))
MODEL_LENGTH_FACTORS = [0.5, 1.0]


def _mpc_worker(args):
    """Run all episodes for one grid point, return packed result."""
    horizon, recompute, lf, init_thetas = args

    all_states, all_actions, all_costs = [], [], []
    for theta in init_thetas:
        agent, env = make_cartpole_mpc(
            model_args={'length': 0.5 * lf},
            agent_args={
                'proposal_args': {'tsteps': horizon},
                'recompute_interval': recompute,
            },
        )
        env.reset(np.array([0.0, 0.0, theta, 0.0]))
        _, _, history = run_simulation(agent, env, n_steps=1000, interval=None)
        states, actions, costs = history.get_state_action_cost()
        all_states.append(states)
        all_actions.append(actions)
        all_costs.append(costs)

    return pack_mpc_results(all_states, all_actions, all_costs, horizon, recompute, lf, 1000, SEED)


def run_mpc_sweep(n_episodes=10, output_dir=RESULTS_DIR):
    """Run full MPC grid sweep, saving each result to disk as it completes."""

    # Pre-generate initial angles for reproducibility
    rng = np.random.RandomState(SEED)
    init_thetas = rng.uniform(-0.1, 0.1, size=n_episodes)

    # One job per grid point, skip invalid recompute > horizon
    args_list = [
        (h, r, lf, init_thetas)
        for lf in MODEL_LENGTH_FACTORS
        for h in HORIZONS
        for r in RECOMPUTE_INTERVALS
        if r <= h
    ]

    # Save each result to disk as it arrives
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    saver = lambda r: save_mpc_results(r, output_dir, timestamp=timestamp)

    run_pool(_mpc_worker, args_list, on_result=saver)
    print(f"Saved {len(args_list)} grid points to {output_dir}")
