"""Sweep model mass mismatch for 2D point mass MPC tracking."""

from configs import DT, SEED
import numpy as np
from agents.dynamics import PointMass2D
from agents.utils import figure_eight
from agents.mpc_python import make_pointmass_mpc
from simulations.simulation import run_simulation, run_pool


def _sweep_worker(args):
    """Run one simulation with mismatched model mass, return integrated errors."""
    model_mass, seed, initial_state = args

    np.random.seed(seed)

    sim_time = 40.0
    horizon_steps = int(0.1 / DT)
    n_steps = int(sim_time / DT)

    agent, env = make_pointmass_mpc(
        agent_args={'horizon_steps': horizon_steps},
        model_args={'mass': model_mass},
    )
    env.reset(initial_state)

    # Run simulation without printing
    agent, env, history = run_simulation(agent, env, n_steps=n_steps, interval=None)

    # Compute integrated errors
    states = history.get_item_history('state')
    positions = states[:, :2]
    s_vals = states[:, 4]

    curve_dist = PointMass2D.cost_curve_distance(positions, figure_eight, 1.5)
    tx, ty = figure_eight(s_vals, scale=1.5)
    targets = np.stack([tx, ty], axis=1)
    target_dist = PointMass2D.cost_tracking_distance(positions, targets)

    curve_int = np.sum(curve_dist) * DT
    target_int = np.sum(target_dist) * DT

    return model_mass, curve_int, target_int


def run_sweep(n_episodes=20, mass_values=None, env_mass=None):
    """Sweep model mass values in parallel, return summary statistics.

    Returns list of dicts with keys: mass_factor, mean_curve_int,
    sem_curve_int, mean_target_int, sem_target_int.
    """
    if env_mass is None:
        env_mass = 0.5
    if mass_values is None:
        mass_values = np.arange(0.2, 2.05, 0.1) * env_mass

    # Ensure env_mass is in the sweep
    if not np.any(np.isclose(mass_values, env_mass)):
        mass_values = np.sort(np.append(mass_values, env_mass))

    # Generate per-episode seeds and initial states
    rng = np.random.RandomState(SEED)
    seeds = rng.randint(0, 2**31, size=n_episodes)
    init_xy = rng.normal(0, 0.1, size=(n_episodes, 2))
    init_v = rng.normal(0, 0.05, size=(n_episodes, 2))
    initial_states = [
        np.array([init_xy[i, 0], init_xy[i, 1], init_v[i, 0], init_v[i, 1], 0.0])
        for i in range(n_episodes)
    ]

    # One job per (mass, episode)
    args_list = [
        (m, int(seeds[i]), initial_states[i])
        for m in mass_values
        for i in range(n_episodes)
    ]

    print(f"Pointmass mismatch sweep: {len(args_list)} jobs "
          f"({len(mass_values)} masses x {n_episodes} episodes)")

    # Collect results, flushing to list as they arrive
    all_results = []
    run_pool(_sweep_worker, args_list, on_result=all_results.append)

    # Group by mass and compute summary statistics
    curve_by_mass = {m: [] for m in mass_values}
    target_by_mass = {m: [] for m in mass_values}
    for mass, curve_int, target_int in all_results:
        curve_by_mass[mass].append(curve_int)
        target_by_mass[mass].append(target_int)

    summary = []
    for m in sorted(mass_values):
        c = np.array(curve_by_mass[m])
        t = np.array(target_by_mass[m])
        n = len(c)
        summary.append({
            'mass_factor': m / env_mass,
            'mean_curve_int': float(np.mean(c)),
            'sem_curve_int': float(np.std(c) / np.sqrt(n)),
            'mean_target_int': float(np.mean(t)),
            'sem_target_int': float(np.std(t) / np.sqrt(n)),
        })

    return summary
