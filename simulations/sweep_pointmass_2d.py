"""2D sweep over rollout horizon and recompute interval for point mass."""

from itertools import product

from configs import DT, SEED
import numpy as np
from agents.dynamics import PointMass2D
from agents.utils import GPForceField, figure_eight
from agents.base import Agent
from agents.mpc_python import TrackMPCProposal, TrackMPCEvaluation, TrackMPCDecision
from simulations.simulation import run_simulation, run_pool

# Per-process shared objects (initialized once, reused across jobs)
_shared = {}


def _init_shared():
    """Build force field and env once per worker process."""
    if _shared:
        return
    force_field = GPForceField(seed=SEED)
    initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    _shared['force_field'] = force_field
    _shared['env'] = PointMass2D(
        force_field, figure_eight,
        stateless=False, dt=DT, initial_state=initial_state,
    )


def _sweep_worker(args):
    """Run one simulation with given horizon, recompute interval, mass factor, and seed."""
    horizon, recompute_interval, mass_factor, seed = args
    _init_shared()

    sim_time = 20.0
    n_steps = int(sim_time / DT)

    np.random.seed(seed)

    env = _shared['env']
    env.reset(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))

    # Build model with (possibly mismatched) mass
    model = PointMass2D(
        _shared['force_field'], figure_eight,
        mass=0.5 * mass_factor, stateless=True, dt=DT,
    )

    proposal = TrackMPCProposal(tsteps=horizon, dt=DT)
    parameters = {'recompute_interval': recompute_interval, 'horizon': horizon}
    agent = Agent(proposal, model, TrackMPCEvaluation(), TrackMPCDecision(),
                  parameters=parameters)

    # Run simulation
    agent, env, history = run_simulation(agent, env, n_steps=n_steps, interval=None)

    # Compute cost rate
    costs = history.get_item_history('cost')
    cost_rate = np.sum(costs) / sim_time

    return horizon, recompute_interval, mass_factor, seed, cost_rate


def run_sweep_2d(horizons=None, recompute_intervals=None, mass_factors=None, n_reps=10):
    """Sweep horizon and recompute interval, return summary grids per mass factor."""
    if horizons is None:
        horizons = list(range(2, 51, 4))
    if recompute_intervals is None:
        recompute_intervals = list(range(1, 6))
    if mass_factors is None:
        mass_factors = [1.0]

    # Build grid of all (horizon, recompute_interval, mass_factor, seed) tuples
    args_list = list(product(horizons, recompute_intervals, mass_factors, range(n_reps)))

    # Run sweep
    raw_results = run_pool(_sweep_worker, args_list)

    # Collect per-cell cost rates, keyed by (mass_factor, horizon, recompute)
    rates_by_cell = {}
    for horizon, recompute, mf, seed, cost_rate in raw_results:
        key = (mf, horizon, recompute)
        if key not in rates_by_cell:
            rates_by_cell[key] = []
        rates_by_cell[key].append(cost_rate)

    # Compute median cost rate grid per mass factor
    grids = {}
    for mf in mass_factors:
        grid = np.zeros((len(horizons), len(recompute_intervals)))
        for hi, h in enumerate(horizons):
            for ri, r in enumerate(recompute_intervals):
                grid[hi, ri] = np.median(rates_by_cell[(mf, h, r)])
        grids[mf] = grid

    return {
        'grids': grids,
        'horizons': horizons,
        'recompute_intervals': recompute_intervals,
        'mass_factors': mass_factors,
    }
