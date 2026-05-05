"""Sweep adaptive vs fixed recompute MPC across mass mismatch levels (pointmass)."""

from configs import DT, SEED
import numpy as np
from agents.mpc_python import make_pointmass_mpc
from simulations.simulation import run_simulation, run_pool

N_STEPS = int(20.0 / DT)

# Condition definitions for adaptive vs fixed comparison
_RECOMPUTE_SHORT = 1
_RECOMPUTE_LONG = 5
_RECOMPUTE_ADAPT = 2

CONDITIONS = [
    dict(recompute=_RECOMPUTE_SHORT, adaptive=False),
    dict(recompute=_RECOMPUTE_LONG, adaptive=False),
    dict(recompute=_RECOMPUTE_ADAPT, adaptive=True),
]

# Auto-generate labels
for _c in CONDITIONS:
    _sec = _c['recompute'] * DT
    if _c['adaptive']:
        _c['label'] = 'Adaptive'
    else:
        _c['label'] = f'Recompute: {_sec:.2f}s'

# Style mapping for plotting
COND_COLORS = ['#D64933', '#E8A838', '#2E86AB']
COND_STYLES = {
    c['label']: dict(color=COND_COLORS[i], marker=['o', 's', '^'][i], ls='-')
    for i, c in enumerate(CONDITIONS)
}


def _adaptive_worker(args):
    """Run one episode, return summary tagged with condition and mismatch."""
    label, mismatch, recompute, adaptive, seed, mismatch_a = args

    np.random.seed(seed)

    # Build adapt_args for adaptive episodes
    adapt_args = None
    if adaptive:
        adapt_args = {
            'adapt_class': 'ODEStepAdaptation',
            'adapt_params': ('recompute',),
            'adapt_kwargs': {'min_error_threshold': 0.20, 'max_recompute': 5},
        }

    agent, env = make_pointmass_mpc(
        model_args={'mass': 0.5 * mismatch},
        agent_args={'recompute_interval': recompute},
        adapt_args=adapt_args,
        seed=seed,
    )

    # Run simulation
    _, _, history = run_simulation(agent, env, n_steps=N_STEPS, interval=None)

    # Extract data from history
    states, _, costs = history.get_state_action_cost()
    recompute_trace = history.get_item_history('recompute_interval')

    # Compute summary metrics
    duration_sec = N_STEPS * DT
    total_cost = float(np.sum(costs))

    # Count replanning events by simulating queue consumption
    n_recomputations = 0
    queue_remaining = 0
    for ri in recompute_trace:
        if queue_remaining == 0:
            n_recomputations += 1
            queue_remaining = int(ri)
        queue_remaining -= 1

    # Time-series traces only for the designated mismatch level
    rh_trace = recompute_trace if mismatch == mismatch_a else None
    cum_trace = None
    if mismatch == mismatch_a:
        cum = np.cumsum(costs[1:])
        elapsed = np.arange(1, len(cum) + 1) * DT
        cum_trace = cum / elapsed

    return (label, mismatch, total_cost, duration_sec, duration_sec,
            n_recomputations, rh_trace, cum_trace)


def run_adaptive_sweep(n_episodes=20, mismatches=None, mismatch_a=2.0):
    """Run all conditions across mismatch levels; return sweep data dict."""
    if mismatches is None:
        mismatches = [0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0]

    rng = np.random.RandomState(SEED)
    seeds = rng.randint(0, 2**31, size=n_episodes)

    # One job per (condition, mismatch, episode)
    args_list = [
        (cond['label'], m, cond['recompute'], cond['adaptive'],
         int(seeds[ep]), mismatch_a)
        for cond in CONDITIONS
        for m in mismatches
        for ep in range(n_episodes)
    ]

    print(f"Pointmass adaptive sweep: {len(args_list)} episodes "
          f"({len(CONDITIONS)} conditions x {len(mismatches)} mismatches "
          f"x {n_episodes} eps)")

    raw_results = run_pool(_adaptive_worker, args_list)

    # Group results by condition and mismatch
    labels = [c['label'] for c in CONDITIONS]
    sweep_len = {lab: {m: [] for m in mismatches} for lab in labels}
    sweep_cost = {lab: {m: [] for m in mismatches} for lab in labels}
    sweep_failure = {lab: {m: [] for m in mismatches} for lab in labels}
    sweep_recomp = {lab: {m: [] for m in mismatches} for lab in labels}
    sweep_rh_traces = {lab: [] for lab in labels}
    sweep_cum_traces = {lab: [] for lab in labels}

    for (label, m, total_cost, duration, failure,
         n_recomp, rh_trace, cum_trace) in raw_results:
        sweep_len[label][m].append(duration)
        sweep_cost[label][m].append(total_cost)
        sweep_failure[label][m].append(failure)
        sweep_recomp[label][m].append(n_recomp)

        if rh_trace is not None:
            sweep_rh_traces[label].append(rh_trace)
        if cum_trace is not None:
            sweep_cum_traces[label].append(cum_trace)

    return dict(
        mismatches=mismatches, mismatch_a=mismatch_a,
        sweep_len=sweep_len, sweep_cost=sweep_cost,
        sweep_failure=sweep_failure, sweep_recomp=sweep_recomp,
        sweep_rh_traces=sweep_rh_traces, sweep_cum_traces=sweep_cum_traces,
    )
