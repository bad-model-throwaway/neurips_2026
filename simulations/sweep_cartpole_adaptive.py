"""Sweep adaptive vs fixed recompute MPC across mismatch levels."""

from configs import DT, SEED, FAILURE_ANGLE
import numpy as np
from agents.mpc import make_mpc
from agents.mujoco_dynamics import MuJoCoCartPoleDynamics
from agents.adaptation import make_adapter
from simulations.simulation import run_simulation, run_pool

N_STEPS = 400
H_CARTPOLE = 44           # Fig 2 matched-best (compute-cheapest) operating point

_RECOMPUTE_SHORT = 1
_RECOMPUTE_LONG = 13
_RECOMPUTE_ADAPT = 3

ADAPT_CLASS = 'ODEStepAdaptation'  # default; smoke tests pass 'TheoryStepAdaptation'

# Condition definitions for adaptive vs fixed comparison
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


def _build_adapt_args(adapt_class):
    if adapt_class == 'ODEStepAdaptation':
        return {
            'adapt_class': 'ODEStepAdaptation',
            'adapt_params': ('recompute',),
            'adapt_kwargs': {'min_error_threshold': 0.08, 'relax_step': 0.05},
        }
    if adapt_class == 'TheoryStepAdaptation':
        # Bounds per plan.md cross-cutting concern #3, validated against
        # data/results/grid_cartpole.pkl: H 50-70 sits inside the
        # ≥90%-survival plateau at mm=1.5; H<44 has 0% survival even matched.
        return {
            'adapt_class': 'TheoryStepAdaptation',
            'adapt_params': ('recompute', 'horizon'),
            'adapt_kwargs': {
                'max_recompute': 8,
                'min_horizon': 30,
                'max_horizon': 138,
                'noise_floor_window': 10,
                'a_step': 0.1,
            },
        }
    raise ValueError(f"unknown adapt_class: {adapt_class!r}")


def _adaptive_worker(args):
    """Run one episode, return summary tagged with condition and mismatch."""
    label, mismatch, recompute, adaptive, initial_theta, mismatch_a, adapt_class = args

    adapt_args = _build_adapt_args(adapt_class) if adaptive else None

    env = MuJoCoCartPoleDynamics(stateless=False)
    agent = make_mpc('cartpole', H=H_CARTPOLE, R=recompute, mismatch_factor=mismatch)
    if adapt_args is not None:
        agent.adaptation = make_adapter(adapt_args)
    env.reset(np.array([0.0, 0.0, initial_theta, 0.0]))

    # Run simulation
    _, _, history = run_simulation(agent, env, n_steps=N_STEPS, interval=None)

    # Extract data from history
    states, _, costs = history.get_state_action_cost()
    recompute_trace = history.get_item_history('recompute_interval')
    horizon_trace = history.get_item_history('horizon')

    # Compute summary metrics
    duration_sec = N_STEPS * DT
    total_cost = float(np.sum(costs))

    # Failure time: first step where angle exceeds threshold
    angles = np.degrees(np.abs(states[:, 2]))
    failed = np.where(angles > FAILURE_ANGLE)[0]
    failure_sec = float(failed[0] * DT) if len(failed) > 0 else duration_sec

    # Replan boundaries: queue empty → new plan of length recompute_interval
    # is enqueued and a horizon-H rollout is computed. Σ ℓₖ over replan events
    # is the machine-independent compute proxy used in Fig 3 (captures both R
    # and H knobs; H is constant for fixed conditions, varies for theory).
    n_recomputations = 0
    total_rollout_steps = 0
    queue_remaining = 0
    for ri, h in zip(recompute_trace, horizon_trace):
        if queue_remaining == 0:
            n_recomputations += 1
            total_rollout_steps += int(h)
            queue_remaining = int(ri)
        queue_remaining -= 1

    # Per-step R and H traces are now captured at every mismatch (cheap, ~kB/episode)
    # for diagnostics and supplementary figures. cum_trace is still mismatch_a-only
    # because the cumulative-cost panel highlights one regime.
    r_trace = np.asarray(recompute_trace, dtype=np.int16)
    h_trace = np.asarray(horizon_trace, dtype=np.int16)
    cum_trace = None
    if mismatch == mismatch_a:
        cum = np.cumsum(costs[1:])
        elapsed = np.arange(1, len(cum) + 1) * DT
        cum_trace = cum / elapsed

    last_states = np.asarray(states[-20:])

    return (label, mismatch, total_cost, duration_sec, failure_sec,
            n_recomputations, r_trace, h_trace, cum_trace, last_states,
            total_rollout_steps)


def run_adaptive_sweep(n_episodes=30, mismatches=None, mismatch_a=2.5,
                       adapt_class=None):
    """Run all conditions across mismatch levels; return sweep data dict."""
    if mismatches is None:
        mismatches = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    if adapt_class is None:
        adapt_class = ADAPT_CLASS

    rng = np.random.RandomState(SEED)
    init_thetas = rng.uniform(-0.1, 0.1, size=n_episodes)

    # One job per (condition, mismatch, episode)
    args_list = [
        (cond['label'], m, cond['recompute'], cond['adaptive'],
         init_thetas[ep], mismatch_a, adapt_class)
        for cond in CONDITIONS
        for m in mismatches
        for ep in range(n_episodes)
    ]

    raw_results = run_pool(_adaptive_worker, args_list)

    # Group results by condition and mismatch
    labels = [c['label'] for c in CONDITIONS]
    sweep_len = {lab: {m: [] for m in mismatches} for lab in labels}
    sweep_cost = {lab: {m: [] for m in mismatches} for lab in labels}
    sweep_failure = {lab: {m: [] for m in mismatches} for lab in labels}
    sweep_recomp = {lab: {m: [] for m in mismatches} for lab in labels}
    sweep_rollout_steps = {lab: {m: [] for m in mismatches} for lab in labels}
    last_states_dict = {lab: {m: [] for m in mismatches} for lab in labels}
    # New: full R and H traces keyed by mismatch (one list of arrays per mm).
    sweep_R_full = {lab: {m: [] for m in mismatches} for lab in labels}
    sweep_H_full = {lab: {m: [] for m in mismatches} for lab in labels}
    # Legacy: R-only traces, mismatch_a only — kept for backward compat with
    # visualization/figures.py and analysis.py.
    sweep_rh_traces = {lab: [] for lab in labels}
    sweep_cum_traces = {lab: [] for lab in labels}

    for (label, m, total_cost, duration, failure,
         n_recomp, r_trace, h_trace, cum_trace,
         last_states, rollout_steps) in raw_results:
        sweep_len[label][m].append(duration)
        sweep_cost[label][m].append(total_cost)
        sweep_failure[label][m].append(failure)
        sweep_recomp[label][m].append(n_recomp)
        sweep_rollout_steps[label][m].append(rollout_steps)
        last_states_dict[label][m].append(last_states)

        sweep_R_full[label][m].append(r_trace)
        sweep_H_full[label][m].append(h_trace)
        if m == mismatch_a:
            sweep_rh_traces[label].append(r_trace)
        if cum_trace is not None:
            sweep_cum_traces[label].append(cum_trace)

    return dict(
        mismatches=mismatches, mismatch_a=mismatch_a,
        sweep_len=sweep_len, sweep_cost=sweep_cost,
        sweep_failure=sweep_failure, sweep_recomp=sweep_recomp,
        sweep_rollout_steps=sweep_rollout_steps,
        last_states=last_states_dict,
        sweep_R_full=sweep_R_full, sweep_H_full=sweep_H_full,
        sweep_rh_traces=sweep_rh_traces, sweep_cum_traces=sweep_cum_traces,
        env='cartpole',
    )
