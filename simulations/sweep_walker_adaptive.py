"""Sweep adaptive vs fixed recompute MPC across mismatch levels — walker.

Walker runs at `ENV_DT['walker'] = 0.01` s; N_STEPS=1000 → 10 s episodes.
Failure cut-off is `torso_z < 0.7` (state[..., 18]).
"""
from __future__ import annotations

import numpy as np

from agents.mpc import make_mpc
from agents.mujoco_dynamics import WalkerDynamics
from agents.adaptation import make_adapter
from configs import ENV_DT, SEED
from simulations.simulation import run_simulation, run_pool


DT = ENV_DT['walker']  # walker ctrl_dt = 0.01 s
N_STEPS = 800          # 8 s episodes (matches Fig 2 grid)
H_WALKER = 67           # Fig 2 matched-best (compute-cheapest) operating point
FAIL_TORSO_Z = 0.7      # cut-off for "fell"

_RECOMPUTE_SHORT = 1
_RECOMPUTE_LONG  = 10
_RECOMPUTE_ADAPT = 7

ADAPT_CLASS = 'ODEStepAdaptation'  # default; smoke tests pass 'TheoryStepAdaptation'

CONDITIONS = [
    dict(recompute=_RECOMPUTE_SHORT, adaptive=False),
    dict(recompute=_RECOMPUTE_LONG,  adaptive=False),
    dict(recompute=_RECOMPUTE_ADAPT, adaptive=True),
]

for _c in CONDITIONS:
    _sec = _c['recompute'] * DT
    if _c['adaptive']:
        _c['label'] = 'Adaptive'
    else:
        _c['label'] = f'Recompute: {_sec:.2f}s'

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
            'adapt_kwargs': {'min_error_threshold': 0.15, 'relax_step': 0.05},
        }
    if adapt_class == 'TheoryStepAdaptation':
        # Bounds per plan.md cross-cutting concern #3 / test_walker_midswitch.
        # Validated against data/results/grid_walker.pkl: R=5 has ≥99%
        # survival at H=82 (mm=1.0); H 50-80 sits inside the stable plateau.
        # Sweep starts R=8/H=82 → first adapt step clips to bounds (benign).
        return {
            'adapt_class': 'TheoryStepAdaptation',
            'adapt_params': ('recompute', 'horizon'),
            'adapt_kwargs': {
                'max_recompute': 9,
                'min_horizon': 30,
                'max_horizon': 150,
                'noise_floor_window': 10,
                'a_step': 0.1,
            },
        }
    raise ValueError(f"unknown adapt_class: {adapt_class!r}")


def _adaptive_worker(args):
    """Run one walker episode, return summary tagged with condition and mismatch.

    `seed` drives the MPPI sampler RNG (initial state is deterministic but the
    sampler is stochastic).
    """
    label, mismatch, recompute, adaptive, seed, mismatch_a, adapt_class = args

    np.random.seed(int(seed))

    adapt_args = _build_adapt_args(adapt_class) if adaptive else None

    env = WalkerDynamics(stateless=False, speed_goal=1.5)
    agent = make_mpc('walker', H=H_WALKER, R=recompute, mismatch_factor=mismatch)
    if adapt_args is not None:
        agent.adaptation = make_adapter(adapt_args)
    env.reset(env.get_default_initial_state())

    _, _, history = run_simulation(agent, env, n_steps=N_STEPS, interval=None)

    # Walker is MJPC-aligned: use the per-step scalar cost directly.
    states = history.get_item_history('state')
    costs = history.get_item_history('cost')
    recompute_trace = history.get_item_history('recompute_interval')
    horizon_trace = history.get_item_history('horizon')

    duration_sec = N_STEPS * DT
    total_cost = float(np.sum(costs))

    fell = np.where(states[:, 18] < FAIL_TORSO_Z)[0]
    failure_sec = float(fell[0] * DT) if len(fell) > 0 else duration_sec

    # Replay agent replan-queue logic to count replanning events and
    # accumulate Σ ℓₖ (machine-independent compute proxy used in Fig 3).
    n_recomputations = 0
    total_rollout_steps = 0
    queue_remaining = 0
    for ri, h in zip(recompute_trace, horizon_trace):
        if queue_remaining == 0:
            n_recomputations += 1
            total_rollout_steps += int(h)
            queue_remaining = int(ri)
        queue_remaining -= 1

    r_trace = np.asarray(recompute_trace, dtype=np.int16)
    h_trace = np.asarray(horizon_trace, dtype=np.int16)
    cum_trace = None
    if mismatch == mismatch_a:
        cum = np.cumsum(costs[1:])
        elapsed = np.arange(1, len(cum) + 1) * DT
        cum_trace = cum / elapsed

    last_states = np.asarray(states[-20:])

    return (label, mismatch, total_cost, duration_sec, failure_sec,
            n_recomputations, r_trace, h_trace, cum_trace,
            last_states, total_rollout_steps)


def run_adaptive_sweep(n_episodes=30, mismatches=None, mismatch_a=1.5,
                       conditions=None, adapt_class=None):
    """Run all conditions across mismatch levels; return sweep data dict.

    Signature matches `sweep_cartpole_adaptive.run_adaptive_sweep`.
    """
    if mismatches is None:
        mismatches = [1.0, 1.3, 1.6, 2.0]
    if conditions is None:
        conditions = CONDITIONS
    if adapt_class is None:
        adapt_class = ADAPT_CLASS

    rng = np.random.RandomState(SEED)
    seeds = rng.randint(0, 2**31 - 1, size=n_episodes)

    args_list = [
        (cond['label'], m, cond['recompute'], cond['adaptive'],
         int(seeds[ep]), mismatch_a, adapt_class)
        for cond in conditions
        for m in mismatches
        for ep in range(n_episodes)
    ]

    raw_results = run_pool(_adaptive_worker, args_list)

    labels = [c['label'] for c in conditions]
    sweep_len        = {lab: {m: [] for m in mismatches} for lab in labels}
    sweep_cost       = {lab: {m: [] for m in mismatches} for lab in labels}
    sweep_failure    = {lab: {m: [] for m in mismatches} for lab in labels}
    sweep_recomp     = {lab: {m: [] for m in mismatches} for lab in labels}
    sweep_rollout_steps = {lab: {m: [] for m in mismatches} for lab in labels}
    last_states_dict  = {lab: {m: [] for m in mismatches} for lab in labels}
    sweep_R_full = {lab: {m: [] for m in mismatches} for lab in labels}
    sweep_H_full = {lab: {m: [] for m in mismatches} for lab in labels}
    sweep_rh_traces  = {lab: [] for lab in labels}
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
        env='walker',
    )
