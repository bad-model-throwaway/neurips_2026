"""Mid-episode mismatch sweep — walker row C for Figure 3.

Episode starts matched (r=1.0). At SWITCH_STEP = N_STEPS // 2 the env
torso mass is scaled by `post_factor` via env.apply_mismatch(kind='torso_mass').
The agent's planning model is never notified. Two conditions, both
initialised at the Fig 2 white-star (H=H_WALKER, R=_RECOMPUTE_ADAPT)
to mirror sweep_walker_summary.py post-#223:

    Fixed (R·dt=0.07s) — no-adapt counterfactual at white-star init
    Adaptive           — TheoryStepAdaptation (joint R+H), white-star init

Same `_build_adapt_args('TheoryStepAdaptation')` as the summary sweep so
the two Figure 3 rows share an identical algorithm config. The Fixed
arm reuses the same adapter in monitor-only mode (`adapt_params=()`)
so the snapshot schema is uniform.

Run from worktree root:

    python -m simulations.sweep_walker_midswitch
    python -m simulations.sweep_walker_midswitch --plot-only
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import visualization.style  # noqa: F401

from agents.mpc import make_mpc
from agents.mujoco_dynamics import WalkerDynamics
from agents.adaptation import make_adapter
from configs import PLOTS_DIR, RESULTS_DIR, SEED
from simulations.simulation import run_simulation, run_pool
from simulations.sweep_walker_adaptive import (
    N_STEPS, H_WALKER, DT, _RECOMPUTE_ADAPT, _build_adapt_args,
)


SWITCH_STEP = N_STEPS // 2
POST_FACTORS = [1.5]  # figure_3 row C plots r=1.5 only; see _MIDSWITCH_REPR_FACTOR
N_EPISODES = 20

CONDITIONS = [
    dict(recompute=_RECOMPUTE_ADAPT, adaptive=False, label='Fixed (R·dt=0.07s)'),
    dict(recompute=_RECOMPUTE_ADAPT, adaptive=True,  label='Adaptive'),
]

PKL_NAME = 'midswitch_walker.pkl'
PNG_NAME = 'midswitch_walker.png'
PDF_NAME = 'midswitch_walker.pdf'

# Match the post-#223 summary palette: Okabe-Ito orange/blue.
COND_COLORS = {
    'Fixed (R·dt=0.07s)': '#E69F00',
    'Adaptive':           '#0072B2',
}


def _midswitch_worker(args):
    """Run one mid-switch episode; return trace dict."""
    label, recompute, adaptive, post_factor, seed = args

    np.random.seed(int(seed))

    env = WalkerDynamics(stateless=False)
    agent = make_mpc('walker', H=H_WALKER, R=recompute, mismatch_factor=1.0)

    adapt_args = _build_adapt_args('TheoryStepAdaptation')
    if not adaptive:
        adapt_args = {**adapt_args, 'adapt_params': ()}
    agent.adaptation = make_adapter(adapt_args)

    env.reset(env.get_default_initial_state())

    switched = [False]

    def env_mismatch_fn(e, step_idx):
        if not switched[0] and step_idx >= SWITCH_STEP:
            e.apply_mismatch(post_factor, kind='torso_mass')
            switched[0] = True

    _, _, history = run_simulation(
        agent, env, n_steps=N_STEPS,
        env_mismatch_fn=env_mismatch_fn, interval=None,
    )

    states, actions, costs = history.get_state_action_cost()
    tau      = np.asarray(history.get_item_history('recompute_interval'), dtype=float)
    horizon  = np.asarray(history.get_item_history('horizon'), dtype=float)
    torso_z  = states[:, 1] if states.ndim == 2 and states.shape[1] > 1 else np.full(N_STEPS, np.nan)

    return dict(
        label=label, post_factor=post_factor,
        cost=costs, tau=tau, horizon=horizon, torso_z=torso_z,
    )


def run_midswitch_sweep():
    """Run all conditions × post_factors × episodes; return sweep dict."""
    rng = np.random.RandomState(SEED)
    seeds = rng.randint(0, 2**31 - 1, size=N_EPISODES)

    args_list = [
        (cond['label'], cond['recompute'], cond['adaptive'], pf, seeds[ep])
        for cond in CONDITIONS
        for pf in POST_FACTORS
        for ep in range(N_EPISODES)
    ]

    raw = run_pool(_midswitch_worker, args_list)

    labels = [c['label'] for c in CONDITIONS]
    traces = {lab: {pf: [] for pf in POST_FACTORS} for lab in labels}
    for r in raw:
        traces[r['label']][r['post_factor']].append(r)

    out_traces = {}
    for lab in labels:
        out_traces[lab] = {}
        for pf in POST_FACTORS:
            eps = traces[lab][pf]
            stack = lambda key, eps=eps: np.stack([np.asarray(e[key]) for e in eps])
            out_traces[lab][pf] = dict(
                cost=stack('cost'),
                tau=stack('tau'),
                horizon=stack('horizon'),
                torso_z=stack('torso_z'),
            )

    return dict(
        labels=labels, post_factors=POST_FACTORS,
        switch_step=SWITCH_STEP, dt=DT, n_steps=N_STEPS,
        traces=out_traces,
    )


def _sanity_plot(sweep, repr_factor=1.5, save=True):
    """Quick two-row plot (error, tau) at a representative post_factor."""
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 4.5), sharex=True)
    t = np.arange(sweep['n_steps']) * sweep['dt']
    switch_t = sweep['switch_step'] * sweep['dt']

    for ax in axes:
        ax.axvline(switch_t, color='0.5', ls=':', lw=0.8)

    for lab in sweep['labels']:
        tr = sweep['traces'][lab][repr_factor]
        color = COND_COLORS[lab]
        ls = '-' if lab == 'Adaptive' else '--'

        axes[0].plot(t, np.nanmean(tr['horizon'], axis=0), color=color, ls=ls, label=lab)
        axes[1].plot(t, np.nanmean(tr['tau'],     axis=0), color=color, ls=ls)

    axes[0].set_ylabel(r'$H$ (steps)')
    axes[1].set_ylabel(r'$\tau$ (steps)')
    axes[1].set_xlabel('Time (s)')
    axes[0].legend(loc='upper right', fontsize=7)
    axes[0].set_title(f'Mid-switch walker (r={repr_factor})')
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        for name in (PNG_NAME, PDF_NAME):
            path = os.path.join(PLOTS_DIR, name)
            fig.savefig(path)
            print(f'Saved: {path}')
    return fig, axes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip sweep; load existing pkl and re-render sanity plot.')
    args = parser.parse_args()

    path = os.path.join(RESULTS_DIR, PKL_NAME)
    if args.plot_only:
        with open(path, 'rb') as f:
            sweep = pickle.load(f)
        print(f'Loaded: {path}')
    else:
        sweep = run_midswitch_sweep()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(sweep, f)
        print(f'Saved: {path}')

    _sanity_plot(sweep)


if __name__ == '__main__':
    main()
