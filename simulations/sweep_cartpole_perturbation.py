"""Perturbation-response sweep for cartpole.

Three controllers are subjected to a single lateral-force impulse at t = T/2
on a matched-dynamics episode (r = 1.0):

    Fixed (R=1)           — performance ceiling
    Adaptive (R=4 init)   — paper's proposed method
    Fixed (R=6)           — impulse-sensitive weak-control arm

Fixed controllers attach a monitoring-only `ODEStepAdaptation(adapt=())` so
the agent `snapshot()` emits the same error/running_error/threshold keys
across all three conditions, keeping the plotter schema uniform.

The impulse is specified in Newtons and converted to ctrl units via the
cartpole actuator gear (XML gear=10), then added to the agent action before
`env.step()`. MuJoCo still clips `action + impulse` to `ctrlrange=[-1, 1]`.
"""
from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import visualization.style  # noqa: F401

from agents.mpc import make_mpc
from agents.mujoco_dynamics import MuJoCoCartPoleDynamics
from agents.adaptation import make_adapter
from configs import DT, PLOTS_DIR, RESULTS_DIR, SEED
from simulations.simulation import run_simulation, run_pool
from simulations.sweep_cartpole_adaptive import N_STEPS


CONDITIONS = [
    dict(recompute=1, adaptive=False, label='Fixed (R=1)'),
    dict(recompute=4, adaptive=True,  label='Adaptive (R=4 init)'),
    dict(recompute=6, adaptive=False, label='Fixed (R=6)'),
]

# Impulse in Newtons → ctrl units via actuator gear (cartpole.xml gear=10).
IMPULSE_N = 5.0
CARTPOLE_GEAR = 10.0
IMPULSE_DURATION_STEPS = 10
IMPULSE_START_FRAC = 0.5
NOMINAL_MISMATCH = 1.0
N_EPISODES = 20

PKL_NAME = 'perturbation_cartpole.pkl'
PNG_NAME = 'perturbation_cartpole.png'
PDF_NAME = 'perturbation_cartpole.pdf'

COND_COLORS = {
    'Fixed (R=1)':           '#009E73',
    'Adaptive (R=4 init)':   '#0072B2',
    'Fixed (R=6)':           '#D55E00',
}


def _build_perturbation_array(n_steps):
    """Impulse as [n_steps, 1] array in ctrl units (additive to action pre-gear)."""
    arr = np.zeros((n_steps, 1))
    start = int(IMPULSE_START_FRAC * n_steps)
    end = start + IMPULSE_DURATION_STEPS
    arr[start:end, 0] = IMPULSE_N / CARTPOLE_GEAR
    return arr


def _perturbation_worker(args):
    """Run one episode under the perturbation protocol; return trace dict."""
    label, recompute, adaptive, initial_theta = args

    env = MuJoCoCartPoleDynamics(stateless=False)
    agent = make_mpc('cartpole', H=50, R=recompute, mismatch_factor=NOMINAL_MISMATCH)

    # Fixed conditions still attach a monitor-only adapter so the snapshot
    # schema (including `error`) is identical across all three.
    if adaptive:
        adapt_args = {
            'adapt_class': 'ODEStepAdaptation',
            'adapt_params': ('recompute',),
            'adapt_kwargs': {'min_error_threshold': 0.08, 'relax_step': 0.05},
        }
    else:
        adapt_args = {
            'adapt_class': 'ODEStepAdaptation',
            'adapt_params': (),
            'adapt_kwargs': {'min_error_threshold': 0.08, 'relax_step': 0.05},
        }
    agent.adaptation = make_adapter(adapt_args)

    env.reset(np.array([0.0, 0.0, initial_theta, 0.0]))

    perturbation = _build_perturbation_array(N_STEPS)
    _, _, history = run_simulation(
        agent, env, n_steps=N_STEPS, perturbation=perturbation, interval=None,
    )

    states, actions, costs = history.get_state_action_cost()
    theta = states[:, 2]
    x = states[:, 0]
    tau = np.asarray(history.get_item_history('recompute_interval'), dtype=float)
    error = _to_float_nan(history.get_item_history('error'))
    running_error = _to_float_nan(history.get_item_history('running_error'))

    return dict(
        label=label, theta=theta, x=x, action=actions, cost=costs,
        tau=tau, error=error, running_error=running_error,
    )


def _to_float_nan(seq):
    """Convert a mixed None/float history column to a clean float array."""
    return np.array([np.nan if v is None else float(v) for v in seq])


def run_perturbation_sweep():
    """Run all conditions × episodes; return dict keyed by condition label."""
    rng = np.random.RandomState(SEED)
    init_thetas = rng.uniform(-0.05, 0.05, size=N_EPISODES)

    args_list = [
        (cond['label'], cond['recompute'], cond['adaptive'], init_thetas[ep])
        for cond in CONDITIONS
        for ep in range(N_EPISODES)
    ]

    raw_results = run_pool(_perturbation_worker, args_list)

    by_label = {c['label']: [] for c in CONDITIONS}
    for r in raw_results:
        by_label[r['label']].append(r)

    out = dict(
        labels=[c['label'] for c in CONDITIONS],
        mismatch=NOMINAL_MISMATCH,
        impulse_n=IMPULSE_N,
        impulse_duration_steps=IMPULSE_DURATION_STEPS,
        impulse_start_step=int(IMPULSE_START_FRAC * N_STEPS),
        dt=DT,
        n_steps=N_STEPS,
        min_error_threshold=0.08,
        tau_bounds=(1, 15),
        traces={},
    )
    for lab, eps in by_label.items():
        stack = lambda key: np.stack([np.asarray(e[key]) for e in eps])
        out['traces'][lab] = dict(
            theta=stack('theta'),
            x=stack('x'),
            action=stack('action'),
            cost=stack('cost'),
            tau=stack('tau'),
            error=stack('error'),
            running_error=stack('running_error'),
        )
    return out


def _sanity_plot(sweep, save=True):
    """Quick three-row plot (theta, error, tau) to eyeball the sweep."""
    fig, axes = plt.subplots(3, 1, figsize=(5.0, 6.0), sharex=True)
    t = np.arange(sweep['n_steps']) * sweep['dt']
    impulse_band = (
        sweep['impulse_start_step'] * sweep['dt'],
        (sweep['impulse_start_step'] + sweep['impulse_duration_steps']) * sweep['dt'],
    )

    for ax in axes:
        ax.axvspan(*impulse_band, color='0.85', zorder=0, linewidth=0)

    for lab in sweep['labels']:
        traces = sweep['traces'][lab]
        color = COND_COLORS[lab]
        ls = '-' if 'Adaptive' in lab else '--'

        theta_mean = np.nanmean(traces['theta'], axis=0)
        axes[0].plot(t, theta_mean, color=color, ls=ls, label=lab)

        err_mean = np.nanmean(traces['error'], axis=0)
        axes[1].plot(t, err_mean, color=color, ls=ls, label=lab)

        tau_mean = np.nanmean(traces['tau'], axis=0)
        axes[2].plot(t, tau_mean, color=color, ls=ls, label=lab)

    axes[0].axhline(0.0, color='k', lw=0.5, alpha=0.3)
    axes[0].set_ylabel(r'$\theta$ (rad)')
    axes[1].axhline(sweep['min_error_threshold'], color='k', lw=0.8, ls=':')
    axes[1].set_ylabel(r'$e_t$')
    for yb in sweep['tau_bounds']:
        axes[2].axhline(yb, color='k', lw=0.8, ls=':')
    axes[2].set_ylabel(r'$\tau$ (steps)')
    axes[2].set_xlabel('Time (s)')
    axes[0].legend(loc='upper right', fontsize=7)
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.tight_layout()

    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        for name in (PNG_NAME, PDF_NAME):
            path = os.path.join(PLOTS_DIR, name)
            fig.savefig(path)
            print(f"Saved: {path}")
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
        print(f"Loaded: {path}")
    else:
        sweep = run_perturbation_sweep()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(sweep, f)
        print(f"Saved: {path}")

    _sanity_plot(sweep)


if __name__ == '__main__':
    main()
