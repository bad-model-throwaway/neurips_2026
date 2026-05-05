"""Perturbation-response sweep for walker.

Three controllers subjected to a single external torso-force impulse at
t = T/2 on a matched-dynamics episode (r = 1.0):

    Fixed (R=1)              — performance ceiling
    Adaptive (R=8 init)      — paper's proposed method
    Fixed (R=R_PERT)         — impulse-sensitive weak-control arm

Perturbation: env-level `xfrc_applied` hook writes directly to
`env._mj_data.xfrc_applied[torso_body, :]` so MuJoCo applies a pure x-axis
force on the torso during the impulse window. (An additive perturbation to
the 6 joint motors would be a torque imbalance, not a body-level force.)
The planner sees only the induced state drift, not the force itself.

Fixed controllers attach a monitoring-only `ODEStepAdaptation(adapt=())`
so the agent `snapshot()` emits the same error/running_error/threshold
keys across all three conditions, keeping the plotter schema uniform
with cartpole's perturbation dict.
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
from simulations.sweep_walker_adaptive import DT, N_STEPS, H_WALKER
from simulations.sweep_walker_summary import R_INIT


# R_PERT: weak-control recompute interval — large enough that at matched
# dynamics the walker stays upright pre-impulse, but loses recovery
# authority under the external torso force.
R_PERT = 10

CONDITIONS = [
    dict(recompute=1,      adaptive=False, label='Fixed (R=1)'),
    dict(recompute=R_INIT, adaptive=True,
         label=f'Adaptive (R={R_INIT} init)'),
    dict(recompute=R_PERT, adaptive=False, label=f'Fixed (R={R_PERT})'),
]


# Impulse on torso's +x axis (forward push) during a stance window at
# mid-episode, applied via env._mj_data.xfrc_applied[torso, 0].
IMPULSE_N = 300.0
IMPULSE_DURATION_STEPS = 10   # 0.1 s at walker DT=0.01
IMPULSE_START_FRAC = 0.5
NOMINAL_MISMATCH = 1.0
N_EPISODES = 20

PKL_NAME = 'perturbation_walker.pkl'
PNG_NAME = 'perturbation_walker.png'
PDF_NAME = 'perturbation_walker.pdf'

COND_COLORS = {
    'Fixed (R=1)':                 '#009E73',
    f'Adaptive (R={R_INIT} init)': '#0072B2',
    f'Fixed (R={R_PERT})':         '#D55E00',
}


def _make_torso_impulse_fn(torso_body_id, impulse_n, start_step, duration_steps):
    """Return an env-level perturbation callable for `run_simulation`.

    The returned callable zeros `xfrc_applied[torso]` each step and writes
    `impulse_n` to the x-axis slot during `[start_step, start_step + duration_steps)`.
    The caller owns captured `torso_body_id`; the env is passed fresh each call
    to keep the closure side-effect free with respect to module state.
    """
    end_step = start_step + duration_steps

    def fn(env, step_idx):
        env._mj_data.xfrc_applied[torso_body_id, :] = 0.0
        if start_step <= step_idx < end_step:
            env._mj_data.xfrc_applied[torso_body_id, 0] = impulse_n

    return fn


def _perturbation_worker(args):
    """Run one walker episode under the impulse protocol; return trace dict."""
    label, recompute, adaptive, seed = args

    np.random.seed(int(seed))

    env = WalkerDynamics(stateless=False, speed_goal=1.5)
    agent = make_mpc('walker', H=H_WALKER, R=recompute,
                     mismatch_factor=NOMINAL_MISMATCH)

    # Fixed conditions attach a monitor-only adapter so the snapshot schema
    # is uniform across all three (plotter relies on `error`, `running_error`,
    # `threshold`).
    if adaptive:
        adapt_args = {
            'adapt_class': 'ODEStepAdaptation',
            'adapt_params': ('recompute',),
            'adapt_kwargs': {'min_error_threshold': 0.15, 'relax_step': 0.05},
        }
    else:
        adapt_args = {
            'adapt_class': 'ODEStepAdaptation',
            'adapt_params': (),
            'adapt_kwargs': {'min_error_threshold': 0.15, 'relax_step': 0.05},
        }
    agent.adaptation = make_adapter(adapt_args)

    env.reset(env.get_default_initial_state())

    start_step = int(IMPULSE_START_FRAC * N_STEPS)
    impulse_fn = _make_torso_impulse_fn(
        torso_body_id=env._torso_id,
        impulse_n=IMPULSE_N,
        start_step=start_step,
        duration_steps=IMPULSE_DURATION_STEPS,
    )

    _, _, history = run_simulation(
        agent, env, n_steps=N_STEPS,
        env_perturbation_fn=impulse_fn, interval=None,
    )

    states = history.get_item_history('state')
    actions = history.get_item_history('action')
    cost = history.get_item_history('cost')

    torso_z = states[:, 18].astype(float)
    com_vx  = states[:, 20].astype(float)
    tau = np.asarray(history.get_item_history('recompute_interval'), dtype=float)
    error = _to_float_nan(history.get_item_history('error'))
    running_error = _to_float_nan(history.get_item_history('running_error'))

    fell = np.where(torso_z < 0.7)[0]
    fall_time = float(fell[0] * DT) if len(fell) > 0 else float('nan')

    return dict(
        label=label, torso_z=torso_z, com_vx=com_vx,
        action=actions, cost=cost, tau=tau,
        error=error, running_error=running_error,
        fall_time=fall_time,
    )


def _to_float_nan(seq):
    """Convert a mixed None/float history column to a clean float array."""
    return np.array([np.nan if v is None else float(v) for v in seq])


def run_perturbation_sweep():
    """Run all conditions × episodes; return dict keyed by condition label.

    Walker-specific traces: `torso_z`, `com_vx`, `fall_time`
    (per-episode vector, NaN if never fell).
    """
    rng = np.random.RandomState(SEED)
    seeds = rng.randint(0, 2**31 - 1, size=N_EPISODES)

    args_list = [
        (cond['label'], cond['recompute'], cond['adaptive'], int(seeds[ep]))
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
        min_error_threshold=0.15,
        tau_bounds=(1, 15),
        traces={},
    )
    for lab, eps in by_label.items():
        stack = lambda key: np.stack([np.asarray(e[key]) for e in eps])
        out['traces'][lab] = dict(
            torso_z=stack('torso_z'),
            com_vx=stack('com_vx'),
            action=stack('action'),
            cost=stack('cost'),
            tau=stack('tau'),
            error=stack('error'),
            running_error=stack('running_error'),
            fall_time=np.asarray([e['fall_time'] for e in eps], dtype=float),
        )
    return out


def _sanity_plot(sweep, save=True):
    """Quick three-row plot (torso_z, error, tau) to eyeball the sweep."""
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

        z_mean = np.nanmean(traces['torso_z'], axis=0)
        axes[0].plot(t, z_mean, color=color, ls=ls, label=lab)

        err_mean = np.nanmean(traces['error'], axis=0)
        axes[1].plot(t, err_mean, color=color, ls=ls, label=lab)

        tau_mean = np.nanmean(traces['tau'], axis=0)
        axes[2].plot(t, tau_mean, color=color, ls=ls, label=lab)

    axes[0].axhline(0.7, color='k', lw=0.5, alpha=0.4, ls=':')
    axes[0].set_ylabel('Torso z (m)')
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
        if not os.path.exists(path):
            raise SystemExit(
                f"No cached sweep found at {path}. Run without --plot-only to "
                "produce it."
            )
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
