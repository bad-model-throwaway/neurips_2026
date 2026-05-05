"""Analyze R/H trajectories across the 3 envs from smoke pkls.

Reads data/results/_smoke_summary_{env}.pkl, extracts sweep_R_full /
sweep_H_full, and produces:
- Per-env figures: time series of R(t) and H(t) (mean ± SE across episodes),
  one column per mismatch level.
- Stdout summary: time-to-first-R-recovery, fraction-of-episode-below-init,
  H ramp magnitude, and matched-mismatch trigger silence test.
"""
from __future__ import annotations

import os
import pickle
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from configs import RESULTS_DIR, PLOTS_DIR

ENVS = [
    ('cartpole',         dict(dt=0.02,  H_init=44, R_init=3)),
    ('walker',           dict(dt=0.01,  H_init=67, R_init=7)),
    ('humanoid_balance', dict(dt=0.015, H_init=85, R_init=8)),
]


def _episode_stats(R_arr, H_arr, R_init, H_init):
    """Per-episode metrics from one R/H trace."""
    R = np.asarray(R_arr, dtype=float)
    H = np.asarray(H_arr, dtype=float)
    below = R < R_init
    if below.any():
        first_drop = int(np.argmax(below))     # first step where R<R_init
    else:
        first_drop = -1
    return dict(
        R_mean=float(R.mean()),
        R_min=float(R.min()),
        frac_below_R_init=float(below.mean()),
        first_R_drop_step=first_drop,
        H_mean=float(H.mean()),
        H_max=float(H.max()),
        H_above_init_frac=float((H > H_init).mean()),
        H_below_init_frac=float((H < H_init).mean()),
    )


def _print_env_summary(env_name, sweep, env_info):
    R_full = sweep['sweep_R_full']
    H_full = sweep['sweep_H_full']
    H_init, R_init = env_info['H_init'], env_info['R_init']
    dt = env_info['dt']

    print(f'\n=== {env_name} (H_init={H_init}, R_init={R_init}, dt={dt}) ===')
    for lab in R_full:
        if 'Adaptive' not in lab:
            continue
        for r in sweep['mismatches']:
            r_eps = R_full[lab][r]
            h_eps = H_full[lab][r]
            if not r_eps:
                continue
            stats = [_episode_stats(re, he, R_init, H_init)
                     for re, he in zip(r_eps, h_eps)]
            agg = {k: np.mean([s[k] for s in stats]) for k in stats[0]}
            # First-drop reported in seconds (median of episodes that did drop)
            drops = [s['first_R_drop_step'] for s in stats
                     if s['first_R_drop_step'] >= 0]
            t_first = (np.median(drops) * dt) if drops else float('nan')
            n_silent = sum(1 for s in stats if s['frac_below_R_init'] == 0
                           and s['H_above_init_frac'] == 0)
            print(
                f'  r={r:>4.2f}: '
                f'R_mean={agg["R_mean"]:.2f}/{R_init} '
                f'(frac<init={agg["frac_below_R_init"]:.2f}, '
                f't_first_drop={t_first:.2f}s, '
                f'silent_eps={n_silent}/{len(stats)})  '
                f'H_mean={agg["H_mean"]:.1f}/{H_init} '
                f'(H>init={agg["H_above_init_frac"]:.2f}, '
                f'H<init={agg["H_below_init_frac"]:.2f}, '
                f'H_max={agg["H_max"]:.0f})'
            )


def _plot_env_trajectories(env_name, sweep, env_info, out_dir):
    """One figure per env: 2 rows (R, H) × N_mismatch columns. Mean ± SE."""
    R_full = sweep['sweep_R_full']
    H_full = sweep['sweep_H_full']
    mismatches = list(sweep['mismatches'])
    H_init, R_init = env_info['H_init'], env_info['R_init']
    dt = env_info['dt']

    adapt_lab = next((l for l in R_full if 'Adaptive' in l), None)
    if adapt_lab is None:
        return

    fig, axes = plt.subplots(2, len(mismatches),
                             figsize=(2.7 * len(mismatches), 4.0),
                             sharex='col')
    if len(mismatches) == 1:
        axes = axes[:, None]

    for col, r in enumerate(mismatches):
        r_eps = R_full[adapt_lab][r]
        h_eps = H_full[adapt_lab][r]
        if not r_eps:
            continue
        Rmat = np.stack([np.asarray(a, dtype=float) for a in r_eps], axis=0)
        Hmat = np.stack([np.asarray(a, dtype=float) for a in h_eps], axis=0)
        n = Rmat.shape[0]
        t = np.arange(Rmat.shape[1]) * dt

        R_lim = (0.5, max(R_init + 1, Rmat.max()) + 0.5)
        H_lim = (max(0, Hmat.min() - 5), Hmat.max() + 5)
        for ax, mat, lim, ref in [
            (axes[0, col], Rmat, R_lim, R_init),
            (axes[1, col], Hmat, H_lim, H_init),
        ]:
            mean = mat.mean(axis=0)
            se = mat.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
            ax.plot(t, mean, lw=1.2, color='#0072B2')
            ax.fill_between(t, mean - se, mean + se, alpha=0.25, color='#0072B2',
                            linewidth=0)
            ax.axhline(ref, color='0.4', ls='--', lw=0.7, label='init')
            ax.set_ylim(lim)
            ax.grid(True, alpha=0.2)

        axes[0, col].set_title(f'r = {r:.2f}', fontsize=10)
        axes[1, col].set_xlabel('time (s)')

    axes[0, 0].set_ylabel(f'R (init={R_init})')
    axes[1, 0].set_ylabel(f'H (init={H_init})')
    fig.suptitle(f'{env_name} — Adaptive R(t) and H(t), mean ± SE (N={n})',
                 fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(out_dir, f'adapt_dyn_{env_name}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out_path}')


def main():
    out_dir = PLOTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    for env_name, env_info in ENVS:
        path = os.path.join(RESULTS_DIR, f'_smoke_summary_{env_name}.pkl')
        if not os.path.exists(path):
            print(f'(skip) {path} missing')
            continue
        with open(path, 'rb') as f:
            sweep = pickle.load(f)
        _print_env_summary(env_name, sweep, env_info)
        _plot_env_trajectories(env_name, sweep, env_info, out_dir)


if __name__ == '__main__':
    main()
