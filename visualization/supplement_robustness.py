"""Supplementary figures for hyperparameter robustness (issue #195).

Two figures:
  figure_supp_rob_grid()    — Fig 2 type: stability heatmaps at focal mismatch
                               2 rows (Walker, Humanoid) × 5 conditions
  figure_supp_rob_summary() — Fig 3 type: replan interval + cost vs mismatch
                               4 rows × 5 conditions

PKLs are read from cartpole_mpc_dev's cluster sweep output.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm
from configs import FIGURES_DIR, RESULTS_DIR
from visualization import heatmaps

SUPP_PKL_DIR = (
    '/oscar/data/dbadre/akikumot/COLLABORATIONS/'
    'cartpole_mpc_dev/cluster/sweep_supp_robustness/'
    'data/results/sweep_supp'
)

# Five tier-1 conditions in display order
CONDITIONS = [
    dict(
        label='N=10',
        grid_walker='grid_headline_walker_N_10.pkl',
        grid_humanoid='grid_headline_humanoid_balance_N_10.pkl',
        grid_cartpole='grid_headline_cartpole_N_10.pkl',
        summary_walker='summary_walker_N_10.pkl',
        summary_humanoid='summary_humanoid_balance_N_10.pkl',
        summary_cartpole='summary_cartpole_N_10.pkl',
    ),
    dict(
        label=r'$\sigma$=½×',
        grid_walker='grid_headline_walker_sigma_0p25.pkl',
        grid_humanoid='grid_headline_humanoid_balance_sigma_0p125.pkl',
        grid_cartpole='grid_headline_cartpole_sigma_0p15.pkl',
        summary_walker='summary_walker_sigma_0p25.pkl',
        summary_humanoid='summary_humanoid_balance_sigma_0p125.pkl',
        summary_cartpole='summary_cartpole_sigma_0p15.pkl',
    ),
    dict(
        label=r'$\sigma$=2×',
        grid_walker='grid_headline_walker_sigma_1p0.pkl',
        grid_humanoid='grid_headline_humanoid_balance_sigma_0p5.pkl',
        grid_cartpole='grid_headline_cartpole_sigma_0p6.pkl',
        summary_walker='summary_walker_sigma_1p0.pkl',
        summary_humanoid='summary_humanoid_balance_sigma_0p5.pkl',
        summary_cartpole='summary_cartpole_sigma_0p6.pkl',
    ),
    dict(
        label='No nominal',
        grid_walker='grid_headline_walker_include_nominal_False.pkl',
        grid_humanoid='grid_headline_humanoid_balance_include_nominal_False.pkl',
        grid_cartpole='grid_headline_cartpole_include_nominal_False.pkl',
        summary_walker='summary_walker_include_nominal_False.pkl',
        summary_humanoid='summary_humanoid_balance_include_nominal_False.pkl',
        summary_cartpole='summary_cartpole_include_nominal_False.pkl',
    ),
    dict(
        label=r'$\sigma$-mix',
        grid_walker='grid_headline_walker_sigma_mix_1p0_0p25.pkl',
        grid_humanoid='grid_headline_humanoid_balance_sigma_mix_0p5_0p25.pkl',
        grid_cartpole='grid_headline_cartpole_sigma_mix_0p6_0p25.pkl',
        summary_walker='summary_walker_sigma_mix_1p0_0p25.pkl',
        summary_humanoid='summary_humanoid_balance_sigma_mix_0p5_0p25.pkl',
        summary_cartpole='summary_cartpole_sigma_mix_0p6_0p25.pkl',
    ),
]

# Figure 3 colors: red=Fixed frequent, orange=Fixed infrequent, blue=Adaptive
_FIG3_FIXED_COLORS = ['#D64933', '#E8A838']  # matches Fig 3 palette[0,1]

# 5 blue shades for Adaptive across hyperparameter conditions (light→dark)
_ADAPTIVE_BLUES = ['#afd4ee', '#74b3d8', '#2E86AB', '#1d6690', '#0d4a6e']

# Main-paper summary pkl filenames (for Fixed reference lines)
_MAIN_SUMMARY = {
    'cartpole': 'summary_cartpole.pkl',
    'walker':   'summary_walker.pkl',
    'humanoid': 'summary_humanoid_balance.pkl',
}

_ENV_META = {
    'cartpole': dict(
        label='CartPole',
        focal_r=1.3,
        xlabel=r'Mismatch ratio $r$ (pole length)',
        dt=0.02,
    ),
    'walker': dict(
        label='Walker',
        focal_r=1.6,        # closest grid factor to publication focal r=1.5
        xlabel=r'Mismatch ratio $r$ (torso mass)',
        dt=0.01,
    ),
    'humanoid': dict(
        label='Humanoid Balance',
        focal_r=1.2,
        xlabel=r'Mismatch ratio $r$ (gravity)',
        dt=0.015,
    ),
}


def _load(fname: str):
    path = os.path.join(SUPP_PKL_DIR, fname)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def _stability_mask(mean_cost, K=4.0):
    """Return boolean mask where cost ≤ K × best-matched cost."""
    matched_best = float(np.nanmin(mean_cost[0]))
    tau = K * matched_best
    return mean_cost <= tau, matched_best


def _draw_heatmap_cell(ax, grid, focal_r, dt, show_xlabel, show_ylabel,
                       norm=None, K=4.0):
    """Draw a single H×R heatmap at focal_r on ax. Returns (im, norm)."""
    factors = list(grid['mismatch_factors'])
    idx = int(np.argmin(np.abs(np.array(factors) - focal_r)))

    mean_cost = np.asarray(grid['mean_cost'])   # (n_m, n_H, n_R)
    H_values  = np.asarray(grid['H_values'])
    R_values  = np.asarray(grid['R_values'])

    cost_per_s = mean_cost / dt
    mat = np.ma.array(cost_per_s[idx], mask=~np.isfinite(cost_per_s[idx]))

    if norm is None:
        flat = cost_per_s[np.isfinite(cost_per_s) & (cost_per_s > 0)]
        if flat.size:
            norm = LogNorm(vmin=float(flat.min()), vmax=float(flat.max()))
        else:
            norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

    cmap = matplotlib.colormaps['viridis'].copy()
    cmap.set_bad('#cccccc')
    im = ax.imshow(mat.T, aspect='equal', cmap=cmap, norm=norm, origin='lower')

    # Pink boundary outline around stability region (cost ≤ K × matched_best)
    stability_mask, _ = _stability_mask(mean_cost, K)
    heatmaps._draw_region_outline(ax, stability_mask[idx])

    n_H, n_R = len(H_values), len(R_values)
    xt = sorted({0, n_H // 2, n_H - 1})
    yt = sorted({0, n_R // 2, n_R - 1})
    ax.set_xticks(xt)
    ax.set_xticklabels([f'{H_values[i]*dt:.2f}' for i in xt], fontsize=6)
    ax.set_yticks(yt)
    ax.set_yticklabels([f'{R_values[i]*dt:.2f}' for i in yt], fontsize=6)
    if show_xlabel:
        ax.set_xlabel('Planning horizon (s)', fontsize=7)
    if show_ylabel:
        ax.set_ylabel('Replan interval (s)', fontsize=7)

    return im, norm


def _load_main(env_key):
    """Load main-paper summary pkl for Fixed reference lines."""
    fname = _MAIN_SUMMARY.get(env_key)
    if fname is None:
        return None
    path = os.path.join(RESULTS_DIR, fname)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def _fixed_labels(summary):
    """Return (fixed_frequent_label, fixed_infrequent_label) — first two non-Adaptive labels."""
    labels = [l for l in summary['sweep_cost'].keys() if 'Adaptive' not in l]
    return (labels[0], labels[1]) if len(labels) >= 2 else (labels[0], labels[0])


def _adaptive_label(summary):
    """Return the key corresponding to the Adaptive strategy in this summary pkl."""
    labels = list(summary['sweep_cost'].keys())
    for lab in labels:
        if 'Adaptive' in lab:
            return lab
    return labels[-1]


def _episode_duration(summary):
    return float(np.asarray(
        next(iter(next(iter(summary['sweep_len'].values())).values()))
    )[0])


def _replan_series(summary, label):
    """Mean±SEM replan interval (s) vs mismatch for one label."""
    mism = list(summary['mismatches'])
    dur  = _episode_duration(summary)
    ms_list, se_list = [], []
    for r in mism:
        nrep = np.asarray(summary['sweep_recomp'][label][r], dtype=float)
        intv = dur / np.maximum(nrep, 1.0)
        ms_list.append(float(intv.mean()))
        se_list.append(float(intv.std(ddof=1) / np.sqrt(len(intv))) if len(intv) > 1 else 0.0)
    return np.array(ms_list), np.array(se_list)


def _physical_success_from_episodes(episodes, specs):
    """Per-episode physical success from last-states window.

    episodes: list of (n_terminal, state_dim) arrays (one per seed).
    specs: list of clause dicts from heatmaps.PHYSICAL_SUCCESS.
    Returns float array of shape (n_episodes,).
    """
    stacked = np.stack(episodes, axis=0)   # (n_ep, n_window, state_dim)
    masks = [heatmaps._eval_clause(stacked, s) for s in specs]
    return np.logical_and.reduce(masks).astype(float)


def _success_series(summary, label, env=None):
    """Physical success rate if sweep_last_states present, else first-failure fallback.

    Physical criterion (Fig 2/3 convention) is used when sweep_last_states and
    env are both available. Falls back to first-failure-time for legacy pickles
    (e.g. supp hyperparameter-condition pkls without last_states).
    """
    mism = list(summary['mismatches'])
    sweep_env = env or summary.get('env', '')
    sweep_ls = summary.get('sweep_last_states') or summary.get('last_states')
    use_physical = (
        sweep_ls is not None
        and sweep_env in heatmaps.PHYSICAL_SUCCESS
        and label in sweep_ls
    )
    ms_list, se_list = [], []
    for r in mism:
        if use_physical:
            specs = heatmaps.PHYSICAL_SUCCESS[sweep_env]
            succ = _physical_success_from_episodes(sweep_ls[label][r], specs)
        else:
            fail = np.asarray(summary['sweep_failure'][label][r])
            lens = np.asarray(summary['sweep_len'][label][r])
            succ = (fail >= lens).astype(float)
        n = len(succ)
        ms_list.append(float(succ.mean()))
        se_list.append(float(succ.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0)
    return np.array(ms_list), np.array(se_list)


def figure_supp_rob_grid(output_dir=FIGURES_DIR, savefig=True):
    """Supp Fig: stability heatmaps at focal mismatch — 3 envs × 5 conditions.

    Each cell shows the H×R cost heatmap at the representative mismatch
    for that env (cartpole r≈1.3, walker r≈1.5, humanoid r=1.2). White
    hatching marks the stability region (cost ≤ 4× matched-best).
    Saved as figS_rob_grid.svg / .pdf.
    """
    if not savefig:
        return
    os.makedirs(output_dir, exist_ok=True)

    n_cond = len(CONDITIONS)
    cell_w, cell_h = 1.8, 1.8

    env_rows = [
        ('cartpole', 'grid_cartpole', 0),
        ('walker',   'grid_walker',   1),
        ('humanoid', 'grid_humanoid', 2),
    ]
    # Drop cartpole row if no pkls available
    if all(_load(c['grid_cartpole']) is None for c in CONDITIONS):
        env_rows = [('walker', 'grid_walker', 0), ('humanoid', 'grid_humanoid', 1)]

    n_env = len(env_rows)
    fig, axes = plt.subplots(
        n_env, n_cond,
        figsize=(cell_w * n_cond + 0.5, cell_h * n_env + 1.0),
        constrained_layout=True,
    )
    if n_env == 1:
        axes = axes[np.newaxis, :]

    for env_key, pkl_key, row in env_rows:
        meta = _ENV_META[env_key]
        # Compute shared norm across conditions for this env (cost/s)
        grids = [_load(c[pkl_key]) for c in CONDITIONS]
        all_costs = np.concatenate([
            (np.asarray(g['mean_cost']) / meta['dt']).ravel() for g in grids if g is not None
        ])
        all_costs = all_costs[np.isfinite(all_costs) & (all_costs > 0)]
        shared_norm = LogNorm(vmin=all_costs.min(), vmax=all_costs.max()) if len(all_costs) else None

        for col, cond in enumerate(CONDITIONS):
            ax = axes[row, col]
            grid = grids[col]
            show_xl = (row == n_env - 1)
            show_yl = (col == 0)
            if grid is None:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, color='#888')
                ax.set_xticks([]); ax.set_yticks([])
                continue
            im, _ = _draw_heatmap_cell(
                ax, grid, focal_r=meta['focal_r'], dt=meta['dt'],
                show_xlabel=show_xl, show_ylabel=show_yl,
                norm=shared_norm,
            )
            # Column header (top row only)
            if row == 0:
                ax.set_title(cond['label'], fontsize=8)
            # Row label (left column only)
            if col == 0:
                ax.set_ylabel(f"{meta['label']}\nReplan interval (s)", fontsize=7)

        # Shared colorbar for this env row
        fig.colorbar(im, ax=axes[row, :], shrink=0.8, pad=0.01,
                     label=f'mean cost / s  (r≈{meta["focal_r"]})', aspect=20)

        # Panel letter (A/B/C) on leftmost axis
        letter = 'ABC'[row]
        axes[row, 0].text(
            -0.20, 1.18, letter,
            transform=axes[row, 0].transAxes,
            fontweight='bold', va='bottom', ha='right',
            fontsize=matplotlib.rcParams['axes.titlesize'] * 1.2,
        )

    out_svg = os.path.join(output_dir, 'figS4.svg')
    out_pdf = os.path.join(output_dir, 'figS4.pdf')
    fig.savefig(out_svg, dpi=300)
    plt.close(fig)
    try:
        from visualization.svgtools import svg_to_pdf
        svg_to_pdf(out_svg, out_pdf)
    except Exception as e:
        print(f'figure_supp_rob_grid: PDF conversion skipped ({e})')
    print(f'figure_supp_rob_grid: saved {out_svg}')


def figure_supp_rob_summary(output_dir=FIGURES_DIR, savefig=True):
    """Supp Fig: replan interval + success rate vs mismatch — 2 rows × n_env cols.

    Layout mirrors Figure 3: top row = replan interval, bottom row = success rate.
    Each panel shows the Adaptive strategy across all 5 hyperparameter conditions
    as distinct colored lines, so the reader can judge condition-to-condition
    robustness directly within each panel.

    Columns: Walker, Humanoid Balance (+ Cartpole when pkls are available).
    Saved as figS_rob_summary.svg / .pdf.
    """
    if not savefig:
        return
    os.makedirs(output_dir, exist_ok=True)

    env_cols = [
        ('walker',   'summary_walker',   r'Mismatch ratio $r$ (torso mass)'),
        ('humanoid', 'summary_humanoid', r'Mismatch ratio $r$ (gravity)'),
    ]
    # Add cartpole column when pkls are present
    if any(_load(c['summary_cartpole']) is not None for c in CONDITIONS
           if 'summary_cartpole' in c):
        env_cols.insert(0, ('cartpole', 'summary_cartpole',
                            r'Mismatch ratio $r$ (pole length)'))

    n_env = len(env_cols)
    panel_w, panel_h = 2.4, 1.95
    fig, axes = plt.subplots(
        2, n_env,
        figsize=(panel_w * n_env + 0.3, panel_h * 2),
        layout='constrained',
        sharex='col',
    )
    if n_env == 1:
        axes = axes.reshape(2, 1)

    row_ylabels = ['Replan interval (s)', 'Success rate']
    row_letters = ['A', 'B']

    for col, (env_key, pkl_key, xlabel) in enumerate(env_cols):
        supp_summaries = [_load(c[pkl_key]) for c in CONDITIONS]
        main_summary   = _load_main(env_key)
        any_data = main_summary is not None or any(s is not None for s in supp_summaries)

        env_label = _ENV_META[env_key]['label'] if env_key in _ENV_META else env_key.capitalize()
        axes[0, col].set_title(env_label, fontsize=9)

        for row in range(2):
            ax = axes[row, col]
            if not any_data:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, color='#888')
                ax.set_xticks([]); ax.set_yticks([])
                continue

            # --- Fixed reference lines from main-paper pkl (Fig 3 colors) ---
            if main_summary is not None:
                ff_lab, fi_lab = _fixed_labels(main_summary)
                mism_main = list(main_summary['mismatches'])
                for lab, color, disp, mkr in [
                    (ff_lab, _FIG3_FIXED_COLORS[0], 'Fixed frequent',   'o'),
                    (fi_lab, _FIG3_FIXED_COLORS[1], 'Fixed infrequent', 's'),
                ]:
                    if row == 0:
                        m, s = _replan_series(main_summary, lab)
                    else:
                        m, s = _success_series(main_summary, lab, env=env_key)
                    ax.plot(mism_main, m, marker=mkr, ms=4.5, lw=1.3,
                            color=color, label=disp, zorder=3)
                    ax.fill_between(mism_main, np.maximum(m - s, 0), m + s,
                                    color=color, alpha=0.2, lw=0, zorder=2)

            # --- Adaptive lines from supp pkls (blue shades per condition) ---
            for i, (cond, summary) in enumerate(zip(CONDITIONS, supp_summaries)):
                if summary is None:
                    continue
                lab  = _adaptive_label(summary)
                mism = list(summary['mismatches'])
                color = _ADAPTIVE_BLUES[i % len(_ADAPTIVE_BLUES)]
                if row == 0:
                    m, s = _replan_series(summary, lab)
                else:
                    m, s = _success_series(summary, lab, env=env_key)
                ax.plot(mism, m, marker='^', ms=3.5, lw=1.2, linestyle='--',
                        color=color, label=f'Adaptive ({cond["label"]})', zorder=4)
                ax.fill_between(mism, np.maximum(m - s, 0), m + s,
                                color=color, alpha=0.15, lw=0, zorder=2)

            if row == 1:
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlabel(xlabel, fontsize=7)
            if col == 0:
                ax.set_ylabel(row_ylabels[row], fontsize=7)
            else:
                ax.tick_params(axis='y', labelleft=False)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.2)

            if env_key == 'humanoid' and row == 0:
                ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, prune='both'))
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))

        plt.setp(axes[0, col].get_xticklabels(), visible=False)

    # Legend only in top-left panel; row letters A/B
    axes[0, 0].legend(fontsize=5, loc='best', ncol=1)
    letter_fs = matplotlib.rcParams['axes.titlesize'] * 1.2
    for row, letter in enumerate(row_letters):
        axes[row, 0].text(-0.28, 1.04, letter,
                          transform=axes[row, 0].transAxes,
                          fontweight='bold', va='bottom', ha='right',
                          fontsize=letter_fs)

    out_svg = os.path.join(output_dir, 'figS5.svg')
    out_pdf = os.path.join(output_dir, 'figS5.pdf')
    fig.savefig(out_svg, dpi=300)
    plt.close(fig)
    try:
        from visualization.svgtools import svg_to_pdf
        svg_to_pdf(out_svg, out_pdf)
    except Exception as e:
        print(f'figure_supp_rob_summary: PDF conversion skipped ({e})')
    print(f'figure_supp_rob_summary: saved {out_svg}')
