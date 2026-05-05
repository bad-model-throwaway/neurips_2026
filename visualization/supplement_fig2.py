"""Figure 2 supplements (issue #181, EPOpt/RWRL reporting protocol).

Three supplements alongside the main Figure 2:

  Supp A — p90 cost heatmap (EPOpt/Mankowitz convention).
  Supp B — continuous physical success-rate heatmap (no threshold).
  Supp C — cross-criterion sensitivity: physical vs K' ∈ {2, 4, 8}.
"""

import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from configs import FIG_FMT, FIGURES_DIR

from visualization import heatmaps


_PANEL_ORDER = heatmaps._PANEL_ORDER


def _draw_supp_letters(fig, axes, grids_by_env):
    """Place A/B/C panel letters in axes-relative coords above each row.

    Using axes-relative coords lets constrained_layout reserve the space
    automatically, so the letters sit cleanly above the column-title row
    without colliding with the rotated env-name ylabel.
    """
    for row_idx, (env_key, _, letter) in enumerate(_PANEL_ORDER):
        if grids_by_env.get(env_key) is None:
            continue
        axes[row_idx, 0].text(
            -0.20, 1.18, letter,
            transform=axes[row_idx, 0].transAxes,
            fontweight='bold', va='bottom', ha='right',
            fontsize=matplotlib.rcParams['axes.titlesize'] * 1.2,
        )


def _per_seed_p90(sweep_result):
    """p90 cost per cell, rescaled to a per-second rate for plot consistency."""
    all_costs = np.asarray(sweep_result['all_costs'])
    return np.nanpercentile(all_costs, 90, axis=-1) / float(sweep_result['dt'])


def supplement_fig2_p90(grids_by_env, output_dir=FIGURES_DIR, savefig=True):
    """Supp A — p90 cost heatmap, same layout as Figure 2.

    Replaces mean cost with the 90th-percentile cost across seeds (EPOpt
    convention). Hatch is the per-seed physical success rule from Fig 2,
    so the viable-region overlay is directly comparable.
    """
    if not savefig:
        return
    os.makedirs(output_dir, exist_ok=True)

    ncols, nrows = 4, 3
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.1 * ncols + 1.8, 2.1 * nrows + 0.6),
        constrained_layout=True,
    )
    for row_idx, (env_key, env_label, letter) in enumerate(_PANEL_ORDER):
        result = grids_by_env.get(env_key)
        if result is None:
            for ax in axes[row_idx]:
                ax.axis('off')
            continue
        env_dt = result.get('dt', 0.02)
        p90 = _per_seed_p90(result)
        heatmaps.plot_heatmap_row(
            axes[row_idx], result, env_label, env_dt,
            cbar_label='p90 cost / s',
            stability_K=4.0,
            criterion='physical',
            success_threshold=0.9,
            letter=None,
            show_col_titles=True,
            show_xlabel=False,
            show_row_label=True,
            ylabel_text=None,
            colorbar=True,
            cbar_stability_suffix=False,
            plot_matrix=p90,
        )

    fig.supxlabel('Planning horizon (s)')
    fig.supylabel('Replan interval (s)')
    _draw_supp_letters(fig, axes, grids_by_env)

    fig.savefig(output_dir + 'figS1' + FIG_FMT, dpi=300)
    plt.close(fig)


def supplement_fig2_success_rate(grids_by_env, output_dir=FIGURES_DIR,
                                 savefig=True):
    """Supp B — continuous physical success-rate heatmap.

    Plots P(success_physical) ∈ [0, 1] as the colormap (no hatch). The
    threshold-0.9 hatch from the main figure becomes the level set
    `rate >= 0.9` of this map.
    """
    if not savefig:
        return
    os.makedirs(output_dir, exist_ok=True)

    ncols, nrows = 4, 3
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.1 * ncols + 1.8, 2.1 * nrows + 0.6),
        constrained_layout=True,
    )
    for row_idx, (env_key, env_label, letter) in enumerate(_PANEL_ORDER):
        result = grids_by_env.get(env_key)
        if result is None:
            for ax in axes[row_idx]:
                ax.axis('off')
            continue
        env_dt = result.get('dt', 0.02)
        rate = heatmaps.compute_success_rate(result, criterion='physical')
        heatmaps.plot_heatmap_row(
            axes[row_idx], result, env_label, env_dt,
            cbar_label='P(success)',
            stability_K=None,           # no contour overlay
            letter=None,
            show_col_titles=True,
            show_xlabel=False,
            show_row_label=True,
            ylabel_text=None,
            colorbar=True,
            cbar_stability_suffix=False,
            plot_matrix=rate,
            plot_norm=Normalize(vmin=0.0, vmax=1.0),
        )

    fig.supxlabel('Planning horizon (s)')
    fig.supylabel('Replan interval (s)')
    _draw_supp_letters(fig, axes, grids_by_env)

    fig.savefig(output_dir + 'figS2' + FIG_FMT, dpi=300)
    plt.close(fig)


# Tolerance levels swept in Supp C. The middle entry (0.2) matches the
# main-figure rule. The other two test the question: "does the viable
# region survive a ±10 ppt wiggle in the threshold?"
_TOLERANCE_LEVELS = (0.1, 0.2, 0.3)


def _threshold_for_tolerance(env_key, tolerance):
    """Translate a tolerance fraction into the env's success threshold.

    walker / humanoid_balance:  threshold = (1 − tolerance) × MJCF_DEFAULT
        — "body part stays within `tolerance` of natural pose height."
    cartpole:                   threshold = tolerance × FAILURE_ANGLE
        — "pole stays within `tolerance` of the failure-angle band of upright."
    """
    if env_key == 'walker':
        return (1.0 - tolerance) * 1.30
    if env_key == 'humanoid_balance':
        return (1.0 - tolerance) * 1.472
    if env_key == 'cartpole':
        return tolerance * 0.5  # FAILURE_ANGLE = 0.5 rad ≈ 30°
    raise KeyError(f'no tolerance mapping for env={env_key!r}')


def _success_mask_at_threshold(sweep_result, threshold, success_threshold=0.9):
    """Per-cell hatch mask with the env's first physical clause's threshold
    overridden, all other clauses kept at their default values.

    The first clause is the "natural pose" rule (cartpole |theta|, walker
    torso_z, humanoid head_z); subsequent clauses (e.g. walker speed)
    stay fixed because the supp's question is "is the height/angle rule
    sensitive?", not "do all clauses jointly survive perturbation?"
    """
    specs = list(heatmaps.PHYSICAL_SUCCESS[sweep_result['env']])
    specs[0] = {**specs[0], 'threshold': threshold}
    last = np.asarray(sweep_result['last_states'])
    masks = [heatmaps._eval_clause(last, s) for s in specs]
    success = np.logical_and.reduce(masks)
    rate = success.mean(axis=-1)
    return rate >= success_threshold


def supplement_fig2_threshold_sensitivity(grids_by_env, output_dir=FIGURES_DIR,
                                          savefig=True, mismatch_col=-1):
    """Supp C — viable-region sensitivity to the body-pose threshold.

    For each env the H×R mean-cost heatmap at the largest mismatch column
    is shown three times, with the hatch coming from a tighter (10%),
    default (20%), and looser (30%) deviation from the env's natural pose.
    Walker has a second success clause (com_vx >= 0.5 m/s) which is held
    fixed across the three columns — this supp only tests sensitivity of
    the body-pose threshold, not joint sensitivity of every clause.
    """
    if not savefig:
        return
    os.makedirs(output_dir, exist_ok=True)

    ncols, nrows = len(_TOLERANCE_LEVELS), 3
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.1 * ncols + 1.8, 2.1 * nrows + 0.6),
        constrained_layout=True,
    )
    for row_idx, (env_key, env_label, letter) in enumerate(_PANEL_ORDER):
        result = grids_by_env.get(env_key)
        if result is None:
            for ax in axes[row_idx]:
                ax.axis('off')
            continue
        env_dt = result.get('dt', 0.02)
        # Rescale to per-second to match the main fig and figS1.
        mean_cost = np.asarray(result['mean_cost']) / env_dt
        H_values  = np.asarray(result['H_values'])
        R_values  = np.asarray(result['R_values'])
        col = mismatch_col if mismatch_col >= 0 else (
            len(result['mismatch_factors']) + mismatch_col
        )

        # Per-row LogNorm shared across the three columns for this env.
        valid = mean_cost[col][np.isfinite(mean_cost[col])]
        positive = valid[valid > 0]
        if positive.size and positive.min() < positive.max():
            norm = LogNorm(vmin=float(positive.min()),
                           vmax=float(positive.max()))
        else:
            mean = float(np.nanmean(mean_cost[col]))
            norm = Normalize(vmin=0.9 * mean, vmax=1.1 * mean)

        cmap = matplotlib.colormaps['viridis']

        for col_idx, tolerance in enumerate(_TOLERANCE_LEVELS):
            ax = axes[row_idx, col_idx]
            mat = np.ma.array(mean_cost[col],
                              mask=~np.isfinite(mean_cost[col]))
            ax.imshow(mat.T, aspect='equal', cmap=cmap, norm=norm,
                      origin='lower')

            thr  = _threshold_for_tolerance(env_key, tolerance)
            mask = _success_mask_at_threshold(result, thr)
            heatmaps._draw_region_outline(ax, mask[col])
            if mask[col].any():
                H_grid, R_grid = np.meshgrid(H_values, R_values, indexing='ij')
                ratio = np.where(mask[col], H_grid / R_grid, np.inf)
                h_i, r_i = np.unravel_index(np.argmin(ratio), ratio.shape)
                ax.scatter(h_i, r_i, marker='*', s=55,
                           facecolor='white', edgecolor='white',
                           linewidths=0.6, zorder=5)

            xtick_idx = sorted({0, len(H_values) // 2, len(H_values) - 1})
            ytick_idx = sorted({0, len(R_values) // 2, len(R_values) - 1})
            ax.set_xticks(xtick_idx)
            ax.set_xticklabels([f'{H_values[i] * env_dt:.2f}' for i in xtick_idx])
            ax.set_yticks(ytick_idx)
            ax.set_yticklabels([f'{R_values[i] * env_dt:.2f}' for i in ytick_idx])
            if col_idx == 0:
                ax.set_ylabel(env_label)
            else:
                ax.tick_params(axis='y', labelleft=False)
            if row_idx == 0:
                unit = 'rad' if env_key == 'cartpole' else 'm'
                tag  = ('tighter' if tolerance < 0.2
                        else 'default' if tolerance == 0.2
                        else 'looser')
                ax.set_title(
                    f'{tag} ({int(tolerance*100)}%)\nthr={thr:.3f} {unit}',
                    fontsize=9,
                )

    fig.supxlabel('Planning horizon (s)')
    fig.supylabel('Replan interval (s)')
    _draw_supp_letters(fig, axes, grids_by_env)

    fig.suptitle(
        'Viable-region sensitivity to the body-pose threshold '
        '(highest mismatch column per env; other clauses held at default)',
        y=1.02, fontsize=11,
    )
    fig.savefig(output_dir + 'figS3' + FIG_FMT,
                dpi=300, bbox_inches='tight')
    plt.close(fig)


def render_all(grids_by_env, output_dir=FIGURES_DIR):
    supplement_fig2_p90(grids_by_env, output_dir=output_dir)
    supplement_fig2_success_rate(grids_by_env, output_dir=output_dir)
    supplement_fig2_threshold_sensitivity(grids_by_env, output_dir=output_dir)
