"""Manuscript Figure 1 v2: scalar-LQR eigenvalues + MPC cost-rate heatmaps.

A 4x4 grid combining the toy LQR closed-loop eigenvalue surface with the
empirical (H, R) cost-rate heatmaps from the three MuJoCo environments.

    Row A: |A_tau(ell)| over the (ell, tau) plane under four
           model mis-specifications, ordered by increasing mismatch from
           the truth (a = 1.5, b = 1.0).
    Row B: CartPole cost-rate heatmaps over (planning horizon H,
           replan interval R) at the four canonical mismatch factors.
    Row C: Walker cost-rate heatmaps (same axes).
    Row D: Humanoid Balance cost-rate heatmaps (same axes).

Rows B-D reuse `visualization.heatmaps.plot_heatmap_row`, which renders
the same panels used by `visualization.figures.figure_2`.
"""

import os
import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from configs import DT, FIGURES_DIR, FIG_FMT, RESULTS_DIR
from supplement.shared import DEFAULT_PARAMS, apply_style
from supplement.section8_joint import _joint_eigenvalue
from visualization import heatmaps


a, b, q, r = DEFAULT_PARAMS['a'], DEFAULT_PARAMS['b'], DEFAULT_PARAMS['q'], DEFAULT_PARAMS['r']

OUTPUT_DIR = FIGURES_DIR
OUTPUT_STEM = 'fig1_v2'

# (ell, tau) ranges for the eigenvalue heatmaps. Match manuscript_fig1.py.
ELL_MAX = 15
TAU_MAX = 10

# Four mis-specifications, increasing mismatch from truth (a = 1.5, b = 1.0).
# Col 0 is the matched model. Cols 1-3 are the three contour cases used in
# manuscript_fig1.py::plot_eigenvalue_heatmap, ordered |a - a_hat| ascending.
EIGEN_CASES = [
    (1.5, 1.0),
    (1.3, 0.9),
    (1.2, 0.9),
    (1.1, 0.9),
]

_ENV_ROWS = [
    ('cartpole',         'CartPole'),
    ('walker',           'Walker'),
    ('humanoid_balance', 'Humanoid Balance'),
]


def _eigenvalue_grid(ells, taus, a_hat, b_hat):
    """Compute |A_tau(ell)| over the (ell, tau) grid; entries with tau > ell are NaN."""
    grid = np.full((len(taus), len(ells)), np.nan)
    for i, tau in enumerate(taus):
        for j, ell in enumerate(ells):
            if tau <= ell:
                grid[i, j] = abs(_joint_eigenvalue(ell, tau, a, b, a_hat, b_hat, q, r))
    return grid


def _draw_eigenvalue_row(ax_row, show_xlabel=True, show_ylabel=True):
    """Render the four |A_tau(ell)| heatmaps into the row of axes.

    Each panel uses one (a_hat, b_hat) from EIGEN_CASES; all four share a
    single colormap range so a row colorbar is comparable across panels.
    """
    ells = np.arange(1, ELL_MAX + 1)
    taus = np.arange(1, TAU_MAX + 1)

    ims = []
    for col, (a_hat, b_hat) in enumerate(EIGEN_CASES):
        ax = ax_row[col]
        grid = _eigenvalue_grid(ells, taus, a_hat, b_hat)

        # Clip large overshoots so the colormap saturates rather than masking
        im = ax.imshow(
            np.clip(grid, 0, 2.0),
            origin='lower', aspect='auto',
            extent=[0.5, ELL_MAX + 0.5, 0.5, TAU_MAX + 0.5],
            cmap='RdYlGn_r', vmin=0, vmax=1.5,
        )
        ims.append(im)

        if abs(a_hat - a) < 1e-12 and abs(b_hat - b) < 1e-12:
            ax.set_title('No mismatch')
        else:
            ax.set_title(rf'$\hat{{a}}={a_hat},\ \hat{{b}}={b_hat}$')

        if show_xlabel:
            ax.set_xlabel(r'Horizon $\ell$')

        if col == 0 and show_ylabel:
            ax.set_ylabel('Toy LQR\n' + r'Replan interval $\tau$')
        else:
            ax.tick_params(axis='y', labelleft=False)

    cbar = ax_row[-1].figure.colorbar(
        ims[-1], ax=ax_row, shrink=0.85, pad=0.02, label=r'$|A_\tau(\ell)|$',
    )
    cbar.ax.tick_params(labelsize=matplotlib.rcParams['ytick.labelsize'])


def _load_grid(env_name):
    """Load a (H, R, mismatch) grid sweep result pickle."""
    path = os.path.join(RESULTS_DIR, f'grid_{env_name}.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)


def figure_1_v2(savefig=True, stability_K=4.0, criterion='physical',
                success_threshold=0.9):
    """Compose the 4x4 figure: one eigenvalue row + three env cost-rate rows."""
    apply_style()

    # Manuscript-scale fonts to match figure_2 reading distance
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 11,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    })

    ncols, nrows = 4, 4
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.1 * ncols + 1.8, 2.1 * nrows + 0.6),
        constrained_layout=True,
    )

    # Row A: scalar-LQR eigenvalue heatmaps
    _draw_eigenvalue_row(axes[0])

    # Rows B-D: per-env (H, R) cost-rate heatmaps from the cached grid sweeps.
    # Bottom row (humanoid) shows the x-axis label; the middle two rows hide
    # it so the row stack reads as a single block with axes labelled at the
    # outer edges.
    for row_offset, (env_key, env_label) in enumerate(_ENV_ROWS):
        row_idx = row_offset + 1
        is_bottom_row = row_offset == len(_ENV_ROWS) - 1

        result = _load_grid(env_key)
        env_dt = result.get('dt', DT)
        heatmaps.plot_heatmap_row(
            axes[row_idx], result, env_label, env_dt,
            cbar_label=heatmaps.CBAR_LABELS_SHORT.get(env_key),
            stability_K=stability_K,
            letter=None,
            show_col_titles=True,
            show_xlabel=is_bottom_row,
            show_row_label=True,
            ylabel_text='Replan interval (s)',
            colorbar=True,
            cbar_stability_suffix=False,
            criterion=criterion,
            success_threshold=success_threshold,
        )

    # Panel letters in axes-relative coords (matches figure_2 convention)
    letter_fs = matplotlib.rcParams['axes.titlesize'] * 1.2
    for row_idx, letter in enumerate(['A', 'B', 'C', 'D']):
        axes[row_idx, 0].text(
            -0.20, 1.18, letter,
            transform=axes[row_idx, 0].transAxes,
            fontweight='bold', va='bottom', ha='right',
            fontsize=letter_fs,
        )

    if savefig:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig.savefig(os.path.join(OUTPUT_DIR, OUTPUT_STEM + FIG_FMT), dpi=300)
        fig.savefig(os.path.join(OUTPUT_DIR, OUTPUT_STEM + '.pdf'), dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    figure_1_v2()
