"""Manuscript Figure 1: closed-loop eigenvalue analysis under model mismatch.

Six panels combining diagnostics from supplement sections 2, 3, and 8:
    A) True eigenvalue vs lookahead horizon
    B) Coarse eigenvalue vs replan interval (compounding under mismatch)
    C) Heatmap of |A_tau(ell)| over (ell, tau) plane
    D) Marginal horizon effect: Delta_ell |A_tau| vs ell
    E) |A_tau(ell)| vs tau for several ell
    F) Budget-constrained |A_tau| along ell/tau = B
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from supplement.shared import (
    DEFAULT_PARAMS,
    optimal_K, greedy_gain, wrong_riccati,
    C_TRUTH, LS_TRUTH, apply_style,
)
from supplement.section3_stale import coarse_eigenvalue
from supplement.section8_joint import _joint_eigenvalue


a, b, q, r = DEFAULT_PARAMS['a'], DEFAULT_PARAMS['b'], DEFAULT_PARAMS['q'], DEFAULT_PARAMS['r']
Kstar = optimal_K(a, b, q, r)

OUTPUT_DIR = 'data/figures'
OUTPUT_STEM = 'fig1'

# Mismatch fixed for panels C, D, E (joint eigenvalue surface)
A_HAT_JOINT = 1.2
B_HAT_JOINT = 0.9

ELL_MAX_JOINT = 15
TAU_MAX_JOINT = 10


def plot_true_eigenvalue_vs_horizon(ax):
    """Panel A: |lambda_true| vs lookahead horizon for several mismatches."""
    K0 = 0.0
    max_ell = 12
    ells = list(range(1, max_ell + 1))

    mismatch_cases = [
        (1.3, b, r'$\hat{a}=1.3$', '#d62728', '-'),
        (a,   b, r'$\hat{a} = a$', C_TRUTH, LS_TRUTH),
        (1.7, b, r'$\hat{a}=1.7$', '#2ca02c', '-'),
    ]

    # Iterate over mismatch cases, computing wrong-VI terminal cost then true eigenvalue
    for a_hat, b_hat, label, color, ls in mismatch_cases:
        eigenvalues = []
        for ell in ells:
            K = K0
            for _ in range(ell - 1):
                K = wrong_riccati(K, a_hat, b_hat, q, r)
            L_hat = -a_hat * b_hat * K / (r + b_hat**2 * K)
            eigenvalues.append(abs(a + b * L_hat))
        ax.plot(ells, eigenvalues, marker='o', color=color, linestyle=ls,
                markersize=3, label=label)

    ax.axhline(1.0, color='#d62728', linestyle='--', alpha=0.5, label=r'$|A_\tau| = 1$')
    ax.set_xlabel(r'Horizon $\ell$')
    ax.set_ylabel(r'$|A_\tau|$')
    ax.set_title('Eigenvalue by horizon')
    ax.set_xticks(ells)
    ax.legend()


def plot_coarse_eigenvalue_vs_tau(ax):
    """Panel B: |A_tau| vs replan interval for several mismatches at fixed K_tilde = K*."""
    Kt = Kstar
    taus = np.arange(1, 11)

    mismatch_cases = [
        (1.3, b, r'$\hat{a}=1.3$', '#d62728', '-'),
        (a,   b, r'$\hat{a} = a$', C_TRUTH, LS_TRUTH),
        (1.7, b, r'$\hat{a}=1.7$', '#2ca02c', '-'),
    ]

    # Compute |A_tau| under each mismatch using the agent's greedy gain on K*
    for ah, bh, label, color, ls in mismatch_cases:
        L_hat = greedy_gain(Kt, ah, bh, r)
        A_vals = [abs(coarse_eigenvalue(tau, a, b, ah, bh, L_hat)) for tau in taus]
        ax.plot(taus, A_vals, marker='o', color=color, linestyle=ls,
                markersize=3, label=label)

    ax.axhline(1.0, color='#d62728', linestyle='--', alpha=0.5, label=r'$|A_\tau| = 1$')
    ax.set_xlabel(r'Replan interval $\tau$')
    ax.set_ylabel(r'$|A_\tau|$')
    ax.set_title('Eigenvalue by replan interval')
    ax.set_yscale('log')
    ax.legend()


def _eigenvalue_grid(ells, taus, a_hat, b_hat):
    """Compute |A_tau(ell)| over the (ell, tau) grid; entries with tau > ell are NaN."""
    grid = np.full((len(taus), len(ells)), np.nan)
    for i, tau in enumerate(taus):
        for j, ell in enumerate(ells):
            if tau <= ell:
                grid[i, j] = abs(_joint_eigenvalue(ell, tau, a, b, a_hat, b_hat, q, r))
    return grid


def plot_eigenvalue_heatmap(ax, fig):
    """Panel C: heatmap of |A_tau(ell)| with stability contours for three mismatch levels.

    The heatmap shows |A_tau| under the central mismatch; the three contour lines
    overlay the |A_tau| = 1 boundary for lesser, central, and greater mismatch.
    """
    contour_cases = [
        ((1.3, B_HAT_JOINT), r'$\hat{a}=1.3$', '#1f77b4'),
        ((A_HAT_JOINT, B_HAT_JOINT), rf'$\hat{{a}}={A_HAT_JOINT}$', '#e91e63'),
        ((1.1, B_HAT_JOINT), r'$\hat{a}=1.1$', 'black'),
    ]

    ells = np.arange(1, ELL_MAX_JOINT + 1)
    taus = np.arange(1, TAU_MAX_JOINT + 1)

    # Background uses the central case
    a_hat_mid, b_hat_mid = contour_cases[1][0]
    heatmap = _eigenvalue_grid(ells, taus, a_hat_mid, b_hat_mid)
    im = ax.imshow(np.clip(heatmap, 0, 2.0), origin='lower', aspect='auto',
                   extent=[0.5, ELL_MAX_JOINT + 0.5, 0.5, TAU_MAX_JOINT + 0.5],
                   cmap='RdYlGn_r', vmin=0, vmax=1.5)

    # Overlay |A_tau| = 1 contour for each mismatch case; build proxy legend entries
    handles = []
    for (a_h, b_h), label, color in contour_cases:
        surface = _eigenvalue_grid(ells, taus, a_h, b_h)
        ax.contour(ells, taus, surface, levels=[1.0], colors=color, linewidths=2)
        handles.append(plt.Line2D([0], [0], color=color, linewidth=2, label=label))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(r'$|A_\tau|$')

    ax.set_xlabel(r'Horizon $\ell$')
    ax.set_ylabel(r'Replan interval $\tau$')
    ax.set_title(r'Eigenvalue in $(\tau, \ell)$ plane')
    ax.legend(handles=handles, loc='upper left')


def plot_marginal_horizon_effect(ax):
    """Panel D: forward difference |Delta_ell |A_tau|| vs ell for tau in 1..8.

    Plots magnitudes on a log y-axis: the sign is suppressed because Delta is
    negative throughout (eigenvalue magnitude decreases monotonically with
    additional horizon under the central mismatch).
    """
    ells = np.arange(1, ELL_MAX_JOINT + 1)
    taus_show = list(range(1, 9))
    colors = plt.cm.Blues(np.linspace(0.3, 0.95, len(taus_show)))

    for tau, color in zip(taus_show, colors):
        A_vals = np.array([_joint_eigenvalue(ell, tau, a, b, A_HAT_JOINT, B_HAT_JOINT, q, r)
                           for ell in ells])
        delta = np.abs(np.diff(np.abs(A_vals)))
        label = rf'$\tau = {tau}$' if tau in (1, 8) else None
        ax.plot(ells[:-1], delta, 'o-', color=color, markersize=3, label=label)

    ax.set_xlabel(r'Horizon $\ell$')
    ax.set_ylabel(r'$|\Delta_\ell |A_\tau||$')
    ax.set_title('Sensitivity to horizon')
    ax.set_yscale('log')
    ax.legend()


def plot_compounding_with_tau(ax):
    """Panel E: |A_tau(ell)| vs tau for several ell at fixed mismatch."""
    ells_show = [1, 3, 5, 8]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']
    taus_fine = np.arange(1, TAU_MAX_JOINT + 1)

    for ell, color in zip(ells_show, colors):
        A_vals = []
        for tau in taus_fine:
            if tau <= ell:
                A_vals.append(abs(_joint_eigenvalue(ell, tau, a, b, A_HAT_JOINT, B_HAT_JOINT, q, r)))
            else:
                A_vals.append(np.nan)
        ax.plot(taus_fine, A_vals, 'o-', color=color, markersize=3, label=rf'$\ell = {ell}$')

    ax.axhline(1, color='#d62728', linestyle='--', linewidth=1)
    ax.set_xlabel(r'Replan interval $\tau$')
    ax.set_ylabel(r'$|A_\tau|$')
    ax.set_title('Sensitivity to replan interval')
    ax.set_yscale('log')
    ax.legend()


def plot_budget_constrained(ax):
    """Panel F: |A_tau(ell)| along the budget contour ell/tau = B for several mismatches."""
    B = 5
    mismatch_cases = [
        (a, b, 'Correct', C_TRUTH, LS_TRUTH),
        (1.3, 1.0, r'$\hat{a}=1.3$', '#ff7f0e', '-'),
        (1.2, 0.9, r'$\hat{a}=1.2, \hat{b}=0.9$', '#2ca02c', '-'),
        (1.0, 0.8, r'$\hat{a}=1.0, \hat{b}=0.8$', '#d62728', '-'),
    ]

    # Walk ell upward; for each ell, choose the smallest tau satisfying ell/tau <= B
    for a_h, b_h, label, color, ls in mismatch_cases:
        ells_budget, A_budget = [], []
        for ell in range(1, 20):
            tau = max(1, int(np.ceil(ell / B)))
            if tau <= ell:
                A = _joint_eigenvalue(ell, tau, a, b, a_h, b_h, q, r)
                ells_budget.append(ell)
                A_budget.append(abs(A))
        ax.plot(ells_budget, A_budget, marker='o', color=color, linestyle=ls,
                markersize=3, label=label)

    ax.axhline(1, color='#d62728', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel(rf'Horizon $\ell$ (along budget $\ell/\tau = {B}$)')
    ax.set_ylabel(r'$|A_\tau|$')
    ax.set_title(rf'Budget-constrained eigenvalue ($B = {B}$)')
    ax.set_yscale('log')
    ax.legend()


def figure_1(savefig=True):
    """Compose the six panels into a 2x3 manuscript figure."""
    apply_style()

    # Override compact supplement defaults with manuscript-scale font sizes
    plt.rcParams.update({
        'font.size': 13,
        'axes.titlesize': 14,
        'axes.labelsize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
    })

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    plot_true_eigenvalue_vs_horizon(axes[0, 0])
    plot_coarse_eigenvalue_vs_tau(axes[0, 1])
    plot_eigenvalue_heatmap(axes[0, 2], fig)
    plot_marginal_horizon_effect(axes[1, 0])
    plot_compounding_with_tau(axes[1, 1])
    plot_budget_constrained(axes[1, 2])

    fig.tight_layout()
    if savefig:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig.savefig(f'{OUTPUT_DIR}/{OUTPUT_STEM}.svg')
        fig.savefig(f'{OUTPUT_DIR}/{OUTPUT_STEM}.pdf')
    plt.close(fig)


if __name__ == '__main__':
    figure_1()
