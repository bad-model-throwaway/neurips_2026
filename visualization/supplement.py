"""Supplementary figure generation."""

import os
import numpy as np
import matplotlib.pyplot as plt

from configs import *
from analysis import (
    compute_cost_rates, compute_recompute_intervals, compute_efficiency,
    compute_rh_traces_sec, get_episode_lengths,
)
from simulations.dataio import filter_by_mpc_settings, create_heatmap_data
from simulations.sweep_cartpole_adaptive import CONDITIONS, COND_STYLES
from simulations import sweep_cartpole_perturbation as scp
from simulations import sweep_walker_perturbation as swp
from visualization.plots_cartpole import (
    plot_cost_and_duration_vs_mismatch, plot_timeseries_by_condition,
)
from visualization.plots_sweep import plot_heatmap_row, plot_metric_vs_mismatch



def supplement_1(all_data, output_dir=FIGURES_DIR, savefig=True):
    """Cost and duration vs model mismatch at fixed MPC settings."""
    if not savefig:
        return
    os.makedirs(output_dir, exist_ok=True)

    filtered = filter_by_mpc_settings(all_data, horizon_sec=1.0, recompute_sec=0.02)
    filtered = sorted(filtered, key=lambda x: x['model_length_factor'])

    length_factors = [d['model_length_factor'] for d in filtered]
    mean_costs = [d['mean_cost'] for d in filtered]
    sem_costs = [d['sem_cost'] for d in filtered]
    mean_durations = [d['mean_duration_sec'] for d in filtered]
    sem_durations = [d['sem_duration_sec'] for d in filtered]

    # Panel A: cost and duration vs mismatch
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    plot_cost_and_duration_vs_mismatch(
        ax1, ax2, length_factors, mean_costs, sem_costs,
        mean_durations, sem_durations, label='MPC (H=1.0s; R=0.02s)'
    )

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')
    fig.tight_layout()
    fig.savefig(output_dir + 'suppl1_A' + FIG_FMT, dpi=300)
    plt.close(fig)


def supplement_2(all_data, output_dir=FIGURES_DIR, savefig=True):
    """Heatmaps of cost and duration over horizon x recompute interval."""
    if not savefig:
        return
    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({'axes.spines.top': True, 'axes.spines.right': True})

    length_factors = sorted(set(d['model_length_factor'] for d in all_data))
    n_cols = len(length_factors)
    subplot_size = 3.5

    # Panel A: cost heatmaps
    fig, axes = plt.subplots(1, n_cols, figsize=(subplot_size * n_cols + 1.5, subplot_size))
    if n_cols == 1:
        axes = [axes]
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    plot_heatmap_row(
        axes, all_data, length_factors, 'mean_cost', cbar_ax,
        create_heatmap_data, cmap='viridis_r', log_scale=True,
        cbar_label='Total Episode Cost'
    )
    fig.tight_layout(rect=[0, 0, 0.90, 1])
    fig.savefig(output_dir + 'suppl2_A' + FIG_FMT, dpi=300)
    plt.close(fig)

    # Panel B: duration heatmaps
    fig, axes = plt.subplots(1, n_cols, figsize=(subplot_size * n_cols + 1.5, subplot_size))
    if n_cols == 1:
        axes = [axes]
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    plot_heatmap_row(
        axes, all_data, length_factors, 'mean_duration_sec', cbar_ax,
        create_heatmap_data, cmap='viridis', log_scale=False,
        cbar_label='Episode Length (s)'
    )
    fig.tight_layout(rect=[0, 0, 0.90, 1])
    fig.savefig(output_dir + 'suppl2_B' + FIG_FMT, dpi=300)
    plt.close(fig)

    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})


def supplement_3(sweep, output_dir=FIGURES_DIR, savefig=True):
    """Adaptive vs fixed recompute MPC across mismatch levels."""
    if not savefig:
        return
    os.makedirs(output_dir, exist_ok=True)

    MISMATCHES = sweep['mismatches']
    labels = [c['label'] for c in CONDITIONS]

    # Panel A1: recompute interval over time
    rh_sec_traces = compute_rh_traces_sec(sweep)
    fig, ax = plt.subplots()
    plot_timeseries_by_condition(
        ax, rh_sec_traces, DT, COND_STYLES,
        ylabel='Recompute Interval (s)', title='Recompute Interval'
    )
    fig.tight_layout()
    fig.savefig(output_dir + 'suppl3_A1' + FIG_FMT, dpi=300)
    plt.close(fig)

    # Panel A2: cost rate over time
    fig, ax = plt.subplots()
    plot_timeseries_by_condition(
        ax,
        {lab: np.array(traces) for lab, traces in sweep['sweep_cum_traces'].items()},
        DT, COND_STYLES,
        ylabel='Cost Rate (cost / s)', title='Average Cost Rate'
    )
    fig.tight_layout()
    fig.savefig(output_dir + 'suppl3_A2' + FIG_FMT, dpi=300)
    plt.close(fig)

    # Panel B1: average recompute interval vs mismatch
    recomp_intervals = compute_recompute_intervals(sweep, labels, MISMATCHES)
    fig, ax = plt.subplots()
    plot_metric_vs_mismatch(
        ax, MISMATCHES, recomp_intervals, COND_STYLES,
        ylabel='Avg Recompute Interval (s)', title='Average Recompute Interval'
    )
    fig.tight_layout()
    fig.savefig(output_dir + 'suppl3_B1' + FIG_FMT, dpi=300)
    plt.close(fig)

    # Panel B2: efficiency vs mismatch
    efficiency = compute_efficiency(sweep, labels, MISMATCHES)
    fig, ax = plt.subplots()
    plot_metric_vs_mismatch(
        ax, MISMATCHES, efficiency, COND_STYLES,
        ylabel='Cost Rate x Recompute Rate',
        title='Efficiency Score (lower = better)'
    )
    fig.tight_layout()
    fig.savefig(output_dir + 'suppl3_B2' + FIG_FMT, dpi=300)
    plt.close(fig)

    # Panel C1: cost rate vs mismatch
    cost_rates = compute_cost_rates(sweep, labels, MISMATCHES)
    fig, ax = plt.subplots()
    plot_metric_vs_mismatch(
        ax, MISMATCHES, cost_rates, COND_STYLES,
        ylabel='Cost Rate (cost / s)', title='Cost Rate'
    )
    fig.tight_layout()
    fig.savefig(output_dir + 'suppl3_C1' + FIG_FMT, dpi=300)
    plt.close(fig)

    # Panel C2: episode length vs mismatch (use failure time if available)
    episode_data = get_episode_lengths(sweep)
    fig, ax = plt.subplots()
    plot_metric_vs_mismatch(
        ax, MISMATCHES, episode_data, COND_STYLES,
        ylabel='Episode Length (s)', title='Episode Length',
        ylim=(0, 45), yticks=[0, 10, 20, 30, 40]
    )
    fig.tight_layout()
    fig.savefig(output_dir + 'suppl3_C2' + FIG_FMT, dpi=300)
    plt.close(fig)


def _draw_perturbation_panel(perturbation, palette, state_key, state_label,
                              output_path, panel_letter,
                              state_hline=None, fall_marker_y=None):
    """Render one env's three-axis perturbation stack (state, error, replan).

    state_key picks the top-axis trace from `traces[label]`
    ('theta' for cartpole, 'torso_z' for walker). state_hline draws a
    reference line on the state axis. fall_marker_y, when provided,
    plots an × at (fall_time, fall_marker_y) for each finite entry of
    `traces[label]['fall_time']` — visualises individual falls the mean
    trace would mask.
    """
    labels = perturbation['labels']
    dt = perturbation['dt']
    n_steps = perturbation['n_steps']
    t = np.arange(n_steps) * dt
    band = (
        perturbation['impulse_start_step'] * dt,
        (perturbation['impulse_start_step']
         + perturbation['impulse_duration_steps']) * dt,
    )
    eps_tol = perturbation['min_error_threshold']
    tau_lo, tau_hi = perturbation['tau_bounds']

    fig, (ax_state, ax_err, ax_tau) = plt.subplots(
        3, 1, figsize=(4.2, 5.4), sharex=True, layout='constrained',
    )

    for ax in (ax_state, ax_err, ax_tau):
        ax.axvspan(*band, color='0.88', zorder=0, linewidth=0)

    for lab in labels:
        traces = perturbation['traces'][lab]
        color = palette[lab]
        ls = '-' if 'Adaptive' in lab else '--'

        state_mean = np.nanmean(traces[state_key], axis=0)
        ax_state.plot(t, state_mean, color=color, ls=ls, lw=1.5, label=lab)

        err_mean = np.nanmean(traces['error'], axis=0)
        ax_err.plot(t, err_mean, color=color, ls=ls, lw=1.5)

        tau_mean_sec = np.nanmean(traces['tau'], axis=0) * dt
        ax_tau.plot(t, tau_mean_sec, color=color, ls=ls, lw=1.5)

        if fall_marker_y is not None and 'fall_time' in traces:
            fall_times = np.asarray(traces['fall_time'], dtype=float)
            finite = fall_times[np.isfinite(fall_times)]
            if finite.size:
                ax_state.scatter(
                    finite, np.full_like(finite, fall_marker_y),
                    marker='x', s=28, color=color, lw=1.2, zorder=5,
                )

    if state_hline is not None:
        ax_state.axhline(state_hline, color='k', lw=0.5, alpha=0.4)
    ax_state.set_ylabel(state_label)
    ax_state.legend(loc='upper right')
    ax_state.grid(True, alpha=0.2)

    ax_err.axhline(eps_tol, color='k', lw=0.8, ls=':', label=r'$\varepsilon$')
    ax_err.set_ylabel(r'$e_t$')
    ax_err.grid(True, alpha=0.2)

    ax_tau.axhline(tau_lo * dt, color='k', lw=0.8, ls=':')
    ax_tau.axhline(tau_hi * dt, color='k', lw=0.8, ls=':')
    ax_tau.set_ylabel('Replan interval (s)')
    ax_tau.set_xlabel('Time (s)')
    ax_tau.grid(True, alpha=0.2)

    ax_state.text(
        -0.18, 1.02, panel_letter, transform=ax_state.transAxes,
        fontsize=12 * SCALE_TEXT, fontweight='bold', va='top', ha='right',
    )

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def supplement_4(cartpole_perturbation, walker_perturbation=None,
                 output_dir=FIGURES_DIR, savefig=True):
    """Perturbation response for cartpole (A) and walker (B).

    Three controllers per env subjected to a mid-episode impulse at matched
    dynamics. Each panel stacks state observable, prediction error e_t
    (with tolerance ε dashed), replan interval τ·dt (with clip bounds dashed).
    Shaded band = impulse window. Adaptive solid, fixed dashed.

    Cartpole: 5 N / 0.2 s lateral impulse; state observable is θ (rad).
    Walker: external torso force during a stance window; state observable
    is torso_z (m), with a dotted line at 0.7 marking the fall threshold.

    Pass `walker_perturbation=None` to render only Panel A.
    """
    if not savefig:
        return
    os.makedirs(output_dir, exist_ok=True)

    _draw_perturbation_panel(
        cartpole_perturbation, scp.COND_COLORS,
        state_key='theta', state_label=r'$\theta$ (rad)',
        output_path=output_dir + 'suppl4_A' + FIG_FMT,
        panel_letter='A', state_hline=0.0,
    )

    if walker_perturbation is not None:
        _draw_perturbation_panel(
            walker_perturbation, swp.COND_COLORS,
            state_key='torso_z', state_label='Torso z (m)',
            output_path=output_dir + 'suppl4_B' + FIG_FMT,
            panel_letter='B', state_hline=0.7, fall_marker_y=0.7,
        )
