"""Manuscript figure generation. One function per figure, one SVG per panel."""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from configs import *
from simulations import sweep_cartpole_summary as scs
from simulations import sweep_walker_summary as sws
from simulations import sweep_humanoid_balance_summary as shbs
from visualization.plots_cartpole import plot_cost_and_duration_vs_mismatch
from visualization import heatmaps


# Keyed by env so the schedule + success-rate panels can be reused across
# cartpole / walker / humanoid columns without branching internally.
_FIG3_ENV_STYLES = {
    'cartpole': dict(palette=scs.COND_COLORS,
                     xlabel=r'Mismatch ratio $r$ (pole length)'),
    'walker':   dict(palette=sws.COND_COLORS,
                     xlabel=r'Mismatch ratio $r$ (torso mass)'),
    'humanoid': dict(palette=shbs.COND_COLORS,
                     xlabel=r'Mismatch ratio $r$ (gravity)'),
}

# Operating points for per-replan compute time: t_ms = alpha * N * H + intercept,
# fit by simulations.diagnostics.run_timing_model_probe and stored in
# data/results/timing_models.pkl. (N, H) here matches PROPOSAL_CONFIGS['<env>']['N']
# and the H_<env> constants in simulations/sweep_<env>_adaptive.py.
_OPERATING_NH = {
    'cartpole': (30, 44),
    'walker':   (30, 67),
    'humanoid': (30, 85),  # 'humanoid' style key, env name 'humanoid_balance'
}

_TIMING_MODEL_PATH = os.path.join(RESULTS_DIR, 'timing_models.pkl')


def _t_ms_per_replan(env_style_key):
    """Return single-thread MuJoCo replan time (ms) for the env's operating (N, H).

    env_style_key is the key used by _FIG3_ENV_STYLES ('cartpole'/'walker'/'humanoid');
    the on-disk timing pickle is keyed by env_name ('humanoid_balance' for humanoid).
    """
    import pickle
    pkl_key = 'humanoid_balance' if env_style_key == 'humanoid' else env_style_key
    with open(_TIMING_MODEL_PATH, 'rb') as f:
        models = pickle.load(f)
    N, H = _OPERATING_NH[env_style_key]
    p = models[pkl_key]
    return float(p['alpha'] * N * H + p['intercept'])


_PARETO_MARKERS = ('o', 's', '^')  # matched-fast, slow, adaptive (palette order)

# Reader-facing condition labels by palette index. Keep the underlying
# COND_COLORS keys untouched (they index pickle dicts) — only the legend text
# changes.
_FIG3_DISPLAY_LABELS = (
    'Fixed frequent recompute',    # palette[0] — matched-fast Fixed
    'Fixed infrequent recompute',  # palette[1] — slow Fixed
    'Adaptive',                    # palette[2] — paper method
)

# Main-figure (2-condition) view: Fixed (at Fig 2 matched-best star) + Adaptive.
# Indexes into palette = COND_COLORS values in their dict-insertion order.
# COND_COLORS now has exactly two entries (Fixed + Adaptive) per env after the
# 2-line refactor (May 5 2026); keep indices (0, 1).
_FIG3_MAIN_PALETTE_INDICES = (0, 1)
_FIG3_MAIN_DISPLAY_LABELS = (
    'Fixed (matched-best)',
    'Adaptive',
)


def _physical_success_from_last_states(episodes, specs):
    """Per-episode physical success from last-states window.

    episodes: list of (n_terminal, state_dim) arrays (one per seed).
    specs: list of clause dicts from heatmaps.PHYSICAL_SUCCESS.
    Returns float array of shape (n_episodes,).
    """
    stacked = np.stack(episodes, axis=0)   # (n_ep, n_window, state_dim)
    masks = [heatmaps._eval_clause(stacked, s) for s in specs]
    return np.logical_and.reduce(masks).astype(float)


def _draw_success_rate_panel(ax, sweep, letter, env='cartpole',
                             show_legend=True, show_xlabel=True,
                             show_ylabel=True):
    """Per-seed physical success rate vs r, K-free.

    Uses sweep_last_states + PHYSICAL_SUCCESS criteria (Fig 2 convention).
    Plots only the main-figure (2-condition) view: Fixed (matched-best) + Adaptive.
    """
    palette = _FIG3_ENV_STYLES[env]['palette']
    all_labels = list(palette.keys())
    labels = [all_labels[i] for i in _FIG3_MAIN_PALETTE_INDICES]
    mism    = list(sweep['mismatches'])

    sweep_env = sweep['env']
    sweep_ls = sweep.get('last_states') or sweep['sweep_last_states']
    phys_specs = heatmaps.PHYSICAL_SUCCESS[sweep_env]

    for li, lab in enumerate(labels):
        rates, ses = [], []
        for r in mism:
            success = _physical_success_from_last_states(
                sweep_ls[lab][r], phys_specs)
            n = len(success)
            p = float(success.mean())
            rates.append(p)
            # Binomial SE for a Bernoulli proportion
            ses.append(float(np.sqrt(p * (1.0 - p) / n)) if n > 0 else 0.0)
        rates = np.asarray(rates); ses = np.asarray(ses)
        color = palette[lab]
        marker = _PARETO_MARKERS[(_FIG3_MAIN_PALETTE_INDICES[li]) % len(_PARETO_MARKERS)]
        display = _FIG3_MAIN_DISPLAY_LABELS[li]
        ax.fill_between(mism, np.clip(rates - ses, 0.0, 1.0),
                        np.clip(rates + ses, 0.0, 1.0),
                        color=color, alpha=0.2, linewidth=0)
        ax.plot(mism, rates, '-', marker=marker, ms=4.5, lw=1.3,
                color=color, label=display)

    if show_xlabel:
        ax.set_xlabel('Model mismatch')
    if show_ylabel:
        ax.set_ylabel('Success rate')
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0.9, color='0.6', ls=':', lw=0.7, zorder=0)
    if show_legend:
        # lower-left lands in the empty region below the success-rate plateau
        # for cartpole r<=1.6; small font + framealpha avoids overlap with the
        # cliff at r>=1.7.
        ax.legend(loc='lower left', fontsize=7, frameon=True,
                  framealpha=0.9, borderpad=0.4, handletextpad=0.5,
                  borderaxespad=0.4)
    ax.grid(True, alpha=0.2)
    if letter is not None:
        _panel_letter(ax, letter)


_ENV_TITLES = {
    'cartpole': 'Cartpole (pole length)',
    'walker':   'Walker (torso mass)',
    'humanoid': 'Humanoid balance (gravity)',
}


def _compose_figure_3_panels(cartpole_summary, walker_summary, humanoid_summary,
                             cartpole_midswitch, walker_midswitch,
                             humanoid_midswitch, output_path):
    """3×3 stack: rows = (compute schedule, success rate, mid-switch trace);
    cols = envs (cartpole, walker, humanoid balance).

    Rows A and B share x per column (mismatch ratio r); row C uses time and
    stays independent. Row B shares y across columns ([0, 1]); rows A and C
    keep per-env scales because compute budgets and cost units differ.
    A single legend in the cartpole success-rate panel covers all three rows
    because they share the Fixed/Adaptive condition palette.
    """
    panel_w, panel_h = 2.4, 1.95
    fig, axes = plt.subplots(
        3, 3,
        figsize=(panel_w * 3 + 0.3, panel_h * 3),
        layout='constrained',
    )
    # Share x between the schedule and success rows (mismatch ratio).
    for c in range(3):
        axes[1, c].sharex(axes[0, c])
    # Share y across the success row (all in [0, 1]).
    for c in range(1, 3):
        axes[1, c].sharey(axes[1, 0])

    summary_pairs = [
        (cartpole_summary, 'cartpole'),
        (walker_summary,   'walker'),
        (humanoid_summary, 'humanoid'),
    ]
    midswitch_pairs = [
        (cartpole_midswitch, 'cartpole'),
        (walker_midswitch,   'walker'),
        (humanoid_midswitch, 'humanoid'),
    ]

    for col, (sweep, env) in enumerate(summary_pairs):
        axes[0, col].set_title(_ENV_TITLES[env])
        if sweep is None:
            _draw_stub(axes[0, col], 'schedule', None)
            _draw_stub(axes[1, col], 'success',  None)
            continue
        # Single shared legend lives in the cartpole success-rate panel where
        # lines plateau near 1.0 and the lower-left is empty until the
        # failure cliff — the only consistently free region across panels.
        _draw_compute_schedule_panel(
            axes[0, col], sweep, letter=None, env=env,
            show_legend=False, show_xlabel=False, show_title=False,
            show_ylabel=(col == 0),
        )
        _draw_success_rate_panel(
            axes[1, col], sweep, letter=None, env=env,
            show_legend=(col == 0 and env == 'cartpole'),
            show_xlabel=True,
            show_ylabel=(col == 0),
        )
        # Each schedule panel keeps its own y scale (different env compute
        # budgets); the success row shares y across columns by sharey above.
        if col != 0:
            axes[1, col].tick_params(axis='y', labelleft=False)
        plt.setp(axes[0, col].get_xticklabels(), visible=False)

    for col, (sweep, env) in enumerate(midswitch_pairs):
        if sweep is None:
            _draw_stub(axes[2, col], 'mid-switch', None)
            continue
        _draw_midswitch_panel(
            axes[2, col], sweep, env=env,
            show_legend=False,
            show_ylabel=(col == 0),
        )
        if col != 0:
            axes[2, col].tick_params(axis='y', labelleft=False)

    # Row letters at the upper-left of column-0 panels — Fig 2 convention.
    letter_fs = matplotlib.rcParams['axes.titlesize'] * 1.2
    for row, lab in [(0, 'A'), (1, 'B'), (2, 'C')]:
        axes[row, 0].text(
            -0.28, 1.04, lab, transform=axes[row, 0].transAxes,
            fontweight='bold', va='bottom', ha='right', fontsize=letter_fs,
        )

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _draw_compute_schedule_panel(ax, sweep, letter, env='cartpole',
                                 show_legend=True, show_xlabel=True,
                                 show_ylabel=True, show_title=True):
    """Lookahead-steps schedule per condition vs mismatch r.

    x = mismatch r (matches Fig 2's "what the world does" axis).
    y = total planner lookahead steps per episode (Σ_k ℓ_k, ×10³).
        Machine-independent compute proxy that responds to BOTH knobs the
        joint adapter moves (R reduces n_replans; H reduces per-replan
        rollout length). Older wall-clock metric used a fixed t_replan
        computed at H_init and silently hid theory's H knob.
    """
    palette = _FIG3_ENV_STYLES[env]['palette']
    all_labels = list(palette.keys())
    labels = [all_labels[i] for i in _FIG3_MAIN_PALETTE_INDICES]
    mism    = list(sweep['mismatches'])
    h_init = _OPERATING_NH[env][1]
    has_rollout_field = 'sweep_rollout_steps' in sweep

    for li, lab in enumerate(labels):
        means, ses = [], []
        for r in mism:
            if has_rollout_field and sweep['sweep_rollout_steps'][lab][r]:
                per_seed = np.asarray(sweep['sweep_rollout_steps'][lab][r]) / 1000.0
            else:
                per_seed = np.asarray(sweep['sweep_recomp'][lab][r]) * float(h_init) / 1000.0
            n = len(per_seed)
            means.append(float(per_seed.mean()))
            ses.append(float(per_seed.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0)
        means = np.asarray(means); ses = np.asarray(ses)
        color = palette[lab]
        marker = _PARETO_MARKERS[(_FIG3_MAIN_PALETTE_INDICES[li]) % len(_PARETO_MARKERS)]
        display = _FIG3_MAIN_DISPLAY_LABELS[li]
        ax.fill_between(mism, np.maximum(means - ses, 0.0), means + ses,
                        color=color, alpha=0.2, linewidth=0)
        ax.plot(mism, means, '-', marker=marker, ms=4.5, lw=1.3,
                color=color, label=display)

    if show_xlabel:
        ax.set_xlabel('Model mismatch')
    if show_ylabel:
        ax.set_ylabel(r'Total planning cost ($\times 10^3$)', fontsize=8)
    ax.set_ylim(bottom=0)
    if show_legend:
        ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.2)
    # show_title kept for backward compat; t_replan no longer meaningful
    # under the lookahead-steps axis, so it's a no-op now.
    del show_title
    if letter is not None:
        _panel_letter(ax, letter)


_MIDSWITCH_REPR_FACTOR = {
    'cartpole': 1.5,
    'walker':   1.5,
    'humanoid': 1.2,  # midswitch sweep grid lacks 1.5; 1.2 is the closest available
}


# Shared by populated and stub panels so they line up after savefig(bbox='tight').
_FIG3_PANEL_SIZE = (5.0, 2.7)
_FIG3_LETTER_XY = (-0.18, 1.02)


def _panel_letter(ax, letter):
    """Place a bold panel letter outside the upper-left of `ax`.

    Uses axes coordinates so the letter sits at a consistent position
    across panels regardless of y-label width. Savefig with tight bbox
    preserves the letter because it is part of the figure.
    """
    x, y = _FIG3_LETTER_XY
    ax.text(
        x, y, letter, transform=ax.transAxes,
        fontsize=12 * SCALE_TEXT, fontweight='bold', va='top', ha='right',
    )


def _stub_panel(path, label='Placeholder', letter=None,
                figsize=_FIG3_PANEL_SIZE):
    """Save a minimal placeholder panel.

    Italic grey label centered on a blank axis-less canvas. Matches the
    visual weight of the data panels without advertising missing data.
    """
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    ax.text(
        0.5, 0.5, label,
        ha='center', va='center', transform=ax.transAxes,
        style='italic', color='#777777',
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if letter is not None:
        _panel_letter(ax, letter)
    fig.savefig(path, dpi=300, bbox_inches=None)
    plt.close(fig)


def _plot_single_heatmap(ax, data, horizons_sec, recomputes_sec, title, norm, cmap='viridis'):
    """Plot a single cost rate heatmap on the given axis."""
    im = ax.imshow(data, aspect='auto', cmap=cmap, norm=norm, origin='lower')

    x_ticks = range(0, len(horizons_sec), max(1, len(horizons_sec) // 6))
    ax.set_xticks(list(x_ticks))
    ax.set_xticklabels([f'{horizons_sec[i]:.1f}' for i in x_ticks])

    y_ticks = range(0, len(recomputes_sec), max(1, len(recomputes_sec) // 5))
    ax.set_yticks(list(y_ticks))
    ax.set_yticklabels([f'{recomputes_sec[i]:.2f}' for i in y_ticks])

    ax.set_xlabel('Horizon (s)')
    ax.set_ylabel('Recompute Interval (s)')
    ax.set_title(title)
    return im


def _plot_discrimination(ax, landscapes, factor_key, xlabel):
    """Plot cost discrimination ratio vs mismatch factor."""
    factors = [d[factor_key] for d in landscapes]
    discs = [d['discrimination'] for d in landscapes]
    sems = [d['discrimination_sem'] for d in landscapes]

    ax.errorbar(factors, discs, yerr=sems, fmt='o-', markersize=4, capsize=3, lw=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Best / Median Cost')
    ax.set_title('Cost Discrimination')
    ax.axhline(1.0, ls=':', color='gray', alpha=0.5)
    ax.grid(alpha=0.3)


def _plot_action_cost(ax, landscapes, highlight_factors, factor_key, action_key, xlabel):
    """Plot first action vs evaluated cost for selected mismatch levels."""
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(highlight_factors)))

    for factor, color in zip(highlight_factors, colors):
        entry = min(landscapes, key=lambda d: abs(d[factor_key] - factor))
        actions = entry[action_key]
        evals = entry['evaluations']

        n_bins = 30
        bin_edges = np.linspace(np.percentile(actions, 2), np.percentile(actions, 98), n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_idx = np.digitize(actions, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        bin_medians = np.array([
            np.median(evals[bin_idx == b]) if np.any(bin_idx == b) else np.nan
            for b in range(n_bins)
        ])
        valid = ~np.isnan(bin_medians)

        label_val = f'{factor:.1f}' if factor != int(factor) else f'{int(factor)}'
        ax.plot(
            bin_centers[valid], bin_medians[valid],
            '-', color=color, lw=1.5, label=f'{label_val}',
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Evaluated Cost')
    ax.set_title('Action-Cost Landscape')
    ax.legend(title='Model/Env', loc='best')
    ax.grid(alpha=0.3)


def figure_diagnostics(mismatch_sweep=None,
                       cartpole_landscapes=None,
                       output_dir=FIGURES_DIR, savefig=True):
    """Diagnostic panels (not in current manuscript).

    Cost-vs-mismatch, cost discrimination, and action-cost landscape
    breakdowns. Kept runnable for sanity checks.

    mismatch_sweep: list of dicts from run_mismatch_sweep (cartpole)
    cartpole_landscapes: list of dicts with keys length_factor, first_actions, evaluations, discrimination
    """
    if not savefig:
        return
    os.makedirs(output_dir, exist_ok=True)

    if mismatch_sweep is not None:
        length_factors = [d['length_factor'] for d in mismatch_sweep]
        mean_cost_rates = [d['mean_cost_rate'] for d in mismatch_sweep]
        sem_cost_rates = [d['sem_cost_rate'] for d in mismatch_sweep]
        mean_durations = [d['mean_duration_sec'] for d in mismatch_sweep]
        sem_durations = [d['sem_duration_sec'] for d in mismatch_sweep]

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        plot_cost_and_duration_vs_mismatch(
            ax1, ax2, length_factors, mean_cost_rates, sem_cost_rates,
            mean_durations, sem_durations, label='MPC (H=1.0s; R=0.02s)',
            cost_label='Cost Rate (cost/s)',
        )

        ax1.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir + 'figdiag_A' + FIG_FMT, dpi=300)
        plt.close(fig)
    else:
        _stub_panel(output_dir + 'figdiag_A' + FIG_FMT, 'Cartpole: cost rate vs mismatch')

    if cartpole_landscapes is not None:
        fig, ax = plt.subplots()
        _plot_discrimination(ax, cartpole_landscapes, 'length_factor',
                             'Pole Length: Model / Env')
        fig.tight_layout()
        fig.savefig(output_dir + 'figdiag_B' + FIG_FMT, dpi=300)
        plt.close(fig)
    else:
        _stub_panel(output_dir + 'figdiag_B' + FIG_FMT, 'Cartpole: cost discrimination')

    if cartpole_landscapes is not None:
        fig, ax = plt.subplots()
        _plot_action_cost(
            ax, cartpole_landscapes,
            highlight_factors=[0.5, 1.0, 2.0],
            factor_key='length_factor',
            action_key='first_actions',
            xlabel='First Action (force, N)',
        )
        fig.tight_layout()
        fig.savefig(output_dir + 'figdiag_C' + FIG_FMT, dpi=300)
        plt.close(fig)
    else:
        _stub_panel(output_dir + 'figdiag_C' + FIG_FMT, 'Cartpole: action-cost landscape')

_FIG2_ROW_LABELS = [
    ('cartpole',         'CartPole',         'A'),
    ('walker',           'Walker',           'B'),
    ('humanoid_balance', 'Humanoid Balance', 'C'),
]


def _draw_fig2_stub_row(ax_row, env_label, letter):
    """Fill a row of 4 axes with a single italic grey placeholder.

    Keeps the composed figure valid when a grid pickle is missing.
    """
    for i, ax in enumerate(ax_row):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax_row[0].text(
        2.0, 0.5, f'{env_label} — grid_{env_label.lower().replace(" ", "_")}.pkl not found',
        ha='center', va='center', transform=ax_row[0].transAxes,
        style='italic', color='#777777',
    )
    ax_row[0].text(
        -0.28, 1.04, letter,
        transform=ax_row[0].transAxes,
        fontweight='bold', va='bottom', ha='right',
        fontsize=plt.rcParams['axes.titlesize'] * 1.2,
    )


def _compose_figure_2(results_by_env, dt, output_path, stability_K=4.0,
                      criterion='physical', success_threshold=0.9):
    """Single-figure composition of the Figure 2 3×4 heatmap grid.

    Uses one `plt.subplots(3, 4)` call so panel dimensions stay uniform
    across rows. Each env row
    gets its own LogNorm colorbar on the right. Panel letters are drawn
    via axes-coords bold text (fig3 convention). The env name rides in
    the leftmost ylabel rather than as a separate `fig.text` rotation
    because constrained_layout does not reserve space for free-floating
    text and it would overlap the `R·dt` label.
    """
    # Figsize overshoots so constrained_layout can honour aspect='equal' without
    # collapsing panels; layout crops the excess.
    ncols, nrows = 4, 3
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.1 * ncols + 1.8, 2.1 * nrows + 0.6),
        constrained_layout=True,
    )

    for row_idx, (env_key, env_label, letter) in enumerate(_FIG2_ROW_LABELS):
        ax_row = axes[row_idx]
        result = results_by_env.get(env_key)
        if result is None:
            _draw_fig2_stub_row(ax_row, env_label, letter)
            continue
        env_dt = result.get('dt', dt)
        heatmaps.plot_heatmap_row(
            ax_row, result, env_label, env_dt,
            cbar_label=heatmaps.CBAR_LABELS_SHORT.get(env_key),
            stability_K=stability_K,
            letter=None,                 # drawn below in fig coords
            show_col_titles=True,
            show_xlabel=False,           # supxlabel handles it
            show_row_label=True,         # env name on leftmost ylabel
            ylabel_text=None,            # supylabel handles "Replan interval"
            colorbar=True,
            cbar_stability_suffix=False,
            criterion=criterion,
            success_threshold=success_threshold,
        )

    fig.supxlabel('Planning horizon (s)')
    fig.supylabel('Replan interval (s)')

    # Panel letters in axes-relative coords so constrained_layout reserves
    # the space and they always sit clearly above the column title row.
    for row_idx, (env_key, _, letter) in enumerate(_FIG2_ROW_LABELS):
        if results_by_env.get(env_key) is None:
            continue
        axes[row_idx, 0].text(
            -0.20, 1.18, letter,
            transform=axes[row_idx, 0].transAxes,
            fontweight='bold', va='bottom', ha='right',
            fontsize=matplotlib.rcParams['axes.titlesize'] * 1.2,
        )

    fig.savefig(output_path, dpi=300, bbox_inches=None)
    plt.close(fig)


def figure_2(cartpole_grid=None, walker_grid=None, humanoid_balance_grid=None,
             output_dir=FIGURES_DIR, savefig=True, stability_K=4.0,
             criterion='physical', success_threshold=0.9):
    """Manuscript Figure 2 — 3×4 heatmap grid.

    Rows: A = CartPole, B = Walker, C = Humanoid Balance. Columns: four
    mismatch levels (col 0 matched, cols 1–3 increasing). White `///`
    hatches mark cells where the per-seed success rate is at least
    `success_threshold`. With `criterion='physical'` (default) success
    is a K-free physical end-state criterion (issue #181); with
    `criterion='k_cost'` it is per-seed cost ≤ `stability_K × matched_best`.
    """
    if not savefig:
        return
    os.makedirs(output_dir, exist_ok=True)

    results_by_env = {
        'cartpole':         cartpole_grid,
        'walker':           walker_grid,
        'humanoid_balance': humanoid_balance_grid,
    }
    heatmaps.build_figure_2(results_by_env, DT, output_dir,
                            fig_fmt=FIG_FMT, stability_K=stability_K,
                            criterion=criterion,
                            success_threshold=success_threshold)
    _compose_figure_2(results_by_env, DT,
                      output_dir + 'fig2_final' + FIG_FMT,
                      stability_K=stability_K,
                      criterion=criterion,
                      success_threshold=success_threshold)


def _draw_stub(ax, label, letter):
    """Draw a minimal placeholder onto a pre-existing axis."""
    ax.text(
        0.5, 0.5, label,
        ha='center', va='center', transform=ax.transAxes,
        style='italic', color='#777777',
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if letter is not None:
        _panel_letter(ax, letter)


def _draw_midswitch_panel(ax, sweep, env='cartpole',
                          show_legend=True, show_xlabel=True,
                          show_ylabel=True, show_title=True):
    """Mid-switch cost-rate timeseries (1 s rolling window) at the representative
    post-switch factor. Uses the same 2-line view as the summary row (Fixed at
    white star + Adaptive at white star) — palette and labels match
    _draw_success_rate_panel so the figure-wide legend covers all rows.
    """
    repr_factor = _MIDSWITCH_REPR_FACTOR[env]
    dt = sweep['dt']
    t = np.arange(sweep['n_steps']) * dt
    window = max(1, round(1.0 / dt))  # 1-second rolling window in steps

    def _rolling_mean(arr):
        """Causal 1-s rolling mean along axis=1, returned as cost/sec."""
        out = np.empty_like(arr, dtype=float)
        cs = np.nancumsum(arr, axis=1)
        out[:, :window] = cs[:, :window] / np.arange(1, window + 1)
        out[:, window:] = (cs[:, window:] - cs[:, :-window]) / window
        return out / dt

    switch_t = sweep['switch_step'] * dt
    ax.axvspan(0, switch_t, color='0.93', zorder=0)
    ax.axvline(switch_t, color='0.4', ls=':', lw=0.8)
    ax.text(switch_t, 1.0, f' r={repr_factor}', transform=ax.get_xaxis_transform(),
            fontsize=7, color='0.4', va='top', clip_on=False)

    # Same 2-line palette/labels as the summary panels (post-#223). The
    # midswitch sweep stores trace data under the summary's COND_COLORS keys,
    # so palette lookup here also keys the trace dict.
    palette = _FIG3_ENV_STYLES[env]['palette']
    all_labels = list(palette.keys())
    labels = [all_labels[i] for i in _FIG3_MAIN_PALETTE_INDICES]
    for li, lab in enumerate(labels):
        rate_mat = _rolling_mean(sweep['traces'][lab][repr_factor]['cost'])
        mean = np.nanmean(rate_mat, axis=0)
        sem  = np.nanstd(rate_mat, ddof=1, axis=0) / np.sqrt(rate_mat.shape[0])
        color = palette[lab]
        display = _FIG3_MAIN_DISPLAY_LABELS[li]
        ax.plot(t, mean, color=color, lw=1.5, label=display)
        ax.fill_between(t, np.maximum(mean - sem, 0), mean + sem,
                        color=color, alpha=0.2, linewidth=0)
    if show_xlabel:
        ax.set_xlabel('Time (s)')
    if show_ylabel:
        ax.set_ylabel('Cost/s')
    ax.set_ylim(bottom=0)
    if show_legend:
        ax.legend(loc='upper left', fontsize=7)
    ax.grid(True, alpha=0.2)


def figure_3(cartpole_summary=None, walker_summary=None, humanoid_summary=None,
             cartpole_midswitch=None, walker_midswitch=None,
             humanoid_midswitch=None, output_dir=FIGURES_DIR, savefig=True):
    """Manuscript Figure 3 — 3×3 stack of compute, success, and mid-switch.

    Row A: mean compute time per episode vs model mismatch r per env.
    Row B: per-seed physical success rate vs r per env.
    Row C: 1-s rolling cost rate over time at the representative post-switch
    factor; switch line marks the perturbation onset.

    Cols: cartpole, walker, humanoid balance.
    """
    if not savefig:
        return
    os.makedirs(output_dir, exist_ok=True)
    _compose_figure_3_panels(
        cartpole_summary, walker_summary, humanoid_summary,
        cartpole_midswitch, walker_midswitch, humanoid_midswitch,
        output_dir + 'fig3_final' + FIG_FMT,
    )


# --------------------------------------------------------------------------- #
# Table and statistics helpers for Figure 3 manuscript reporting              #
# --------------------------------------------------------------------------- #

def _fig3_success(sweep, lab, r):
    """Per-episode physical success array for (sweep, label, r)."""
    sweep_env = sweep['env']
    sweep_ls = sweep.get('last_states') or sweep['sweep_last_states']
    specs = heatmaps.PHYSICAL_SUCCESS[sweep_env]
    return _physical_success_from_last_states(sweep_ls[lab][r], specs)


def print_fig3_table(cartpole_summary=None, walker_summary=None,
                     humanoid_summary=None):
    """Print cost/s ± SE and success-rate table for Figure 3.

    Rows: each env at r=1.0 (matched) and r=mismatch_a (differentiated).
    Columns: Fixed frequent | Fixed infrequent | Adaptive.
    """
    summaries = [
        (cartpole_summary, 'Cartpole',  'cartpole'),
        (walker_summary,   'Walker',    'walker'),
        (humanoid_summary, 'Humanoid',  'humanoid_balance'),
    ]
    col_hdr = ['Fixed frequent', 'Fixed infrequent', 'Adaptive']

    print()
    print('Figure 3 — cost/s and success rate')
    print('-' * 100)
    print(f'{"Env":<12} {"r":>5}' + ''.join(f'  {h:>28}' for h in col_hdr))
    print('-' * 100)

    for sweep, env_display, _env_key in summaries:
        if sweep is None:
            print(f'{env_display:<12}  (no data)')
            continue
        labels = list(sweep['sweep_cost'].keys())
        mismatch_a = sweep.get('mismatch_a')
        rs = [1.0] + ([mismatch_a] if mismatch_a is not None else [])
        for r in rs:
            if r not in sweep['sweep_cost'][labels[0]]:
                continue
            cells = []
            for lab in labels:
                costs = np.asarray(sweep['sweep_cost'][lab][r])
                lens  = np.asarray(sweep['sweep_len'][lab][r])
                cs = costs / lens
                mean_c = cs.mean()
                se_c = cs.std(ddof=1) / np.sqrt(len(cs)) if len(cs) > 1 else 0.0
                pct = 100.0 * _fig3_success(sweep, lab, r).mean()
                cells.append(f'{mean_c:.3f}±{se_c:.3f} ({pct:.0f}%)')
            print(f'{env_display:<12} {r:>5.2f}' + ''.join(f'  {c:>28}' for c in cells))

    print('-' * 100)
    print('Values: mean cost/s ± SE  (success %)')
    print()


def print_fig3_stats(cartpole_summary=None, walker_summary=None,
                     humanoid_summary=None):
    """Print Adaptive vs Fixed-infrequent statistics at mismatch_a per env.

    Reports: cost/s improvement (%), Welch t-test, success rates,
    Fisher's exact test (two-tailed).
    """
    from scipy.stats import ttest_ind, fisher_exact

    summaries = [
        (cartpole_summary, 'Cartpole',  'cartpole'),
        (walker_summary,   'Walker',    'walker'),
        (humanoid_summary, 'Humanoid',  'humanoid_balance'),
    ]

    print()
    print('Figure 3 — Adaptive vs Fixed-infrequent at mismatch_a')
    print('-' * 88)
    print(f'{"Env":<12} {"r":>5}  {"improv%":>9}  {"t-test p":>10}'
          f'  {"succ_adapt":>11}  {"succ_infreq":>12}  {"Fisher p":>10}')
    print('-' * 88)

    for sweep, env_display, _env_key in summaries:
        if sweep is None:
            print(f'{env_display:<12}  (no data)')
            continue
        labels = list(sweep['sweep_cost'].keys())
        mismatch_a = sweep.get('mismatch_a')
        if mismatch_a is None or mismatch_a not in sweep['sweep_cost'][labels[0]]:
            print(f'{env_display:<12}  (mismatch_a not in data)')
            continue

        # labels[1] = Fixed infrequent, labels[2] = Adaptive
        lab_infreq = labels[1]
        lab_adapt  = labels[2]
        r = mismatch_a

        costs_i = np.asarray(sweep['sweep_cost'][lab_infreq][r])
        lens_i  = np.asarray(sweep['sweep_len'][lab_infreq][r])
        costs_a = np.asarray(sweep['sweep_cost'][lab_adapt][r])
        lens_a  = np.asarray(sweep['sweep_len'][lab_adapt][r])
        ci = costs_i / lens_i
        ca = costs_a / lens_a

        improvement = 100.0 * (ci.mean() - ca.mean()) / ci.mean()
        _, p_t = ttest_ind(ci, ca, equal_var=False)

        succ_i = _fig3_success(sweep, lab_infreq, r)
        succ_a = _fig3_success(sweep, lab_adapt, r)
        n_si, n_i = int(succ_i.sum()), len(succ_i)
        n_sa, n_a = int(succ_a.sum()), len(succ_a)
        _, p_f = fisher_exact([[n_sa, n_a - n_sa], [n_si, n_i - n_si]],
                              alternative='two-sided')

        print(f'{env_display:<12} {r:>5.2f}  {improvement:>+8.1f}%  {p_t:>10.4f}'
              f'  {100*succ_a.mean():>10.1f}%  {100*succ_i.mean():>11.1f}%  {p_f:>10.4f}')

    print('-' * 88)
    print('improv% = (Fixed-infreq − Adaptive) / Fixed-infreq × 100')
    print('t-test: two-sample Welch (cost/s); Fisher: two-tailed contingency')
    print()


_ENV_STYLE_KEY = {
    'cartpole':         'cartpole',
    'walker':           'walker',
    'humanoid_balance': 'humanoid',
}


def _draw_cum_trace_panel(ax, sweep, env='cartpole',
                          show_legend=True, show_xlabel=True, show_ylabel=True):
    """Cumulative mean cost/s time series at mismatch_a, mean ± SE across seeds."""
    from configs import ENV_DT
    palette = _FIG3_ENV_STYLES[env]['palette']
    labels  = list(palette.keys())

    sweep_env = sweep.get('env', env)
    dt = ENV_DT.get(sweep_env, ENV_DT['cartpole'])

    for li, lab in enumerate(labels):
        traces = sweep['sweep_cum_traces'][lab]
        if not traces:
            continue
        mat  = np.asarray(traces, dtype=float)   # (n_ep, n_steps-1)
        n_t  = mat.shape[1]
        t    = np.arange(1, n_t + 1) * dt
        mean = mat.mean(axis=0)
        se   = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
        color  = palette[lab]
        marker = _PARETO_MARKERS[li % len(_PARETO_MARKERS)]
        display = _FIG3_DISPLAY_LABELS[li] if li < len(_FIG3_DISPLAY_LABELS) else lab
        ax.plot(t, mean, color=color, lw=1.3, label=display)
        ax.fill_between(t, np.maximum(mean - se, 0), mean + se,
                        color=color, alpha=0.2, linewidth=0)

    if show_xlabel:
        ax.set_xlabel('Time (s)')
    if show_ylabel:
        ax.set_ylabel('Cumulative cost/s')
    ax.set_ylim(bottom=0)
    if show_legend:
        ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.2)


def _compose_supp_fig3(cartpole_summary, walker_summary, humanoid_summary,
                       output_path):
    """1×3 cumulative cost-rate supplement for Figure 3."""
    panel_w, panel_h = 2.4, 1.95
    fig, axes = plt.subplots(
        1, 3,
        figsize=(panel_w * 3 + 0.3, panel_h),
        layout='constrained',
    )
    mismatch_a_by_env = {}
    pairs = [
        (cartpole_summary, 'cartpole', 'A'),
        (walker_summary,   'walker',   'B'),
        (humanoid_summary, 'humanoid', 'C'),
    ]
    for col, (sweep, env, _letter) in enumerate(pairs):
        ma = sweep.get('mismatch_a') if sweep is not None else None
        title_r = f' (r={ma})' if ma is not None else ''
        axes[col].set_title(_ENV_TITLES[env] + title_r)
        if sweep is None:
            _draw_stub(axes[col], 'cum traces', None)
            continue
        _draw_cum_trace_panel(
            axes[col], sweep, env=env,
            show_legend=(col == 0),
            show_ylabel=(col == 0),
        )
        if col != 0:
            axes[col].tick_params(axis='y', labelleft=False)

    letter_fs = matplotlib.rcParams['axes.titlesize'] * 1.2
    for col, (_, _, lab) in enumerate(pairs):
        axes[col].text(-0.28, 1.04, lab, transform=axes[col].transAxes,
                       fontweight='bold', va='bottom', ha='right', fontsize=letter_fs)

    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def figure_supp3(cartpole_summary=None, walker_summary=None,
                 humanoid_summary=None, output_dir=FIGURES_DIR, savefig=True):
    """Supplement Figure 3 — cumulative cost/s time series at mismatch_a (1×3).

    Shows the running-mean cost rate over the episode at the differentiated
    mismatch level for each env. Parallels Figure 4's mid-switch panel.
    """
    if not savefig:
        return
    os.makedirs(output_dir, exist_ok=True)
    _compose_supp_fig3(
        cartpole_summary, walker_summary, humanoid_summary,
        output_dir + 'fig3_supp' + FIG_FMT,
    )


