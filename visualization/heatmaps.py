"""Figure 2 — 3×4 (env × mismatch) cost heatmap grid.

Rows are environments (cartpole, walker, humanoid_balance); columns
are mismatch factors (col 1 = exact match, cols 2–4 = increasing mismatch).
Each row gets its own LogNorm colormap (shared across the 4 columns of the
row, never across rows) because the environments use different cost
formulas and units.

A white `///` hatch overlay marks every (H, R) cell inside the
region-of-stability: `mean_cost ≤ stability_K × matched_best`, where
`matched_best = min(mean_cost[mismatch == 1.0])` per env.

Consumes the result dict schema returned by `simulations.sweep_grid.run_grid_sweep`.
"""

import copy

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize


# Default boundary outline for "viable" cells (P(success) >= success_threshold).
# `#e91e63` (Material Design pink 500) is far enough from viridis's
# purple→blue→green→yellow range to read cleanly across the colormap,
# including the dark purple low-cost cells where the viable region usually
# sits. Linewidth tuned to read clearly at 2.1" panel size.
REGION_OUTLINE_COLOR = '#e91e63'
REGION_OUTLINE_LINEWIDTH = 1.6


def _draw_region_outline(ax, mask, color=REGION_OUTLINE_COLOR,
                         linewidth=REGION_OUTLINE_LINEWIDTH):
    """Trace a rectilinear contour around the True cells in a 2D boolean mask.

    `mask` has shape (n_H, n_R). Coordinates assume the imshow used
    `mat.T` and origin='lower' — i.e., x is H and y is R. Each cell occupies
    a unit square centred on its integer index. For every True cell we add
    line segments only along edges that face a False cell (or a grid border).
    Uses a single LineCollection so the figure stays light.
    """
    n_H, n_R = mask.shape
    segments = []
    for hi in range(n_H):
        for ri in range(n_R):
            if not mask[hi, ri]:
                continue
            x0, x1 = hi - 0.5, hi + 0.5
            y0, y1 = ri - 0.5, ri + 0.5
            if hi == 0 or not mask[hi - 1, ri]:
                segments.append([(x0, y0), (x0, y1)])
            if hi == n_H - 1 or not mask[hi + 1, ri]:
                segments.append([(x1, y0), (x1, y1)])
            if ri == 0 or not mask[hi, ri - 1]:
                segments.append([(x0, y0), (x1, y0)])
            if ri == n_R - 1 or not mask[hi, ri + 1]:
                segments.append([(x0, y1), (x1, y1)])
    if segments:
        ax.add_collection(LineCollection(
            segments, colors=color, linewidths=linewidth,
            capstyle='round', joinstyle='round',
        ))


# Colorbar labels report cost as a per-second rate so the colormap shares
# time units with the H·dt and R·dt axes; the underlying `mean_cost` field
# stores per-step costs and is rescaled by 1/dt at plot time.
# CartPole uses a bounded tolerance cost in [0, 1]/step; Walker/Humanoid
# are unbounded MJPC residuals with different weight scales.
CBAR_LABELS = {
    'cartpole':         'mean cost / s  (1 − upright·centered·small_vel)',
    'walker':           'mean cost / s  (MJPC residual, 4-term)',
    'humanoid_balance': 'mean cost / s  (MJPC residual, balance)',
}

# Terse variant for the composed figure: the rightmost colorbar sits
# flush against the figure edge so longer strings get clipped.
CBAR_LABELS_SHORT = {
    'cartpole':         'mean cost / s',
    'walker':           'mean cost / s',
    'humanoid_balance': 'mean cost / s',
}


def _row_norm(mean_cost_row):
    """Build the per-row color normalization.

    Returns (norm, used_log: bool). Falls back to linear Normalize when the
    row is flat or non-positive (LogNorm would crash).
    """
    valid = mean_cost_row[np.isfinite(mean_cost_row)]
    positive = valid[valid > 0]

    if positive.size == 0:
        mean = float(np.nanmean(mean_cost_row)) if valid.size else 1.0
        return Normalize(vmin=0.9 * mean, vmax=1.1 * mean), False

    vmin = float(positive.min())
    vmax = float(positive.max())
    if vmin >= vmax:
        mean = float(np.mean(positive))
        return Normalize(vmin=0.9 * mean, vmax=1.1 * mean), False

    return LogNorm(vmin=vmin, vmax=vmax), True


def _stability_mask(mean_cost, K):
    """Boolean mask (n_mismatch, n_H, n_R) for cells at or below K × matched_best.

    matched_best is the min over the matched-model slice (mismatch = 1.0, col 0).
    Returns (mask, tau=K*matched_best, matched_best). mask[i, j, k] is True
    where a hatch should be drawn.
    """
    matched_best = float(np.nanmin(mean_cost[0]))
    tau = K * matched_best
    with np.errstate(invalid='ignore'):
        mask = np.isfinite(mean_cost) & (mean_cost <= tau)
    return mask, tau, matched_best


# Per-env physical success criteria for the K-free rule (issue #181).
# Each entry is a *list* of conjuncted clauses; every clause must hold
# for a seed to count as successful. Each clause is averaged over the
# last `n_terminal_states` simulation steps before comparison.
#
# The shared "natural pose" rule for body height:
#   walker / humanoid_balance:  body_z >= 0.8 * MJCF default
#                                       (within 20% of natural pose).
# The cartpole analogue:
#   cartpole:                   |theta| <= 0.2 * FAILURE_ANGLE
#                                       (within 20% of failure-angle band).
# Walker also enforces a forward-motion clause because its task is run,
# not stand: the controller must move forward at >=33% of target speed
# (1.5 m/s -> 0.5 m/s) to qualify as "not stalled" — see manuscript
# §app:per_env, "moves forward at 1.5 m/s, growing without bound as the
# walker falls or stalls."
PHYSICAL_SUCCESS = {
    'cartpole': [
        dict(idx=2,  op='abs_le', threshold=0.1, label='|theta|'),
    ],
    'walker': [
        dict(idx=18, op='ge', threshold=0.8 * 1.30, label='torso_z'),
        dict(idx=20, op='ge', threshold=0.5,        label='com_vx'),
    ],
    'humanoid_balance': [
        dict(idx=55, op='ge', threshold=0.8 * 1.472, label='head_z'),
    ],
}


def _eval_clause(last_states, spec):
    """Evaluate one clause against a `last_states` array.

    `last_states` shape: (..., n_window, state_dim). Returns a boolean array
    of shape (..., ): True where the per-seed window-mean of the relevant
    state component satisfies the clause.
    """
    avg = last_states[..., spec['idx']].mean(axis=-1)
    op  = spec['op']
    thr = spec['threshold']
    if op == 'abs_le':
        return np.abs(avg) <= thr
    if op == 'ge':
        return avg >= thr
    if op == 'le':
        return avg <= thr
    raise ValueError(f"unknown comparator {op!r}")


def _per_seed_physical_success(sweep_result):
    """Per-seed physical success array, shape (n_mismatch, n_H, n_R, n_reps).

    For each clause in PHYSICAL_SUCCESS[env], averages the relevant state
    component over the last-states window and applies the clause's rule.
    A seed succeeds iff every clause holds (logical AND).

    Raises KeyError if the env isn't registered, ValueError if last_states
    is missing.
    """
    env = sweep_result.get('env')
    if env not in PHYSICAL_SUCCESS:
        raise KeyError(
            f'physical success criterion not registered for env={env!r}; '
            f'add it to PHYSICAL_SUCCESS in visualization/heatmaps.py'
        )
    if 'last_states' not in sweep_result:
        raise ValueError(
            f'sweep_result for {env!r} has no last_states field; re-run the '
            f'grid sweep with the issue #181 schema'
        )
    last = np.asarray(sweep_result['last_states'])
    masks = [_eval_clause(last, spec) for spec in PHYSICAL_SUCCESS[env]]
    return np.logical_and.reduce(masks)


def _per_seed_k_cost_success(sweep_result, K):
    """Per-seed cost-based success array, shape (n_mismatch, n_H, n_R, n_reps).

    success_seed = (cost_seed <= K * matched_best), where matched_best is the
    min mean_cost over the mismatch=1.0 slice (same baseline as the legacy
    cell-mean hatch).
    """
    all_costs = np.asarray(sweep_result['all_costs'])
    mean_cost = np.asarray(sweep_result['mean_cost'])
    matched_best = float(np.nanmin(mean_cost[0]))
    tau = K * matched_best
    with np.errstate(invalid='ignore'):
        return np.isfinite(all_costs) & (all_costs <= tau)


def compute_success_rate(sweep_result, criterion='physical', K=4.0):
    """Per-cell success rate, shape (n_mismatch, n_H, n_R), in [0, 1].

    criterion='physical' uses the env's PHYSICAL_SUCCESS spec.
    criterion='k_cost'   uses per-seed cost <= K * matched_best.
    """
    if criterion == 'physical':
        success = _per_seed_physical_success(sweep_result)
    elif criterion == 'k_cost':
        success = _per_seed_k_cost_success(sweep_result, K)
    else:
        raise ValueError(f'unknown criterion {criterion!r}')
    return success.mean(axis=-1)


def compute_success_mask(sweep_result, criterion='physical', K=4.0,
                         success_threshold=0.9):
    """Boolean hatch mask, shape (n_mismatch, n_H, n_R).

    True where the per-cell success rate >= success_threshold.
    """
    rate = compute_success_rate(sweep_result, criterion=criterion, K=K)
    return rate >= success_threshold


def plot_heatmap_row(
    ax_row, sweep_result, env_label, dt,
    cbar_label=None,
    stability_K=4.0,
    stability_hatch=None,                       # legacy, unused
    stability_color=REGION_OUTLINE_COLOR,
    letter=None,
    show_col_titles=True,
    show_xlabel=True,
    show_row_label=True,
    colorbar=True,
    cbar_stability_suffix=True,
    criterion='physical',
    success_threshold=0.9,
    plot_matrix=None,
    plot_norm=None,
    plot_cmap_name='viridis',
    xlabel_text='Planning horizon (s)',
    ylabel_text='Replan interval (s)',
):
    """Render one (env) row of Figure 2 into the 4 axes of `ax_row`.

    Parameters
    ----------
    ax_row : sequence of 4 matplotlib Axes
        One axis per mismatch column.
    sweep_result : dict
        Schema from simulations.sweep_grid.run_grid_sweep — needs
        'H_values', 'R_values', 'mismatch_factors', 'mean_cost', 'env'.
        `mean_cost` has shape (n_mismatch, n_H, n_R).
    env_label : str
        Row label (e.g. 'CartPole'). Used for the leftmost ylabel when
        `show_row_label=True`.
    dt : float
        Control timestep (seconds); ticks are in H·dt and R·dt.
    cbar_label : str or None
        Overrides CBAR_LABELS[sweep_result['env']] if given. The stability
        threshold is appended automatically when `stability_K` is truthy.
    stability_K : float or None
        If truthy, overlay a white `///` hatch on every cell where
        mean_cost ≤ K × matched_best (matched_best = min mean_cost over
        the mismatch=1.0 slice). Pass `None` to disable hatching.
    stability_hatch, stability_color : str
        Hatch pattern and edge color for the overlay rectangles.
    letter : str or None
        Optional bold panel letter drawn in axes coords of the leftmost
        axis (matches the Figure 3 letter convention).
    show_col_titles : bool
        Put `r = <factor>` titles on each column axis.
    show_xlabel : bool
        Put `H·dt [s]` label on the x-axis of each column.
    show_row_label : bool
        Prefix the leftmost ylabel with `env_label`. Disable when the
        composed figure draws the row label via `fig.text`.
    colorbar : bool
        Attach a shared colorbar to the rightmost axis of the row.

    Returns
    -------
    (ims, norm, used_log, stats)
        ims     : list of 4 imshow handles
        norm    : matplotlib Normalize used across the row
        used_log: True if LogNorm; False if the flat-row fallback fired
        stats   : dict with keys 'matched_best', 'tau', 'stable_fraction'
                  (stable_fraction is a list per column, or None if
                  stability_K is falsy).
    """
    ax_row = list(ax_row)
    mean_cost = np.asarray(sweep_result['mean_cost'])
    factors   = list(sweep_result['mismatch_factors'])
    H_values  = np.asarray(sweep_result['H_values'])
    R_values  = np.asarray(sweep_result['R_values'])

    # `plot_matrix` lets supp panels swap in p90 / success_rate / etc. while
    # reusing the rest of the layout code. Hatch logic still keys off the
    # original mean_cost / sweep_result so the masks stay comparable across
    # main fig and supps. We rescale to a per-second rate so the colormap
    # shares time units with the H·dt and R·dt axes (sweep_result stores
    # per-step values). Supps that pass `plot_matrix` are responsible for
    # scaling their own data and can pass `plot_norm` to bypass _row_norm.
    if plot_matrix is None:
        env_dt_for_scale = float(sweep_result.get('dt', dt))
        cost_for_imshow = mean_cost / env_dt_for_scale
    else:
        cost_for_imshow = np.asarray(plot_matrix)
    assert cost_for_imshow.shape == (len(factors), len(H_values), len(R_values)), (
        f"plot matrix shape {cost_for_imshow.shape} != "
        f"({len(factors)}, {len(H_values)}, {len(R_values)})"
    )
    assert len(ax_row) == len(factors) == 4, (
        f"Figure 2 row expects 4 axes and 4 mismatch factors; "
        f"got {len(ax_row)} axes, {len(factors)} factors"
    )

    if plot_norm is not None:
        norm, used_log = plot_norm, isinstance(plot_norm, LogNorm)
    else:
        norm, used_log = _row_norm(cost_for_imshow)
        if not used_log:
            env_key = sweep_result.get('env', env_label.lower())
            print(
                f"[heatmaps] {env_key}: flat or non-positive row detected — "
                f"falling back to linear Normalize (LogNorm unusable)."
            )

    matched_best = float(np.nanmin(mean_cost[0]))
    tau = None
    if criterion == 'physical':
        if stability_K is None:
            mask = None
            stable_fraction = None
        else:
            mask = compute_success_mask(
                sweep_result, criterion='physical',
                success_threshold=success_threshold,
            )
            stable_fraction = [
                float(np.nanmean(mask[i])) for i in range(len(factors))
            ]
    elif criterion == 'k_cost':
        if stability_K is None:
            mask = None
            stable_fraction = None
        else:
            mask = compute_success_mask(
                sweep_result, criterion='k_cost', K=stability_K,
                success_threshold=success_threshold,
            )
            tau = stability_K * matched_best
            stable_fraction = [
                float(np.nanmean(mask[i])) for i in range(len(factors))
            ]
    elif criterion == 'cell_mean':
        # Legacy: cell-mean cost <= K * matched_best, no per-seed aggregation.
        if stability_K is None:
            mask = None
            stable_fraction = None
        else:
            mask, tau, matched_best = _stability_mask(mean_cost, stability_K)
            stable_fraction = [
                float(np.nanmean(mask[i])) for i in range(len(factors))
            ]
    else:
        raise ValueError(f'unknown criterion {criterion!r}')

    cmap = copy.copy(matplotlib.colormaps[plot_cmap_name])
    cmap.set_bad(color='#cccccc')

    # Endpoints + midpoint only — 4-column row at publication width cannot
    # fit 5+ tick labels per axis without overlap.
    n_H = len(H_values)
    n_R = len(R_values)
    xtick_idx = sorted({0, n_H // 2, n_H - 1})
    ytick_idx = sorted({0, n_R // 2, n_R - 1})
    x_ticklabels = [f'{H_values[i] * dt:.2f}' for i in xtick_idx]
    y_ticklabels = [f'{R_values[i] * dt:.2f}' for i in ytick_idx]

    ims = []
    for col_idx, (ax, factor) in enumerate(zip(ax_row, factors)):
        mat = np.ma.array(cost_for_imshow[col_idx],
                          mask=~np.isfinite(cost_for_imshow[col_idx]))
        # Transpose so H is on x-axis and R is on y-axis. aspect='equal'
        # gives each (H, R) cell a unit square (square axes for 10×10 grid).
        im = ax.imshow(mat.T, aspect='equal', cmap=cmap, norm=norm,
                       origin='lower')
        ims.append(im)

        if mask is not None:
            _draw_region_outline(ax, mask[col_idx], color=stability_color)
            # Mark the cheapest stable cell by total planning steps per
            # episode (∝ H/R; same quantity as Fig 3 row 1's Σℓₖ axis).
            if mask[col_idx].any():
                H_grid, R_grid = np.meshgrid(H_values, R_values, indexing='ij')
                ratio = np.where(mask[col_idx], H_grid / R_grid, np.inf)
                h_i, r_i = np.unravel_index(np.argmin(ratio), ratio.shape)
                ax.scatter(h_i, r_i, marker='*', s=55,
                           facecolor='white', edgecolor='white',
                           linewidths=0.6, zorder=5)

        ax.set_xticks(xtick_idx)
        ax.set_xticklabels(x_ticklabels, rotation=0)
        ax.set_yticks(ytick_idx)
        ax.set_yticklabels(y_ticklabels)

        if show_xlabel and xlabel_text:
            ax.set_xlabel(xlabel_text)
        else:
            ax.set_xlabel('')

        if col_idx == 0:
            parts = []
            if show_row_label and env_label:
                parts.append(env_label)
            if ylabel_text:
                parts.append(ylabel_text)
            ax.set_ylabel('\n'.join(parts))
        else:
            ax.set_ylabel('')
            ax.tick_params(axis='y', labelleft=False)

        if show_col_titles:
            if abs(factor - 1.0) < 1e-12:
                ax.set_title('No mismatch')
            else:
                ax.set_title(f'Mismatch ratio = {factor:.2f}')

    if letter is not None:
        ax_row[0].text(
            -0.28, 1.04, letter,
            transform=ax_row[0].transAxes,
            fontweight='bold', va='bottom', ha='right',
            fontsize=matplotlib.rcParams['axes.titlesize'] * 1.2,
        )

    if colorbar:
        base_label = cbar_label if cbar_label is not None else CBAR_LABELS.get(
            sweep_result.get('env', ''), 'cost'
        )
        if mask is not None and cbar_stability_suffix:
            if criterion == 'physical':
                specs = PHYSICAL_SUCCESS.get(sweep_result.get('env', '')) or []
                clause_strs = []
                for spec in specs:
                    name = spec.get('label', f's_{{{spec["idx"]}}}')
                    if spec['op'] == 'abs_le':
                        clause_strs.append(rf'$|{name}|\leq{spec["threshold"]:g}$')
                    elif spec['op'] == 'ge':
                        clause_strs.append(rf'${name}\geq{spec["threshold"]:g}$')
                    elif spec['op'] == 'le':
                        clause_strs.append(rf'${name}\leq{spec["threshold"]:g}$')
                rule = ', '.join(clause_strs) if clause_strs else 'physical'
                label = (
                    f'{base_label}\n'
                    rf'outlined: $P(\mathrm{{success}})\geq{success_threshold:g}$, '
                    f'{rule}'
                )
            elif criterion == 'k_cost':
                label = (
                    f'{base_label}\n'
                    rf'outlined: $P(c\leq{stability_K:g}\times$ matched best$)'
                    rf'\geq{success_threshold:g}$'
                )
            else:  # cell_mean
                label = (
                    f'{base_label}\n'
                    rf'outlined: cost $\leq$ {stability_K:g}$\times$ matched best'
                )
        else:
            label = base_label
        cbar = ax_row[-1].figure.colorbar(
            ims[-1], ax=ax_row, shrink=0.85, pad=0.02, label=label,
        )
        cbar.ax.tick_params(labelsize=matplotlib.rcParams['ytick.labelsize'])

    return ims, norm, used_log, {
        'matched_best': matched_best,
        'tau': tau,
        'stable_fraction': stable_fraction,
    }


_PANEL_ORDER = [
    ('cartpole',         'CartPole',         'A'),
    ('walker',           'Walker',           'B'),
    ('humanoid_balance', 'Humanoid Balance', 'C'),
]


# Physical parameter scaled by `r` on each env's mismatch axis.
PERTURBATIONS = {
    'cartpole':         'pole length',
    'walker':           'torso mass',
    'humanoid_balance': 'gravity',
}


def _row_label_with_perturbation(env_key, env_label):
    """Return just the env name. The perturbation (pole length / torso mass /
    gravity) is documented in the manuscript caption, not duplicated in the
    figure label. Kept as a function for API stability with prior call sites.
    """
    return env_label


def build_figure_2_panel(sweep_result, env_label, dt, save_path,
                         stability_K=4.0, criterion='physical',
                         success_threshold=0.9):
    """Render one env's 1×4 heatmap row as a standalone SVG.

    Carries env name in the leftmost ylabel and shows column titles
    and tick labels everywhere, so the file is usable on its own.
    """
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.2),
                             constrained_layout=True)
    plot_heatmap_row(
        axes, sweep_result, env_label, dt,
        stability_K=stability_K,
        show_col_titles=True,
        show_xlabel=True,
        show_row_label=True,
        colorbar=True,
        criterion=criterion,
        success_threshold=success_threshold,
    )
    fig.savefig(save_path)
    plt.close(fig)


def build_figure_2(results_by_env, dt, output_dir, fig_fmt='.svg',
                   stability_K=4.0, criterion='physical',
                   success_threshold=0.9):
    """Save Figure 2 as three separate env panels: fig2_A/B/C.

    `output_dir` must end with '/'. Missing entries in `results_by_env`
    are skipped (the composed figure in figures.py stubs them).
    """
    for env_key, env_label, panel_id in _PANEL_ORDER:
        result = results_by_env.get(env_key)
        if result is None:
            continue
        env_dt = result.get('dt', dt)
        save_path = f"{output_dir}fig2_{panel_id}{fig_fmt}"
        build_figure_2_panel(
            result, _row_label_with_perturbation(env_key, env_label),
            env_dt, save_path, stability_K=stability_K,
            criterion=criterion, success_threshold=success_threshold,
        )
