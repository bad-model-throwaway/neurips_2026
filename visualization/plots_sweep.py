import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from visualization import style  # noqa: F401  (applies publication rcParams)

def plot_sweep(results, env_mass=0.5, figsize=(8, 4), save=True):
    """Plot integrated error against model mass parameter.

    results: list of (model_mass, curve_int, target_int) tuples
    env_mass: true environment mass [kg], shown as vertical line
    """
    masses = np.array([r[0] for r in results])
    curve_int = np.array([r[1] for r in results])
    target_int = np.array([r[2] for r in results])

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(masses, curve_int, 'o-', label='Integrated track error')
    ax.plot(masses, target_int, 'o-', label='Integrated target error')
    ax.axvline(env_mass, color='k', linestyle='--', alpha=0.5, label='True mass')
    ax.set_xlabel('Model mass [kg]')
    ax.set_ylabel('Integrated distance [m·s]')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(save if isinstance(save, str) else 'sweep.svg')
    return fig, ax

def plot_metric_vs_mismatch(ax, mismatches, values_by_label, styles,
                            ylabel='', title='', ylim=None, yticks=None):
    """Plot mean +/- SEM of a metric across mismatch levels for multiple conditions.

    mismatches: list of mismatch values (x-axis)
    values_by_label: dict {label: dict {mismatch: list of values}}
    styles: dict {label: dict(color=..., marker=..., ls=...)}
    """
    for label, vals_by_m in values_by_label.items():
        means, sems = [], []
        for m in mismatches:
            vals = vals_by_m[m]
            means.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        ax.errorbar(
            mismatches, means, yerr=sems, label=label,
            markersize=6, capsize=4, capthick=1.5, lw=1.5,
            markerfacecolor=styles[label]['color'],
            **styles[label]
        )

    ax.set_xlabel('Pole Length: Model / Environment')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)







def plot_sweep_2d(curve_grid, target_grid, horizons, recompute_intervals, save='sweep_2d.svg'):
    """Plot heatmaps of median integrated error over horizon x recompute interval."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, data, title in [(axes[0], curve_grid, 'Median track error'), (axes[1], target_grid, 'Median target error')]:
        im = ax.imshow(data, origin='lower', aspect='auto')
        ax.set_xticks(range(len(recompute_intervals)))
        ax.set_xticklabels(recompute_intervals)
        ax.set_yticks(range(len(horizons)))
        ax.set_yticklabels(horizons)
        ax.set_xlabel('Recompute interval [steps]')
        ax.set_ylabel('Horizon [steps]')
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label='[m·s]')

    plt.tight_layout()
    if save:
        plt.savefig(save)
    return fig, axes



def plot_error_distribution(curve_errors, target_errors, figsize=(8, 5), save=True):
    """Plot KDE with carpet plot for integrated error distributions.

    curve_errors: 1D array of integrated track errors across all runs
    target_errors: 1D array of integrated target errors across all runs
    """
    from scipy.stats import gaussian_kde

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    for ax, errors, label in [(axes[0], curve_errors, 'Track error'),
                               (axes[1], target_errors, 'Target error')]:
        # KDE
        kde = gaussian_kde(errors)
        x_grid = np.linspace(errors.min() * 0.9, errors.max() * 1.1, 200)
        ax.fill_between(x_grid, kde(x_grid), alpha=0.3)
        ax.plot(x_grid, kde(x_grid))

        # Carpet (rug) plot
        ax.plot(errors, np.zeros_like(errors), '|', color='k', markersize=10, alpha=0.5)

        ax.set_ylabel('Density')
        ax.set_title(label)
        ax.grid(alpha=0.3)

    axes[1].set_xlabel('Integrated distance [m·s]')
    plt.tight_layout()
    if save:
        plt.savefig(save if isinstance(save, str) else 'error_distribution.svg')
    return fig, axes


def plot_heatmap_row(axes, all_data, length_factors, value_key, cbar_ax,
                     create_heatmap_data_func, cmap='viridis', log_scale=False,
                     cbar_label=''):
    """Plot a row of heatmaps across mismatch levels with shared colorbar.

    axes: array of axes, one per length factor
    all_data: list of result dicts from load_all_results
    length_factors: list of model/env ratios
    value_key: which field to plot ('mean_cost' or 'mean_duration_sec')
    cbar_ax: axis for colorbar
    create_heatmap_data_func: function(all_data, lf, value_key) -> (data, h, r)
    cmap: colormap name
    log_scale: use LogNorm for color scale
    cbar_label: colorbar label text
    """
    # Compute global range for consistent color scale
    all_vals = [d[value_key] for d in all_data]
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)

    if log_scale:
        vmin = max(vmin, 1)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    images = []
    for col, lf in enumerate(length_factors):
        ax = axes[col]
        data, horizons_sec, recomputes_sec = create_heatmap_data_func(all_data, lf, value_key)

        if data is None:
            continue

        if log_scale:
            data = np.clip(data, vmin, None)

        # Draw heatmap
        im_kwargs = dict(aspect='auto', cmap=cmap, origin='lower')
        if log_scale:
            im_kwargs['norm'] = norm
        else:
            im_kwargs['vmin'] = vmin
            im_kwargs['vmax'] = vmax
        im = ax.imshow(data, **im_kwargs)
        images.append(im)

        # Tick labels (every 4th)
        x_ticks = range(0, len(horizons_sec), 4)
        ax.set_xticks(list(x_ticks))
        ax.set_xticklabels([f'{horizons_sec[i]:.1f}' for i in x_ticks])

        y_ticks = range(0, len(recomputes_sec), 4)
        ax.set_yticks(list(y_ticks))
        ax.set_yticklabels([f'{recomputes_sec[i]:.2f}' for i in y_ticks])

        ax.set_xlabel('Horizon (s)')
        if col == 0:
            ax.set_ylabel('Recompute Interval (s)')

        lf_str = f'{int(lf)}' if lf == int(lf) else f'{lf}'
        ax.set_title(f'Pole Length: Model/Env = {lf_str}')

    # Shared colorbar
    if images:
        cbar = plt.colorbar(images[0], cax=cbar_ax)
        cbar.set_label(cbar_label)
        cbar.ax.tick_params()