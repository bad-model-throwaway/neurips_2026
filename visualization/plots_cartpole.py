import numpy as np
import matplotlib.pyplot as plt

from visualization import style  # noqa: F401  (applies publication rcParams)


def plot_cartpole_history(history, dt=0.02, title=None):
    """Plot CartPole simulation history. Returns the figure.

    Left column: angle, position, force, cost (always shown).
    Right column: adaptation panels when present in history keys.
    Falls back to single-column layout when no adaptation panels exist.
    Caller is responsible for savefig/close on the returned figure.
    """
    states, actions, costs = history.get_state_action_cost()
    time = np.arange(len(states)) * dt

    # Collect available adaptation panels
    # threshold gets overlaid on error panels rather than its own panel
    adapt_candidates = [
        'recompute_interval', 'horizon',
        'running_error', 'error', 'state_pe', 'cost_pe', 'cost_surprise',
        'prediction_error', 'mean_error',
        'E_curr', 'eta', 'sigma',
    ]
    adapt_panels = [k for k in adapt_candidates if k in history.keys]

    # Look up threshold series if recorded
    threshold = history.get_item_history('threshold') if 'threshold' in history.keys else None
    log_eps = history.get_item_history('log_epsilon') if 'log_epsilon' in history.keys else None

    # Panel dispatch
    PANEL_FUNCS = {
        'recompute_interval': lambda ax, t, d: plot_cartpole_recompute(ax, t, d, dt),
        'horizon':            lambda ax, t, d: plot_cartpole_horizon(ax, t, d, dt),
        'running_error':      lambda ax, t, d: plot_cartpole_running_error(ax, t, d, threshold),
        'error':              plot_cartpole_prediction_error,
        'state_pe':           plot_cartpole_state_pe,
        'cost_pe':            plot_cartpole_cost_pe,
        'cost_surprise':      plot_cartpole_cost_surprise,
        'prediction_error':   plot_cartpole_prediction_error,
        'mean_error':         lambda ax, t, d: plot_cartpole_adaptation_state(ax, t, d, threshold),
        'E_curr':             plot_cartpole_divergence,
        'eta':                plot_cartpole_noise_floor,
        'sigma':              lambda ax, t, d: plot_cartpole_sigma(ax, t, d, log_eps),
    }

    # Two-column layout when adaptation panels exist; otherwise single-column.
    n_core = 4
    if adapt_panels:
        n_right = len(adapt_panels)
        n_rows = max(n_core, n_right)
        fig, axes = plt.subplots(n_rows, 2, figsize=(16, 2 * n_rows), sharex=True)

        # Core panels in left column
        plot_cartpole_angle(axes[0, 0], time, states)
        plot_cartpole_position(axes[1, 0], time, states)
        plot_cartpole_force(axes[2, 0], time, actions)
        plot_cartpole_cost(axes[3, 0], time, costs)

        # Hide unused left-column rows
        for i in range(n_core, n_rows):
            axes[i, 0].set_visible(False)

        # Adaptation panels in right column
        for i, key in enumerate(adapt_panels):
            data = history.get_item_history(key)
            PANEL_FUNCS[key](axes[i, 1], time, data)

        # Hide unused right-column rows
        for i in range(n_right, n_rows):
            axes[i, 1].set_visible(False)

        # X-axis labels on bottom visible row of each column
        axes[n_core - 1, 0].set_xlabel('Time (s)')
        axes[n_right - 1, 1].set_xlabel('Time (s)')

    else:
        fig, axes = plt.subplots(n_core, 1, figsize=(10, 2 * n_core), sharex=True)
        plot_cartpole_angle(axes[0], time, states)
        plot_cartpole_position(axes[1], time, states)
        plot_cartpole_force(axes[2], time, actions)
        plot_cartpole_cost(axes[3], time, costs)
        axes[-1].set_xlabel('Time (s)')

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_cartpole_angle(ax, time, states, failure_threshold=30):
    """Pole angle over time with failure thresholds.

    failure_threshold: angle in degrees
    """
    ax.plot(time, np.degrees(states[:, 2]), 'b-', lw=1.5, label='Pole angle')
    ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)
    ax.axhline(failure_threshold, color='r', ls='--', lw=1.5, label='Failure threshold')
    ax.axhline(-failure_threshold, color='r', ls='--', lw=1.5)
    ax.set_ylabel('Angle (deg)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_cartpole_position(ax, time, states, position_bound=None):
    """Cart position over time with boundary limits."""
    if position_bound is None:
        from configs import POSITION_BOUND
        position_bound = POSITION_BOUND
    ax.plot(time, states[:, 0], 'm-', lw=1.5, label='Cart position')
    ax.axhline(0, color='k', ls='-', lw=0.5, alpha=0.3)
    ax.axhline(position_bound, color='r', ls='--', lw=1.5, label='Boundary')
    ax.axhline(-position_bound, color='r', ls='--', lw=1.5)
    ax.set_ylabel('Position (m)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_cartpole_force(ax, time, actions):
    """Control force over time with force limits."""
    ax.plot(time, actions, 'g-', lw=1, label='Control force')
    ax.axhline(0, color='k', ls='--', lw=0.5, alpha=0.5)
    ax.axhline(10.0, color='r', ls=':', lw=0.5, alpha=0.5, label='Force limits')
    ax.axhline(-10.0, color='r', ls=':', lw=0.5, alpha=0.5)
    ax.set_ylabel('Force (N)')
    ax.set_ylim(-12, 12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_cartpole_cost(ax, time, costs):
    """Instantaneous cost over time."""
    ax.plot(time, costs, color='orange', lw=1, label='Cost')
    ax.set_ylabel('Cost')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_cartpole_prediction_error(ax, time, errors):
    """Prediction error over time.

    errors: list or array of scalar prediction errors, same length as time.
        NaN values are skipped.
    """
    errors = np.asarray(errors, dtype=float)
    mask = ~np.isnan(errors)

    if not np.any(mask):
        ax.text(0.5, 0.5, 'No prediction error data', transform=ax.transAxes,
                ha='center', va='center', alpha=0.5)
        return

    ax.plot(time[mask], errors[mask], 'purple', lw=1, alpha=0.7, label='Prediction error')
    ax.axhline(0, color='k', ls='--', lw=0.5, alpha=0.3)
    ax.set_ylabel('Prediction Error')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_cartpole_recompute(ax, time, recompute_history, dt=0.02):
    """Recompute interval over time (steps, drawn as step function)."""
    recompute = np.array(recompute_history)
    ax.plot(time[:len(recompute)], recompute, 'brown', lw=1.5,
            drawstyle='steps-post', label='Recompute interval')
    ax.set_ylabel('Replan interval')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_cartpole_horizon(ax, time, horizon_history, dt=0.02):
    """Horizon setting over time (steps, drawn as step function)."""
    horizon = np.array(horizon_history)
    ax.plot(time[:len(horizon)], horizon, 'teal', lw=1.5,
            drawstyle='steps-post', label='Horizon')
    ax.set_ylabel('Horizon')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_cartpole_adaptation_state(ax, time, mean_errors, threshold=None):
    """Mean prediction error with threshold over time.

    mean_errors: array of mean error values (None/NaN before warmup)
    threshold: array of threshold values, or None to omit
    """
    mean_errors = np.array(mean_errors, dtype=float)
    mask = ~np.isnan(mean_errors)

    if not np.any(mask):
        ax.text(0.5, 0.5, 'No mean error data', transform=ax.transAxes,
                ha='center', va='center', alpha=0.5)
        return

    ax.plot(time[mask], mean_errors[mask], 'purple', lw=1.5, label='Mean error')
    if threshold is not None:
        threshold = np.array(threshold, dtype=float)
        tmask = ~np.isnan(threshold)
        if np.any(tmask):
            ax.plot(time[tmask], threshold[tmask], 'r--', lw=1.5, alpha=0.7, label='Threshold')
    ax.set_ylabel('Mean Error')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_cartpole_running_error(ax, time, data, threshold=None):
    """Running error EMA over time with optional threshold overlay."""
    data = np.array(data, dtype=float)
    mask = ~np.isnan(data)
    if np.any(mask):
        ax.plot(time[mask], data[mask], 'purple', lw=1.5, label='Running error')
    if threshold is not None:
        threshold = np.array(threshold, dtype=float)
        tmask = ~np.isnan(threshold)
        if np.any(tmask):
            ax.plot(time[tmask], threshold[tmask], 'r--', lw=1.5, alpha=0.7, label='Threshold')
    ax.set_ylabel('Running Error')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_cartpole_state_pe(ax, time, data):
    """State prediction error over time."""
    data = np.array(data, dtype=float)
    mask = ~np.isnan(data)
    if np.any(mask):
        ax.plot(time[mask], data[mask], 'tab:blue', lw=1, alpha=0.7, label='State PE')
    ax.set_ylabel('State PE')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_cartpole_cost_pe(ax, time, data):
    """Cost prediction error over time."""
    data = np.array(data, dtype=float)
    mask = ~np.isnan(data)
    if np.any(mask):
        ax.plot(time[mask], data[mask], 'tab:orange', lw=1, alpha=0.7, label='Cost PE')
    ax.set_ylabel('Cost PE')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_cartpole_cost_surprise(ax, time, data):
    """Cost surprise over time."""
    data = np.array(data, dtype=float)
    mask = ~np.isnan(data)
    if np.any(mask):
        ax.plot(time[mask], data[mask], 'tab:red', lw=1, alpha=0.7, label='Cost surprise')
    ax.axhline(0, color='k', ls='--', lw=0.5, alpha=0.3)
    ax.set_ylabel('Cost Surprise')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_cartpole_divergence(ax, time, E_history):
    """End-of-window prediction-error magnitude E_n on a log axis."""
    data = np.array(E_history, dtype=float)
    mask = ~np.isnan(data) & (data > 0)
    plotted = False
    if np.any(mask):
        ax.semilogy(time[mask], data[mask], 'tab:purple', lw=1.0, alpha=0.8,
                    drawstyle='steps-post', label=r'$E_n = \|x - \hat{x}\|$')
        plotted = True
    ax.set_ylabel('Divergence')
    if plotted:
        ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_cartpole_noise_floor(ax, time, eta_history):
    """Online noise-floor estimate eta on a log axis."""
    data = np.array(eta_history, dtype=float)
    mask = ~np.isnan(data) & (data > 0)
    plotted = False
    if np.any(mask):
        ax.semilogy(time[mask], data[mask], 'tab:gray', lw=1.5,
                    drawstyle='steps-post', label=r'$\eta$')
        plotted = True
    ax.set_ylabel('Noise floor')
    if plotted:
        ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_cartpole_sigma(ax, time, sigma_history, log_epsilon=None):
    """Trigger signal sigma in log-domain with the log-epsilon threshold."""
    data = np.array(sigma_history, dtype=float)
    mask = ~np.isnan(data)
    plotted = False
    if np.any(mask):
        ax.plot(time[mask], data[mask], 'tab:red', lw=1.5,
                drawstyle='steps-post', label=r'$\sigma$')
        plotted = True
    if log_epsilon is not None:
        thresh = np.array(log_epsilon, dtype=float)
        tmask = ~np.isnan(thresh)
        if np.any(tmask):
            ax.plot(time[tmask], thresh[tmask], 'k--', lw=1.0, alpha=0.7,
                    label=r'$\log\epsilon$')
            plotted = True
    ax.axhline(0, color='gray', lw=0.5, alpha=0.3)
    ax.set_ylabel(r'$\sigma$ (log)')
    if plotted:
        ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)


def plot_cost_and_duration_vs_mismatch(ax_cost, ax_dur, length_factors, mean_costs,
                                       sem_costs, mean_durations, sem_durations,
                                       label='MPC',
                                       cost_label='Total Episode Cost'):
    """Dual y-axis plot of cost and episode duration vs model mismatch.

    ax_cost: left y-axis (cost)
    ax_dur: right y-axis (episode length); typically ax_cost.twinx()
    """
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    ax_cost.errorbar(
        length_factors, mean_costs, yerr=sem_costs,
        marker='o', markersize=6, capsize=4, capthick=1.5,
        color=colors[0], markerfacecolor=colors[0],
        linestyle='none', label=f'{label}: Cost'
    )
    ax_cost.set_xlabel('Pole Length: Model/Env')
    ax_cost.set_ylabel(cost_label)

    ax_dur.errorbar(
        length_factors, mean_durations, yerr=sem_durations,
        marker='s', markersize=6, capsize=4, capthick=1.5,
        color=colors[1], markerfacecolor=colors[1],
        linestyle='none', label=f'{label}: Episode Length'
    )
    ax_dur.set_ylabel('Episode Length (s)')
    ax_dur.set_ylim(0, 45)
    ax_dur.set_yticks([0, 10, 20, 30, 40])




def plot_timeseries_by_condition(ax, traces_by_label, dt, styles, ylabel='',
                                title=''):
    """Plot mean +/- SEM time series for multiple conditions.

    traces_by_label: dict {label: array of shape (n_episodes, n_steps)}
    dt: timestep for x-axis conversion to seconds
    styles: dict {label: dict(color=..., ...)}
    """
    for label, traces in traces_by_label.items():
        mean = np.mean(traces, axis=0)
        sem = np.std(traces, axis=0) / np.sqrt(traces.shape[0])
        t = np.arange(mean.shape[0]) * dt
        ax.plot(t, mean, lw=1.5, color=styles[label]['color'], label=label)
        ax.fill_between(t, mean - sem, mean + sem,
                        color=styles[label]['color'], alpha=0.2)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)


