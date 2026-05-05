import math
import numpy as np
import matplotlib.pyplot as plt
from agents.dynamics import PointMass2D

from configs import SCALE_TEXT

plt.ion()
plt.rcParams.update({
'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'font.size': 10 * SCALE_TEXT,
})

def plot_panel_force_field(ax, force_field, curve_func=None, extent=2.0,
                           n_grid=20, n_potential=50, curve_scale=1.5):
    """Draw force field vectors over potential heatmap on existing axis.

    force_field: GPForceField instance
    curve_func: parametric curve function (s) -> (x, y), or None to skip
    """

    # Compute potential on fine grid for heatmap
    pot_grid = np.linspace(-extent, extent, n_potential)
    pot_xx, pot_yy = np.meshgrid(pot_grid, pot_grid)
    pot_positions = np.stack([pot_xx.ravel(), pot_yy.ravel()], axis=1)
    potential = force_field.potential(pot_positions).reshape(n_potential, n_potential)

    # Compute forces on coarse grid for quiver
    quiv_grid = np.linspace(-extent, extent, n_grid)
    quiv_xx, quiv_yy = np.meshgrid(quiv_grid, quiv_grid)
    positions = np.stack([quiv_xx.ravel(), quiv_yy.ravel()], axis=1)
    forces = force_field.force_vectorized(positions)
    fx = forces[:, 0].reshape(n_grid, n_grid)
    fy = forces[:, 1].reshape(n_grid, n_grid)

    # Plot potential heatmap and force vectors
    ax.imshow(potential, extent=[-extent, extent, -extent, extent],
              origin='lower', cmap='viridis', alpha=0.2)
    ax.quiver(quiv_xx, quiv_yy, fx * 0.75, fy * 0.75, alpha=0.7)

    # Plot curve if provided
    if curve_func is not None:
        s_vals = np.linspace(0, 1, 200)
        curve_x, curve_y = curve_func(s_vals, scale=curve_scale)
        ax.plot(curve_x, curve_y, 'r-', linewidth=2, label='curve')

        # Add orientation marker at s=0.3
        s_marker = 0.3
        ds = 0.01
        mx, my = curve_func(s_marker, scale=curve_scale)
        mx_next, my_next = curve_func(s_marker + ds, scale=curve_scale)
        dx, dy = mx_next - mx, my_next - my
        norm = np.sqrt(dx**2 + dy**2)
        dx, dy = dx / norm, dy / norm
        ax.annotate('', xy=(mx + dx * 0.01, my + dy * 0.01), xytext=(mx, my),
                    arrowprops=dict(arrowstyle='-|>', color='r', lw=2,
                                    mutation_scale=20))
        ax.legend()

    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(alpha=0.3)


def plot_force_field(force_field, curve_func=None, extent=2.0, n_grid=20,
                     n_potential=50, curve_scale=1.5, figsize=(6, 5)):
    """Plot force field vectors over potential heatmap (standalone figure)."""
    fig, ax = plt.subplots(figsize=figsize)
    plot_panel_force_field(ax, force_field, curve_func, extent, n_grid, n_potential, curve_scale)
    ax.set_title('Force field')
    plt.tight_layout()
    return fig, ax


def plot_tracking_summary(states, dt, force_field, curve_func, curve_scale,
                          title='', history=None):
    """Combined tracking plot: error panels (left) and trajectory on force field (right).

    states: array of shape (n_steps, 5) with columns [x, y, vx, vy, s]
    history: optional History object. When adaptation keys are present,
        extra panels are added below the core error panels.
    """
    from visualization.plots_cartpole import (
        plot_cartpole_recompute, plot_cartpole_horizon,
        plot_cartpole_running_error, plot_cartpole_prediction_error,
        plot_cartpole_state_pe, plot_cartpole_cost_pe,
        plot_cartpole_cost_surprise, plot_cartpole_adaptation_state,
    )

    positions = states[:, :2]
    s_vals = states[:, 4]
    time = np.arange(len(states)) * dt
    curve_dist = PointMass2D.cost_curve_distance(positions, curve_func, curve_scale)
    tx, ty = curve_func(s_vals, scale=curve_scale)
    target_dist = PointMass2D.cost_tracking_distance(positions, np.stack([tx, ty], axis=1))

    # Detect adaptation panels from history
    adapt_candidates = [
        'recompute_interval', 'horizon',
        'running_error', 'error', 'state_pe', 'cost_pe', 'cost_surprise',
        'prediction_error', 'mean_error',
    ]

    # Look up threshold series if recorded
    threshold = None
    if history is not None and 'threshold' in history.keys:
        threshold = history.get_item_history('threshold')

    panel_funcs = {
        'recompute_interval': lambda ax, t, d: plot_cartpole_recompute(ax, t, d, dt),
        'horizon':            lambda ax, t, d: plot_cartpole_horizon(ax, t, d, dt),
        'running_error':      lambda ax, t, d: plot_cartpole_running_error(ax, t, d, threshold),
        'error':              plot_cartpole_prediction_error,
        'state_pe':           plot_cartpole_state_pe,
        'cost_pe':            plot_cartpole_cost_pe,
        'cost_surprise':      plot_cartpole_cost_surprise,
        'prediction_error':   plot_cartpole_prediction_error,
        'mean_error':         lambda ax, t, d: plot_cartpole_adaptation_state(ax, t, d, threshold),
    }
    adapt_panels = []
    if history is not None:
        adapt_panels = [k for k in adapt_candidates if k in history.keys]

    # Layout
    #   Top zone (rows 0-1): core error panels left, heatmap right (spans 2 rows)
    #   Adaptation zone (rows 2+): panels split evenly across both columns
    n_core = 2
    n_adapt = len(adapt_panels)
    n_adapt_left = math.ceil(n_adapt / 2)
    n_adapt_right = n_adapt - n_adapt_left
    n_adapt_rows = n_adapt_left
    n_rows = n_core + n_adapt_rows

    fig = plt.figure(figsize=(12, 3 * n_rows))
    gs = fig.add_gridspec(n_rows, 2, width_ratios=[1, 1])

    # Top-left: instantaneous error
    ax_inst = fig.add_subplot(gs[0, 0])
    ax_inst.plot(time, curve_dist, label='Distance from track')
    ax_inst.plot(time, target_dist, label='Distance from target')
    ax_inst.set_ylabel('Distance [m]')
    ax_inst.legend()
    ax_inst.grid(alpha=0.3)

    # Top-left: integrated error
    ax_int = fig.add_subplot(gs[1, 0], sharex=ax_inst)
    ax_int.plot(time, np.cumsum(curve_dist) * dt, label='Integrated track error')
    ax_int.plot(time, np.cumsum(target_dist) * dt, label='Integrated target error')
    ax_int.set_ylabel('Integrated distance [m*s]')
    ax_int.legend()
    ax_int.grid(alpha=0.3)

    # Top-right: force field with trajectory (spans 2 rows)
    ax_traj = fig.add_subplot(gs[0:2, 1])
    plot_panel_force_field(ax_traj, force_field, curve_func, curve_scale=curve_scale)
    add_plot_dynamics(ax_traj, [states.tolist()])
    if title:
        ax_traj.set_title(title)

    # Adaptation zone: split panels across left and right columns below core
    adapt_left = adapt_panels[:n_adapt_left]
    adapt_right = adapt_panels[n_adapt_left:]

    left_axes = [ax_inst, ax_int]
    right_axes = []

    for i, key in enumerate(adapt_left):
        ax = fig.add_subplot(gs[n_core + i, 0], sharex=ax_inst)
        data = history.get_item_history(key)
        panel_funcs[key](ax, time, data)
        left_axes.append(ax)

    for i, key in enumerate(adapt_right):
        ax = fig.add_subplot(gs[n_core + i, 1], sharex=ax_inst)
        data = history.get_item_history(key)
        panel_funcs[key](ax, time, data)
        right_axes.append(ax)

    # Hide unused right-column adaptation cells
    for i in range(n_adapt_right, n_adapt_rows):
        ax_empty = fig.add_subplot(gs[n_core + i, 1])
        ax_empty.set_visible(False)

    # X-axis labels on bottom visible panel of each column
    left_axes[-1].set_xlabel('Time [s]')
    if right_axes:
        right_axes[-1].set_xlabel('Time [s]')

    fig.tight_layout()
    return fig


def plot_pointmass_error_vs_mismatch(ax, summary):
    """Plot integrated tracking errors vs mass mismatch factor with error bars.

    summary: list of dicts with keys mass_factor, mean_curve_int,
        sem_curve_int, mean_target_int, sem_target_int.
    """
    mass_factors = [d['mass_factor'] for d in summary]
    mean_curve = [d['mean_curve_int'] for d in summary]
    sem_curve = [d['sem_curve_int'] for d in summary]
    mean_target = [d['mean_target_int'] for d in summary]
    sem_target = [d['sem_target_int'] for d in summary]

    ax.errorbar(mass_factors, mean_curve, yerr=sem_curve,
                marker='o', markersize=6, capsize=4, capthick=1.5,
                linestyle='none', label='Track error')
    ax.errorbar(mass_factors, mean_target, yerr=sem_target,
                marker='s', markersize=6, capsize=4, capthick=1.5,
                linestyle='none', label='Target error')
    ax.axvline(1.0, color='k', linestyle='--', alpha=0.5, label='True mass')
    ax.set_xlabel('Mass: Model / Environment')
    ax.set_ylabel('Integrated Distance (m·s)')
    ax.legend()
    ax.grid(alpha=0.3)


def add_plot_dynamics(ax, histories, cmap='gist_heat', cmap_range=(0.0, 0.6)):
    """Overlay trajectory traces on existing plot with temporal coloring.

    ax: matplotlib axis to plot on
    histories: list of state histories, each history is list of states [x, y, vx, vy, s]
    cmap: colormap for temporal progression (same for all trajectories)
    cmap_range: (min, max) range of colormap to use (0-1), for truncating
    """
    colormap = plt.colormaps[cmap]
    cmin, cmax = cmap_range

    for history in histories:
        states = np.array(history)
        x, y = states[:, 0], states[:, 1]
        n_points = len(x)

        # Plot segments with color corresponding to time
        for j in range(n_points - 1):
            t_norm = j / (n_points - 1)
            c_val = cmin + t_norm * (cmax - cmin)
            ax.plot(x[j:j+2], y[j:j+2], '-', color=colormap(c_val),
                    linewidth=1.5, alpha=0.8)

        # Mark starting and ending points
        ax.plot(x[0], y[0], 'o', color=colormap(cmin), markersize=6)
        ax.plot(x[-1], y[-1], 'o', color=colormap(cmax), markersize=6)

