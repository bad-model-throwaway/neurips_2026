import os

import numpy as np
import mujoco

from agents.dynamics import PointMass2D
from agents.utils import figure_eight


def _pointmass_tracking_converged(history, curve_scale=1.5):
    """Assert that tracking error converged over final 100 steps."""
    states = history.get_item_history('state')
    final_dist = PointMass2D.cost_curve_distance(states[-100:, :2], figure_eight, curve_scale)
    mean_dist = np.mean(final_dist)
    assert mean_dist < 1.0, f"Tracking diverged: mean curve distance = {mean_dist:.3f}"


def _cartpole_pole_stayed_upright(history):
    """Assert that the pole angle stayed within bounds throughout the simulation."""
    states = history.get_item_history('state')
    max_theta_deg = np.max(np.degrees(np.abs(states[:, 2])))
    from configs import FAILURE_ANGLE
    assert max_theta_deg < FAILURE_ANGLE, f"Pole exceeded angle bound: max |theta| = {max_theta_deg:.1f} deg"

def _cartpole_cart_stayed_bounded(history):
    """Assert that the cart position stayed within bounds throughout the simulation."""
    from configs import POSITION_BOUND
    states = history.get_item_history('state')
    max_x = np.max(np.abs(states[:, 0]))
    assert max_x < POSITION_BOUND, f"Cart exceeded position bound: max |x| = {max_x:.1f} m"



def _walker_stayed_upright(history, min_torso_z=0.8):
    """Assert the walker's torso z stayed above min_torso_z for the whole trajectory.

    Torso z lives in `state[..., 18]` — the first of the three MJPC sensor slots
    appended by `WalkerDynamics._state_from_data`. The probe's "fell" cutoff is
    0.7 m; 0.8 is a conservative margin below the matched-dynamics mean of 1.23.
    """
    states = history.get_item_history('state')
    min_z = float(np.min(states[:, 18]))
    assert min_z > min_torso_z, f"Walker fell: min torso_z = {min_z:.3f} m (< {min_torso_z})"


def _walker_forward_progress(history, min_mean_vx=0.7, tail_frac=0.5):
    """Assert mean COM x-velocity over the final `tail_frac` of the trajectory exceeds `min_mean_vx`.

    `state[..., 20]` is the MJPC `com_vx` sensor slot. Probe matched-dynamics
    whole-window mean was 1.1–1.4; a tail-half floor of 0.7 catches a planner
    that stalls out while leaving ~3σ of headroom.
    """
    states = history.get_item_history('state')
    n = len(states)
    tail = states[int((1.0 - tail_frac) * n):]
    mean_vx = float(np.mean(tail[:, 20]))
    assert mean_vx > min_mean_vx, (
        f"Walker stalled: tail mean com_vx = {mean_vx:.3f} m/s (< {min_mean_vx})"
    )


# Manuscript Figure 2 physical-success criteria, mirrored from
# visualization/heatmaps.py:PHYSICAL_SUCCESS. Each clause asserts that the
# *mean over the last `n_terminal_states` steps* satisfies a per-env rule.
# Used by adapter tests so test thresholds match the figure's success
# definition, not stricter ad hoc bounds.
_N_TERMINAL_STATES = 20

_FIG2_PHYSICAL_SUCCESS = {
    'cartpole': [
        dict(idx=2,  op='abs_le', threshold=0.1,           label='|theta|'),
    ],
    'walker': [
        dict(idx=18, op='ge',     threshold=0.8 * 1.30,    label='torso_z'),
        dict(idx=20, op='ge',     threshold=0.5,           label='com_vx'),
    ],
    'humanoid_balance': [
        dict(idx=55, op='ge',     threshold=0.8 * 1.472,   label='head_z'),
    ],
}


def _assert_fig2_physical_success(history, env_name, n_terminal=_N_TERMINAL_STATES):
    """Assert all Fig-2 physical-success clauses hold for the trajectory tail.

    Note: Fig 2 criteria measure *quality of solution* (stayed near nominal pose),
    not catastrophic failure. For test pass/fail, use `_assert_fall_safety`.
    """
    if env_name not in _FIG2_PHYSICAL_SUCCESS:
        raise KeyError(f"Fig 2 success criterion not registered for {env_name!r}")
    states = np.asarray(history.get_item_history('state'))
    if len(states) < n_terminal:
        raise ValueError(f"trajectory shorter than n_terminal={n_terminal}")
    tail = states[-n_terminal:]
    for spec in _FIG2_PHYSICAL_SUCCESS[env_name]:
        avg = float(np.mean(tail[:, spec['idx']]))
        op, thr, label = spec['op'], spec['threshold'], spec['label']
        if op == 'abs_le':
            ok = abs(avg) <= thr
            cmp = f"|{label}| = {abs(avg):.3f} (must be <= {thr})"
        elif op == 'ge':
            ok = avg >= thr
            cmp = f"{label} = {avg:.3f} (must be >= {thr})"
        elif op == 'le':
            ok = avg <= thr
            cmp = f"{label} = {avg:.3f} (must be <= {thr})"
        else:
            raise ValueError(f"unknown op {op!r}")
        assert ok, f"[{env_name}] Fig-2 success failed: {cmp}"


# Operational fall thresholds — match the production sweeps' "fell" cutoffs:
# simulations/sweep_grid.py uses _FAIL_TORSO_Z=0.7, _FAIL_HEAD_Z=0.8;
# cartpole uses configs.FAILURE_ANGLE (degrees) for the pole-fell cutoff.
# Tests use these for pass/fail because they measure "controller didn't lose
# stability" — the right gate for a regression test, distinct from Fig 2's
# "stayed near nominal pose" which measures solution quality.
_FALL_THRESHOLDS = {
    'cartpole':         dict(idx=2,  op='abs_le_deg',  threshold=None,  label='|theta|'),
    'walker':           dict(idx=18, op='ge',          threshold=0.7,   label='torso_z'),
    'humanoid_balance': dict(idx=55, op='ge',          threshold=0.8,   label='head_z'),
}


def _assert_fall_safety(history, env_name, n_terminal=_N_TERMINAL_STATES):
    """Assert tail-window mean of the safety state is on the safe side of the
    operational fall threshold (the same cutoff sweep_grid.py uses to flag
    a seed as "fell"). Pass/fail gate for tests.
    """
    if env_name not in _FALL_THRESHOLDS:
        raise KeyError(f"fall threshold not registered for {env_name!r}")
    spec = _FALL_THRESHOLDS[env_name]
    states = np.asarray(history.get_item_history('state'))
    tail = states[-n_terminal:]
    avg = float(np.mean(tail[:, spec['idx']]))
    op, label = spec['op'], spec['label']
    if op == 'abs_le_deg':
        from configs import FAILURE_ANGLE
        avg_deg = float(np.degrees(abs(avg)))
        ok = avg_deg < FAILURE_ANGLE
        msg = f"{label} = {avg_deg:.2f} deg (must be < {FAILURE_ANGLE})"
    elif op == 'ge':
        ok = avg >= spec['threshold']
        msg = f"{label} = {avg:.3f} (must be >= {spec['threshold']})"
    else:
        raise ValueError(f"unknown op {op!r}")
    assert ok, f"[{env_name}] FALL: {msg}"


def _log_fig2_metric(history, env_name, n_terminal=_N_TERMINAL_STATES):
    """Print the Fig-2 metric value(s) for informational logging — does not
    affect pass/fail. Useful for tracking solution quality alongside the
    fall-safety gate.
    """
    if env_name not in _FIG2_PHYSICAL_SUCCESS:
        return
    states = np.asarray(history.get_item_history('state'))
    if len(states) < n_terminal:
        return
    tail = states[-n_terminal:]
    parts = []
    for spec in _FIG2_PHYSICAL_SUCCESS[env_name]:
        avg = float(np.mean(tail[:, spec['idx']]))
        thr = spec['threshold']
        op = spec['op']
        if op == 'abs_le':
            v = abs(avg)
            ok = '✓' if v <= thr else '✗'
            parts.append(f"|{spec['label']}|={v:.3f}≤{thr}{ok}")
        elif op == 'ge':
            ok = '✓' if avg >= thr else '✗'
            parts.append(f"{spec['label']}={avg:.3f}≥{thr:.3f}{ok}")
        elif op == 'le':
            ok = '✓' if avg <= thr else '✗'
            parts.append(f"{spec['label']}={avg:.3f}≤{thr}{ok}")
    print(f"[{env_name}] Fig-2: " + "  ".join(parts))


_ENV_DT = {'cartpole': 0.02, 'walker': 0.01, 'humanoid_balance': 0.015}


def _save_test_artifacts(history, env_name, test_name):
    """Save a per-test diagnostic plot and an MP4 rollout.

    Plot lands in TESTS_PLOTS_DIR, video lands in TESTS_VIDEOS_DIR. Cartpole
    plots use `plot_cartpole_history`; walker/humanoid use a generic 4-panel
    (safety-state, cost, recompute, horizon). Video uses `record_rollout_video`
    against a stateless dynamics instance.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from configs import TESTS_PLOTS_DIR, TESTS_VIDEOS_DIR
    os.makedirs(TESTS_PLOTS_DIR, exist_ok=True)
    os.makedirs(TESTS_VIDEOS_DIR, exist_ok=True)
    dt = _ENV_DT[env_name]

    if env_name == 'cartpole':
        from visualization.plots_cartpole import plot_cartpole_history
        fig = plot_cartpole_history(history, dt=dt, title=test_name)
    else:
        fig = _plot_history_generic(history, env_name, dt=dt, title=test_name)
    fig.savefig(os.path.join(TESTS_PLOTS_DIR, f'{test_name}.svg'))
    plt.close(fig)

    dyn = _build_stateless_dynamics(env_name)
    states = np.asarray(history.get_item_history('state'))
    camera = 'side' if env_name == 'walker' else -1
    path = os.path.join(TESTS_VIDEOS_DIR, f'{test_name}.mp4')
    record_rollout_video(dyn, states, path, camera=camera)


def _build_stateless_dynamics(env_name):
    """Build a stateless dynamics instance suitable for rendering frames."""
    from agents.mujoco_dynamics import (
        MuJoCoCartPoleDynamics, WalkerDynamics, HumanoidStandDynamics,
    )
    if env_name == 'cartpole':
        return MuJoCoCartPoleDynamics(stateless=True)
    if env_name == 'walker':
        return WalkerDynamics(stateless=True, speed_goal=1.5)
    if env_name == 'humanoid_balance':
        return HumanoidStandDynamics(stateless=True, mode='balance')
    raise ValueError(f"unknown env {env_name!r}")


_ADAPT_PANEL_CANDIDATES = [
    'recompute_interval', 'horizon',
    'running_error', 'error', 'state_pe', 'cost_pe', 'cost_surprise',
    'prediction_error', 'mean_error',
    'E_curr', 'eta', 'sigma',
]


def _build_adapt_panel_funcs(history, dt):
    """Return (available_keys, panel_func_dict) mirroring plot_cartpole_history.

    The visualization.plots_cartpole helpers take (ax, time, data) and are
    not actually cartpole-specific — they plot scalar time series. Reusing
    them keeps walker/humanoid diagnostics visually identical to cartpole.
    """
    from visualization.plots_cartpole import (
        plot_cartpole_recompute, plot_cartpole_horizon,
        plot_cartpole_divergence, plot_cartpole_noise_floor,
        plot_cartpole_sigma, plot_cartpole_running_error,
        plot_cartpole_state_pe, plot_cartpole_cost_pe,
        plot_cartpole_cost_surprise, plot_cartpole_prediction_error,
        plot_cartpole_adaptation_state,
    )

    available = [k for k in _ADAPT_PANEL_CANDIDATES if k in history.keys]
    threshold = (history.get_item_history('threshold')
                 if 'threshold' in history.keys else None)
    log_eps = (history.get_item_history('log_epsilon')
               if 'log_epsilon' in history.keys else None)

    funcs = {
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
    return available, funcs


def _plot_history_generic(history, env_name, dt, title=None):
    """Diagnostic plot for walker/humanoid mirroring plot_cartpole_history.

    Left column: safety state (torso_z / head_z) and cost. Right column:
    adaptation diagnostics (E_curr / eta / sigma / recompute / horizon /
    prediction-error variants) when present in history. Falls back to
    single-column when no adaptation keys are recorded.
    """
    import matplotlib.pyplot as plt
    states = np.asarray(history.get_item_history('state'))
    costs = np.asarray(history.get_item_history('cost'))
    t = np.arange(len(states)) * dt
    spec = _FALL_THRESHOLDS[env_name]
    safety = states[:, spec['idx']]
    fig2_thr = _FIG2_PHYSICAL_SUCCESS[env_name][0]['threshold']

    adapt_panels, panel_funcs = _build_adapt_panel_funcs(history, dt)
    n_core = 2  # safety + cost

    if adapt_panels:
        n_right = len(adapt_panels)
        n_rows = max(n_core, n_right)
        fig, axes = plt.subplots(n_rows, 2, figsize=(16, 2 * n_rows), sharex=True)
        left_axes = axes[:, 0]
        right_axes = axes[:, 1]
    else:
        n_rows = n_core
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, 2 * n_rows), sharex=True)
        left_axes = axes
        right_axes = None

    left_axes[0].plot(t, safety, color='C0', lw=0.8)
    left_axes[0].axhline(spec['threshold'], color='r', ls='--', lw=0.8, label='fall')
    left_axes[0].axhline(fig2_thr, color='gray', ls=':', lw=0.8, label='Fig 2')
    left_axes[0].set_ylabel(spec['label'])
    left_axes[0].legend(fontsize=7, loc='lower right')

    left_axes[1].plot(t, costs, color='C1', lw=0.8)
    left_axes[1].set_ylabel('cost')

    if right_axes is not None:
        for i in range(n_core, n_rows):
            left_axes[i].set_visible(False)
        for i, key in enumerate(adapt_panels):
            data = history.get_item_history(key)
            panel_funcs[key](right_axes[i], t, data)
        for i in range(len(adapt_panels), n_rows):
            right_axes[i].set_visible(False)
        left_axes[n_core - 1].set_xlabel('time (s)')
        right_axes[len(adapt_panels) - 1].set_xlabel('time (s)')
    else:
        left_axes[-1].set_xlabel('time (s)')

    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def assert_valid_rollout_video(path, min_size_bytes=10_000, expected_frames=None):
    """Verify a rollout video exists, is non-trivial, and contains frames.

    A bare `os.path.exists(path)` check passes on a 0-byte write failure. This
    helper also checks file size and that imageio can decode at least one
    frame (optionally matching `expected_frames` exactly).
    """
    import imageio.v2 as imageio  # lazy (no hard dep at import time)

    assert os.path.exists(path), f"Video not written: {path}"
    size = os.path.getsize(path)
    assert size >= min_size_bytes, (
        f"Video suspiciously small: {path} is {size} bytes (< {min_size_bytes})"
    )

    reader = imageio.get_reader(path)
    try:
        n_frames = reader.count_frames()
    finally:
        reader.close()
    assert n_frames > 0, f"Video has no frames: {path}"
    if expected_frames is not None:
        assert n_frames == expected_frames, (
            f"Video frame count mismatch: {path} has {n_frames}, "
            f"expected {expected_frames}"
        )


def record_rollout_video(dynamics, states, path, fps=30, size=(480, 480), camera=-1):
    """Render each state via `mujoco.Renderer` and encode to `path`.

    `dynamics` must expose `_mj_model` and `_set_data_state(data, state)`.
    `states` is an `[T, state_dim]` array (e.g. `history.get_item_history('state')`).
    Parent dirs of `path` are created if missing. `size` is `(height, width)`.
    `camera` is forwarded to `renderer.update_scene` — pass an XML camera name
    (e.g. `'side'`) to use a tracking camera so a moving body stays in frame.
    """
    import imageio.v2 as imageio  # lazy so the default suite has no hard dep

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    height, width = size
    data = mujoco.MjData(dynamics._mj_model)
    renderer = mujoco.Renderer(dynamics._mj_model, height, width)
    try:
        frames = []
        for state in states:
            dynamics._set_data_state(data, state)
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())
    finally:
        renderer.close()
    imageio.mimsave(path, frames, fps=fps)
