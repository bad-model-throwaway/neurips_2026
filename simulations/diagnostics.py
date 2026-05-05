"""Diagnostic probes for the MPC paper pipeline.

Functions
---------
run_control_quality_probe(env_name) -> dict
run_timing_model_probe(env_name) -> dict
run_mismatch_sensitivity_probe(env_name) -> dict
"""

import os
import sys
import time
import multiprocessing
from collections import defaultdict

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import configs
from agents.mujoco_dynamics import MISMATCH_FACTORS

_ENV_CANONICAL = {
    'cartpole':         'CartPole',
    'walker':           'Walker',
    'humanoid_balance': 'HumanoidBalance',
}

_ENV_OFFSET = {
    'cartpole':         0,
    'walker':           1,
    'humanoid_balance': 2,
}

_ENV_ACTION_DIM = {
    'CartPole':        1,
    'Walker':          6,
    'HumanoidBalance': 21,
}

_PROBE_CONFIGS = {
    'CartPole': {'H': 50, 'R': 1, 'n_steps': 1000},
}

_N_REPS_QUALITY  = 3
_N_REPS_MISMATCH = 2


def _run_quality_episode(cfg):
    """Worker: single matched-parameter quality episode."""
    import sys as _sys
    import numpy as _np
    import mujoco as _mujoco

    repo_root = cfg['repo_root']
    if repo_root not in _sys.path:
        _sys.path.insert(0, repo_root)

    from agents.mpc import make_mpc
    from agents.mujoco_dynamics import MuJoCoCartPoleDynamics

    env_name      = cfg['env_name']
    H, R, n_steps = cfg['H'], cfg['R'], cfg['n_steps']
    rng           = _np.random.RandomState(cfg['seed'])

    env   = MuJoCoCartPoleDynamics(stateless=False)
    data  = _mujoco.MjData(env._mj_model)
    _mujoco.mj_resetData(env._mj_model, data)
    _mujoco.mj_forward(env._mj_model, data)
    state0    = env._state_from_data(data)
    state0[2] = rng.uniform(-0.1, 0.1)
    env.reset(state0)

    agent = make_mpc(env_name, H=H, R=R)
    agent.model.reset(state0)

    costs  = []
    states = []
    for _ in range(n_steps):
        action = agent.interact(env.state, env.cost)
        env.step(action)
        costs.append(float(env.cost))
        states.append(env.state.copy())

    mean_cost    = float(_np.mean(costs))
    traj_summary = float(_np.max(_np.abs([s[2] for s in states])))

    return {
        'env_name':     env_name,
        'rep':          cfg['rep'],
        'mean_cost':    mean_cost,
        'traj_summary': traj_summary,
    }


def _run_mismatch_episode(cfg):
    """Worker: single mismatch-sensitivity episode."""
    import sys as _sys
    import numpy as _np
    import mujoco as _mujoco

    repo_root = cfg['repo_root']
    if repo_root not in _sys.path:
        _sys.path.insert(0, repo_root)

    from agents.mpc import make_mpc
    from agents.mujoco_dynamics import MuJoCoCartPoleDynamics

    env_name      = cfg['env_name']
    factor        = cfg['factor']
    H, R, n_steps = cfg['H'], cfg['R'], cfg['n_steps']
    rng           = _np.random.RandomState(cfg['seed'])

    env   = MuJoCoCartPoleDynamics(stateless=False)
    data  = _mujoco.MjData(env._mj_model)
    _mujoco.mj_resetData(env._mj_model, data)
    _mujoco.mj_forward(env._mj_model, data)
    state0    = env._state_from_data(data)
    state0[2] = rng.uniform(-0.1, 0.1)
    env.reset(state0)

    agent = make_mpc(env_name, H=H, R=R, mismatch_factor=factor)
    agent.model.reset(state0)

    costs = []
    for _ in range(n_steps):
        action = agent.interact(env.state, env.cost)
        env.step(action)
        costs.append(float(env.cost))

    return {
        'env_name':  env_name,
        'factor':    factor,
        'rep':       cfg['rep'],
        'mean_cost': float(_np.mean(costs)),
    }


def run_control_quality_probe(env_name):
    """Run matched-parameter quality probe for one environment.

    Returns per-env summary stats (baseline_cost, baseline_traj_metric, H, R, N).
    For CartPole, baseline_traj_metric is max |pole angle| over episode (rad).
    """
    from agents.mpc import PROPOSAL_CONFIGS

    env_can    = _ENV_CANONICAL[env_name]
    pcfg       = _PROBE_CONFIGS[env_can]
    env_offset = _ENV_OFFSET[env_name]
    N          = PROPOSAL_CONFIGS[env_name]['N']

    jobs = []
    for rep in range(_N_REPS_QUALITY):
        seed = configs.SEED + env_offset * 1000 + rep
        jobs.append({
            'repo_root': _REPO_ROOT,
            'env_name':  env_name,
            'rep':       rep,
            'seed':      seed,
            **pcfg,
        })

    n_workers = configs.N_WORKERS
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(_run_quality_episode, jobs)

    return {
        'env':                  env_name,
        'baseline_cost':        float(np.mean([r['mean_cost']    for r in results])),
        'baseline_traj_metric': float(np.mean([r['traj_summary'] for r in results])),
        'n_steps':              pcfg['n_steps'],
        'H':                    pcfg['H'],
        'R':                    pcfg['R'],
        'N':                    N,
    }


def _build_timing_model(env_name):
    """Construct a stateless dynamics model and return (model, initial_state).

    Mirrors the env-class dispatch used by `make_mpc` so the timing fit reflects
    the same MuJoCo step pipeline used by the actual planner.
    """
    import mujoco
    from agents.mujoco_dynamics import (
        MuJoCoCartPoleDynamics, WalkerDynamics, HumanoidStandDynamics,
    )

    if env_name == 'cartpole':
        model = MuJoCoCartPoleDynamics(stateless=True)
    elif env_name == 'walker':
        model = WalkerDynamics(stateless=True)
    elif env_name == 'humanoid_balance':
        model = HumanoidStandDynamics(stateless=True, mode='balance')
    else:
        raise ValueError(f"timing probe not configured for env_name={env_name!r}")

    data = mujoco.MjData(model._mj_model)
    mujoco.mj_resetData(model._mj_model, data)
    mujoco.mj_forward(model._mj_model, data)
    state = model._state_from_data(data)
    return model, state


def run_timing_model_probe(env_name):
    """Fit time_ms = alpha * N*H + intercept on a single-threaded (N, H) sweep.

    The (N, H) grid brackets the operating points used in the Figure 2/3
    sweeps (cartpole H=53, walker H=82, humanoid H=30; N=30 from
    PROPOSAL_CONFIGS) so the fit interpolates rather than extrapolates.
    """
    env_can    = _ENV_CANONICAL[env_name]
    action_dim = _ENV_ACTION_DIM[env_can]

    N_SAMPLES_GRID = [10, 30, 50, 100, 200, 500]
    HORIZON_GRID   = [10, 30, 50, 75, 100]
    N_TRIALS       = 3

    model, state = _build_timing_model(env_name)

    nh_vals = []
    t_vals  = []

    for N in N_SAMPLES_GRID:
        for H in HORIZON_GRID:
            proposals = np.random.uniform(-1.0, 1.0, size=(N, action_dim, H))
            _ = model.query(state, proposals)  # warm-up JIT/caches
            times = []
            for _ in range(N_TRIALS):
                t0 = time.perf_counter()
                _ = model.query(state, proposals)
                times.append((time.perf_counter() - t0) * 1000.0)
            nh_vals.append(N * H)
            t_vals.append(float(np.median(times)))

    nh_arr = np.array(nh_vals, dtype=float)
    t_arr  = np.array(t_vals,  dtype=float)

    A = np.column_stack([nh_arr, np.ones_like(nh_arr)])
    coeffs, _, _, _ = np.linalg.lstsq(A, t_arr, rcond=None)
    alpha, intercept = float(coeffs[0]), float(coeffs[1])

    t_pred = alpha * nh_arr + intercept
    ss_res = float(np.sum((t_arr - t_pred) ** 2))
    ss_tot = float(np.sum((t_arr - t_arr.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    return {
        'env':         env_name,
        'alpha':       alpha,
        'intercept':   intercept,
        'r_squared':   r2,
        'nh_grid':     nh_vals,
        't_ms_grid':   t_vals,
        'n_samples':   N_SAMPLES_GRID,
        'horizon':     HORIZON_GRID,
        'n_trials':    N_TRIALS,
    }


def run_mismatch_sensitivity_probe(env_name):
    """Sweep MISMATCH_FACTORS and report cost sensitivity.

    factors[0] == 1.0 (exact match) is the baseline used for cost_ratio.
    """
    env_can    = _ENV_CANONICAL[env_name]
    pcfg       = _PROBE_CONFIGS[env_can]
    factors    = MISMATCH_FACTORS[env_can]
    env_offset = _ENV_OFFSET[env_name]

    jobs = []
    for fi, factor in enumerate(factors):
        for rep in range(_N_REPS_MISMATCH):
            # Independent per-factor seeds (no common random numbers across
            # factors): diagnostics are read as per-factor point estimates,
            # not across-factor contrasts.
            seed = configs.SEED + env_offset * 1000 + fi * 100 + rep
            jobs.append({
                'repo_root': _REPO_ROOT,
                'env_name':  env_name,
                'factor':    factor,
                'rep':       rep,
                'seed':      seed,
                'H':         pcfg['H'],
                'R':         pcfg['R'],
                'n_steps':   500,
            })

    n_workers = configs.N_WORKERS
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(_run_mismatch_episode, jobs)

    raw = defaultdict(list)
    for r in results:
        raw[r['factor']].append(r['mean_cost'])

    mean_costs  = [float(np.mean(raw[f])) for f in factors]
    baseline    = mean_costs[0]
    cost_ratios = [c / baseline if baseline > 1e-12 else float('inf')
                   for c in mean_costs]

    return {
        'env':        env_name,
        'factors':    list(factors),
        'mean_cost':  mean_costs,
        'cost_ratio': cost_ratios,
    }


def fit_and_save_timing_models(envs=('cartpole', 'walker', 'humanoid_balance'),
                               out_path=None):
    """Run the timing probe across envs and save a single dict to pickle.

    The pickle is keyed by env_name and consumed by Figure 3 compute-aware
    aggregation: per-replan ms = alpha * N * H + intercept, total compute per
    episode = n_replans * per-replan ms / 1000.
    """
    import pickle
    if out_path is None:
        out_path = os.path.join(configs.RESULTS_DIR, 'timing_models.pkl')

    models = {}
    for env in envs:
        r = run_timing_model_probe(env)
        models[env] = r
        print(f"  {env:18s}: alpha={r['alpha']:.4e} ms  "
              f"intercept={r['intercept']:.2f} ms  "
              f"R2={r['r_squared']:.4f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(models, f)
    print(f"\nSaved timing models -> {out_path}")
    return models


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', default='timing',
                    choices=['timing', 'quality', 'mismatch', 'all'])
    ap.add_argument('--envs', nargs='+',
                    default=['cartpole', 'walker', 'humanoid_balance'])
    args = ap.parse_args()

    if args.probe in ('timing', 'all'):
        print('=== Timing Model ===')
        fit_and_save_timing_models(envs=tuple(args.envs))

    if args.probe in ('quality', 'all'):
        print('\n=== Control Quality ===')
        for env in args.envs:
            r = run_control_quality_probe(env)
            print(f"  {env:18s}: baseline_cost={r['baseline_cost']:.4f}  "
                  f"traj_metric={r['baseline_traj_metric']:.4f}  "
                  f"H={r['H']}  R={r['R']}  N={r['N']}")

    if args.probe in ('mismatch', 'all'):
        print('\n=== Mismatch Sensitivity ===')
        for env in args.envs:
            r = run_mismatch_sensitivity_probe(env)
            print(f"  {env}:")
            for f, mc, cr in zip(r['factors'], r['mean_cost'], r['cost_ratio']):
                print(f"    factor={f:.2f}  mean_cost={mc:.4f}  ratio={cr:.3f}")
