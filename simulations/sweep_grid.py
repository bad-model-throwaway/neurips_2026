"""Unified (H, R, mismatch, rep) grid sweep for CartPole, Walker, and HumanoidStand.

Result schema returned by run_grid_sweep:
    {
        'env':              str,
        'H_values':         np.ndarray,   # shape (n_H,)
        'R_values':         np.ndarray,   # shape (n_R,)
        'mismatch_factors': list[float],  # length n_mismatch
        'mean_cost':        np.ndarray,   # shape (n_mismatch, n_H, n_R)
        'std_cost':         np.ndarray,   # shape (n_mismatch, n_H, n_R)
        'all_costs':        np.ndarray,   # shape (n_mismatch, n_H, n_R, n_reps)
        'failure_sec':      np.ndarray,   # shape (n_mismatch, n_H, n_R, n_reps), float32
                                          # — per-seed seconds-to-first-physical-failure
                                          #   (full duration when env never fell)
        'cost_traj':        np.ndarray,   # shape (n_m, n_H, n_R, n_reps, n_steps), float32
        'terminal_states':  np.ndarray,   # shape (n_m, n_H, n_R, n_reps, state_dim)
        'last_states':      np.ndarray,   # shape (n_m, n_H, n_R, n_reps, N_TERMINAL_STATES, state_dim)
        'n_terminal_states': int,
        'dt':               float,
        'n_reps':           int,
    }
"""

import os
import sys
import numpy as np
import multiprocessing as mp
import pickle
import argparse

from tqdm import tqdm

from configs import N_WORKERS as _CFG_N_WORKERS, SEED, RESULTS_DIR, ENV_DT, FAILURE_ANGLE
from agents.mujoco_dynamics import MISMATCH_FACTORS


N_TERMINAL_STATES = 20

import math as _math
_FAIL_ANGLE_RAD = _math.radians(FAILURE_ANGLE)
_FAIL_TORSO_Z   = 0.7
_FAIL_HEAD_Z    = 0.8
_HEAD_Z_IDX     = 55       # humanoid state index for head_z
_WALKER_Z_IDX   = 18       # walker state index for torso_z

_FAILURE_PREDICATES = {
    'cartpole':                lambda s: abs(s[2])           > _FAIL_ANGLE_RAD,
    'cartpole_quadratic':      lambda s: abs(s[2])           > _FAIL_ANGLE_RAD,
    'walker':                  lambda s: s[_WALKER_Z_IDX]    < _FAIL_TORSO_Z,
    'humanoid_balance':        lambda s: s[_HEAD_Z_IDX]      < _FAIL_HEAD_Z,
    'humanoid_stand':          lambda s: s[_HEAD_Z_IDX]      < _FAIL_HEAD_Z,
    'humanoid_stand_gravity':  lambda s: s[_HEAD_Z_IDX]      < _FAIL_HEAD_Z,
}


DEFAULT_GRIDS = {
    'cartpole_quadratic': dict(
        H=[30, 36, 44, 53, 64, 78, 94, 114, 138, 170],
        R=list(range(1, 11)),
        mismatch=MISMATCH_FACTORS['CartPoleQuadratic'], reps=100,
        n_steps=400,
    ),
    'cartpole': dict(
        H=[30, 36, 44, 53, 64, 78, 94, 114, 138, 170],
        R=list(range(1, 11)),
        mismatch=MISMATCH_FACTORS['CartPole'], reps=100,
        n_steps=400,
    ),
    'walker': dict(
        H=[30, 37, 45, 55, 67, 82, 100, 122, 150, 183],
        R=list(range(1, 11)),
        mismatch=MISMATCH_FACTORS['Walker'], reps=100,
        n_steps=800,
    ),
    'humanoid_stand': dict(
        H=[15, 20, 25, 30, 40, 50, 65, 85, 110, 135],
        R=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        mismatch=MISMATCH_FACTORS['HumanoidStand'], reps=30,
        n_steps=400,
    ),
    'humanoid_balance': dict(
        H=[15, 20, 25, 30, 40, 50, 65, 85, 110, 135],
        R=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        mismatch=MISMATCH_FACTORS['HumanoidBalance'], reps=100,
        n_steps=300,
    ),
    'humanoid_stand_gravity': dict(
        H=[15, 20, 25, 30, 40, 50, 65, 85, 110, 135],
        R=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        mismatch=MISMATCH_FACTORS['HumanoidStandGravity'], reps=30,
        n_steps=400,
    ),
}

SMOKE_GRIDS = {
    'cartpole_quadratic': dict(
        H=[30, 53, 78, 114, 170],
        R=list(range(1, 11)),
        mismatch=MISMATCH_FACTORS['CartPoleQuadratic'], reps=3,
        n_steps=400,
    ),
    'cartpole': dict(
        H=[30, 53, 78, 114, 170],
        R=list(range(1, 11)),
        mismatch=MISMATCH_FACTORS['CartPole'], reps=3,
        n_steps=400,
        proposal='spline_ps',
        N=30,
        proposal_kwargs=dict(
            P=3, sigma=0.1, interp='cubic',
            include_nominal=True, clip=True,
        ),
        decision='spline_ps_argmin',
    ),
    'walker': dict(
        H=[30, 45, 82, 122, 183],
        R=list(range(1, 11)),
        mismatch=MISMATCH_FACTORS['Walker'], reps=3,
        n_steps=800,
    ),
    'humanoid_stand': dict(
        H=[15, 30, 50, 85, 135],
        R=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        mismatch=MISMATCH_FACTORS['HumanoidStand'], reps=3,
        n_steps=400,
    ),
    'humanoid_balance': dict(
        H=[15, 30, 50, 85, 135],
        R=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        mismatch=MISMATCH_FACTORS['HumanoidBalance'], reps=3,
        n_steps=300,
    ),
    'humanoid_stand_gravity': dict(
        H=[15, 30, 50, 85, 135],
        R=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        mismatch=MISMATCH_FACTORS['HumanoidStandGravity'], reps=3,
        n_steps=400,
    ),
}

_ENV_OFFSETS = {'cartpole': 0, 'walker': 1, 'humanoid_stand': 2,
                'humanoid_balance': 3, 'humanoid_stand_gravity': 4,
                'cartpole_quadratic': 5}


def _run_episode_worker(cfg):
    """Run one episode and return mean cost."""
    import sys as _sys
    import numpy as _np
    import mujoco as _mujoco

    repo_root = cfg['repo_root']
    if repo_root not in _sys.path:
        _sys.path.insert(0, repo_root)

    from agents.mujoco_dynamics import (
        MuJoCoCartPoleDynamics, WalkerDynamics, HumanoidStandDynamics,
    )
    from agents.mpc import make_mpc, PROPOSAL_CONFIGS

    env_name = cfg['env_name']
    H        = cfg['H']
    R        = cfg['R']
    factor   = cfg['factor']
    n_steps  = cfg['n_steps']
    seed     = cfg['seed']
    proposal = cfg.get('proposal', None)
    N        = cfg.get('N', None)
    prop_kw  = cfg.get('proposal_kwargs', None)
    decision = cfg.get('decision', None)

    # MPC needs to execute R actions from a plan of length H; R > H is
    # undefined. Skip those cells — upstream aggregation leaves them NaN.
    if R > H:
        return {
            'env_name':       env_name,
            'factor':         factor,
            'H':              H,
            'R':              R,
            'seed':           seed,
            'rep':            cfg['rep'],
            'mean_cost':      float('nan'),
            'failure_sec':    float('nan'),
            'cost_traj':      None,
            'terminal_state': None,
            'last_states':    None,
            '_fi':            cfg['_fi'],
            '_hi':            cfg['_hi'],
            '_ri':            cfg['_ri'],
        }

    rng = _np.random.default_rng(seed)
    _np.random.seed(seed)

    if env_name in ('cartpole', 'cartpole_quadratic'):
        env_kwargs = PROPOSAL_CONFIGS.get(env_name, {}).get('env_kwargs', {}) or {}
        env    = MuJoCoCartPoleDynamics(stateless=False, **env_kwargs)
        data   = _mujoco.MjData(env._mj_model)
        _mujoco.mj_resetData(env._mj_model, data)
        _mujoco.mj_forward(env._mj_model, data)
        state0    = env._state_from_data(data)
        state0[2] = rng.uniform(-0.1, 0.1)
        env.reset(state0)

    elif env_name == 'walker':
        env_kwargs = PROPOSAL_CONFIGS.get('walker', {}).get('env_kwargs', {}) or {}
        env    = WalkerDynamics(stateless=False, **env_kwargs)
        state0 = env.get_default_initial_state()
        env.reset(state0)

    elif env_name == 'humanoid_stand':
        env    = HumanoidStandDynamics(stateless=False)
        state0 = env.get_default_initial_state()
        env.reset(state0)

    elif env_name == 'humanoid_balance':
        env_kwargs = PROPOSAL_CONFIGS.get('humanoid_balance', {}).get('env_kwargs', {}) or {}
        env    = HumanoidStandDynamics(stateless=False, **env_kwargs)
        state0 = env.get_default_initial_state()
        env.reset(state0)

    elif env_name == 'humanoid_stand_gravity':
        env    = HumanoidStandDynamics(stateless=False)
        state0 = env.get_default_initial_state()
        env.reset(state0)

    else:
        raise ValueError(f"Unknown env: {env_name}")

    mpc_kwargs = dict(
        N=N,
        mismatch_factor=factor,
        proposal=proposal,
        proposal_kwargs=prop_kw,
        decision=decision,
    )
    if decision is not None:
        mpc_kwargs['decision'] = decision
    agent = make_mpc(env_name, H, R, **mpc_kwargs)

    costs = []
    state_dim   = int(_np.asarray(env.state).size)
    last_states = _np.full((N_TERMINAL_STATES, state_dim), _np.nan, dtype=float)
    fail_predicate = _FAILURE_PREDICATES.get(env_name)
    first_fail_step = None
    dt = ENV_DT[env_name]
    for step_idx in range(n_steps):
        action = agent.interact(env.state, env.cost)
        env.step(action)
        costs.append(float(env.cost))
        s = env.state
        if first_fail_step is None and fail_predicate is not None and fail_predicate(s):
            first_fail_step = step_idx
        if step_idx >= n_steps - N_TERMINAL_STATES:
            last_states[step_idx - (n_steps - N_TERMINAL_STATES)] = _np.asarray(s)

    # The +1 aligns with sweep_*_adaptive.py timing: the adaptive records
    # env.state BEFORE env.step, while this loop samples AFTER env.step. Without
    # the +1 the grid's failure_sec would lag the adaptive's by one timestep.
    duration_sec = n_steps * dt
    if first_fail_step is not None:
        failure_sec = float((first_fail_step + 1) * dt)
    else:
        failure_sec = float(duration_sec)

    return {
        'env_name':       env_name,
        'factor':         factor,
        'H':              H,
        'R':              R,
        'rep':            cfg['rep'],
        'mean_cost':      float(_np.mean(costs)),
        'failure_sec':    failure_sec,
        'cost_traj':      _np.asarray(costs, dtype=_np.float32),
        'terminal_state': _np.asarray(env.state, dtype=float).copy(),
        'last_states':    last_states,
        '_fi':            cfg['_fi'],
        '_hi':            cfg['_hi'],
        '_ri':            cfg['_ri'],
    }


def run_grid_sweep(
    env_name,
    H_values,
    R_values,
    mismatch_factors,
    n_reps,
    n_steps=1000,
    n_workers=None,
    proposal=None,
    N=None,
    proposal_kwargs=None,
    decision=None,
):
    """Run a full (H, R, mismatch, rep) grid and return aggregated results."""
    H_arr = np.array(H_values)
    R_arr = np.array(R_values)
    n_mismatch = len(mismatch_factors)
    n_H        = len(H_arr)
    n_R        = len(R_arr)

    n_workers = n_workers if n_workers is not None else _CFG_N_WORKERS

    env_offset = _ENV_OFFSETS[env_name]
    repo_root  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    jobs = []
    for fi, factor in enumerate(mismatch_factors):
        for hi, H in enumerate(H_arr):
            for ri, R in enumerate(R_arr):
                for rep in range(n_reps):
                    # Common random numbers: seed depends on (env, rep) only,
                    # so each rep evaluates the same initial state and noise
                    # stream across every (factor, H, R) cell. Variance
                    # reduction for across-cell comparisons at the cost of
                    # correlated cell estimates.
                    seed = SEED + env_offset * 1000 + rep
                    jobs.append({
                        'repo_root':       repo_root,
                        'env_name':        env_name,
                        'H':               int(H),
                        'R':               int(R),
                        'factor':          float(factor),
                        'n_steps':         n_steps,
                        'seed':            seed,
                        'rep':             rep,
                        'proposal':        proposal,
                        'N':               N,
                        'proposal_kwargs': proposal_kwargs,
                        'decision':        decision,
                        '_fi': fi, '_hi': hi, '_ri': ri,
                    })

    all_costs        = np.full((n_mismatch, n_H, n_R, n_reps), np.nan)
    all_failure_sec  = np.full((n_mismatch, n_H, n_R, n_reps), np.nan, dtype=np.float32)
    all_cost_traj    = None
    all_terminal     = None
    all_last_states  = None

    ctx = mp.get_context('spawn')
    # maxtasksperchild=20 recycles workers before MuJoCo/numpy memory bloat
    # accumulates — without it long pools (~16k episodes) hit OOM.
    with ctx.Pool(processes=n_workers, maxtasksperchild=20) as pool:
        for result in tqdm(
            pool.imap_unordered(_run_episode_worker, jobs),
            total=len(jobs),
            desc=f'{env_name} sweep',
        ):
            fi, hi, ri, rep = result['_fi'], result['_hi'], result['_ri'], result['rep']
            all_costs[fi, hi, ri, rep]       = result['mean_cost']
            all_failure_sec[fi, hi, ri, rep] = result.get('failure_sec', np.nan)

            if result.get('cost_traj') is not None:
                if all_cost_traj is None:
                    n_steps_local = result['cost_traj'].shape[0]
                    state_dim     = result['terminal_state'].shape[0]
                    all_cost_traj = np.full(
                        (n_mismatch, n_H, n_R, n_reps, n_steps_local),
                        np.nan, dtype=np.float32,
                    )
                    all_terminal = np.full(
                        (n_mismatch, n_H, n_R, n_reps, state_dim), np.nan,
                    )
                    all_last_states = np.full(
                        (n_mismatch, n_H, n_R, n_reps, N_TERMINAL_STATES, state_dim),
                        np.nan,
                    )
                all_cost_traj[fi, hi, ri, rep]   = result['cost_traj']
                all_terminal[fi, hi, ri, rep]    = result['terminal_state']
                all_last_states[fi, hi, ri, rep] = result['last_states']

    out = {
        'env':              env_name,
        'H_values':         H_arr,
        'R_values':         R_arr,
        'mismatch_factors': list(mismatch_factors),
        'mean_cost':        np.nanmean(all_costs, axis=-1),
        'std_cost':         np.nanstd(all_costs, axis=-1),
        'all_costs':        all_costs,
        'failure_sec':      all_failure_sec,
        'dt':               ENV_DT[env_name],
        'n_reps':           n_reps,
    }
    if all_cost_traj is not None:
        out['cost_traj']       = all_cost_traj
        out['terminal_states'] = all_terminal
        out['last_states']     = all_last_states
        out['n_terminal_states'] = N_TERMINAL_STATES
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run H×R×mismatch grid sweep for one environment.'
    )
    parser.add_argument('--env',
                        choices=['cartpole', 'walker', 'humanoid_stand',
                                 'humanoid_balance', 'humanoid_stand_gravity'],
                        required=True)
    parser.add_argument('--smoke', action='store_true',
                        help='Use SMOKE_GRIDS instead of DEFAULT_GRIDS.')
    parser.add_argument('--n-workers', type=int, default=None)
    args = parser.parse_args()

    grids = SMOKE_GRIDS if args.smoke else DEFAULT_GRIDS
    g     = grids[args.env]

    print(f'Starting {args.env} sweep '
          f'(H={len(g["H"])} × R={len(g["R"])} × '
          f'mismatch={len(g["mismatch"])} × reps={g["reps"]})')

    result = run_grid_sweep(
        args.env,
        g['H'], g['R'], g['mismatch'],
        n_reps=g['reps'],
        n_steps=g.get('n_steps', 1000),
        n_workers=args.n_workers,
        proposal=g.get('proposal'),
        N=g.get('N'),
        proposal_kwargs=g.get('proposal_kwargs'),
        decision=g.get('decision'),
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f'grid_{args.env}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(result, f)

    mc = result['mean_cost']
    print(f'Done. mean_cost shape={mc.shape}  '
          f'range=[{np.nanmin(mc):.4f}, {np.nanmax(mc):.4f}]')
    print(f'Saved to {out_path}')

    from configs import FIGURES_DIR
    from visualization.heatmaps import build_figure_2_panel
    panel_ids = {'cartpole': 'A', 'walker': 'B', 'humanoid_stand': 'C'}
    env_labels = {'cartpole': 'CartPole', 'walker': 'Walker',
                  'humanoid_stand': 'Humanoid Stand'}
    if args.env in panel_ids:
        panel_id = panel_ids[args.env]
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig_path = os.path.join(FIGURES_DIR, f'fig2_{panel_id}.svg')
        build_figure_2_panel(result, env_labels[args.env], result['dt'], fig_path)
        print(f'Saved {fig_path}')
