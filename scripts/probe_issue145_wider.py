"""Wider probe — quadratic cost, DEFAULT 10x10 grid x 5 mismatch x 7 reps.

Config:
  Cost:     Q = diag(1.0, 0.1, 3.0, 1.0), R_scalar = 0.1  (strict quadratic)
  Planner:  project default — spline_ps, N=30, P=3, sigma=0.3, cubic
  Grid:     H = DEFAULT_GRIDS['cartpole'] (10 pts),
            R = [1..10],
            r (mismatch) = [1.0, 1.15, 1.3, 1.5, 1.8],
            reps = 7, n_steps = 400
  Parallel: 12 workers, spawn context.

Output:
  data/results/grid_cartpole_quadratic_probe.pkl
    schema = run_grid_sweep-compatible
    {env, H_values, R_values, mismatch_factors, mean_cost, std_cost, dt, n_reps}

Estimated wall time: ~35-40 min on 12 local cores.
"""

import os
import sys
import time
import pickle
import numpy as np
import multiprocessing as mp

from tqdm import tqdm

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from configs import RESULTS_DIR, ENV_DT


H_VALUES = [30, 36, 44, 53, 64, 78, 94, 114, 138, 170]
R_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
MISMATCH = [1.0, 1.15, 1.3, 1.5, 1.8]
REPS = 7
N_STEPS = 400
SEED_BASE = 0

N_WORKERS = 12
OUT_PATH = os.path.join(RESULTS_DIR, 'grid_cartpole_quadratic_probe.pkl')

COST_KWARGS = {
    'cost_type': 'quadratic',
    'Q_diag': (1.0, 0.1, 3.0, 1.0),
    'R_scalar': 0.1,
}


def _run_episode(cfg):
    import sys as _sys
    repo_root = cfg['repo_root']
    if repo_root not in _sys.path:
        _sys.path.insert(0, repo_root)

    import numpy as _np
    import mujoco as _mujoco
    from agents.mujoco_dynamics import MuJoCoCartPoleDynamics
    from agents import mpc as _mpc

    H = cfg['H']
    R = cfg['R']
    factor = cfg['factor']
    seed = cfg['seed']

    # R > H is undefined for MPC (need R actions from a plan of length H).
    if R > H:
        return {**cfg, 'mean_cost': float('nan')}

    # Inject cost-matched env_kwargs into the cartpole proposal config so
    # make_mpc builds a planning model with matching cost.
    pcfg = _mpc.PROPOSAL_CONFIGS['cartpole']
    pcfg['env_kwargs'] = dict(cfg['cost_kwargs'])

    rng = _np.random.default_rng(seed)
    _np.random.seed(seed)

    env = MuJoCoCartPoleDynamics(stateless=False, **cfg['cost_kwargs'])
    data = _mujoco.MjData(env._mj_model)
    _mujoco.mj_resetData(env._mj_model, data)
    _mujoco.mj_forward(env._mj_model, data)
    state0 = env._state_from_data(data)
    state0[2] = rng.uniform(-0.1, 0.1)
    env.reset(state0)

    agent = _mpc.make_mpc('cartpole', H, R, mismatch_factor=factor)

    costs = []
    for _ in range(cfg['n_steps']):
        action = agent.interact(env.state, env.cost)
        env.step(action)
        costs.append(float(env.cost))

    return {**cfg, 'mean_cost': float(_np.mean(costs))}


def main():
    t0 = time.time()

    cfgs = []
    for fi, r in enumerate(MISMATCH):
        for hi, H in enumerate(H_VALUES):
            for ri, R in enumerate(R_VALUES):
                for rep in range(REPS):
                    seed = SEED_BASE + (
                        fi * 10_000 + hi * 1_000 + ri * 100 + rep
                    )
                    cfgs.append(dict(
                        repo_root=_repo_root,
                        H=H, R=R, factor=float(r),
                        rep=rep, seed=seed,
                        n_steps=N_STEPS,
                        cost_kwargs=dict(COST_KWARGS),
                        _fi=fi, _hi=hi, _ri=ri,
                    ))

    print(f"Wider probe")
    print(f"  Cost:   Q_diag={COST_KWARGS['Q_diag']}, R_scalar={COST_KWARGS['R_scalar']}")
    print(f"  H ({len(H_VALUES)}): {H_VALUES}")
    print(f"  R ({len(R_VALUES)}): {R_VALUES}")
    print(f"  r ({len(MISMATCH)}): {MISMATCH}")
    print(f"  reps={REPS}, n_steps={N_STEPS}")
    print(f"  Total episodes: {len(cfgs)}")
    print(f"  Workers: {N_WORKERS}\n")

    ctx = mp.get_context('spawn')
    results = []
    with ctx.Pool(N_WORKERS) as pool:
        for res in tqdm(pool.imap_unordered(_run_episode, cfgs),
                        total=len(cfgs), ncols=80):
            results.append(res)

    n_r, n_H, n_R = len(MISMATCH), len(H_VALUES), len(R_VALUES)
    per_cell = [[[[] for _ in range(n_R)] for _ in range(n_H)] for _ in range(n_r)]
    for res in results:
        per_cell[res['_fi']][res['_hi']][res['_ri']].append(res['mean_cost'])
    mean_cost = np.full((n_r, n_H, n_R), np.nan, dtype=float)
    std_cost  = np.full((n_r, n_H, n_R), np.nan, dtype=float)
    for fi in range(n_r):
        for hi in range(n_H):
            for ri in range(n_R):
                vals = [v for v in per_cell[fi][hi][ri] if not np.isnan(v)]
                if len(vals) > 0:
                    mean_cost[fi, hi, ri] = float(np.mean(vals))
                    std_cost[fi, hi, ri]  = float(np.std(vals))

    out = {
        'env': 'cartpole',
        'H_values': np.asarray(H_VALUES),
        'R_values': np.asarray(R_VALUES),
        'mismatch_factors': list(MISMATCH),
        'mean_cost': mean_cost,
        'std_cost': std_cost,
        'dt': float(ENV_DT.get('cartpole')),
        'n_reps': REPS,
        'cost_kwargs': COST_KWARGS,
        'n_steps': N_STEPS,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(out, f)
    print(f"\nSaved {OUT_PATH}")

    print("\n--- Summary ---")
    for fi, r in enumerate(MISMATCH):
        M = mean_cost[fi]
        valid = ~np.isnan(M)
        if valid.any():
            flat_idx = int(np.nanargmin(M))
            hi, ri = np.unravel_index(flat_idx, M.shape)
            print(f"r={r:>4.2f}  argmin = (H={H_VALUES[hi]:>3}, R={R_VALUES[ri]:>2}) "
                  f"cost={M[hi, ri]:.3f}")

    print("\nR=1 cost vs H (shows inverted-U structure):")
    header = f"{'H':>4}  " + "  ".join(f"r={r:<4.2f}" for r in MISMATCH)
    print(header)
    ri_1 = R_VALUES.index(1)
    for hi, H in enumerate(H_VALUES):
        row = f"{H:>4}  " + "  ".join(
            f"{mean_cost[fi, hi, ri_1]:>7.3f}" for fi in range(n_r)
        )
        print(row)

    print(f"\nTotal wall time: {(time.time() - t0)/60:.1f} min")


if __name__ == '__main__':
    main()
