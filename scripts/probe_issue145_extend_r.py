"""r-axis extension probe — add r ∈ {2.0, 2.5, 3.0} at 10 reps.

Extends `grid_cartpole_quadratic_probe.pkl` along the mismatch axis. Keeps
the existing r ∈ {1.0, 1.15, 1.3, 1.5, 1.8} rows (7 reps each) untouched.

New cells: 3 r × 10×10 H/R × 10 reps = 3000 episodes. ETA ~15 min on 12 cores.

Output: overwrites data/results/grid_cartpole_quadratic_probe.pkl with the
combined 8-level r grid. Also records per-mismatch rep counts since the high-r
rows have 10 reps vs 7 for the low-r rows.
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
NEW_R   = [2.0, 2.5, 3.0]
REPS    = 10
N_STEPS = 400
N_WORKERS = 12

# Seed scheme must not collide with the original probe, which used
# fi * 10_000 + hi * 1_000 + ri * 100 + rep (fi ∈ {0..4}, rep < 7).
# Offset the new levels past fi=4 so seed spaces stay disjoint.
SEED_BASE = 500_000

COST_KWARGS = {
    'cost_type': 'quadratic',
    'Q_diag': (1.0, 0.1, 3.0, 1.0),
    'R_scalar': 0.1,
}

OUT_PATH = os.path.join(RESULTS_DIR, 'grid_cartpole_quadratic_probe.pkl')


def _run_episode(cfg):
    import sys as _sys
    repo_root = cfg['repo_root']
    if repo_root not in _sys.path:
        _sys.path.insert(0, repo_root)

    import numpy as _np
    import mujoco as _mujoco
    from agents.mujoco_dynamics import MuJoCoCartPoleDynamics
    from agents import mpc as _mpc

    H, R, factor, seed = cfg['H'], cfg['R'], cfg['factor'], cfg['seed']
    if R > H:
        return {**cfg, 'mean_cost': float('nan')}

    _mpc.PROPOSAL_CONFIGS['cartpole']['env_kwargs'] = dict(cfg['cost_kwargs'])

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
    for fi, r in enumerate(NEW_R):
        for hi, H in enumerate(H_VALUES):
            for ri, R in enumerate(R_VALUES):
                for rep in range(REPS):
                    seed = SEED_BASE + fi * 10_000 + hi * 1_000 + ri * 100 + rep
                    cfgs.append(dict(
                        repo_root=_repo_root,
                        H=H, R=R, factor=float(r),
                        rep=rep, seed=seed,
                        n_steps=N_STEPS,
                        cost_kwargs=dict(COST_KWARGS),
                        _fi=fi, _hi=hi, _ri=ri,
                    ))

    print(f"r-extension probe")
    print(f"  New r ({len(NEW_R)}): {NEW_R}")
    print(f"  reps={REPS}, n_steps={N_STEPS}")
    print(f"  Total new episodes: {len(cfgs)}")
    print(f"  Workers: {N_WORKERS}\n")

    ctx = mp.get_context('spawn')
    results = []
    with ctx.Pool(N_WORKERS) as pool:
        for res in tqdm(pool.imap_unordered(_run_episode, cfgs),
                        total=len(cfgs), ncols=80):
            results.append(res)

    n_r_new, n_H, n_R = len(NEW_R), len(H_VALUES), len(R_VALUES)
    per = [[[[] for _ in range(n_R)] for _ in range(n_H)] for _ in range(n_r_new)]
    for res in results:
        per[res['_fi']][res['_hi']][res['_ri']].append(res['mean_cost'])
    mean_new = np.full((n_r_new, n_H, n_R), np.nan, dtype=float)
    std_new  = np.full((n_r_new, n_H, n_R), np.nan, dtype=float)
    for fi in range(n_r_new):
        for hi in range(n_H):
            for ri in range(n_R):
                vals = [v for v in per[fi][hi][ri] if not np.isnan(v)]
                if vals:
                    mean_new[fi, hi, ri] = float(np.mean(vals))
                    std_new[fi, hi, ri]  = float(np.std(vals))

    with open(OUT_PATH, 'rb') as f:
        prev = pickle.load(f)
    assert list(prev['H_values']) == H_VALUES, "H axis mismatch"
    assert list(prev['R_values']) == R_VALUES, "R axis mismatch"

    mismatch_combined = list(prev['mismatch_factors']) + list(NEW_R)
    mean_combined = np.concatenate([prev['mean_cost'], mean_new], axis=0)
    std_combined  = np.concatenate([prev['std_cost'],  std_new],  axis=0)

    n_reps_per_r = [int(prev['n_reps'])] * len(prev['mismatch_factors']) + [REPS] * n_r_new

    out = {
        'env': 'cartpole',
        'H_values': np.asarray(H_VALUES),
        'R_values': np.asarray(R_VALUES),
        'mismatch_factors': mismatch_combined,
        'mean_cost': mean_combined,
        'std_cost':  std_combined,
        'dt': float(ENV_DT.get('cartpole')),
        'n_reps': max(prev['n_reps'], REPS),
        'n_reps_per_r': n_reps_per_r,
        'cost_kwargs': COST_KWARGS,
        'n_steps': N_STEPS,
    }
    with open(OUT_PATH, 'wb') as f:
        pickle.dump(out, f)
    print(f"\nSaved {OUT_PATH} (now {len(mismatch_combined)} mismatch levels)")

    print("\n--- Combined summary ---")
    print(f"{'r':<6} {'n_reps':<7} argmin (H,R)          cost")
    for fi, r in enumerate(mismatch_combined):
        M = mean_combined[fi]
        if (~np.isnan(M)).any():
            flat = int(np.nanargmin(M))
            hi, ri = np.unravel_index(flat, M.shape)
            print(f"{r:<6.2f} {n_reps_per_r[fi]:<7} "
                  f"(H={H_VALUES[hi]:>3}, R={R_VALUES[ri]:>2})   "
                  f"cost={M[hi, ri]:.3f}")

    print("\nR=1 cost vs H across all r:")
    header = f"{'H':>4}  " + "  ".join(f"r={r:<4.2f}" for r in mismatch_combined)
    print(header)
    ri_1 = R_VALUES.index(1)
    for hi, H in enumerate(H_VALUES):
        row = f"{H:>4}  " + "  ".join(
            f"{mean_combined[fi, hi, ri_1]:>7.3f}" for fi in range(len(mismatch_combined))
        )
        print(row)

    print(f"\nTotal wall time (extension only): {(time.time() - t0)/60:.1f} min")


if __name__ == '__main__':
    main()
