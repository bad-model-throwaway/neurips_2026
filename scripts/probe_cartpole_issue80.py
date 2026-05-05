"""Cartpole PS-spline mismatch tuning probe.

Two phases:

  --phase=mismatch_range
      Fixed (H, R) = (80, 4) — plateau region of the PS-spline smoke
      grid. Sweep r ∈ {1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0}, 3 seeds.
      Purpose: find the cliff in r at a single "easy" cell.

  --phase=hr_grid
      2D (H, R) slice at narrower r ∈ {1.0, 2.5, 3.0, 3.5}.
      H ∈ {30, 80, 170} × R ∈ {1, 4, 8} = 9 cells, 3 seeds.
      Purpose: confirm the r-cliff found at the fixed cell holds across H
      and R, and check whether the best (H, R) shifts with r.

n_steps=500 (10 s at dt=0.02) — the legacy 1000-step default dilutes
late-failure signal for cartpole stabilisation.

Planner config mirrors SMOKE_GRIDS['cartpole'] (spline_ps, N=30, P=3
cubic, sigma=0.1, include_nominal=True, clip=True, decision=
spline_ps_argmin) so probe results predict what the eventual full grid
will see.

Outputs (suffix differs by phase):
  data/results/issue_80_probe_<phase>.csv
  data/results/issue_80_probe_<phase>_trajectories.pkl
  data/results/issue_80_probe_<phase>_summary.txt
"""

import argparse
import csv
import os
import pickle
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs import RESULTS_DIR, SEED
from simulations.simulation import run_pool


H_FIX   = 80
R_FIX   = 4
N_FIX   = 30
N_STEPS = 500   # 10 s at dt=0.02
CTRL_DT = 0.02

# Seeds mirror sweep_grid convention for cartpole: SEED + env_offset*1000
# + rep, with env_offset['cartpole']=0 — so rep k uses seed 42+k. This
# means probe rep-k shares its initial state with smoke rep-k at any (H,
# R, r) cell.
N_REPS = 3
SEEDS  = [SEED + rep for rep in range(N_REPS)]

MISMATCH_WIDE = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

# Narrow r band bracketing the cliff, crossed with a coarse (H, R) slice
# that spans short / default / long H and tight / moderate / sparse R.
MISMATCH_NARROW = [1.0, 2.5, 3.0, 3.5]
H_GRID = [30, 80, 170]
R_GRID = [1, 4, 8]

# Match SMOKE_GRIDS['cartpole'] planner config.
PROPOSAL        = 'spline_ps'
PROPOSAL_KWARGS = dict(P=3, sigma=0.1, interp='cubic',
                       include_nominal=True, clip=True)
DECISION        = 'spline_ps_argmin'

# "Balanced" episode = final cost ≤ 0.1 (pole roughly upright, cart
# roughly centred, slow). Matches the tolerance bands in the cost
# function well enough for a binary success flag.
BALANCED_FINAL_COST = 0.10


def _run_probe_episode(cfg):
    import numpy as _np
    import mujoco as _mujoco

    repo_root = cfg['repo_root']
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from agents.mpc import make_mpc
    from agents.mujoco_dynamics import MuJoCoCartPoleDynamics

    rng = _np.random.default_rng(cfg['seed'])
    _np.random.seed(cfg['seed'])

    # Match sweep_grid._run_episode_worker initial state convention.
    env  = MuJoCoCartPoleDynamics(stateless=False)
    data = _mujoco.MjData(env._mj_model)
    _mujoco.mj_resetData(env._mj_model, data)
    _mujoco.mj_forward(env._mj_model, data)
    state0    = env._state_from_data(data)
    state0[2] = rng.uniform(-0.1, 0.1)
    env.reset(state0)

    agent = make_mpc(
        'cartpole', cfg['H'], cfg['R'],
        N=cfg['N'],
        mismatch_factor=cfg['mismatch_factor'],
        proposal=cfg['proposal'],
        proposal_kwargs=cfg['proposal_kwargs'],
        decision=cfg['decision'],
    )

    t0    = time.time()
    costs = _np.empty(cfg['n_steps'], dtype=_np.float64)
    theta = _np.empty(cfg['n_steps'] + 1, dtype=_np.float32)
    x     = _np.empty(cfg['n_steps'] + 1, dtype=_np.float32)
    theta[0] = env.state[2]
    x[0]     = env.state[0]
    for t in range(cfg['n_steps']):
        action = agent.interact(env.state, env.cost)
        env.step(action)
        costs[t]   = float(env.cost)
        theta[t+1] = env.state[2]
        x[t+1]     = env.state[0]
    wallclock_s = time.time() - t0

    # Stability ≡ final 2 s (100 steps) averages below threshold.
    tail      = costs[-100:]
    tail_cost = float(_np.mean(tail))
    balanced  = tail_cost <= BALANCED_FINAL_COST

    return {
        'H': cfg['H'], 'R': cfg['R'], 'N': cfg['N'],
        'seed': cfg['seed'],
        'mismatch_factor': cfg['mismatch_factor'],
        'n_steps': cfg['n_steps'],
        'ctrl_dt': CTRL_DT,
        'mean_cost': float(_np.mean(costs)),
        'tail_cost': tail_cost,
        'balanced':  bool(balanced),
        'final_theta': float(theta[-1]),
        'max_abs_theta': float(_np.max(_np.abs(theta))),
        'final_x': float(x[-1]),
        'wallclock_s': wallclock_s,
        '_cost':  costs.astype(_np.float32),
        '_theta': theta,
        '_x':     x,
    }


def _build_jobs_mismatch_range():
    jobs = []
    for r in MISMATCH_WIDE:
        for seed in SEEDS:
            jobs.append({
                'env_name': 'cartpole',
                'H': H_FIX, 'R': R_FIX, 'N': N_FIX,
                'seed': seed,
                'n_steps': N_STEPS,
                'mismatch_factor': r,
                'proposal': PROPOSAL,
                'proposal_kwargs': PROPOSAL_KWARGS,
                'decision': DECISION,
                'repo_root': REPO_ROOT,
            })
    return jobs


def _build_jobs_hr_grid():
    jobs = []
    for r in MISMATCH_NARROW:
        for H in H_GRID:
            for R in R_GRID:
                for seed in SEEDS:
                    jobs.append({
                        'env_name': 'cartpole',
                        'H': H, 'R': R, 'N': N_FIX,
                        'seed': seed,
                        'n_steps': N_STEPS,
                        'mismatch_factor': r,
                        'proposal': PROPOSAL,
                        'proposal_kwargs': PROPOSAL_KWARGS,
                        'decision': DECISION,
                        'repo_root': REPO_ROOT,
                    })
    return jobs


_CSV_COLS = [
    'H', 'R', 'N', 'seed', 'mismatch_factor', 'n_steps', 'ctrl_dt',
    'mean_cost', 'tail_cost', 'balanced',
    'final_theta', 'max_abs_theta', 'final_x', 'wallclock_s',
]


def _write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in _CSV_COLS})


def _format_summary_mismatch_range(rows):
    lines = [
        f'Cartpole mismatch probe @ (H={H_FIX}, R={R_FIX})',
        '=' * 78,
        f'Fixed: H={H_FIX}, R={R_FIX}, N={N_FIX}, ctrl_dt={CTRL_DT}, '
        f'n_steps={N_STEPS} ({N_STEPS*CTRL_DT:.1f} s), seeds={SEEDS}.',
        f'Planner: {PROPOSAL}, P={PROPOSAL_KWARGS["P"]}, '
        f'sigma={PROPOSAL_KWARGS["sigma"]}, interp={PROPOSAL_KWARGS["interp"]}.',
        f'"balanced" = mean cost over final 2 s <= {BALANCED_FINAL_COST}.',
        '',
        f'   {"r":>4s}  {"mean_cost":>12s}  {"tail_cost":>12s}  '
        f'{"balanced":>8s}  {"max|theta|":>10s}  {"wall_s":>8s}',
        '   ' + '-' * 70,
    ]
    by_r = {}
    for row in rows:
        by_r.setdefault(row['mismatch_factor'], []).append(row)
    for r in sorted(by_r):
        arm = by_r[r]
        def m(key):
            a = np.array([x[key] for x in arm], dtype=float)
            return a.mean(), a.std(ddof=0)
        mc_m, mc_s = m('mean_cost')
        tc_m, tc_s = m('tail_cost')
        mth_m, _   = m('max_abs_theta')
        wc_m, _    = m('wallclock_s')
        bal        = sum(1 for x in arm if x['balanced'])
        lines.append(
            f'   {r:>4.1f}  {mc_m:>5.3f}±{mc_s:<4.3f}  '
            f'{tc_m:>5.3f}±{tc_s:<4.3f}  '
            f'{bal:>3d}/{len(arm):<4d}  '
            f'{mth_m:>8.2f}   '
            f'{wc_m:>6.1f}'
        )
    lines += [
        '',
        'Reading the table:',
        '  mean_cost — averaged over all 500 steps (10 s).',
        '  tail_cost — averaged over final 100 steps (2 s): cleaner',
        '               success/fail signal than the full-episode mean.',
        '  balanced  — tail_cost <= 0.10 across all seeds.',
        '  max|theta|— largest |pole angle| seen in the episode (pi/2 ≈ 1.57 = horizontal).',
    ]
    return '\n'.join(lines) + '\n'


def _format_summary_hr_grid(rows):
    """For each r, print the (H, R) grid of tail_cost + balanced rate."""
    lines = [
        'Cartpole (H, R) × narrow mismatch probe',
        '=' * 78,
        f'Fixed: N={N_FIX}, ctrl_dt={CTRL_DT}, n_steps={N_STEPS} '
        f'({N_STEPS*CTRL_DT:.1f} s), seeds={SEEDS}.',
        f'H ∈ {H_GRID}, R ∈ {R_GRID}, r ∈ {MISMATCH_NARROW}.',
        f'Planner: {PROPOSAL}, P={PROPOSAL_KWARGS["P"]}, '
        f'sigma={PROPOSAL_KWARGS["sigma"]}, interp={PROPOSAL_KWARGS["interp"]}.',
        f'"balanced" = mean cost over final 2 s <= {BALANCED_FINAL_COST}.',
        '',
    ]
    by_r = {}
    for row in rows:
        by_r.setdefault(row['mismatch_factor'], []).append(row)

    def cell(arm_cell, key_agg):
        arr = np.array([x[key_agg] for x in arm_cell], dtype=float)
        return arr.mean(), arr.std(ddof=0)

    for r in sorted(by_r):
        arm = by_r[r]
        lines.append(f'-- r = {r:.1f} ' + '-' * 60)

        lines.append(f'  tail_cost (mean ± sd over {len(SEEDS)} seeds)')
        lines.append('    H\\R  ' + '  '.join(f'R={R:<10d}' for R in R_GRID))
        for H in H_GRID:
            parts = [f'    H={H:<3d}']
            for R in R_GRID:
                cells = [x for x in arm if x['H'] == H and x['R'] == R]
                tc_m, tc_s = cell(cells, 'tail_cost')
                parts.append(f'{tc_m:>5.3f}±{tc_s:<5.3f} ')
            lines.append(''.join(parts))

        lines.append(f'  balanced rate (tail_cost <= {BALANCED_FINAL_COST})')
        lines.append('    H\\R  ' + '  '.join(f'R={R:<10d}' for R in R_GRID))
        for H in H_GRID:
            parts = [f'    H={H:<3d}']
            for R in R_GRID:
                cells = [x for x in arm if x['H'] == H and x['R'] == R]
                bal = sum(1 for c in cells if c['balanced'])
                parts.append(f'{bal}/{len(cells):<10d}')
            lines.append(''.join(parts))

        agg = {}
        for H in H_GRID:
            for R in R_GRID:
                cells = [x for x in arm if x['H'] == H and x['R'] == R]
                agg[(H, R)] = np.mean([x['tail_cost'] for x in cells])
        best_agg = min(agg, key=lambda k: agg[k])
        lines.append(
            f'  best (H, R) by seed-avg tail_cost: H={best_agg[0]}, '
            f'R={best_agg[1]}, tail_cost={agg[best_agg]:.3f}'
        )
        lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase',
                    choices=['mismatch_range', 'hr_grid'],
                    default='mismatch_range')
    ap.add_argument('--dry-run', action='store_true',
                    help='Run one job only to time a cell.')
    ap.add_argument('--n-workers', type=int, default=2,
                    help='Pool size (default 2).')
    args = ap.parse_args()

    out_dir = os.path.join(REPO_ROOT, RESULTS_DIR.lstrip('./'))
    os.makedirs(out_dir, exist_ok=True)

    if args.phase == 'mismatch_range':
        jobs = _build_jobs_mismatch_range()
        fmt  = _format_summary_mismatch_range
    else:
        jobs = _build_jobs_hr_grid()
        fmt  = _format_summary_hr_grid

    if args.dry_run:
        jobs = jobs[:1]

    print(f'Probe phase={args.phase}: {len(jobs)} jobs on {args.n_workers} workers')
    t0 = time.time()
    rows = run_pool(_run_probe_episode, jobs,
                    n_processes=args.n_workers, verbose=1)
    print(f'Total wallclock: {time.time() - t0:.1f}s')

    rows.sort(key=lambda r: (r['mismatch_factor'], r['H'], r['R'], r['seed']))

    csv_path  = os.path.join(out_dir, f'issue_80_probe_{args.phase}.csv')
    traj_path = os.path.join(out_dir, f'issue_80_probe_{args.phase}_trajectories.pkl')
    sum_path  = os.path.join(out_dir, f'issue_80_probe_{args.phase}_summary.txt')

    _write_csv(rows, csv_path)
    print(f'Wrote {csv_path}')

    traj = {
        'phase': args.phase,
        'N': N_FIX, 'n_steps': N_STEPS, 'ctrl_dt': CTRL_DT,
        'proposal': PROPOSAL,
        'proposal_kwargs': dict(PROPOSAL_KWARGS),
        'decision': DECISION,
        'mismatch_factors': sorted({r['mismatch_factor'] for r in rows}),
        'seeds': SEEDS,
        'runs': [
            {k: r[k] for k in ('mismatch_factor', 'H', 'R', 'seed',
                               '_cost', '_theta', '_x')}
            for r in rows
        ],
    }
    with open(traj_path, 'wb') as f:
        pickle.dump(traj, f)
    print(f'Wrote {traj_path}')

    text = fmt(rows)
    with open(sum_path, 'w') as f:
        f.write(text)
    print(f'Wrote {sum_path}')
    print()
    print(text)


if __name__ == '__main__':
    main()
