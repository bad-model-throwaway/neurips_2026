"""Walker mismatch range probe (running arm).

One-axis torso-mass mismatch sweep at a fixed (H, R) near the smoke-grid
U-shape minimum, used to pick walker-specific MISMATCH_FACTORS['Walker'].

Fixed cell: H=60, R=1, reps=3.  Running arm only (speed_goal=1.5).
Mismatch:  r ∈ {1.0, 1.3, 1.6, 1.8, 2.0, 2.3, 2.6}  → 7 levels.
Jobs: 7 × 3 = 21 episodes.

n_steps=1000 (10 s at ctrl_dt=0.01) — walker default; stand arm and
foot-friction axis are out of scope.

Planner config mirrors PROPOSAL_CONFIGS['walker'] (spline_ps, N=30, P=3
cubic, sigma=0.5, include_nominal=True, clip=True, decision=
spline_ps_argmin) so probe results predict what the eventual full grid
will see.

Outputs:
  data/results/issue_88_probe_mismatch_range.csv
  data/results/issue_88_probe_mismatch_range_trajectories.pkl
  data/results/issue_88_probe_mismatch_range_summary.txt
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


H_FIX      = 60
R_FIX      = 1
N_FIX      = 30
N_STEPS    = 1000    # 10 s at ctrl_dt=0.01
CTRL_DT    = 0.01
SPEED_GOAL = 1.5     # running arm
KIND       = 'torso_mass'

# Seeds mirror sweep_grid convention for walker: SEED + env_offset*1000
# + rep, with env_offset['walker']=3 — so rep k uses seed 42 + 3000 + k.
# This means probe rep-k shares its initial state / noise stream with
# sweep_grid rep-k at any (H, R, r) cell.
N_REPS = 3
SEEDS  = [SEED + 3 * 1000 + rep for rep in range(N_REPS)]

MISMATCH_LEVELS = [1.0, 1.3, 1.6, 1.8, 2.0, 2.3, 2.6]

# Match PROPOSAL_CONFIGS['walker'].
PROPOSAL        = 'spline_ps'
PROPOSAL_KWARGS = dict(P=3, sigma=0.5, interp='cubic',
                       include_nominal=True, clip=True)
DECISION        = 'spline_ps_argmin'

# "Fell" heuristic: torso dropped below 0.7 m at any point (same as
# probe_walker_mismatch.py). Binary per-episode; summary reports fall
# rate across seeds per r level.
FELL_Z_THRESHOLD = 0.7


def _run_probe_episode(cfg):
    import numpy as _np

    repo_root = cfg['repo_root']
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from agents.mpc import make_mpc
    from agents.mujoco_dynamics import WalkerDynamics

    _np.random.seed(cfg['seed'])

    env = WalkerDynamics(stateless=False, speed_goal=cfg['speed_goal'])
    env.reset(env.get_default_initial_state())

    # make_mpc applies mismatch to the planning model only
    # (WalkerDynamics.apply_mismatch with kind='torso_mass').
    agent = make_mpc(
        'walker', cfg['H'], cfg['R'],
        N=cfg['N'],
        mismatch_factor=cfg['mismatch_factor'],
        proposal=cfg['proposal'],
        proposal_kwargs=cfg['proposal_kwargs'],
        decision=cfg['decision'],
    )

    t0      = time.time()
    costs   = _np.empty(cfg['n_steps'], dtype=_np.float64)
    torso_z = _np.empty(cfg['n_steps'] + 1, dtype=_np.float32)
    com_vx  = _np.empty(cfg['n_steps'] + 1, dtype=_np.float32)
    torso_z[0] = env.state[18]
    com_vx[0]  = env.state[20]
    for t in range(cfg['n_steps']):
        action = agent.interact(env.state, env.cost)
        env.step(action)
        costs[t]     = float(env.cost)
        torso_z[t+1] = env.state[18]
        com_vx[t+1]  = env.state[20]
    wallclock_s = time.time() - t0

    fell = bool(_np.any(torso_z < FELL_Z_THRESHOLD))

    return {
        'H': cfg['H'], 'R': cfg['R'], 'N': cfg['N'],
        'seed': cfg['seed'],
        'speed_goal': cfg['speed_goal'],
        'mismatch_factor': cfg['mismatch_factor'],
        'kind': cfg['kind'],
        'n_steps': cfg['n_steps'],
        'ctrl_dt': CTRL_DT,
        'mean_cost':    float(_np.mean(costs)),
        'torso_z_mean': float(_np.mean(torso_z)),
        'torso_z_min':  float(_np.min(torso_z)),
        'mean_vx':      float(_np.mean(com_vx)),
        'fell':         fell,
        'wallclock_s':  wallclock_s,
        '_cost':    costs.astype(_np.float32),
        '_torso_z': torso_z,
        '_com_vx':  com_vx,
    }


def _build_jobs():
    jobs = []
    for r in MISMATCH_LEVELS:
        for seed in SEEDS:
            jobs.append({
                'H': H_FIX, 'R': R_FIX, 'N': N_FIX,
                'seed': seed,
                'speed_goal': SPEED_GOAL,
                'n_steps': N_STEPS,
                'mismatch_factor': r,
                'kind': KIND,
                'proposal': PROPOSAL,
                'proposal_kwargs': PROPOSAL_KWARGS,
                'decision': DECISION,
                'repo_root': REPO_ROOT,
            })
    return jobs


_CSV_COLS = [
    'H', 'R', 'N', 'seed', 'speed_goal', 'mismatch_factor', 'kind',
    'n_steps', 'ctrl_dt',
    'mean_cost', 'torso_z_mean', 'torso_z_min', 'mean_vx', 'fell',
    'wallclock_s',
]


def _write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in _CSV_COLS})


def _format_summary(rows):
    lines = [
        f'Walker mismatch probe @ (H={H_FIX}, R={R_FIX})',
        '=' * 78,
        f'Fixed: H={H_FIX}, R={R_FIX}, N={N_FIX}, ctrl_dt={CTRL_DT}, '
        f'n_steps={N_STEPS} ({N_STEPS*CTRL_DT:.1f} s).',
        f'Speed goal: {SPEED_GOAL} (running arm). Kind: {KIND}.',
        f'Seeds: {SEEDS} (walker env_offset=3).',
        f'Planner: {PROPOSAL}, N={N_FIX}, '
        f'P={PROPOSAL_KWARGS["P"]}, sigma={PROPOSAL_KWARGS["sigma"]}, '
        f'interp={PROPOSAL_KWARGS["interp"]}.',
        f'"fell" = torso_z < {FELL_Z_THRESHOLD} m at any step.',
        '',
        f'   {"r":>4s}  {"mean_cost":>16s}  {"torso_z_min":>16s}  '
        f'{"mean_vx":>16s}  {"fell":>6s}  {"wall_s":>7s}',
        '   ' + '-' * 76,
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
        zm_m, zm_s = m('torso_z_min')
        vx_m, vx_s = m('mean_vx')
        wc_m, _    = m('wallclock_s')
        fell_n     = sum(1 for x in arm if x['fell'])
        lines.append(
            f'   {r:>4.1f}  {mc_m:>7.4f} ± {mc_s:<6.4f}  '
            f'{zm_m:>7.3f} ± {zm_s:<6.3f}  '
            f'{vx_m:>7.3f} ± {vx_s:<6.3f}  '
            f'{fell_n:>3d}/{len(arm):<2d}  '
            f'{wc_m:>6.1f}'
        )
    lines += [
        '',
        'Reading the table:',
        '  mean_cost   — averaged over all 1000 steps (10 s).',
        '  torso_z_min — lowest torso height seen (< 0.7 m ⇒ fell).',
        '  mean_vx     — mean forward velocity of torso (target ≈ 1.5 m/s).',
        '  fell        — #seeds where torso ever dropped below 0.7 m.',
    ]
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true',
                    help='Run one job only (timing a single cell).')
    ap.add_argument('--n-workers', type=int, default=12,
                    help='Pool size (default 12).')
    args = ap.parse_args()

    out_dir = os.path.join(REPO_ROOT, RESULTS_DIR.lstrip('./'))
    os.makedirs(out_dir, exist_ok=True)

    jobs = _build_jobs()
    if args.dry_run:
        jobs = jobs[:1]

    print(f'Probe: {len(jobs)} jobs on {args.n_workers} workers')
    t0   = time.time()
    rows = run_pool(_run_probe_episode, jobs,
                    n_processes=args.n_workers, verbose=1)
    print(f'Total wallclock: {time.time() - t0:.1f}s')

    rows.sort(key=lambda r: (r['mismatch_factor'], r['seed']))

    tag      = 'mismatch_range'
    csv_path  = os.path.join(out_dir, f'issue_88_probe_{tag}.csv')
    traj_path = os.path.join(out_dir, f'issue_88_probe_{tag}_trajectories.pkl')
    sum_path  = os.path.join(out_dir, f'issue_88_probe_{tag}_summary.txt')

    _write_csv(rows, csv_path)
    print(f'Wrote {csv_path}')

    traj = {
        'phase': 'mismatch_range',
        'H': H_FIX, 'R': R_FIX, 'N': N_FIX,
        'speed_goal': SPEED_GOAL, 'kind': KIND,
        'n_steps': N_STEPS, 'ctrl_dt': CTRL_DT,
        'proposal': PROPOSAL,
        'proposal_kwargs': dict(PROPOSAL_KWARGS),
        'decision': DECISION,
        'mismatch_factors': sorted({r['mismatch_factor'] for r in rows}),
        'seeds': SEEDS,
        'runs': [
            {k: r[k] for k in ('mismatch_factor', 'seed',
                               '_cost', '_torso_z', '_com_vx')}
            for r in rows
        ],
    }
    with open(traj_path, 'wb') as f:
        pickle.dump(traj, f)
    print(f'Wrote {traj_path}')

    text = _format_summary(rows)
    with open(sum_path, 'w') as f:
        f.write(text)
    print(f'Wrote {sum_path}')
    print()
    print(text)


if __name__ == '__main__':
    main()
