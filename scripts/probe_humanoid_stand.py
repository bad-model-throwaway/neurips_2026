"""Humanoid-stand PS-spline probe — matched + torso-mass mismatch.

Two phases:
  --phase=matched  : sweep N ∈ N_VALUES at r=1.0, H=23, 3 seeds.
                     Acceptance: humanoid stands ≥5 s at N ≤ ~500.
  --phase=mismatch : fix N at MJPC value, sweep r × H.
                     Extracts long-horizon degradation signal.

Writes per-run rows to data/results/issue_74_probe_<phase>.csv plus a
summary txt sibling.

"Stood" metric: fraction of episode steps with |r_height| ≤ 0.1 (one
kSmoothAbsLoss p-unit), where r_height = head_z − feet_avg_z − height_goal.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs import RESULTS_DIR
from simulations.simulation import run_pool


# MJPC humanoid-stand anchor: H=23 planner steps at ctrl_dt=0.015 s (~0.35 s horizon).
H_MJPC = 23
R_MJPC = 1

# 5 s sim time = 334 ctrl steps at dt=0.015; round to 400 for 6 s buffer.
N_STEPS_5S = 400  # 6 s; acceptance metric uses first 334 steps (5 s).

# Matched screen: MJPC default (N=10) first, escalate up to 500 if needed.
N_VALUES_MATCHED = [10, 30, 100, 300, 500]

# Mismatch screen: N=100 = minimum N where matched dynamics reliably stood;
# MJPC's N=10 fell 0/3 in our Python pipeline.
N_MISMATCH = 100
MISMATCH_FACTORS = [1.0, 1.3, 1.6, 2.0]
# Multiple H values to stress the long-horizon story (MJPC + 2×, 3×).
H_VALUES_MISMATCH = [23, 46, 69]

SEEDS_MATCHED  = [0, 1, 2]
SEEDS_MISMATCH = [0]


def _run_probe_episode(cfg):
    import numpy as _np
    import mujoco as _mujoco

    repo_root = cfg['repo_root']
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from agents.mpc import make_mpc
    from agents.mujoco_dynamics import HumanoidStandDynamics
    from simulations.simulation import run_simulation

    env_name         = cfg['env_name']
    H                = cfg['H']
    R                = cfg['R']
    N                = cfg['N']
    n_steps          = cfg['n_steps']
    seed             = cfg['seed']
    mismatch_factor  = cfg['mismatch_factor']
    height_goal      = cfg['height_goal']

    _np.random.seed(seed)

    env = HumanoidStandDynamics(stateless=False, height_goal=height_goal)
    env.reset(env.get_default_initial_state())

    # Planning model inherits the same MJCF but can be mismatched.
    agent = make_mpc(env_name, H, R, N=N, mismatch_factor=mismatch_factor)

    t0 = time.time()
    _, env_out, history = run_simulation(
        agent, env, n_steps=n_steps, interval=None,
    )
    wallclock_s = time.time() - t0

    states  = history.get_item_history('state')   # (T+1, state_dim)
    actions = history.get_item_history('action')  # (T, action_dim)

    # Residual 0: r_height = head_z − feet_avg_z − height_goal (scalar).
    # state layout: extras start at qpos+qvel = 28+27 = 55.
    head_z     = states[:, 55]
    feet_avg_z = states[:, 56]
    r_height   = head_z - feet_avg_z - height_goal

    # "Stood" = |r_height| ≤ 0.1 for the full first 5 s (334 ctrl steps).
    # Also report the full-episode stood fraction.
    ctrl_dt = 0.015
    n_step_5s = int(round(5.0 / ctrl_dt))  # 334
    rh_first5 = r_height[:n_step_5s + 1]
    stood_full_5s = bool(_np.all(_np.abs(rh_first5) <= 0.1))
    stood_fraction_5s   = float(_np.mean(_np.abs(rh_first5) <= 0.1))
    stood_fraction_full = float(_np.mean(_np.abs(r_height) <= 0.1))

    # H-averaged cost: mean cost over the episode.
    costs = _np.array([env_out.cost_function(x, ctrl=a)
                       for x, a in zip(states[1:], actions)])
    mean_cost = float(_np.mean(costs))

    row = {
        'env': env_name,
        'phase': cfg.get('phase', 'unknown'),
        'N': N, 'seed': seed, 'H': H, 'R': R,
        'mismatch_factor': mismatch_factor,
        'n_steps': n_steps,
        'ctrl_dt': ctrl_dt,
        'height_goal': height_goal,
        'mean_cost': mean_cost,
        'final_head_z': float(states[-1, 55]),
        'final_feet_avg_z': float(states[-1, 56]),
        'final_r_height': float(r_height[-1]),
        'min_head_z': float(_np.min(states[:, 55])),
        'stood_full_5s': stood_full_5s,
        'stood_frac_5s': stood_fraction_5s,
        'stood_frac_full': stood_fraction_full,
        'wallclock_s': wallclock_s,
    }
    return row


def _build_matched_jobs():
    jobs = []
    for N in N_VALUES_MATCHED:
        for seed in SEEDS_MATCHED:
            jobs.append({
                'env_name': 'humanoid_stand',
                'phase': 'matched',
                'H': H_MJPC, 'R': R_MJPC, 'N': N,
                'seed': seed,
                'n_steps': N_STEPS_5S,
                'mismatch_factor': 1.0,
                'height_goal': 1.4,
                'repo_root': REPO_ROOT,
            })
    return jobs


def _build_mismatch_jobs():
    jobs = []
    for r in MISMATCH_FACTORS:
        for H in H_VALUES_MISMATCH:
            for seed in SEEDS_MISMATCH:
                jobs.append({
                    'env_name': 'humanoid_stand',
                    'phase': 'mismatch',
                    'H': H, 'R': R_MJPC, 'N': N_MISMATCH,
                    'seed': seed,
                    'n_steps': N_STEPS_5S,
                    'mismatch_factor': r,
                    'height_goal': 1.4,
                    'repo_root': REPO_ROOT,
                })
    return jobs


_CSV_COLS = [
    'env', 'phase', 'N', 'seed', 'H', 'R', 'mismatch_factor',
    'n_steps', 'ctrl_dt', 'height_goal',
    'mean_cost', 'final_head_z', 'final_feet_avg_z', 'final_r_height',
    'min_head_z', 'stood_full_5s', 'stood_frac_5s', 'stood_frac_full',
    'wallclock_s',
]


def _write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in _CSV_COLS})


def _format_matched_summary(rows):
    """Group by N, report mean±std across seeds."""
    lines = ['Humanoid-stand matched screen probe',
             '=' * 72,
             f'Common: H={H_MJPC}, R={R_MJPC}, ctrl_dt=0.015 s, n_steps={N_STEPS_5S} (6 s), '
             f'seeds {SEEDS_MATCHED}.',
             f'"stood_full_5s" = |r_height| ≤ 0.1 m for every step of the first 5 s.',
             '']
    lines.append(f'   {"N":>4s}  {"stood5s":>10s}  {"stood_frac_5s":>14s}  '
                 f'{"mean_cost":>14s}  {"final_head_z":>14s}  {"wallclock_s":>12s}')
    lines.append('   ' + '-' * 76)
    by_N = {}
    for r in rows:
        by_N.setdefault(r['N'], []).append(r)
    for N in sorted(by_N):
        arm = by_N[N]
        stood_count = sum(1 for r in arm if r['stood_full_5s'])
        sf = np.array([r['stood_frac_5s'] for r in arm])
        mc = np.array([r['mean_cost'] for r in arm])
        fh = np.array([r['final_head_z'] for r in arm])
        wc = np.array([r['wallclock_s'] for r in arm])
        lines.append(
            f'   {N:>4d}  {stood_count:>4d}/{len(arm):d}      '
            f'{sf.mean():>6.3f}±{sf.std(ddof=0):<5.3f}  '
            f'{mc.mean():>7.3f}±{mc.std(ddof=0):<5.3f}  '
            f'{fh.mean():>7.3f}±{fh.std(ddof=0):<5.3f}  '
            f'{wc.mean():>6.1f}±{wc.std(ddof=0):<4.1f}'
        )
    return '\n'.join(lines) + '\n'


def _format_mismatch_summary(rows):
    """One row per (r, H)."""
    lines = ['Humanoid-stand torso-mass mismatch probe',
             '=' * 72,
             f'Fixed: N={N_MISMATCH}, R={R_MJPC}, ctrl_dt=0.015 s, n_steps={N_STEPS_5S} (6 s), '
             f'seed={SEEDS_MISMATCH[0]}.',
             '']
    lines.append(f'   {"r":>4s}  {"H":>4s}  {"stood_frac_5s":>14s}  '
                 f'{"mean_cost":>12s}  {"final_r_height":>14s}  {"wallclock_s":>12s}')
    lines.append('   ' + '-' * 76)
    rows = sorted(rows, key=lambda r: (r['mismatch_factor'], r['H']))
    for r in rows:
        lines.append(
            f'   {r["mismatch_factor"]:>4.1f}  {r["H"]:>4d}  '
            f'{r["stood_frac_5s"]:>14.3f}  '
            f'{r["mean_cost"]:>12.3f}  '
            f'{r["final_r_height"]:>+14.3f}  '
            f'{r["wallclock_s"]:>12.2f}'
        )
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['matched', 'mismatch'], required=True)
    ap.add_argument('--n-subset', type=int, nargs='*', default=None,
                    help='For --phase=matched: only run these N values (default = all).')
    args = ap.parse_args()

    out_dir = os.path.join(REPO_ROOT, RESULTS_DIR.lstrip('./'))
    os.makedirs(out_dir, exist_ok=True)

    if args.phase == 'matched':
        jobs = _build_matched_jobs()
        if args.n_subset is not None:
            jobs = [j for j in jobs if j['N'] in args.n_subset]
        csv_path     = os.path.join(out_dir, 'issue_74_probe_matched.csv')
        summary_path = os.path.join(out_dir, 'issue_74_probe_matched_summary.txt')
        fmt          = _format_matched_summary
    else:
        jobs = _build_mismatch_jobs()
        csv_path     = os.path.join(out_dir, 'issue_74_probe_mismatch.csv')
        summary_path = os.path.join(out_dir, 'issue_74_probe_mismatch_summary.txt')
        fmt          = _format_mismatch_summary

    print(f'Probe phase={args.phase}: {len(jobs)} jobs')
    t0 = time.time()
    rows = run_pool(_run_probe_episode, jobs, verbose=1)
    print(f'Total wallclock: {time.time() - t0:.1f}s')

    rows.sort(key=lambda r: (r['mismatch_factor'], r['H'], r['N'], r['seed']))
    _write_csv(rows, csv_path)
    print(f'Wrote {csv_path}')

    text = fmt(rows)
    with open(summary_path, 'w') as f:
        f.write(text)
    print(f'Wrote {summary_path}')
    print()
    print(text)


if __name__ == '__main__':
    main()
