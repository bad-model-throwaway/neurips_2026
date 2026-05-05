"""Get-up under torso-mass mismatch probe.

Three phases:

  --phase=mismatch_range
      Fixed (H, R) = (40, 1) — matched best cell from the smoke grid.
      Sweep r ∈ {1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0}.
      Purpose: separate "slow get-up" from "failed get-up" by logging
      final_head_z / max_head_z per seed. Reliable-stand band ends
      around r≈2.

  --phase=hr_grid
      2D (H, R) slice at narrower mismatch r ∈ {1.0, 1.5, 2.0, 2.5}.
      H ∈ {40, 80, 130} × R ∈ {1, 3, 8} = 9 (H, R) cells.
      Purpose: test whether the best (H, R) actually shifts with r in
      the reliable-stand band, before committing to new grid constants.

  --phase=flip_check
      Fine-H slice at R=1, r ∈ {1.0, 2.5}, reps=5. H densified over the
      transition region: H ∈ {30, 40, 50, 65, 85}.
      Purpose: confirm whether best H really shifts from ~40 at r=1 to
      ~80 at r=2.5; five reps halves the SEM relative to the reps=3 smoke.

N=30 matches PROPOSAL_CONFIGS['humanoid_stand'] and the smoke sweep.
Per episode we save head_z(t), r_height(t), and cost(t) trajectories.

Outputs (suffix differs by phase):
  data/results/issue_108_probe_<phase>.csv
  data/results/issue_108_probe_<phase>_trajectories.pkl
  data/results/issue_108_probe_<phase>_summary.txt
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

from configs import RESULTS_DIR
from simulations.simulation import run_pool


N_FIX    = 30       # PROPOSAL_CONFIGS default — matches smoke sweep
N_STEPS  = 400      # 6 s at ctrl_dt=0.015 (same as smoke)
SEEDS    = [0, 1, 2]

# mismatch_range: fixed (H, R), wide r sweep.
H_FIX    = 40       # matched-best H from grid_humanoid_stand.pkl
R_FIX    = 1        # matched-best R
MISMATCH_WIDE = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

# hr_grid: narrow r band where the humanoid reliably stands, crossed with
# a coarse (H, R) slice. H spans short / default / long; R spans
# replan-every-step / moderate / sparse.
MISMATCH_NARROW = [1.0, 1.5, 2.0, 2.5]
H_GRID = [40, 80, 130]
R_GRID = [1, 3, 8]

# flip_check: fine H at R=1, endpoints of the reliable-stand band,
# reps=5 to cut bimodal-failure noise roughly in half vs the reps=3 smoke.
H_FINE        = [30, 40, 50, 65, 85]
R_FINE        = 1
MISMATCH_ENDS = [1.0, 2.5]
SEEDS_FINE    = [0, 1, 2, 3, 4]

# "Stood" = |r_height| ≤ 0.1 m (one kSmoothAbsLoss p-unit).
STOOD_TOL = 0.1
# 5 s window for stood-fraction (first 334 ctrl steps).
N_STEP_5S = int(round(5.0 / 0.015))  # 334


def _run_probe_episode(cfg):
    import numpy as _np
    import mujoco as _mujoco  # noqa: F401  (import eagerly so workers fail fast)

    repo_root = cfg['repo_root']
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from agents.mpc import make_mpc
    from agents.mujoco_dynamics import HumanoidStandDynamics
    from simulations.simulation import run_simulation

    _np.random.seed(cfg['seed'])

    env = HumanoidStandDynamics(stateless=False,
                                height_goal=cfg['height_goal'])
    env.reset(env.get_default_initial_state())

    agent = make_mpc('humanoid_stand', cfg['H'], cfg['R'],
                     N=cfg['N'], mismatch_factor=cfg['mismatch_factor'])

    t0 = time.time()
    _, env_out, history = run_simulation(
        agent, env, n_steps=cfg['n_steps'], interval=None,
    )
    wallclock_s = time.time() - t0

    states  = history.get_item_history('state')   # (T+1, state_dim)
    actions = history.get_item_history('action')  # (T, action_dim)

    head_z     = states[:, 55]
    feet_avg_z = states[:, 56]
    r_height   = head_z - feet_avg_z - cfg['height_goal']

    costs = _np.array([env_out.cost_function(x, ctrl=a)
                       for x, a in zip(states[1:], actions)])

    stood_mask   = _np.abs(r_height) <= STOOD_TOL
    stood_mask5s = stood_mask[:N_STEP_5S + 1]
    # First time step at which the humanoid reaches the "stood" band and
    # stays there for the remainder of the episode. np.inf if never stays.
    time_to_stand_s = float('inf')
    if stood_mask.any():
        # earliest i such that stood_mask[i:] is all True
        flipped = stood_mask[::-1]
        first_false_from_end = int(_np.argmax(~flipped)) if (~flipped).any() else 0
        if first_false_from_end == 0:
            # whole tail is True — time_to_stand is when it first entered
            entry = int(_np.argmax(stood_mask))
            time_to_stand_s = entry * 0.015

    row = {
        'phase': 'mismatch_range',
        'H': cfg['H'], 'R': cfg['R'], 'N': cfg['N'],
        'seed': cfg['seed'],
        'mismatch_factor': cfg['mismatch_factor'],
        'n_steps': cfg['n_steps'],
        'ctrl_dt': 0.015,
        'height_goal': cfg['height_goal'],
        'mean_cost': float(_np.mean(costs)),
        'final_head_z':     float(states[-1, 55]),
        'max_head_z':       float(_np.max(head_z)),
        'final_r_height':   float(r_height[-1]),
        'stood_frac_5s':    float(_np.mean(stood_mask5s)),
        'stood_frac_full':  float(_np.mean(stood_mask)),
        'stood_full_5s':    bool(_np.all(stood_mask5s)),
        'time_to_stand_s':  time_to_stand_s,
        'wallclock_s':      wallclock_s,
        # trajectories for later plotting / analysis
        '_head_z':   head_z.astype(_np.float32),
        '_r_height': r_height.astype(_np.float32),
        '_cost':     costs.astype(_np.float32),
    }
    return row


def _build_jobs_mismatch_range():
    jobs = []
    for r in MISMATCH_WIDE:
        for seed in SEEDS:
            jobs.append({
                'env_name': 'humanoid_stand',
                'H': H_FIX, 'R': R_FIX, 'N': N_FIX,
                'seed': seed,
                'n_steps': N_STEPS,
                'mismatch_factor': r,
                'height_goal': 1.4,
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
                        'env_name': 'humanoid_stand',
                        'H': H, 'R': R, 'N': N_FIX,
                        'seed': seed,
                        'n_steps': N_STEPS,
                        'mismatch_factor': r,
                        'height_goal': 1.4,
                        'repo_root': REPO_ROOT,
                    })
    return jobs


def _build_jobs_flip_check():
    jobs = []
    for r in MISMATCH_ENDS:
        for H in H_FINE:
            for seed in SEEDS_FINE:
                jobs.append({
                    'env_name': 'humanoid_stand',
                    'H': H, 'R': R_FINE, 'N': N_FIX,
                    'seed': seed,
                    'n_steps': N_STEPS,
                    'mismatch_factor': r,
                    'height_goal': 1.4,
                    'repo_root': REPO_ROOT,
                })
    return jobs


_CSV_COLS = [
    'phase', 'H', 'R', 'N', 'seed', 'mismatch_factor',
    'n_steps', 'ctrl_dt', 'height_goal',
    'mean_cost', 'final_head_z', 'max_head_z', 'final_r_height',
    'stood_frac_5s', 'stood_frac_full', 'stood_full_5s',
    'time_to_stand_s', 'wallclock_s',
]


def _write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in _CSV_COLS})


def _format_summary_mismatch_range(rows):
    lines = [
        'Humanoid stand-up mismatch-range probe @ (H=40, R=1)',
        '=' * 78,
        f'Fixed: H={H_FIX}, R={R_FIX}, N={N_FIX}, ctrl_dt=0.015 s, '
        f'n_steps={N_STEPS} (6 s), seeds={SEEDS}.',
        '"stood" = |head_z − feet_avg_z − 1.4| ≤ 0.1 m.',
        '',
        f'   {"r":>4s}  {"mean_cost":>12s}  {"final_head_z":>14s}  '
        f'{"max_head_z":>12s}  {"stood_frac5s":>14s}  '
        f'{"t_stand_s":>10s}  {"wall_s":>8s}',
        '   ' + '-' * 78,
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
        fh_m, fh_s = m('final_head_z')
        mh_m, mh_s = m('max_head_z')
        sf_m, sf_s = m('stood_frac_5s')
        wc_m, _    = m('wallclock_s')
        tts_vals = np.array([x['time_to_stand_s'] for x in arm], dtype=float)
        tts_str  = (f'{np.nanmean(np.where(np.isfinite(tts_vals), tts_vals, np.nan)):>5.2f}'
                    if np.isfinite(tts_vals).any() else '   inf')
        lines.append(
            f'   {r:>4.1f}  {mc_m:>6.2f}±{mc_s:<4.2f}  '
            f'{fh_m:>+7.3f}±{fh_s:<4.3f}  '
            f'{mh_m:>+6.3f}±{mh_s:<4.3f}  '
            f'{sf_m:>6.3f}±{sf_s:<5.3f}  '
            f'{tts_str:>10s}  {wc_m:>6.1f}'
        )
    lines += [
        '',
        'Reading the table:',
        '  final_head_z — where the head ends up. Standing ≈ 1.5 m, supine ≈ 0.1 m.',
        '  max_head_z   — did the humanoid ever reach standing height?',
        '  stood_frac5s — fraction of first 5 s spent in the stood band.',
        '  t_stand_s    — first entry into the stood band that persisted to end.',
    ]
    return '\n'.join(lines) + '\n'


def _format_summary_hr_grid(rows):
    """For each r, print the (H, R) grid of mean_cost and success_rate."""
    lines = [
        'Humanoid stand-up (H, R) grid × narrow mismatch probe',
        '=' * 78,
        f'Fixed: N={N_FIX}, ctrl_dt=0.015 s, n_steps={N_STEPS} (6 s), '
        f'seeds={SEEDS}.',
        f'H ∈ {H_GRID}, R ∈ {R_GRID}, r ∈ {MISMATCH_NARROW}.',
        '"success" = final_head_z ≥ 1.0  (humanoid is upright-ish at t=6 s).',
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
        lines.append(f'  mean_cost (mean ± sd over {len(SEEDS)} seeds)')
        lines.append('    H\\R  ' + '  '.join(f'R={R:<10d}' for R in R_GRID))
        for H in H_GRID:
            parts = [f'    H={H:<3d}']
            for R in R_GRID:
                cells = [x for x in arm if x['H'] == H and x['R'] == R]
                mc_m, mc_s = cell(cells, 'mean_cost')
                parts.append(f'{mc_m:>5.1f}±{mc_s:<4.1f}  ')
            lines.append(''.join(parts))
        lines.append(f'  success rate (final_head_z ≥ 1.0)')
        lines.append('    H\\R  ' + '  '.join(f'R={R:<10d}' for R in R_GRID))
        for H in H_GRID:
            parts = [f'    H={H:<3d}']
            for R in R_GRID:
                cells = [x for x in arm if x['H'] == H and x['R'] == R]
                succ = sum(1 for c in cells if c['final_head_z'] >= 1.0)
                parts.append(f'{succ}/{len(cells):<10d}')
            lines.append(''.join(parts))
        best = min(arm, key=lambda x: x['mean_cost'])
        lines.append(
            f'  best cell: H={best["H"]}, R={best["R"]}, '
            f'seed={best["seed"]}, mean_cost={best["mean_cost"]:.2f}, '
            f'final_head_z={best["final_head_z"]:.2f}'
        )
        agg = {}
        for H in H_GRID:
            for R in R_GRID:
                cells = [x for x in arm if x['H'] == H and x['R'] == R]
                agg[(H, R)] = np.mean([x['mean_cost'] for x in cells])
        best_agg = min(agg, key=lambda k: agg[k])
        lines.append(
            f'  best (H, R) by seed-avg cost: H={best_agg[0]}, R={best_agg[1]}, '
            f'avg_cost={agg[best_agg]:.2f}'
        )
        lines.append('')
    return '\n'.join(lines) + '\n'


def _format_summary_flip_check(rows):
    """For each r, print mean_cost ± sd and success rate across H at R=1,
    then show the H-shift contrast (argmin H and SEM-scaled differences).
    """
    lines = [
        'Humanoid stand-up fine-H flip check probe @ R=1, reps=5',
        '=' * 78,
        f'Fixed: N={N_FIX}, R={R_FINE}, ctrl_dt=0.015 s, n_steps={N_STEPS} (6 s), '
        f'seeds={SEEDS_FINE}.',
        f'H ∈ {H_FINE}, r ∈ {MISMATCH_ENDS}.',
        '"success" = final_head_z ≥ 1.0  (humanoid is upright-ish at t=6 s).',
        '',
    ]
    by_r = {}
    for row in rows:
        by_r.setdefault(row['mismatch_factor'], []).append(row)

    for r in sorted(by_r):
        arm = by_r[r]
        lines.append(f'-- r = {r:.1f} ' + '-' * 60)
        lines.append(f'  mean_cost (mean ± sd over {len(SEEDS_FINE)} seeds)')
        header = '    H     ' + '  '.join(f'H={H:<8d}' for H in H_FINE)
        lines.append(header)
        parts = ['           ']
        for H in H_FINE:
            cells = [x for x in arm if x['H'] == H]
            arr = np.array([x['mean_cost'] for x in cells], dtype=float)
            parts.append(f'{arr.mean():>5.1f}±{arr.std(ddof=0):<4.1f}  ')
        lines.append(''.join(parts))
        lines.append(f'  success rate (final_head_z ≥ 1.0)')
        lines.append(header)
        parts = ['           ']
        for H in H_FINE:
            cells = [x for x in arm if x['H'] == H]
            succ = sum(1 for c in cells if c['final_head_z'] >= 1.0)
            parts.append(f'{succ}/{len(cells):<8d}  ')
        lines.append(''.join(parts))
        agg = {H: np.mean([x['mean_cost'] for x in arm if x['H'] == H])
               for H in H_FINE}
        best_H = min(agg, key=lambda k: agg[k])
        lines.append(f'  best H by seed-avg cost: H={best_H}, avg_cost={agg[best_H]:.2f}')
        lines.append('')

    lines.append('-- Flip test ' + '-' * 60)
    lines.append('  cost(H) at R=1, with SEM over 5 reps, and Δ vs best-H-at-r=1:')
    r_lo, r_hi = sorted(MISMATCH_ENDS)[0], sorted(MISMATCH_ENDS)[-1]
    arm_lo = by_r.get(r_lo, [])
    arm_hi = by_r.get(r_hi, [])
    stats = {}
    for r_tag, arm in (('r_lo', arm_lo), ('r_hi', arm_hi)):
        for H in H_FINE:
            cells = [x for x in arm if x['H'] == H]
            arr = np.array([x['mean_cost'] for x in cells], dtype=float)
            mean = arr.mean()
            sem  = arr.std(ddof=0) / np.sqrt(max(1, len(arr)))
            stats[(r_tag, H)] = (mean, sem)
    best_lo_H = min(H_FINE, key=lambda H: stats[('r_lo', H)][0])
    best_hi_H = min(H_FINE, key=lambda H: stats[('r_hi', H)][0])
    lines.append(f'  at r={r_lo}:  best H = {best_lo_H}')
    lines.append(f'  at r={r_hi}:  best H = {best_hi_H}')
    if best_hi_H != best_lo_H:
        m_old, s_old = stats[('r_hi', best_lo_H)]
        m_new, s_new = stats[('r_hi', best_hi_H)]
        diff = m_old - m_new
        sem  = np.sqrt(s_old**2 + s_new**2)
        lines.append(
            f'  at r={r_hi}, cost(H={best_lo_H}) - cost(H={best_hi_H}) = '
            f'{diff:+.2f} (SEM={sem:.2f}, {diff/sem:+.2f}σ)'
        )
    else:
        lines.append('  → no flip: best H unchanged across r.')
    lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase',
                    choices=['mismatch_range', 'hr_grid', 'flip_check'],
                    default='mismatch_range')
    ap.add_argument('--dry-run', action='store_true',
                    help='Run one job only to time a cell.')
    args = ap.parse_args()

    out_dir = os.path.join(REPO_ROOT, RESULTS_DIR.lstrip('./'))
    os.makedirs(out_dir, exist_ok=True)

    if args.phase == 'mismatch_range':
        jobs = _build_jobs_mismatch_range()
        fmt  = _format_summary_mismatch_range
    elif args.phase == 'hr_grid':
        jobs = _build_jobs_hr_grid()
        fmt  = _format_summary_hr_grid
    else:
        jobs = _build_jobs_flip_check()
        fmt  = _format_summary_flip_check

    if args.dry_run:
        jobs = jobs[:1]

    print(f'Probe phase={args.phase}: {len(jobs)} jobs')
    t0 = time.time()
    rows = run_pool(_run_probe_episode, jobs, verbose=1)
    print(f'Total wallclock: {time.time() - t0:.1f}s')

    rows.sort(key=lambda r: (r['mismatch_factor'], r['H'], r['R'], r['seed']))

    csv_path  = os.path.join(out_dir, f'issue_108_probe_{args.phase}.csv')
    traj_path = os.path.join(out_dir, f'issue_108_probe_{args.phase}_trajectories.pkl')
    sum_path  = os.path.join(out_dir, f'issue_108_probe_{args.phase}_summary.txt')

    _write_csv(rows, csv_path)
    print(f'Wrote {csv_path}')

    traj = {
        'phase': args.phase,
        'N': N_FIX, 'n_steps': N_STEPS,
        'ctrl_dt': 0.015, 'height_goal': 1.4,
        'mismatch_factors': sorted({r['mismatch_factor'] for r in rows}),
        'seeds': SEEDS,
        'runs': [
            {k: r[k] for k in ('mismatch_factor', 'H', 'R', 'seed',
                               '_head_z', '_r_height', '_cost')}
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
