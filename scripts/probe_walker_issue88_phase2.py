"""Walker fresh smoke grid probe with candidate H/R and mismatch.

Runs `run_grid_sweep` with candidate H/R and candidate MISMATCH_FACTORS
at reps=3 to validate the new grid shape and gather first data at the
new high-r endpoint.

Candidate grid:
  H = [30, 40, 45, 50, 60, 80, 110, 150, 220]   9 values
       - H=20 dropped (catastrophic at every r in cached grid).
       - densified at the U-bottom (45, 50, 60 bracket the minimum).
       - long-H tail trimmed to the clear-degradation end (220).
  R = [1, 2, 3, 4, 6, 10]                       6 values
       - R=1 dominated 35/40 cells in cached grid.
       - R={1,2,3,4} spans the near-replan band.
       - R=6 is the first clear failure boundary; R=10 is the long-lag
         failure-tail marker.
  mismatch = [1.0, 1.6, 2.0, 2.6]               4 values
  reps=3, n_steps=1000 (10 s at ctrl_dt=0.01).

Planner config inherits from PROPOSAL_CONFIGS['walker']
(spline_ps, N=30, P=3 cubic, σ=0.5, speed_goal=1.5 running arm).

Total jobs: 9 × 6 × 4 × 3 = 648. On 13 cores, ~60 min wallclock.

Outputs:
  data/results/grid_walker_issue88_phase2.pkl   (run_grid_sweep pickle)
  data/results/issue_88_phase2_summary.txt      (H/R heatmap per r)
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs import RESULTS_DIR
from simulations.sweep_grid import run_grid_sweep


# Candidate grid (see module docstring for derivation)
H_CAND = [30, 40, 45, 50, 60, 80, 110, 150, 220]
R_CAND = [1, 2, 3, 4, 6, 10]
MISMATCH_CAND = [1.0, 1.6, 2.0, 2.6]
N_REPS = 3
N_STEPS = 1000   # 10 s at ctrl_dt=0.01


def _format_summary(result):
    H       = [int(x) for x in result['H_values']]
    R       = [int(x) for x in result['R_values']]
    r_vals  = result['mismatch_factors']
    mc      = result['mean_cost']     # (n_r, n_H, n_R)
    std     = result['std_cost']

    lines = [
        'Walker fresh smoke probe (candidate H/R + new r=2.6)',
        '=' * 78,
        f'H = {H}',
        f'R = {R}',
        f'mismatch = {r_vals}',
        f'reps = {result["n_reps"]}, n_steps = {N_STEPS} (10 s), ctrl_dt = {result["dt"]}',
        'Planner: spline_ps (PROPOSAL_CONFIGS["walker"]), running arm (speed_goal=1.5).',
        '',
    ]

    for fi, rv in enumerate(r_vals):
        lines.append(f'-- r = {rv}  (mean_cost, rows=H, cols=R) ' + '-' * 20)
        header = ' H \\ R ' + ''.join(f'  {Ri:>5d}' for Ri in R)
        lines.append(header)
        for hi, Hh in enumerate(H):
            row = ''.join(f'  {mc[fi,hi,ri]:>5.2f}' for ri in range(len(R)))
            lines.append(f' {Hh:>4d}  {row}')
        # best cell
        slab = mc[fi]
        if np.all(np.isnan(slab)):
            lines.append('  (all NaN — grid skipped entirely?)')
        else:
            flat_best = int(np.nanargmin(slab))
            hi_b, ri_b = np.unravel_index(flat_best, slab.shape)
            lines.append(f'  best: H={H[hi_b]}, R={R[ri_b]}, cost={slab[hi_b, ri_b]:.3f}')
        lines.append('')

    lines.append('-- range of mean_cost across R axis (max - min, small => thinnable) --')
    header = ' H \\ r  ' + ''.join(f'  {rv:>5.1f}' for rv in r_vals)
    lines.append(header)
    for hi, Hh in enumerate(H):
        parts = []
        for fi in range(len(r_vals)):
            row = mc[fi, hi, :]
            if np.all(np.isnan(row)):
                parts.append('   nan ')
            else:
                rng = float(np.nanmax(row) - np.nanmin(row))
                parts.append(f'  {rng:>5.2f}')
        lines.append(f' {Hh:>4d}   {"".join(parts)}')
    lines.append('')

    lines.append('-- fraction of (r, H) cells where each R is argmin --')
    counts = np.zeros(len(R), dtype=int)
    total  = 0
    for fi in range(len(r_vals)):
        for hi in range(len(H)):
            row = mc[fi, hi, :]
            if np.all(np.isnan(row)):
                continue
            counts[int(np.nanargmin(row))] += 1
            total += 1
    for ri, Rr in enumerate(R):
        lines.append(f'  R={Rr:>2d}: {counts[ri]:>2d}/{total}')
    lines.append('')

    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-workers', type=int, default=13,
                    help='Pool size (default 13).')
    args = ap.parse_args()

    out_dir = os.path.join(REPO_ROOT, RESULTS_DIR.lstrip('./'))
    os.makedirs(out_dir, exist_ok=True)

    total_jobs = len(H_CAND) * len(R_CAND) * len(MISMATCH_CAND) * N_REPS
    print(f'Probe: {total_jobs} jobs '
          f'({len(H_CAND)} H × {len(R_CAND)} R × {len(MISMATCH_CAND)} r × {N_REPS} reps) '
          f'on {args.n_workers} workers')

    t0 = time.time()
    result = run_grid_sweep(
        'walker',
        H_CAND, R_CAND, MISMATCH_CAND,
        n_reps=N_REPS,
        n_steps=N_STEPS,
        n_workers=args.n_workers,
    )
    wall = time.time() - t0
    print(f'Total wallclock: {wall:.1f}s ({wall/60:.1f} min)')

    out_pkl = os.path.join(out_dir, 'grid_walker_issue88_phase2.pkl')
    with open(out_pkl, 'wb') as f:
        pickle.dump(result, f)
    print(f'Wrote {out_pkl}')

    sum_path = os.path.join(out_dir, 'issue_88_phase2_summary.txt')
    text = _format_summary(result)
    with open(sum_path, 'w') as f:
        f.write(text)
    print(f'Wrote {sum_path}')
    print()
    print(text)


if __name__ == '__main__':
    main()
