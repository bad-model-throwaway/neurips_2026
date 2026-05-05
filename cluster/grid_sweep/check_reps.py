"""One-shot diagnostic: how few reps give the same heatmap as 100?

For each env with a populated <env>_grid_cells/ dir, restrict to cells that
have all 100 reps, then compare:
  mean_n  = mean of reps 0..n-1   for n in {25, 50, 75}
  mean_100 = mean of reps 0..99

Reports per-env:
  * max absolute delta    (mean_n - mean_100) across cells
  * max fractional delta  normalized to the grid's cost range
  * # cells where |delta| > 5% of grid range
"""

import os
import pickle
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, REPO_ROOT)

from simulations.sweep_grid import DEFAULT_GRIDS

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')
ENVS = ['cartpole', 'walker', 'humanoid_balance']


def load_costs(env):
    """Return costs array (n_m, n_H, n_R, n_reps) filled with NaN for missing."""
    g = DEFAULT_GRIDS[env]
    H_values = list(g['H']); R_values = list(g['R'])
    mismatch = list(g['mismatch']); n_reps = g['reps']
    H_idx = {int(h): i for i, h in enumerate(H_values)}
    R_idx = {int(r): i for i, r in enumerate(R_values)}
    m_idx = {float(f): i for i, f in enumerate(mismatch)}

    costs = np.full((len(mismatch), len(H_values), len(R_values), n_reps),
                    np.nan, dtype=float)

    cell_dir = os.path.join(RESULTS_DIR, f'{env}_grid_cells')
    if not os.path.isdir(cell_dir):
        return None

    for fname in os.listdir(cell_dir):
        if not (fname.endswith('.pkl') and fname.startswith('cell_')):
            continue
        with open(os.path.join(cell_dir, fname), 'rb') as f:
            c = pickle.load(f)
        H, R, fac, rep = int(c['H']), int(c['R']), float(c['factor']), int(c['rep'])
        if H in H_idx and R in R_idx and fac in m_idx and rep < n_reps:
            costs[m_idx[fac], H_idx[H], R_idx[R], rep] = c['mean_cost']
    return costs


def report(env, costs):
    n_reps_full = costs.shape[-1]
    filled = np.sum(~np.isnan(costs), axis=-1)
    full_mask = filled == n_reps_full
    n_full = int(full_mask.sum())
    if n_full == 0:
        print(f'{env}: no fully-populated cells — skipping'); return

    c_full = costs[full_mask]
    mean_full = c_full.mean(axis=-1)

    grid_mean_full = np.nanmean(costs, axis=-1)
    grid_range     = float(np.nanmax(grid_mean_full) - np.nanmin(grid_mean_full))

    def summarize(n_sub):
        m_sub = c_full[:, :n_sub].mean(axis=-1)
        delta = m_sub - mean_full
        abs_d = np.abs(delta)
        frac  = abs_d / grid_range
        n_big = int((frac > 0.05).sum())
        print(f'  n={n_sub:>3} vs n={n_reps_full}: max|Δ|={abs_d.max():.4f}  '
              f'max|Δ|/range={frac.max()*100:.2f}%  '
              f'median|Δ|/range={np.median(frac)*100:.2f}%  '
              f'cells>5%: {n_big}/{n_full}')

    print(f'{env}: {n_full} fully-populated cells (of {costs.shape[0]*costs.shape[1]*costs.shape[2]}),'
          f'  grid cost range={grid_range:.3f}')
    sub_ns = [n for n in (25, 50, 75) if n < n_reps_full]
    for n_sub in sub_ns:
        summarize(n_sub)


def main():
    for env in ENVS:
        costs = load_costs(env)
        if costs is None:
            print(f'{env}: cell dir missing — skipping'); continue
        report(env, costs)
        print()


if __name__ == '__main__':
    main()
