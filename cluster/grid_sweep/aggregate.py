"""Aggregate per-cell pickles into the canonical grid_<env>.pkl.

Reads every pickle under results/<env>_grid_cells/ written by run_one_cell.py,
assembles the (n_mismatch, n_H, n_R) mean/std arrays, and writes
results/grid_<env>.pkl with the same schema as data/results/grid_<env>.pkl —
so downstream figure code (visualization.heatmaps) can consume it unchanged
after you copy it there.

Usage:
    python aggregate.py --env cartpole          # production run
    python aggregate.py --env cartpole --smoke  # smoke/timing run

Prints a coverage report and, if any cells are missing, the exact
--array=<indices> argument to pass to sbatch for resubmission.
"""

import argparse
import os
import pickle
import sys

import numpy as np

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs import ENV_DT
from simulations.sweep_grid import DEFAULT_GRIDS, SMOKE_GRIDS, N_TERMINAL_STATES

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True, choices=list(DEFAULT_GRIDS.keys()))
    parser.add_argument('--smoke', action='store_true',
                        help='Read from smoke_cells and write grid_<env>_smoke.pkl.')
    args = parser.parse_args()
    env = args.env

    grids    = SMOKE_GRIDS if args.smoke else DEFAULT_GRIDS
    g        = grids[env]
    H_values = list(g['H'])
    R_values = list(g['R'])
    mismatch = list(g['mismatch'])
    n_reps   = g['reps']

    n_m, n_H, n_R = len(mismatch), len(H_values), len(R_values)
    H_idx = {int(h): i for i, h in enumerate(H_values)}
    R_idx = {int(r): i for i, r in enumerate(R_values)}
    m_idx = {float(f): i for i, f in enumerate(mismatch)}

    costs       = np.full((n_m, n_H, n_R, n_reps), np.nan, dtype=float)
    failure_sec = np.full((n_m, n_H, n_R, n_reps), np.nan, dtype=np.float32)

    # Trajectory and state arrays are allocated lazily on the first
    # per-cell pickle that carries them, because state_dim and n_steps
    # are env-specific and not known up front. Pickles written by older
    # runs without these fields are still loaded — the corresponding
    # rows in the rolled-up arrays just stay NaN.
    cost_traj_arr   = None
    terminal_arr    = None
    last_states_arr = None

    suffix   = 'smoke_cells' if args.smoke else 'grid_cells'
    cell_dir = os.path.join(RESULTS_DIR, f'{env}_{suffix}')
    if not os.path.isdir(cell_dir):
        sys.exit(f'no cell directory at {cell_dir} — run submit_all.sh first')

    n_loaded = n_unknown = 0
    for fname in sorted(os.listdir(cell_dir)):
        if not fname.endswith('.pkl') or not fname.startswith('cell_'):
            continue
        with open(os.path.join(cell_dir, fname), 'rb') as f:
            c = pickle.load(f)
        H, R, factor, rep = int(c['H']), int(c['R']), float(c['factor']), int(c['rep'])
        if H not in H_idx or R not in R_idx or factor not in m_idx or rep >= n_reps:
            n_unknown += 1
            continue
        fi, hi, ri = m_idx[factor], H_idx[H], R_idx[R]
        costs[fi, hi, ri, rep] = c['mean_cost']
        if 'failure_sec' in c and c['failure_sec'] is not None:
            failure_sec[fi, hi, ri, rep] = c['failure_sec']

        if c.get('cost_traj') is not None and c.get('terminal_state') is not None:
            if cost_traj_arr is None:
                n_steps_local = c['cost_traj'].shape[0]
                state_dim     = c['terminal_state'].shape[0]
                cost_traj_arr = np.full(
                    (n_m, n_H, n_R, n_reps, n_steps_local),
                    np.nan, dtype=np.float32,
                )
                terminal_arr = np.full(
                    (n_m, n_H, n_R, n_reps, state_dim), np.nan,
                )
                last_states_arr = np.full(
                    (n_m, n_H, n_R, n_reps, N_TERMINAL_STATES, state_dim),
                    np.nan,
                )
            cost_traj_arr[fi, hi, ri, rep]   = c['cost_traj']
            terminal_arr[fi, hi, ri, rep]    = c['terminal_state']
            last_states_arr[fi, hi, ri, rep] = c['last_states']

        n_loaded += 1

    filled      = np.sum(~np.isnan(costs), axis=-1)   # (n_m, n_H, n_R)
    total_cells = n_m * n_H * n_R
    complete    = int(np.sum(filled == n_reps))
    partial     = int(np.sum((filled > 0) & (filled < n_reps)))
    empty       = int(np.sum(filled == 0))

    print(f'Loaded {n_loaded} per-cell pickles ({n_unknown} ignored).')
    print(f'Grid: {n_m} mismatch × {n_H} H × {n_R} R = {total_cells} cells, '
          f'reps={n_reps} (expected {total_cells * n_reps} episodes).')
    print(f'  complete cells: {complete}/{total_cells}')
    print(f'  partial cells:  {partial}')
    print(f'  empty cells:    {empty}')

    if empty > 0 or partial > 0:
        missing_task_ids = []
        print()
        print('Incomplete (H, R, factor) slots:')
        for hi in range(n_H):
            for ri in range(n_R):
                for fi in range(n_m):
                    f = int(filled[fi, hi, ri])
                    if f < n_reps:
                        task_id = hi * (n_R * n_m) + ri * n_m + fi
                        missing_task_ids.append(str(task_id))
                        print(f'  task={task_id:<4}  H={H_values[hi]:>3}  '
                              f'R={R_values[ri]:>2}  r={mismatch[fi]:<4}  '
                              f'{f}/{n_reps}')
        smoke_flag = ' --smoke' if args.smoke else ''
        print()
        print('Resubmit missing slots:')
        print(f'  sbatch --array={",".join(missing_task_ids)} run_one_cell.sh {env}{smoke_flag}')

    result = {
        'env':              env,
        'H_values':         np.array(H_values),
        'R_values':         np.array(R_values),
        'mismatch_factors': mismatch,
        'mean_cost':        np.nanmean(costs, axis=-1),
        'std_cost':         np.nanstd(costs, axis=-1),
        'all_costs':        costs,
        'failure_sec':      failure_sec,
        'dt':               ENV_DT[env],
        'n_reps':           n_reps,
    }
    if cost_traj_arr is not None:
        result['cost_traj']         = cost_traj_arr
        result['terminal_states']   = terminal_arr
        result['last_states']       = last_states_arr
        result['n_terminal_states'] = N_TERMINAL_STATES

    fname    = f'grid_{env}_smoke.pkl' if args.smoke else f'grid_{env}.pkl'
    out_path = os.path.join(RESULTS_DIR, fname)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp_path = out_path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(result, f)
    os.replace(tmp_path, out_path)

    mc = result['mean_cost']
    print()
    print(f'Wrote {out_path}')
    print(f'mean_cost shape={mc.shape}  '
          f'range=[{np.nanmin(mc):.4f}, {np.nanmax(mc):.4f}]')


if __name__ == '__main__':
    main()
