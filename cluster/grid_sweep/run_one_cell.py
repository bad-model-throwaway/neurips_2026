"""Run a range of rep episodes for one (H, R, factor) grid cell and write per-cell pickles.

Each SLURM array job covers one (H_i, R_i, f_i) triple over a rep range
[rep_start, rep_end). Reps run sequentially on one CPU. Per-cell pickles are
written atomically; existing pickles are skipped so re-running a partially
failed job is safe.

Usage (called by run_one_cell.sh):
    python run_one_cell.py <env> --task-id <id> [--smoke] [--n-steps N]
                                [--rep-start S] [--rep-end E]

    env          : one of cartpole, walker, humanoid_stand, ...
    --task-id    : SLURM_ARRAY_TASK_ID, decoded as
                   H_i = task_id // (n_R * n_f)
                   R_i = (task_id // n_f) % n_R
                   f_i = task_id % n_f
    --smoke      : use SMOKE_GRIDS instead of DEFAULT_GRIDS
    --n-steps N  : override the grid's n_steps (e.g. 50 for a timing probe)
    --rep-start S: starting rep index (default 0). Use with --rep-end to
                   split a cell across multiple SLURM slots so each slot
                   fits in the time cap (worst cell × n_reps may exceed 48 h).
    --rep-end E  : end rep index, exclusive (default n_reps from grid).

Output:
    results/<env>_grid_cells/cell_H{H:03d}_R{R:02d}_f{factor:.2f}_rep{rep:02d}.pkl
    results/<env>_smoke_cells/cell_...pkl  (with --smoke)
    containing {env_name, H, R, factor, rep, seed, n_steps, mean_cost,
                failure_sec, cost_traj, terminal_state, last_states}.

    failure_sec:    float — seconds-to-first-physical-failure (or full
                    duration if env never fell). Same convention as
                    sweep_*_adaptive.py `sweep_failure`. NaN when R > H.
    cost_traj:      np.float32 array of shape (n_steps,) — per-step env.cost
    terminal_state: np.ndarray of shape (state_dim,)     — final env.state
    last_states:    np.ndarray of shape (N_TERMINAL_STATES, state_dim)
                    — env.state from the last N_TERMINAL_STATES timesteps
    All four trajectory/state fields are None when R > H.
"""

import argparse
import os
import pickle
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs import SEED
from simulations.sweep_grid import DEFAULT_GRIDS, SMOKE_GRIDS, _ENV_OFFSETS, _run_episode_worker

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')


def _cell_path(env, H, R, factor, rep, smoke=False):
    suffix   = 'smoke_cells' if smoke else 'grid_cells'
    cell_dir = os.path.join(RESULTS_DIR, f'{env}_{suffix}')
    os.makedirs(cell_dir, exist_ok=True)
    return os.path.join(cell_dir, f'cell_H{H:03d}_R{R:02d}_f{factor:.2f}_rep{rep:02d}.pkl')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('env', choices=list(DEFAULT_GRIDS.keys()))
    parser.add_argument('--task-id', type=int, required=True)
    parser.add_argument('--smoke', action='store_true',
                        help='Use SMOKE_GRIDS instead of DEFAULT_GRIDS.')
    parser.add_argument('--n-steps', type=int, default=None,
                        help='Override episode length (e.g. 50 for a timing probe).')
    parser.add_argument('--rep-start', type=int, default=0,
                        help='Starting rep index (inclusive). Default 0.')
    parser.add_argument('--rep-end', type=int, default=None,
                        help='End rep index (exclusive). Default = grid n_reps.')
    args = parser.parse_args()

    env     = args.env
    task_id = args.task_id

    grids    = SMOKE_GRIDS if args.smoke else DEFAULT_GRIDS
    g        = grids[env]
    H_values = list(g['H'])
    R_values = list(g['R'])
    factors  = list(g['mismatch'])
    n_reps   = g['reps']
    n_steps  = args.n_steps if args.n_steps is not None else g.get('n_steps', 1000)
    n_R      = len(R_values)
    n_f      = len(factors)

    H_i = task_id // (n_R * n_f)
    R_i = (task_id // n_f) % n_R
    f_i = task_id % n_f
    H      = H_values[H_i]
    R      = R_values[R_i]
    factor = factors[f_i]

    env_offset = _ENV_OFFSETS[env]

    rep_start = max(0, args.rep_start)
    rep_end   = args.rep_end if args.rep_end is not None else n_reps
    rep_end   = min(rep_end, n_reps)
    if rep_start >= rep_end:
        print(f'[{env}] empty rep range [{rep_start}, {rep_end}) — nothing to do')
        return

    for rep in range(rep_start, rep_end):
        out_path = _cell_path(env, H, R, factor, rep, smoke=args.smoke)
        if os.path.exists(out_path):
            print(f'[{env}] exists, skipping: {os.path.basename(out_path)}')
            continue

        seed = SEED + env_offset * 1000 + rep
        cfg = {
            'repo_root':       REPO_ROOT,
            'env_name':        env,
            'H':               H,
            'R':               R,
            'factor':          factor,
            'n_steps':         n_steps,
            'seed':            seed,
            'rep':             rep,
            'proposal':        None,
            'N':               None,
            'proposal_kwargs': None,
            'decision':        None,
            '_fi': 0, '_hi': 0, '_ri': 0,
        }

        result = _run_episode_worker(cfg)

        out = {
            'env_name':       env,
            'H':              H,
            'R':              R,
            'factor':         factor,
            'rep':            rep,
            'seed':           seed,
            'n_steps':        n_steps,
            'mean_cost':      result['mean_cost'],
            # Per-seed seconds-to-first-physical-failure (full duration
            # if env never fell). Same convention as sweep_*_adaptive.py
            # `sweep_failure`. NaN when R > H.
            'failure_sec':    result.get('failure_sec'),
            # Trajectory + state fields enable post-hoc K-free physical
            # success criteria. None when R > H (worker short-circuit).
            'cost_traj':      result.get('cost_traj'),
            'terminal_state': result.get('terminal_state'),
            'last_states':    result.get('last_states'),
        }
        tmp_path = out_path + '.tmp'
        with open(tmp_path, 'wb') as f:
            pickle.dump(out, f)
        os.replace(tmp_path, out_path)
        print(f'[{env}] H={H} R={R} f={factor} rep={rep}  mean_cost={result["mean_cost"]:.4f}')


if __name__ == '__main__':
    main()
