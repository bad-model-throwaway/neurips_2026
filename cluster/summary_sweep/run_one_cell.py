"""Run all reps for one (condition, mismatch) summary-sweep cell; write per-rep pickles.

Task-ID decoding:
    task_id = condition_idx * n_mismatches + mismatch_idx

Each SLURM array slot covers one (condition, mismatch) pair and runs
N_EPISODES reps sequentially on one CPU. Per-rep pickles are written
atomically (tmp → rename); existing pickles are skipped so re-running
after a partial failure only re-runs missing reps.

Seed strategy: seed = SEED + condition_idx * 10_000 + mismatch_idx * 1_000 + rep
Deterministic and unique per (env, condition, mismatch, rep). For walker /
humanoid, this seeds the MPPI sampler. For cartpole, it is also fed to the
worker but the initial pole angle is drawn separately from a single
RandomState(SEED).uniform(-0.1, 0.1, size=N_EPISODES) keyed by rep, exactly
matching `sweep_cartpole_adaptive.run_adaptive_sweep` so the fan-out
output is bit-identical to the in-process sweep.

Usage (called by run_one_cell.sh):
    python run_one_cell.py cartpole         --task-id 7
    python run_one_cell.py walker           --task-id 12
    python run_one_cell.py humanoid_balance --task-id 5
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs import SEED
from simulations import sweep_cartpole_summary           as _scs
from simulations import sweep_walker_summary             as _sws
from simulations import sweep_humanoid_balance_summary   as _shbs
from simulations.sweep_cartpole_adaptive         import _adaptive_worker as _cp_worker
from simulations.sweep_walker_adaptive           import _adaptive_worker as _wk_worker
from simulations.sweep_humanoid_balance_adaptive import _adaptive_worker as _hb_worker


_CLUSTER_ADAPT_CLASS = 'TheoryStepAdaptation'


def _cp_make_args(cond, mismatch, rep, n_episodes, seed, mismatch_a):
    """Cartpole worker takes initial_theta as 5th arg.

    Reproduces sweep_cartpole_adaptive.run_adaptive_sweep's draw exactly:
    a single np.random.RandomState(SEED).uniform(-0.1, 0.1, size=n_episodes)
    shared across all (cond, mismatch) cells so that init pose varies only
    with rep — matching the single-job pipeline. Picking the rep-th value
    (rather than drawing from a fresh per-rep RNG) preserves bit-exact
    reproducibility against the in-process sweep.
    """
    init_thetas = np.random.RandomState(int(SEED)).uniform(-0.1, 0.1, size=n_episodes)
    return (cond['label'], mismatch, cond['recompute'], cond['adaptive'],
            float(init_thetas[rep]), mismatch_a, _CLUSTER_ADAPT_CLASS)


def _seed_make_args(cond, mismatch, rep, n_episodes, seed, mismatch_a):
    """Walker / humanoid workers take a per-cell seed as 5th arg.

    `rep` and `n_episodes` are accepted for signature parity with
    `_cp_make_args` but not used: walker/humanoid envs use a deterministic
    initial state (env.get_default_initial_state) and only the MPPI sampler
    needs randomness, which `seed` provides per (cond, mismatch, rep).
    """
    del rep, n_episodes
    return (cond['label'], mismatch, cond['recompute'], cond['adaptive'],
            seed, mismatch_a, _CLUSTER_ADAPT_CLASS)


ENV_CFG: dict[str, dict] = {
    'cartpole': dict(
        conditions  = _scs.CONDITIONS,
        mismatches  = _scs.MISMATCHES,
        n_episodes  = _scs.N_EPISODES,
        mismatch_a  = _scs.MISMATCH_A,
        worker      = _cp_worker,
        make_args   = _cp_make_args,
        cell_subdir = 'cartpole_summary_cells',
    ),
    'walker': dict(
        conditions  = _sws.CONDITIONS,
        mismatches  = _sws.MISMATCHES,
        n_episodes  = _sws.N_EPISODES,
        mismatch_a  = _sws.MISMATCH_A,
        worker      = _wk_worker,
        make_args   = _seed_make_args,
        cell_subdir = 'walker_summary_cells',
    ),
    'humanoid_balance': dict(
        conditions  = _shbs.CONDITIONS,
        mismatches  = _shbs.MISMATCHES,
        n_episodes  = _shbs.N_EPISODES,
        mismatch_a  = _shbs.MISMATCH_A,
        worker      = _hb_worker,
        make_args   = _seed_make_args,
        cell_subdir = 'humanoid_balance_summary_cells',
    ),
}

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')


def _cell_path(cell_subdir: str, cond_idx: int, mismatch: float, rep: int) -> str:
    cell_dir = os.path.join(RESULTS_DIR, cell_subdir)
    os.makedirs(cell_dir, exist_ok=True)
    return os.path.join(cell_dir, f'cell_c{cond_idx:02d}_m{mismatch:.4f}_r{rep:03d}.pkl')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('env', choices=list(ENV_CFG.keys()))
    parser.add_argument('--task-id', type=int, required=True)
    args = parser.parse_args()

    cfg        = ENV_CFG[args.env]
    conditions = cfg['conditions']
    mismatches = cfg['mismatches']
    n_episodes = cfg['n_episodes']
    mismatch_a = cfg['mismatch_a']
    worker     = cfg['worker']
    make_args  = cfg['make_args']
    cell_subdir = cfg['cell_subdir']

    n_m      = len(mismatches)
    cond_idx = args.task_id // n_m
    mism_idx = args.task_id  % n_m
    cond     = conditions[cond_idx]
    mismatch = mismatches[mism_idx]

    print(f'[{args.env}] task={args.task_id}  '
          f'cond="{cond["label"]}"  mismatch={mismatch}  '
          f'({n_episodes} reps)', flush=True)

    for rep in range(n_episodes):
        out_path = _cell_path(cell_subdir, cond_idx, mismatch, rep)
        if os.path.exists(out_path):
            print(f'  rep={rep:03d} exists, skipping', flush=True)
            continue

        seed = int(SEED) + cond_idx * 10_000 + mism_idx * 1_000 + rep
        np.random.seed(seed)  # set global RNG before calling worker

        worker_args = make_args(cond, mismatch, rep, n_episodes, seed, mismatch_a)
        result = worker(worker_args)
        (label_out, _, total_cost, duration_sec,
         failure_sec, n_recomp, rh_trace, cum_trace, last_states,
         rollout_steps) = result

        out = dict(
            env           = args.env,
            label         = label_out,
            condition_idx = cond_idx,
            mismatch      = float(mismatch),
            rep           = rep,
            total_cost    = float(total_cost),
            duration_sec  = float(duration_sec),
            failure_sec   = float(failure_sec),
            n_recomp      = int(n_recomp),
            rollout_steps = int(rollout_steps),
            rh_trace      = rh_trace,
            cum_trace     = cum_trace,
            last_states   = last_states,
        )
        tmp = out_path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(out, f)
        os.replace(tmp, out_path)
        print(f'  rep={rep:03d}  cost={total_cost:.2f}  fail={failure_sec:.2f}s',
              flush=True)


if __name__ == '__main__':
    main()
