"""Aggregate per-cell pickles into canonical summary_<env>.pkl.

Reads every pickle under results/<env>_summary_cells/ written by run_one_cell.py
and assembles the dict-of-dict-of-list schema used by visualization/figures.py:

    mismatches, mismatch_a
    sweep_cost[label][mismatch]    — list of total_cost per episode
    sweep_failure[label][mismatch] — list of failure_sec per episode
    sweep_len[label][mismatch]     — list of duration_sec per episode
    sweep_recomp[label][mismatch]  — list of n_recomputations per episode
    sweep_rh_traces[label]         — list of recompute-interval traces (mismatch_a only)
    sweep_cum_traces[label]        — list of cumulative-cost traces (mismatch_a only)

The output pkl is written to results/ and then should be copied to
data/results/ to be picked up by render_figures.py.

Usage:
    python aggregate.py --env cartpole
    python aggregate.py --env walker
    python aggregate.py --env humanoid_balance

After aggregation, copy to data/results/:
    cp results/summary_<env>.pkl ../../data/results/
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

from simulations import sweep_cartpole_summary           as _scs
from simulations import sweep_walker_summary             as _sws
from simulations import sweep_humanoid_balance_summary   as _shbs

ENV_CFG: dict[str, dict] = {
    'cartpole': dict(
        conditions  = _scs.CONDITIONS,
        mismatches  = _scs.MISMATCHES,
        n_episodes  = _scs.N_EPISODES,
        mismatch_a  = _scs.MISMATCH_A,
        cell_subdir = 'cartpole_summary_cells',
        out_name    = 'summary_cartpole.pkl',
    ),
    'walker': dict(
        conditions  = _sws.CONDITIONS,
        mismatches  = _sws.MISMATCHES,
        n_episodes  = _sws.N_EPISODES,
        mismatch_a  = _sws.MISMATCH_A,
        cell_subdir = 'walker_summary_cells',
        out_name    = 'summary_walker.pkl',
    ),
    'humanoid_balance': dict(
        conditions  = _shbs.CONDITIONS,
        mismatches  = _shbs.MISMATCHES,
        n_episodes  = _shbs.N_EPISODES,
        mismatch_a  = _shbs.MISMATCH_A,
        cell_subdir = 'humanoid_balance_summary_cells',
        out_name    = 'summary_humanoid_balance.pkl',
    ),
}

RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', required=True, choices=list(ENV_CFG.keys()))
    args = parser.parse_args()
    cfg = ENV_CFG[args.env]

    conditions  = cfg['conditions']
    mismatches  = cfg['mismatches']
    n_episodes  = cfg['n_episodes']
    mismatch_a  = cfg['mismatch_a']
    cell_subdir = cfg['cell_subdir']

    labels = [c['label'] for c in conditions]
    n_cond = len(conditions)
    n_m    = len(mismatches)

    # Schema mirrors sweep_*_adaptive.run_adaptive_sweep.
    sweep_len            = {lab: {m: [] for m in mismatches} for lab in labels}
    sweep_cost           = {lab: {m: [] for m in mismatches} for lab in labels}
    sweep_failure        = {lab: {m: [] for m in mismatches} for lab in labels}
    sweep_recomp         = {lab: {m: [] for m in mismatches} for lab in labels}
    sweep_rollout_steps  = {lab: {m: [] for m in mismatches} for lab in labels}
    sweep_last_states    = {lab: {m: [] for m in mismatches} for lab in labels}
    sweep_rh_traces      = {lab: [] for lab in labels}
    sweep_cum_traces     = {lab: [] for lab in labels}

    cell_dir = os.path.join(RESULTS_DIR, cell_subdir)
    if not os.path.isdir(cell_dir):
        sys.exit(f'no cell directory at {cell_dir} — run submit_all.sh first')

    valid_mismatches = {float(m) for m in mismatches}

    n_loaded = n_skipped = 0
    for fname in sorted(os.listdir(cell_dir)):
        if not fname.endswith('.pkl') or not fname.startswith('cell_'):
            continue
        path = os.path.join(cell_dir, fname)
        with open(path, 'rb') as f:
            c = pickle.load(f)

        lab  = c['label']
        mism = float(c['mismatch'])
        if lab not in sweep_cost or mism not in valid_mismatches:
            n_skipped += 1
            continue
        mism_key = min(mismatches, key=lambda m: abs(float(m) - mism))

        sweep_len[lab][mism_key].append(c['duration_sec'])
        sweep_cost[lab][mism_key].append(c['total_cost'])
        sweep_failure[lab][mism_key].append(c['failure_sec'])
        sweep_recomp[lab][mism_key].append(c['n_recomp'])
        sweep_rollout_steps[lab][mism_key].append(c.get('rollout_steps', 0))
        if c.get('last_states') is not None:
            sweep_last_states[lab][mism_key].append(c['last_states'])
        if c.get('rh_trace') is not None:
            sweep_rh_traces[lab].append(c['rh_trace'])
        if c.get('cum_trace') is not None:
            sweep_cum_traces[lab].append(c['cum_trace'])
        n_loaded += 1

    total_expected = n_cond * n_m * n_episodes
    print(f'Loaded {n_loaded} / {total_expected} expected pickles '
          f'({n_skipped} skipped).')

    missing_task_ids = []
    for ci, cond in enumerate(conditions):
        lab = cond['label']
        for mi, m in enumerate(mismatches):
            n = len(sweep_cost[lab][m])
            if n < n_episodes:
                task_id = ci * n_m + mi
                missing_task_ids.append(str(task_id))
                print(f'  incomplete: "{lab[:20]}"  m={m}  '
                      f'{n}/{n_episodes}  (task_id={task_id})')

    if missing_task_ids:
        print()
        print('Resubmit missing slots:')
        print(f'  sbatch --array={",".join(missing_task_ids)} '
              f'run_one_cell.sh {args.env}')
        print()

    out = dict(
        mismatches          = mismatches,
        mismatch_a          = mismatch_a,
        sweep_len           = sweep_len,
        sweep_cost          = sweep_cost,
        sweep_failure       = sweep_failure,
        sweep_recomp        = sweep_recomp,
        sweep_rollout_steps = sweep_rollout_steps,
        sweep_last_states   = sweep_last_states,
        sweep_rh_traces     = sweep_rh_traces,
        sweep_cum_traces    = sweep_cum_traces,
        env                 = args.env,
    )

    out_path = os.path.join(RESULTS_DIR, cfg['out_name'])
    tmp_path = out_path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(out, f)
    os.replace(tmp_path, out_path)
    print(f'Wrote {out_path}')
    print(f'Deploy: cp {out_path} {REPO_ROOT}/data/results/{cfg["out_name"]}')


if __name__ == '__main__':
    main()
