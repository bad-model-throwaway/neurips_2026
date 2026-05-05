"""Humanoid GRAVITY mismatch probe, balance or stand-up.

Pass --mode {balance, standup}. Default balance. For stand-up we use
n_steps=400 (6 s) to complete the get-up phase; for balance 300 (4.5 s).

Outputs (suffix depends on mode):
  data/results/issue_134_probe_<mode>_gravity.csv
  data/results/issue_134_probe_<mode>_gravity_summary.txt
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


N_FIX          = 30
N_STEPS_BY_MODE = {'balance': 300, 'standup': 400}

# Default factor set per mode:
#   balance: two-direction decision sweep (done; g>1 won).
#   standup: g>1 only, mirrors the winning balance range for comparability.
FACTORS_BY_MODE = {
    'balance': [0.5, 0.75, 1.0, 1.33, 2.0],
    'standup': [1.0, 1.25, 1.5, 1.75],
}

H_GRID = [20, 40, 80]
R_GRID = [1, 4]
SEEDS  = [0, 1, 2]

STOOD_TOL = 0.1


def _run_gravity_episode(cfg):
    import numpy as _np

    repo_root = cfg['repo_root']
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from agents.base import Agent
    from agents.mpc import MPCEvaluation, SplinePSProposal, SplinePSArgminDecision
    from agents.mujoco_dynamics import HumanoidStandDynamics
    from simulations.simulation import run_simulation
    from configs import ENV_DT

    _np.random.seed(cfg['seed'])

    env = HumanoidStandDynamics(
        stateless=False, mode=cfg['mode'], height_goal=cfg['height_goal']
    )
    env.reset(env.get_default_initial_state())

    model = HumanoidStandDynamics(
        stateless=True, mode=cfg['mode'], height_goal=cfg['height_goal']
    )
    if cfg['gravity_factor'] != 1.0:
        model.apply_mismatch(cfg['gravity_factor'], kind='gravity')

    mj = model._mj_model
    ctrl_low = _np.asarray(mj.actuator_ctrlrange[:, 0], dtype=float)
    ctrl_high = _np.asarray(mj.actuator_ctrlrange[:, 1], dtype=float)
    env_dt = ENV_DT['humanoid_stand']

    proposal = SplinePSProposal(
        action_dim=21, tsteps=cfg['H'], n_samples=cfg['N'], dt=env_dt,
        ctrl_low=ctrl_low, ctrl_high=ctrl_high,
        P=3, sigma=0.25, interp='cubic', include_nominal=True, clip=True,
    )
    decision = SplinePSArgminDecision(proposal=proposal)
    agent = Agent(
        proposal, model, MPCEvaluation(), decision,
        parameters={'recompute_interval': cfg['R'], 'horizon': cfg['H']},
    )

    t0 = time.time()
    _, env_out, history = run_simulation(
        agent, env, n_steps=cfg['n_steps'], interval=None,
    )
    wallclock_s = time.time() - t0

    states  = history.get_item_history('state')
    actions = history.get_item_history('action')

    head_z     = states[:, env.HEAD_Z_IDX]
    feet_avg_z = states[:, env.FEET_AVG_Z_IDX]
    r_height   = head_z - feet_avg_z - cfg['height_goal']

    costs = _np.array([env_out.cost_function(x, ctrl=a)
                       for x, a in zip(states[1:], actions)])

    stood_mask = _np.abs(r_height) <= STOOD_TOL
    fell = bool(head_z[-1] < 1.0)

    return {
        'H': cfg['H'], 'R': cfg['R'], 'N': cfg['N'],
        'seed': cfg['seed'],
        'gravity_factor': cfg['gravity_factor'],
        'n_steps': cfg['n_steps'],
        'mean_cost':      float(_np.mean(costs)),
        'final_head_z':   float(head_z[-1]),
        'min_head_z':     float(_np.min(head_z)),
        'final_r_height': float(r_height[-1]),
        'stood_frac':     float(_np.mean(stood_mask)),
        'fell':           fell,
        'wallclock_s':    wallclock_s,
    }


def _build_jobs(mode, factors, n_steps):
    jobs = []
    for g in factors:
        for H in H_GRID:
            for R in R_GRID:
                for seed in SEEDS:
                    jobs.append({
                        'H': H, 'R': R, 'N': N_FIX,
                        'seed': seed,
                        'n_steps': n_steps,
                        'gravity_factor': g,
                        'mode': mode,
                        'height_goal': 1.4,
                        'repo_root': REPO_ROOT,
                    })
    return jobs


_CSV_COLS = [
    'H', 'R', 'N', 'seed', 'gravity_factor', 'n_steps',
    'mean_cost', 'final_head_z', 'min_head_z', 'final_r_height',
    'stood_frac', 'fell', 'wallclock_s',
]


def _write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in _CSV_COLS})


def _format_summary(rows, mode, factors, n_steps):
    lines = [
        f'Humanoid {mode.upper()}, GRAVITY mismatch',
        '=' * 78,
        f'N={N_FIX}, ctrl_dt=0.015, n_steps={n_steps} ({n_steps*0.015:.2f} s), seeds={SEEDS}.',
        f'H ∈ {H_GRID}, R ∈ {R_GRID}, g ∈ {factors}.',
        'factor > 1 : planner expects stronger gravity than real.',
        'factor < 1 : planner expects weaker gravity than real.',
        '"fell" = final_head_z < 1.0 m.',
        '',
    ]
    by_g = {}
    for row in rows:
        by_g.setdefault(row['gravity_factor'], []).append(row)
    for g in sorted(by_g):
        arm = by_g[g]
        lines.append(f'-- g_factor = {g} ' + '-' * 55)
        lines.append('  mean_cost (mean ± sd over seeds)')
        lines.append('    H\\R   ' + '  '.join(f'R={R:<10d}' for R in R_GRID))
        for H in H_GRID:
            parts = [f'    H={H:<3d}']
            for R in R_GRID:
                cells = [x for x in arm if x['H'] == H and x['R'] == R]
                arr = np.array([c['mean_cost'] for c in cells], dtype=float)
                parts.append(f'{arr.mean():>6.2f}±{arr.std(ddof=0):<4.2f}  ')
            lines.append(''.join(parts))
        lines.append('  stood_frac (mean over seeds)')
        lines.append('    H\\R   ' + '  '.join(f'R={R:<10d}' for R in R_GRID))
        for H in H_GRID:
            parts = [f'    H={H:<3d}']
            for R in R_GRID:
                cells = [x for x in arm if x['H'] == H and x['R'] == R]
                arr = np.array([c['stood_frac'] for c in cells], dtype=float)
                parts.append(f'{arr.mean():>6.3f}      ')
            lines.append(''.join(parts))
        lines.append(f'  fell count (out of {len(SEEDS)} seeds)')
        lines.append('    H\\R   ' + '  '.join(f'R={R:<10d}' for R in R_GRID))
        for H in H_GRID:
            parts = [f'    H={H:<3d}']
            for R in R_GRID:
                cells = [x for x in arm if x['H'] == H and x['R'] == R]
                nf = sum(1 for c in cells if c['fell'])
                parts.append(f'{nf}/{len(cells):<9d}   ')
            lines.append(''.join(parts))
        lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['balance', 'standup'], default='balance')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    n_steps = N_STEPS_BY_MODE[args.mode]
    factors = FACTORS_BY_MODE[args.mode]

    out_dir = os.path.join(REPO_ROOT, RESULTS_DIR.lstrip('./'))
    os.makedirs(out_dir, exist_ok=True)

    jobs = _build_jobs(args.mode, factors, n_steps)
    if args.dry_run:
        jobs = jobs[:1]

    print(f'Probe gravity (mode={args.mode}): {len(jobs)} jobs')
    t0 = time.time()
    rows = run_pool(_run_gravity_episode, jobs, verbose=1)
    print(f'Total wallclock: {time.time() - t0:.1f}s')

    rows.sort(key=lambda r: (r['gravity_factor'], r['H'], r['R'], r['seed']))

    csv_path = os.path.join(out_dir, f'issue_134_probe_{args.mode}_gravity.csv')
    sum_path = os.path.join(out_dir, f'issue_134_probe_{args.mode}_gravity_summary.txt')

    _write_csv(rows, csv_path)
    print(f'Wrote {csv_path}')

    text = _format_summary(rows, args.mode, factors, n_steps)
    with open(sum_path, 'w') as f:
        f.write(text)
    print(f'Wrote {sum_path}')
    print()
    print(text)


if __name__ == '__main__':
    main()
