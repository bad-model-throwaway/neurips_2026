"""Mismatch sanity probe for walker.

Checks whether walker degrades gracefully under a given misspecification
axis (the Figure 3 property). Fixed N=30, factors ∈ {1.0, 1.3, 1.6, 2.0},
both speed_goal arms, 3 seeds → 24 jobs.

Env stays at factor=1.0; only the planning model is mutated. MJPC-shipped
spline-PS config: σ=0.5, P=3 cubic, H=80, R=1, ctrl_dt=0.01.

CLI:
    --kind {torso_mass, foot_friction}   (default: torso_mass)

Writes data/results/issue_66_walker_mismatch_{kind}.csv and _summary.txt.
"""

import argparse
import csv
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs import RESULTS_DIR
from simulations.simulation import run_pool


ARM = dict(H=80, R=1, n_steps=1000)

SPEED_GOALS = [0.0, 1.5]
FACTORS = [1.0, 1.3, 1.6, 2.0]
N = 30
SEEDS = [0, 1, 2]

PROP_KW = dict(
    P=3, sigma=0.5, interp='cubic',
    include_nominal=True, clip=True,
)


def _run_probe_episode(cfg):
    import numpy as _np

    repo_root = cfg['repo_root']
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from agents.base import Agent
    from agents.mpc import (
        MPCEvaluation, SplinePSProposal, SplinePSArgminDecision,
    )
    from agents.mujoco_dynamics import WalkerDynamics
    from simulations.simulation import run_simulation

    H          = cfg['H']
    R          = cfg['R']
    N_         = cfg['N']
    n_steps    = cfg['n_steps']
    seed       = cfg['seed']
    speed_goal = cfg['speed_goal']
    factor     = cfg['factor']
    kind       = cfg['kind']

    _np.random.seed(seed)

    # Real env: factor=1.0 (unmodified).
    env = WalkerDynamics(stateless=False, speed_goal=speed_goal)
    env.reset(env.get_default_initial_state())

    # Planning model: selected axis scaled by `factor`.
    model = WalkerDynamics(stateless=True, speed_goal=speed_goal)
    model.apply_mismatch(factor, kind=kind)

    mj = model._mj_model
    ctrl_low  = _np.asarray(mj.actuator_ctrlrange[:, 0], dtype=float)
    ctrl_high = _np.asarray(mj.actuator_ctrlrange[:, 1], dtype=float)

    proposal = SplinePSProposal(
        action_dim=int(mj.nu), tsteps=H, n_samples=N_, dt=env.dt,
        ctrl_low=ctrl_low, ctrl_high=ctrl_high, **PROP_KW,
    )
    decision = SplinePSArgminDecision(proposal=proposal)
    agent = Agent(
        proposal, model, MPCEvaluation(), decision,
        parameters={'recompute_interval': R, 'horizon': H},
    )

    _, env_out, history = run_simulation(
        agent, env, n_steps=n_steps, interval=None,
    )
    states = history.get_item_history('state')

    # history['cost'] is env.cost at each snapshot, i.e. the ctrl-inclusive
    # cost produced by the prior env.step (or the reset cost at t=0), which
    # is what MJPC reports per timestep.
    mean_cost = float(_np.mean(history.get_item_history('cost')))

    torso_z = states[:, 18]   # sensor slot: actual torso_position[2]
    com_vx  = states[:, 20]   # sensor slot: torso_subtreelinvel[0]

    # "Did it fall?" heuristic: torso dropped below 0.7 m at any point.
    fell = bool(_np.any(torso_z < 0.7))

    return {
        'env': 'walker', 'kind': kind,
        'speed_goal': speed_goal, 'factor': factor,
        'N': N_, 'seed': seed,
        'H': H, 'R': R, 'n_steps': n_steps,
        'mean_cost':    mean_cost,
        'torso_z_mean': float(_np.mean(torso_z)),
        'torso_z_std':  float(_np.std(torso_z)),
        'torso_z_min':  float(_np.min(torso_z)),
        'mean_vx':      float(_np.mean(com_vx)),
        'fell':         fell,
    }


def _build_jobs(kind):
    jobs = []
    for speed_goal in SPEED_GOALS:
        for factor in FACTORS:
            for seed in SEEDS:
                jobs.append({
                    'kind': kind,
                    'speed_goal': speed_goal, 'factor': factor,
                    'H': ARM['H'], 'R': ARM['R'],
                    'n_steps': ARM['n_steps'],
                    'N': N, 'seed': seed,
                    'repo_root': REPO_ROOT,
                })
    return jobs


_CSV_COLS = [
    'env', 'kind', 'speed_goal', 'factor', 'N', 'seed', 'H', 'R', 'n_steps',
    'mean_cost', 'torso_z_mean', 'torso_z_std', 'torso_z_min',
    'mean_vx', 'fell',
]


def _write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in _CSV_COLS})


def _summarize(rows):
    by_arm = {}
    for r in rows:
        by_arm.setdefault((r['speed_goal'], r['factor']), []).append(r)

    summary = {}
    keys = ['mean_cost', 'torso_z_mean', 'torso_z_std', 'torso_z_min', 'mean_vx']
    for (speed_goal, factor), arm_rows in by_arm.items():
        agg = {'speed_goal': speed_goal, 'factor': factor,
               'n_seeds': len(arm_rows),
               'n_fell':  sum(1 for r in arm_rows if r['fell'])}
        for k in keys:
            vals = np.array([r[k] for r in arm_rows], dtype=float)
            agg[f'{k}_mean'] = float(vals.mean())
            agg[f'{k}_std']  = float(vals.std(ddof=0))
        summary[(speed_goal, factor)] = agg
    return summary


def _format_summary(summary, kind):
    lines = []
    lines.append(f'Walker mismatch probe — {kind} factor')
    lines.append('=' * 80)
    lines.append(f'Config: N={N}, H={ARM["H"]} (0.8 s), R={ARM["R"]}, '
                 f'n_steps={ARM["n_steps"]} (10 s), σ=0.5, P=3 cubic')
    lines.append(f'Seeds per arm: {len(SEEDS)}. Reported mean ± std across seeds. '
                 'n_fell counts seeds where torso_z<0.7 m ever occurred.')
    lines.append('')

    cols = ['mean_cost', 'torso_z_mean', 'torso_z_min', 'mean_vx']
    for speed_goal in SPEED_GOALS:
        lines.append(f'## speed_goal = {speed_goal}')
        header = (f'   {"factor":>6s}  {"n_fell":>6s}  '
                  + '  '.join(f'{c:>26s}' for c in cols))
        lines.append(header)
        lines.append('   ' + '-' * (len(header) - 3))
        for factor in FACTORS:
            agg = summary[(speed_goal, factor)]
            cells = []
            for c in cols:
                m = agg[f'{c}_mean']
                s = agg[f'{c}_std']
                cells.append(f'{m:>10.4f} ± {s:<10.4f}')
            lines.append(f'   {factor:>6.2f}  {agg["n_fell"]:>3d}/{agg["n_seeds"]:<2d}  '
                         + '  '.join(cells))
        lines.append('')

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', default='torso_mass',
                        choices=('torso_mass', 'foot_friction'),
                        help='Misspecification axis to probe.')
    args = parser.parse_args()
    kind = args.kind

    out_dir = os.path.join(REPO_ROOT, RESULTS_DIR.lstrip('./'))
    os.makedirs(out_dir, exist_ok=True)
    csv_path     = os.path.join(out_dir, f'issue_66_walker_mismatch_{kind}.csv')
    summary_path = os.path.join(out_dir, f'issue_66_walker_mismatch_{kind}_summary.txt')

    jobs = _build_jobs(kind)
    print(f'Walker mismatch probe ({kind}): {len(jobs)} jobs '
          f'({len(SPEED_GOALS)} speed_goals × {len(FACTORS)} factors × {len(SEEDS)} seeds)')

    rows = run_pool(_run_probe_episode, jobs, verbose=1)
    rows.sort(key=lambda r: (r['speed_goal'], r['factor'], r['seed']))

    _write_csv(rows, csv_path)
    print(f'Wrote {csv_path}')

    summary = _summarize(rows)
    text = _format_summary(summary, kind)
    with open(summary_path, 'w') as f:
        f.write(text + '\n')
    print(f'Wrote {summary_path}')
    print()
    print(text)


if __name__ == '__main__':
    main()
