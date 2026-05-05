"""H×R pilot probe for walker running arm.

Explores how walker cost behaves across planning horizons and replan
intervals under no mismatch and one mismatch level, before committing to a
final H range for SMOKE_GRIDS['walker']. Running arm only (speed_goal=1.5,
where torso-mass mismatch produces a clean monotone cost swing).

Grid: H × R × factor × reps = 6 × 3 × 2 × 1 = 36 jobs.
Env stays at factor=1.0; only the planning model is mutated
(torso_mass). MJPC-shipped spline-PS config: σ=0.5, P=3 cubic, N=30,
ctrl_dt=0.01, n_steps=1000.

Writes data/results/issue_58_walker_HR_grid.csv and _summary.txt.
"""

import csv
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs import RESULTS_DIR
from simulations.simulation import run_pool


SPEED_GOAL = 1.5
N_STEPS = 1000
N = 30
KIND = 'torso_mass'

HS = [20, 50, 80, 120, 170, 220]
RS = [1, 4, 8]
FACTORS = [1.0, 2.0]
SEEDS = [0]

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

    env = WalkerDynamics(stateless=False, speed_goal=speed_goal)
    env.reset(env.get_default_initial_state())

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

    mean_cost = float(_np.mean(history.get_item_history('cost')))

    torso_z = states[:, 18]
    com_vx  = states[:, 20]

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


def _build_jobs():
    jobs = []
    for H in HS:
        for R in RS:
            for factor in FACTORS:
                for seed in SEEDS:
                    jobs.append({
                        'kind': KIND,
                        'speed_goal': SPEED_GOAL, 'factor': factor,
                        'H': H, 'R': R,
                        'n_steps': N_STEPS,
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


def _format_summary(rows):
    by_key = {(r['H'], r['R'], r['factor']): r for r in rows}

    lines = []
    lines.append(f'Walker H×R pilot probe — torso-mass mismatch')
    lines.append('=' * 80)
    lines.append(f'Config: N={N}, n_steps={N_STEPS} (10 s), '
                 f'speed_goal={SPEED_GOAL}, reps=1 (seed={SEEDS[0]}), '
                 f'σ=0.5, P=3 cubic')
    lines.append(f'H grid: {HS}')
    lines.append(f'R grid: {RS}')
    lines.append(f'factors: {FACTORS}')
    lines.append('')

    hr_label = 'H\\R'

    for factor in FACTORS:
        lines.append(f'## factor = {factor}  (mean_cost)')
        header = '   ' + f'{hr_label:>6s}' + '  ' + '  '.join(f'{R:>10d}' for R in RS)
        lines.append(header)
        lines.append('   ' + '-' * (len(header) - 3))
        for H in HS:
            cells = []
            for R in RS:
                r = by_key.get((H, R, factor))
                cells.append(f'{r["mean_cost"]:>10.4f}' if r else f'{"-":>10s}')
            lines.append(f'   {H:>6d}  ' + '  '.join(cells))
        lines.append('')

    lines.append('## fell flag (torso_z<0.7 m ever) — "." = stood, "x" = fell')
    for factor in FACTORS:
        lines.append(f'factor = {factor}')
        header = '   ' + f'{hr_label:>6s}' + '  ' + '  '.join(f'{R:>3d}' for R in RS)
        lines.append(header)
        for H in HS:
            cells = []
            for R in RS:
                r = by_key.get((H, R, factor))
                cells.append('  x' if (r and r['fell']) else '  .')
            lines.append(f'   {H:>6d}  ' + '  '.join(cells))
        lines.append('')

    lines.append('## mean_vx (target = 1.0)')
    for factor in FACTORS:
        lines.append(f'factor = {factor}')
        header = '   ' + f'{hr_label:>6s}' + '  ' + '  '.join(f'{R:>10d}' for R in RS)
        lines.append(header)
        lines.append('   ' + '-' * (len(header) - 3))
        for H in HS:
            cells = []
            for R in RS:
                r = by_key.get((H, R, factor))
                cells.append(f'{r["mean_vx"]:>10.4f}' if r else f'{"-":>10s}')
            lines.append(f'   {H:>6d}  ' + '  '.join(cells))
        lines.append('')

    return '\n'.join(lines)


def main():
    out_dir = os.path.join(REPO_ROOT, RESULTS_DIR.lstrip('./'))
    os.makedirs(out_dir, exist_ok=True)
    csv_path     = os.path.join(out_dir, 'issue_58_walker_HR_grid.csv')
    summary_path = os.path.join(out_dir, 'issue_58_walker_HR_grid_summary.txt')

    jobs = _build_jobs()
    print(f'Walker H×R pilot probe: {len(jobs)} jobs '
          f'({len(HS)} H × {len(RS)} R × {len(FACTORS)} factors × '
          f'{len(SEEDS)} seeds)')

    rows = run_pool(_run_probe_episode, jobs, verbose=1)
    rows.sort(key=lambda r: (r['factor'], r['H'], r['R']))

    _write_csv(rows, csv_path)
    print(f'Wrote {csv_path}')

    text = _format_summary(rows)
    with open(summary_path, 'w') as f:
        f.write(text + '\n')
    print(f'Wrote {summary_path}')
    print()
    print(text)


if __name__ == '__main__':
    main()
