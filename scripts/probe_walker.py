"""MJPC-shipped walker task probe with spline-PS at tiny-N.

Two speed_goal arms (0.0 stand-still, 1.5 walk) × N ∈ {10, 30, 100} ×
3 seeds = 18 episodes. Uses MJPC-aligned spline-PS (P=3 cubic spline
knots, sigma=0.5, matching MJPC's `sampling_exploration = 0.5`).
Agent horizon 0.8 s at ctrl_dt=0.01 (H=80, R=1) matches MJPC's
shipped walker task config.

Writes per-run rows to data/results/issue_66_walker_probe.csv and a
summary table to data/results/issue_66_walker_probe_summary.txt.
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


# H = 80 steps × 0.01 s = 0.8 s horizon (MJPC walker task.xml).
# n_steps = 1000 steps × 0.01 s = 10 s episode.
ARM = dict(H=80, R=1, n_steps=1000)

SPEED_GOALS = [0.0, 1.5]
N_VALUES = [10, 30, 100]
SEEDS = [0, 1, 2]

# MJPC shipped walker config: sampling_spline_points=3, sampling_exploration=0.5.
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
    N          = cfg['N']
    n_steps    = cfg['n_steps']
    seed       = cfg['seed']
    speed_goal = cfg['speed_goal']

    _np.random.seed(seed)

    # Walker XML timestep 0.0025 × n_substeps=4 → ctrl_dt = 0.01 (MJPC anchor).
    env = WalkerDynamics(stateless=False, speed_goal=speed_goal)
    env.reset(env.get_default_initial_state())

    model = WalkerDynamics(stateless=True, speed_goal=speed_goal)

    mj = model._mj_model
    ctrl_low  = _np.asarray(mj.actuator_ctrlrange[:, 0], dtype=float)
    ctrl_high = _np.asarray(mj.actuator_ctrlrange[:, 1], dtype=float)

    proposal = SplinePSProposal(
        action_dim=int(mj.nu), tsteps=H, n_samples=N, dt=env.dt,
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

    return {
        'env': 'walker',
        'speed_goal': speed_goal,
        'N': N, 'seed': seed,
        'H': H, 'R': R, 'n_steps': n_steps,
        'mean_cost':    mean_cost,
        'torso_z_mean': float(_np.mean(torso_z)),
        'torso_z_std':  float(_np.std(torso_z)),
        'mean_vx':      float(_np.mean(com_vx)),
    }


def _build_jobs():
    jobs = []
    for speed_goal in SPEED_GOALS:
        for N in N_VALUES:
            for seed in SEEDS:
                jobs.append({
                    'speed_goal': speed_goal,
                    'H': ARM['H'], 'R': ARM['R'],
                    'n_steps': ARM['n_steps'],
                    'N': N, 'seed': seed,
                    'repo_root': REPO_ROOT,
                })
    return jobs


_CSV_COLS = [
    'env', 'speed_goal', 'N', 'seed', 'H', 'R', 'n_steps',
    'mean_cost', 'torso_z_mean', 'torso_z_std', 'mean_vx',
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
        by_arm.setdefault((r['speed_goal'], r['N']), []).append(r)

    summary = {}
    keys = ['mean_cost', 'torso_z_mean', 'torso_z_std', 'mean_vx']
    for (speed_goal, N), arm_rows in by_arm.items():
        agg = {
            'speed_goal': speed_goal, 'N': N, 'n_seeds': len(arm_rows),
            'H': arm_rows[0]['H'], 'R': arm_rows[0]['R'],
            'n_steps': arm_rows[0]['n_steps'],
        }
        for k in keys:
            vals = np.array([r[k] for r in arm_rows], dtype=float)
            agg[f'{k}_mean'] = float(vals.mean())
            agg[f'{k}_std']  = float(vals.std(ddof=0))
        summary[(speed_goal, N)] = agg
    return summary


def _format_summary(summary):
    lines = []
    lines.append('Walker + MJPC-shipped spline-PS probe')
    lines.append('=' * 80)
    lines.append(f'Common config: H={ARM["H"]} (0.8 s), R={ARM["R"]}, '
                 f'n_steps={ARM["n_steps"]} (10 s), ctrl_dt=0.01')
    lines.append('Proposal:      P=3, sigma=0.5 (MJPC sampling_exploration), '
                 'interp=cubic, include_nominal=True, clip=True')
    lines.append(f'Seeds per arm: {len(SEEDS)} (seeds {", ".join(str(s) for s in SEEDS)}). '
                 'Reported as mean ± std across seeds.')
    lines.append('')

    cols = ['mean_cost', 'torso_z_mean', 'torso_z_std', 'mean_vx']
    for speed_goal in SPEED_GOALS:
        lines.append(f'## speed_goal = {speed_goal}  (height_goal = 1.2)')
        header = f'   {"N":>4s}  ' + '  '.join(f'{c:>26s}' for c in cols)
        lines.append(header)
        lines.append('   ' + '-' * (len(header) - 3))
        for N in N_VALUES:
            agg = summary[(speed_goal, N)]
            cells = []
            for c in cols:
                m = agg[f'{c}_mean']
                s = agg[f'{c}_std']
                cells.append(f'{m:>10.4f} ± {s:<10.4f}')
            lines.append(f'   {N:>4d}  ' + '  '.join(cells))
        lines.append('')

    return '\n'.join(lines)


def main():
    out_dir = os.path.join(REPO_ROOT, RESULTS_DIR.lstrip('./'))
    os.makedirs(out_dir, exist_ok=True)
    csv_path     = os.path.join(out_dir, 'issue_66_walker_probe.csv')
    summary_path = os.path.join(out_dir, 'issue_66_walker_probe_summary.txt')

    jobs = _build_jobs()
    print(f'Walker probe: {len(jobs)} jobs '
          f'({len(SPEED_GOALS)} speed_goals × {len(N_VALUES)} Ns × {len(SEEDS)} seeds)')

    rows = run_pool(_run_probe_episode, jobs, verbose=1)
    rows.sort(key=lambda r: (r['speed_goal'], r['N'], r['seed']))

    _write_csv(rows, csv_path)
    print(f'Wrote {csv_path}')

    summary = _summarize(rows)
    text = _format_summary(summary)
    with open(summary_path, 'w') as f:
        f.write(text + '\n')
    print(f'Wrote {summary_path}')
    print()
    print(text)


if __name__ == '__main__':
    main()
