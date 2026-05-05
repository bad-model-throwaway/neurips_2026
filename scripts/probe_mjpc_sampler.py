"""MJPC-aligned spline-PS probe for cartpole.

Runs 1 env × N ∈ {10, 30, 100} × 3 seeds = 9 episodes.

Writes per-run rows to data/results/issue_60_probe.csv and a summary
table to data/results/issue_60_probe_summary.txt.
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


ARMS = {
    'cartpole': dict(H=50, R=3, n_steps=300),
}

N_VALUES = [10, 30, 100]
SEEDS = [0, 1, 2]

PROP_KW = dict(
    P=3, sigma=0.1, interp='cubic',
    include_nominal=True, clip=True,
)


def _run_probe_episode(cfg):
    import numpy as _np
    import mujoco as _mujoco

    repo_root = cfg['repo_root']
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from agents.mpc import make_mpc
    from agents.mujoco_dynamics import MuJoCoCartPoleDynamics
    from simulations.simulation import run_simulation

    env_name = cfg['env_name']
    H        = cfg['H']
    R        = cfg['R']
    N        = cfg['N']
    n_steps  = cfg['n_steps']
    seed     = cfg['seed']

    _np.random.seed(seed)

    env  = MuJoCoCartPoleDynamics(stateless=False)
    data = _mujoco.MjData(env._mj_model)
    _mujoco.mj_resetData(env._mj_model, data)
    state0 = env._state_from_data(data)
    state0[2] = 0.05  # mirrors tests/test_cartpole.py::_make_env
    env.reset(state0)
    agent = make_mpc(
        env_name, H, R, N=N,
        proposal='spline_ps', decision='spline_ps_argmin',
        proposal_kwargs=dict(PROP_KW),
    )

    _, env_out, history = run_simulation(
        agent, env, n_steps=n_steps, interval=None,
    )
    states = history.get_item_history('state')

    mean_cost = float(_np.mean([env_out.cost_function(x) for x in states]))

    row = {
        'env': env_name, 'N': N, 'seed': seed,
        'H': H, 'R': R, 'n_steps': n_steps,
        'mean_cost': mean_cost,
    }

    theta = states[:, 2]
    row['max_theta_deg'] = float(_np.max(_np.abs(_np.rad2deg(theta))))
    row['max_abs_x']     = float(_np.max(_np.abs(states[:, 0])))

    return row


def _build_jobs():
    jobs = []
    for env_name, spec in ARMS.items():
        for N in N_VALUES:
            for seed in SEEDS:
                jobs.append({
                    'env_name': env_name,
                    'H': spec['H'], 'R': spec['R'],
                    'n_steps': spec['n_steps'],
                    'N': N, 'seed': seed,
                    'repo_root': REPO_ROOT,
                })
    return jobs


def _csv_columns():
    return [
        'env', 'N', 'seed', 'H', 'R', 'n_steps', 'mean_cost',
        'max_theta_deg', 'max_abs_x',
    ]


def _write_csv(rows, path):
    cols = _csv_columns()
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in cols})


def _summarize(rows):
    """Group by (env, N), compute mean ± std across seeds. Returns dict."""
    by_arm = {}
    for r in rows:
        by_arm.setdefault((r['env'], r['N']), []).append(r)

    summary = {}
    for (env, N), arm_rows in by_arm.items():
        agg = {
            'env': env, 'N': N, 'n_seeds': len(arm_rows),
            'H': arm_rows[0]['H'], 'R': arm_rows[0]['R'],
            'n_steps': arm_rows[0]['n_steps'],
        }
        keys = ['mean_cost', 'max_theta_deg', 'max_abs_x']
        for k in keys:
            vals = np.array([r[k] for r in arm_rows], dtype=float)
            agg[f'{k}_mean'] = float(vals.mean())
            agg[f'{k}_std']  = float(vals.std(ddof=0))
        summary[(env, N)] = agg
    return summary


def _format_summary(summary):
    lines = []
    lines.append('MJPC-aligned spline-PS probe')
    lines.append('=' * 72)
    lines.append('Common config: P=3, sigma=0.1, interp=cubic, include_nominal=True, clip=True')
    lines.append('Seeds per arm: 3 (seeds 0, 1, 2). Reported as mean ± std.')
    lines.append('')

    for env in ('cartpole',):
        spec = ARMS[env]
        lines.append(f'## {env}  (H={spec["H"]}, R={spec["R"]}, n_steps={spec["n_steps"]})')
        lines.append('   ctrl_dt=0.02')
        cols = ['mean_cost', 'max_theta_deg', 'max_abs_x']

        header = f'   {"N":>4s}  ' + '  '.join(f'{c:>26s}' for c in cols)
        lines.append(header)
        lines.append('   ' + '-' * (len(header) - 3))
        for N in N_VALUES:
            agg = summary[(env, N)]
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
    csv_path = os.path.join(out_dir, 'issue_60_probe.csv')
    summary_path = os.path.join(out_dir, 'issue_60_probe_summary.txt')

    jobs = _build_jobs()
    print(f'Probe: {len(jobs)} jobs '
          f'({len(ARMS)} envs × {len(N_VALUES)} Ns × {len(SEEDS)} seeds)')

    rows = run_pool(_run_probe_episode, jobs, verbose=1)

    rows.sort(key=lambda r: (r['env'], r['N'], r['seed']))
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
