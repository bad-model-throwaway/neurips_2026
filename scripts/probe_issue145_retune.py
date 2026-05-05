"""Retune probe: 2×2 (q_theta, N) at long-H cells, 3 seeds.

Disentangles cost-shape vs planner-budget contributions to the long-H
bimodal failure seen in the first robustness check. Also includes a
tolerance baseline at N=30 for reference.

Conditions at (94, 1) and (94, 10), 3 seeds, r=1.0:
    (quad q_theta=10, N=30)   — original proposal (already known: ~2/3 fail)
    (quad q_theta=10, N=60)   — pure N bump on original cost
    (quad q_theta=3,  N=30)   — pure cost retune
    (quad q_theta=3,  N=60)   — both changes
    (tolerance,       N=30)   — baseline
"""

import os
import sys
import time
import numpy as np
import mujoco

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from agents.mujoco_dynamics import MuJoCoCartPoleDynamics
from agents import mpc as mpc_mod

CELLS = [(94, 1), (94, 10)]
SEEDS = [0, 1, 2]
N_STEPS = 400

# (label, cost_type, Q_diag, N)
CONDS = [
    ('tol  N=30',         'tolerance', None,                   30),
    ('q10  N=30',         'quadratic', (1.0, 0.1, 10.0, 1.0),  30),
    ('q10  N=60',         'quadratic', (1.0, 0.1, 10.0, 1.0),  60),
    ('q3   N=30',         'quadratic', (1.0, 0.1,  3.0, 1.0),  30),
    ('q3   N=60',         'quadratic', (1.0, 0.1,  3.0, 1.0),  60),
]


def run_one(H, R, cost_type, Q_diag, N, seed):
    cfg = mpc_mod.PROPOSAL_CONFIGS['cartpole']
    prev = cfg.get('env_kwargs')
    ek = {'cost_type': cost_type}
    if Q_diag is not None:
        ek['Q_diag'] = Q_diag
    cfg['env_kwargs'] = ek
    try:
        rng = np.random.default_rng(seed)
        np.random.seed(seed)
        env = MuJoCoCartPoleDynamics(stateless=False, **ek)
        data = mujoco.MjData(env._mj_model)
        mujoco.mj_resetData(env._mj_model, data)
        mujoco.mj_forward(env._mj_model, data)
        state0 = env._state_from_data(data)
        state0[2] = rng.uniform(-0.1, 0.1)
        env.reset(state0)

        agent = mpc_mod.make_mpc('cartpole', H, R, N=N, mismatch_factor=1.0)
        costs = []
        for _ in range(N_STEPS):
            action = agent.interact(env.state, env.cost)
            env.step(action)
            costs.append(float(env.cost))
        return float(np.mean(costs))
    finally:
        if prev is None:
            cfg.pop('env_kwargs', None)
        else:
            cfg['env_kwargs'] = prev


def main():
    t0 = time.time()
    print(f"Retune probe — long-H, {len(SEEDS)} seeds, {N_STEPS} steps\n")
    print(f"{'cell':<10} {'cond':<12} " + " ".join(f"seed={s:<3}" for s in SEEDS)
          + f"  {'mean':>8} {'std':>8}  {'t(s)':>6}")
    for H, R in CELLS:
        for label, cost_type, Q_diag, N in CONDS:
            t_c = time.time()
            vals = [run_one(H, R, cost_type, Q_diag, N, s) for s in SEEDS]
            m, sd = float(np.mean(vals)), float(np.std(vals))
            dt = time.time() - t_c
            row = f"({H:>3},{R:>2})  {label:<12} " + " ".join(f"{v:>8.3f}" for v in vals)
            row += f"  {m:>8.3f} {sd:>8.3f}  {dt:>6.1f}"
            print(row)
        print()
    print(f"Total wall time: {time.time() - t0:.1f} s")


if __name__ == '__main__':
    main()
