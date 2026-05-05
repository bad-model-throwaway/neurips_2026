"""Aggressive retune probe.

Tests whether strict-quadratic cost can be made planner-friendly at long H
by softening q_theta and giving the spline-PS sampler more resolution.

Aggressive config:
    Q = diag(1, 0.1, 1, 1), R = 0.1    (q_theta from 3 -> 1)
    N = 60                             (vs default 30)
    P = 5 spline knots                 (vs default 3 — finer control over H=94)
    sigma = 0.1                        (vs default 0.3 — tighter exploration)

Reference: tolerance at the cartpole default proposal (N=30, P=3, sigma=0.3).
Cells: (53,1) [sanity — must stay low], (94,1), (94,10). 3 seeds each.
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

CELLS = [(53, 1), (94, 1), (94, 10)]
SEEDS = [0, 1, 2]
N_STEPS = 400

# (label, cost_type, Q_diag, N, proposal_kwargs)
CONDS = [
    ('tol default', 'tolerance', None,
     30, None),
    ('q1 P5 s0.1',  'quadratic', (1.0, 0.1, 1.0, 1.0),
     60, dict(P=5, sigma=0.1, interp='cubic', include_nominal=True, clip=True)),
]


def run_one(H, R, cost_type, Q_diag, N, prop_kw, seed):
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

        agent = mpc_mod.make_mpc(
            'cartpole', H, R, N=N, mismatch_factor=1.0,
            proposal_kwargs=prop_kw,
        )
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
    print(f"Aggressive retune probe — {len(SEEDS)} seeds, {N_STEPS} steps\n")
    print(f"{'cell':<10} {'cond':<14} " + " ".join(f"seed={s:<3}" for s in SEEDS)
          + f"  {'mean':>8} {'std':>8}  {'t(s)':>6}")
    for H, R in CELLS:
        for label, ct, Q, N, pk in CONDS:
            t_c = time.time()
            vals = [run_one(H, R, ct, Q, N, pk, s) for s in SEEDS]
            m, sd = float(np.mean(vals)), float(np.std(vals))
            dt = time.time() - t_c
            row = f"({H:>3},{R:>2})  {label:<14} " + " ".join(f"{v:>8.3f}" for v in vals)
            row += f"  {m:>8.3f} {sd:>8.3f}  {dt:>6.1f}"
            print(row)
        print()
    print(f"Total wall time: {time.time() - t0:.1f} s")


if __name__ == '__main__':
    main()
