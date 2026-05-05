"""Robustness check: 3 reps at the long-H cells that diverged.

Answers: is the (94,1) quadratic-cost disaster robust across seeds, or
single-rep noise? Compares to tolerance at the same seeds.
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
N_STEPS = 400
SEEDS = [0, 1, 2]


def run_one(H, R, cost_type, seed):
    cfg = mpc_mod.PROPOSAL_CONFIGS['cartpole']
    prev = cfg.get('env_kwargs')
    cfg['env_kwargs'] = {'cost_type': cost_type}
    try:
        rng = np.random.default_rng(seed)
        np.random.seed(seed)
        env = MuJoCoCartPoleDynamics(stateless=False, cost_type=cost_type)
        data = mujoco.MjData(env._mj_model)
        mujoco.mj_resetData(env._mj_model, data)
        mujoco.mj_forward(env._mj_model, data)
        state0 = env._state_from_data(data)
        state0[2] = rng.uniform(-0.1, 0.1)
        env.reset(state0)

        agent = mpc_mod.make_mpc('cartpole', H, R, mismatch_factor=1.0)
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
    print(f"Robustness probe — long-H cells, {len(SEEDS)} reps, {N_STEPS} steps")
    print(f"{'cell':<10} {'cost_type':<12} " + " ".join(f"seed={s:<3}" for s in SEEDS)
          + f"  {'mean':>8} {'std':>8}")
    for H, R in CELLS:
        for cost_type in ('tolerance', 'quadratic'):
            vals = [run_one(H, R, cost_type, s) for s in SEEDS]
            m, sd = float(np.mean(vals)), float(np.std(vals))
            row = f"({H:>3},{R:>2})  {cost_type:<12} " + " ".join(f"{v:>8.3f}" for v in vals)
            row += f"  {m:>8.3f} {sd:>8.3f}"
            print(row)
    print(f"\nTotal wall time: {time.time() - t0:.1f} s")


if __name__ == '__main__':
    main()
