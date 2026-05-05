"""Spot check: compare tolerance vs quadratic cost on 5 cells.

Runs 5 (H, R) cells at matched dynamics (r=1.0), 1 rep each, under both the
current tolerance-product cost and the proposed quadratic cost. Prints a
side-by-side table. Purpose: verify the quadratic-cost heatmap would preserve
the qualitative argmin-R / inverted-U structure before committing to the
full local grid.

Cells: (H, R) ∈ {(30,1), (53,1), (94,1), (53,5), (94,10)}.
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

CELLS = [(30, 1), (53, 1), (94, 1), (53, 5), (94, 10)]
N_STEPS = 400   # matches DEFAULT_GRIDS['cartpole']
SEED = 0


def run_one(H, R, cost_type):
    # Inject cost_type into PROPOSAL_CONFIGS so make_mpc builds both the real
    # env and the planning model with matching cost. (cartpole cfg has no
    # env_kwargs by default; we add one transiently here.)
    cfg = mpc_mod.PROPOSAL_CONFIGS['cartpole']
    prev_env_kwargs = cfg.get('env_kwargs')
    cfg['env_kwargs'] = {'cost_type': cost_type}

    try:
        rng = np.random.default_rng(SEED)
        np.random.seed(SEED)

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
        if prev_env_kwargs is None:
            cfg.pop('env_kwargs', None)
        else:
            cfg['env_kwargs'] = prev_env_kwargs


def main():
    print(f"Probe — 5 cells at r=1.0, {N_STEPS} steps, seed={SEED}")
    print(f"{'cell (H,R)':<12} {'tolerance':>12} {'quadratic':>12} {'tol_rank':>9} {'quad_rank':>10} {'t(s)':>6}")

    tol_costs, quad_costs = [], []
    t0 = time.time()
    for H, R in CELLS:
        t_cell = time.time()
        c_tol = run_one(H, R, 'tolerance')
        c_quad = run_one(H, R, 'quadratic')
        tol_costs.append(c_tol)
        quad_costs.append(c_quad)
        dt = time.time() - t_cell
        print(f"({H:>3},{R:>2})    {c_tol:>12.4f} {c_quad:>12.4f} {'':>9} {'':>10} {dt:>6.1f}")

    tol_order = np.argsort(tol_costs)
    quad_order = np.argsort(quad_costs)
    tol_rank = {i: r for r, i in enumerate(tol_order)}
    quad_rank = {i: r for r, i in enumerate(quad_order)}

    print()
    print("Ranking (0 = best / lowest cost):")
    print(f"{'cell':<12} {'tol':>12} {'quad':>12} {'tol_rank':>9} {'quad_rank':>10}")
    for i, (H, R) in enumerate(CELLS):
        print(f"({H:>3},{R:>2})    {tol_costs[i]:>12.4f} {quad_costs[i]:>12.4f} "
              f"{tol_rank[i]:>9} {quad_rank[i]:>10}")

    ranks_tol = np.array([tol_rank[i] for i in range(len(CELLS))])
    ranks_quad = np.array([quad_rank[i] for i in range(len(CELLS))])
    rho = np.corrcoef(ranks_tol, ranks_quad)[0, 1]
    print(f"\nSpearman rank correlation (tolerance vs quadratic): rho = {rho:.3f}")
    print(f"Best cell under tolerance: {CELLS[int(np.argmin(tol_costs))]}")
    print(f"Best cell under quadratic: {CELLS[int(np.argmin(quad_costs))]}")
    print(f"Total wall time: {time.time() - t0:.1f} s")


if __name__ == '__main__':
    main()
