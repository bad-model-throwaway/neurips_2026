"""Render illustrative stand-up rollouts.

Picks configs that together tell the story:
    1. matched-success       — r=1.0, (H=40, R=1), seed=0 — clean baseline.
    2. mild-mismatch-success — r=2.0, (H=40, R=1), seed=0 — still stands.
    3. faceplant-at-high-r   — r=5.0, (H=40, R=1), seed=0 — catastrophic.
    4. short-H-fails-at-r2.5 — r=2.5, (H=40, R=1), seed=2 — falls.
    5. long-H-rescues-at-r2.5 — r=2.5, (H=80, R=1), seed=2 — same seed stands.
    6. foot-friction-mismatch — r=3.0 on foot friction (not torso mass),
       matched else (H=40, R=1, seed=0). Planner assumes 3× grippier feet
       than reality → slippy push-offs.

Writes mp4s to temp/ (created if missing). Serial — Renderer is not
fork-safe and each run is ~40-80 s wall-clock.
"""

import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agents.mpc import make_mpc
from agents.mujoco_dynamics import HumanoidStandDynamics
from simulations.simulation import run_simulation
from tests.shared import record_rollout_video


N_FIX    = 30
N_STEPS  = 400        # 6 s at ctrl_dt=0.015
CAMERA   = 'side'     # trackcom side camera from humanoid_modified.xml


# (label, H, R, mismatch_factor, seed, kind)
# kind:
#   'torso_mass'      — planner's torso body mass × factor (via apply_mismatch).
#   'foot_friction'   — planner's 4 foot geoms' tangential friction × factor.
#   'env_ice'         — REAL env's every-geom tangential friction × factor.
#                       Planner keeps normal friction → "standing up on ice"
#                       where the planner doesn't know the floor is slippery.
CONFIGS = [
    ('01_matched_r1.0_H40_R1_seed0',            40, 1, 1.0, 0, 'torso_mass'),
    ('02_mild_mismatch_r2.0_H40_R1_seed0',      40, 1, 2.0, 0, 'torso_mass'),
    ('03_faceplant_r5.0_H40_R1_seed0',          40, 1, 5.0, 0, 'torso_mass'),
    ('04_shortH_fails_r2.5_H40_R1_seed2',       40, 1, 2.5, 2, 'torso_mass'),
    ('05_longH_rescues_r2.5_H80_R1_seed2',      80, 1, 2.5, 2, 'torso_mass'),
    ('06_foot_friction_r3.0_H40_R1_seed0',      40, 1, 3.0, 0, 'foot_friction'),
    ('07_foot_friction_r0.3_H40_R1_seed0',      40, 1, 0.3, 0, 'foot_friction'),
    # Ice: factor multiplies the real env's every-geom tangential friction.
    # 0.1 ≈ wet ice; 0.02 ≈ fresh black ice. Start at 0.1.
    ('08_ice_env_r0.1_H40_R1_seed0',            40, 1, 0.1, 0, 'env_ice'),
]


def _scale_all_tangential_friction(mj_model, factor):
    """Multiply column 0 (tangential friction) of every geom's friction."""
    mj_model.geom_friction[:, 0] *= factor


def _run(H, R, mismatch_factor, seed, kind):
    np.random.seed(seed)
    env = HumanoidStandDynamics(stateless=False, height_goal=1.4)

    # 'env_ice' modifies the REAL env, not the planner.
    if kind == 'env_ice' and mismatch_factor != 1.0:
        _scale_all_tangential_friction(env._mj_model, mismatch_factor)

    env.reset(env.get_default_initial_state())

    # For non-default planner mismatch axes, build planner with factor=1.0
    # and apply the mismatch manually. make_mpc hardwires kind='torso_mass'.
    if kind == 'torso_mass':
        agent = make_mpc('humanoid_stand', H, R, N=N_FIX,
                         mismatch_factor=mismatch_factor)
    else:
        agent = make_mpc('humanoid_stand', H, R, N=N_FIX,
                         mismatch_factor=1.0)
        if kind == 'foot_friction' and mismatch_factor != 1.0:
            agent.model.apply_mismatch(mismatch_factor, kind=kind)
        # 'env_ice' deliberately leaves the planner untouched.

    _, _, history = run_simulation(
        agent, env, n_steps=N_STEPS, interval=None,
    )
    return env, history.get_item_history('state')


def main():
    out_dir = os.path.join(REPO_ROOT, 'temp')
    os.makedirs(out_dir, exist_ok=True)

    for label, H, R, r, seed, kind in CONFIGS:
        path = os.path.join(out_dir, f'{label}.mp4')
        if os.path.exists(path):
            print(f'[{label}] exists, skipping', flush=True)
            continue
        t0 = time.time()
        print(f'[{label}] H={H} R={R} r={r} seed={seed} kind={kind} …', flush=True)
        env, states = _run(H, R, r, seed, kind)

        head_z = states[:, 55]
        final  = float(head_z[-1])
        max_z  = float(head_z.max())
        outcome = ('STOOD' if final >= 1.3
                   else 'partial' if final >= 0.6
                   else 'FAILED')
        print(f'  final_head_z={final:+.3f}  max={max_z:+.3f}  {outcome}  '
              f'({time.time()-t0:.1f}s)')

        path = os.path.join(out_dir, f'{label}.mp4')
        record_rollout_video(env, states, path, fps=30,
                             size=(480, 480), camera=CAMERA)
        print(f'  wrote {path}')


if __name__ == '__main__':
    main()
