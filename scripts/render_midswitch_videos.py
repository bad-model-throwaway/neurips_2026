"""Render mid-episode mismatch rollout videos for Figure 3 column C/F/I.

For each environment, renders two conditions at the representative mismatch factor:
  - Adaptive (paper's method)
  - Fixed slow (counterfactual at Adaptive's R_init)

A switch is applied at SWITCH_STEP = N_STEPS // 2 (same as the sweeps).
Writes mp4s to temp/midswitch/. Serial — each episode is ~20-120 s wall-clock.

Usage (on a compute node via sbatch):
    python scripts/render_midswitch_videos.py
"""

import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agents.mpc import make_mpc
from agents.mujoco_dynamics import (
    MuJoCoCartPoleDynamics, WalkerDynamics, HumanoidStandDynamics,
)
from agents.adaptation import make_adapter
from configs import DT as CARTPOLE_DT, SEED
from simulations.simulation import run_simulation
from simulations.sweep_cartpole_midswitch import (
    SWITCH_STEP as CP_SWITCH_STEP, N_STEPS as CP_N_STEPS,
)
from simulations.sweep_cartpole_adaptive import H_CARTPOLE
from simulations.sweep_walker_midswitch import (
    SWITCH_STEP as WK_SWITCH_STEP, N_STEPS as WK_N_STEPS, H_WALKER, R_INIT as WK_R_INIT,
)
from simulations.sweep_humanoid_balance_adaptive import (
    N_STEPS as HB_N_STEPS, H_HUMANOID_BALANCE,
    FAIL_HEAD_Z, HEAD_Z_IDX, _EPISODE_TIMEOUT_S, _timeout_handler,
)
from simulations.sweep_humanoid_balance_midswitch import R_INIT as HB_R_INIT
from tests.shared import record_rollout_video

import signal


SEED_VAL = int(SEED)
OUT_DIR = os.path.join(REPO_ROOT, 'temp', 'midswitch')


def _make_adapter(adaptive):
    adapt_args = {
        'adapt_class': 'ODEStepAdaptation',
        'adapt_params': ('recompute',) if adaptive else (),
        'adapt_kwargs': {'min_error_threshold': 0.08, 'relax_step': 0.05},
    }
    return make_adapter(adapt_args)


def _run_cartpole(recompute, adaptive, post_factor):
    np.random.seed(SEED_VAL)
    env = MuJoCoCartPoleDynamics(stateless=False)
    agent = make_mpc('cartpole', H=H_CARTPOLE, R=recompute, mismatch_factor=1.0)
    agent.adaptation = _make_adapter(adaptive)
    env.reset(np.array([0.0, 0.0, 0.02, 0.0]))

    switched = [False]
    switch_step = CP_SWITCH_STEP

    def env_mismatch_fn(e, step_idx):
        if not switched[0] and step_idx >= switch_step:
            e.apply_mismatch(post_factor)
            switched[0] = True

    _, _, history = run_simulation(
        agent, env, n_steps=CP_N_STEPS,
        env_mismatch_fn=env_mismatch_fn, interval=None,
    )
    states = np.array(history.get_item_history('state'))
    return env, states


def _run_walker(recompute, adaptive, post_factor):
    np.random.seed(SEED_VAL)
    env = WalkerDynamics(stateless=False)
    agent = make_mpc('walker', H=H_WALKER, R=recompute, mismatch_factor=1.0)
    agent.adaptation = _make_adapter(adaptive)
    env.reset(env.get_default_initial_state())

    switched = [False]
    switch_step = WK_SWITCH_STEP

    def env_mismatch_fn(e, step_idx):
        if not switched[0] and step_idx >= switch_step:
            e.apply_mismatch(post_factor, kind='torso_mass')
            switched[0] = True

    _, _, history = run_simulation(
        agent, env, n_steps=WK_N_STEPS,
        env_mismatch_fn=env_mismatch_fn, interval=None,
    )
    states = np.array(history.get_item_history('state'))
    return env, states


def _run_humanoid_balance(recompute, adaptive, post_factor):
    import signal
    np.random.seed(SEED_VAL)

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(_EPISODE_TIMEOUT_S)
    try:
        env = HumanoidStandDynamics(stateless=False, mode='balance')
        agent = make_mpc('humanoid_balance', H=H_HUMANOID_BALANCE,
                         R=recompute, mismatch_factor=1.0)
        agent.adaptation = _make_adapter(adaptive)
        env.reset(env.get_default_initial_state())

        hb_switch = HB_N_STEPS // 2
        switched = [False]

        def env_mismatch_fn(e, step_idx):
            if not switched[0] and step_idx >= hb_switch:
                e.apply_mismatch(post_factor, kind='gravity')
                switched[0] = True

        _, _, history = run_simulation(
            agent, env, n_steps=HB_N_STEPS,
            env_mismatch_fn=env_mismatch_fn, interval=None,
        )
        states = np.array(history.get_item_history('state'))
    finally:
        signal.alarm(0)

    return env, states


# (label, env, recompute, adaptive, post_factor). Repr factors mirror
# _MIDSWITCH_REPR_FACTOR in visualization/figures.py.
CONFIGS = [
    ('cp_adaptive_r1.5',    'cartpole',  4, True,  1.5),
    ('cp_fixed_slow_r1.5',  'cartpole',  6, False, 1.5),
    ('cp_fixed_fast_r1.5',  'cartpole',  1, False, 1.5),
    ('wk_adaptive_r2.0',    'walker',    WK_R_INIT, True,  2.0),
    ('wk_fixed_slow_r2.0',  'walker',    WK_R_INIT, False, 2.0),
    ('wk_fixed_fast_r2.0',  'walker',    1,         False, 2.0),
    ('hb_adaptive_r1.2',    'humanoid',  HB_R_INIT, True,  1.2),
    ('hb_fixed_slow_r1.2',  'humanoid',  HB_R_INIT, False, 1.2),
    ('hb_fixed_fast_r1.2',  'humanoid',  1,         False, 1.2),
]

RUNNERS = {
    'cartpole': _run_cartpole,
    'walker':   _run_walker,
    'humanoid': _run_humanoid_balance,
}

CAMERAS = {
    'cartpole': -1,
    'walker':   'side',
    'humanoid': 'side',
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for label, env, recompute, adaptive, post_factor in CONFIGS:
        path = os.path.join(OUT_DIR, f'{label}.mp4')
        if os.path.exists(path):
            print(f'[{label}] exists, skipping', flush=True)
            continue

        cond = 'adaptive' if adaptive else f'fixed_R{recompute}'
        print(f'[{label}] env={env} R={recompute} adaptive={adaptive} r={post_factor} …',
              flush=True)
        t0 = time.time()
        dynamics, states = RUNNERS[env](recompute, adaptive, post_factor)
        print(f'  simulated in {time.time()-t0:.1f}s, rendering …', flush=True)

        record_rollout_video(
            dynamics, states, path,
            fps=30, size=(480, 480),
            camera=CAMERAS[env],
        )
        print(f'  wrote {path}  ({time.time()-t0:.1f}s total)')


if __name__ == '__main__':
    main()
