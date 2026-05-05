"""Cartpole sanity tests: matched dynamics, sustained mismatch, mid-episode switch."""
import numpy as np
import mujoco

from agents.adaptation import TheoryStepAdaptation
from agents.mpc import make_mpc
from agents.mujoco_dynamics import MuJoCoCartPoleDynamics
from simulations.simulation import run_simulation
from tests.shared import (
    _assert_fall_safety, _log_fig2_metric, _save_test_artifacts,
)


_H = 80
_R = 3
_N = 50  # matched-dynamics theory-adapter ran flaky at N=30 (2026-05-04)
_N_STEPS = 600


def _make_env():
    env = MuJoCoCartPoleDynamics(stateless=False)
    data = mujoco.MjData(env._mj_model)
    mujoco.mj_resetData(env._mj_model, data)
    state0 = env._state_from_data(data)
    state0[2] = 0.05  # small pole tilt to start
    env.reset(state0)
    return env


def _run(H=_H, R=_R, N=_N, n_steps=_N_STEPS, mismatch_factor=1.0,
         adaptation=None, env_mismatch_fn=None, seed=42):
    np.random.seed(seed)
    agent = make_mpc('cartpole', H, R, N=N, mismatch_factor=mismatch_factor)
    if adaptation is not None:
        agent.adaptation = adaptation
    env = _make_env()
    _, _, history = run_simulation(agent, env, n_steps=n_steps,
                                   env_mismatch_fn=env_mismatch_fn,
                                   interval=None)
    return history


def test_cartpole_basic():
    history = _run()
    _assert_fall_safety(history, 'cartpole')
    _log_fig2_metric(history, 'cartpole')
    _save_test_artifacts(history, 'cartpole', 'test_cartpole_basic')


def test_cartpole_mismatch():
    history = _run(mismatch_factor=1.5)
    _assert_fall_safety(history, 'cartpole')
    _log_fig2_metric(history, 'cartpole')
    _save_test_artifacts(history, 'cartpole', 'test_cartpole_mismatch')


def test_cartpole_midswitch():
    switch_step = _N_STEPS // 2
    post_factor = 1.5

    switched = [False]
    def env_mismatch_fn(e, step_idx):
        if not switched[0] and step_idx >= switch_step:
            e.apply_mismatch(post_factor)
            switched[0] = True

    adapt = TheoryStepAdaptation(adapt=('recompute', 'horizon'),
                                 warmup_replans=5, noise_floor_window=10,
                                 max_recompute=4,
                                 min_horizon=50, max_horizon=70)
    history = _run(H=60, R=5, adaptation=adapt, env_mismatch_fn=env_mismatch_fn)
    _assert_fall_safety(history, 'cartpole')
    _log_fig2_metric(history, 'cartpole')
    _save_test_artifacts(history, 'cartpole', 'test_cartpole_midswitch')
