"""Humanoid balance sanity tests: matched dynamics, sustained mismatch, mid-episode switch."""
import numpy as np

from agents.adaptation import TheoryStepAdaptation
from agents.mpc import make_mpc
from agents.mujoco_dynamics import HumanoidStandDynamics
from simulations.simulation import run_simulation
from tests.shared import (
    _assert_fall_safety, _log_fig2_metric, _save_test_artifacts,
)


_H = 40
_R = 1
_N = 30
_N_STEPS = 300


def _make_env():
    env = HumanoidStandDynamics(stateless=False, mode='balance')
    env.reset(env.get_default_initial_state())
    return env


def _run(H=_H, R=_R, N=_N, n_steps=_N_STEPS, mismatch_factor=1.0,
         adaptation=None, env_mismatch_fn=None, seed=42):
    np.random.seed(seed)
    agent = make_mpc('humanoid_balance', H, R, N=N,
                     mismatch_factor=mismatch_factor)
    if adaptation is not None:
        agent.adaptation = adaptation
    env = _make_env()
    _, _, history = run_simulation(agent, env, n_steps=n_steps,
                                   env_mismatch_fn=env_mismatch_fn,
                                   interval=None)
    return history


def test_humanoid_basic():
    history = _run()
    _assert_fall_safety(history, 'humanoid_balance')
    _log_fig2_metric(history, 'humanoid_balance')
    _save_test_artifacts(history, 'humanoid_balance', 'test_humanoid_basic')


def test_humanoid_mismatch():
    history = _run(mismatch_factor=1.4)
    _assert_fall_safety(history, 'humanoid_balance')
    _log_fig2_metric(history, 'humanoid_balance')
    _save_test_artifacts(history, 'humanoid_balance', 'test_humanoid_mismatch')


def test_humanoid_midswitch():
    switch_step = _N_STEPS // 2
    post_factor = 1.2

    switched = [False]
    def env_mismatch_fn(e, step_idx):
        if not switched[0] and step_idx >= switch_step:
            e.apply_mismatch(post_factor, kind='gravity')
            switched[0] = True

    adapt = TheoryStepAdaptation(adapt=('recompute', 'horizon'),
                                 warmup_replans=5, noise_floor_window=10,
                                 max_recompute=3,
                                 min_horizon=30, max_horizon=50)
    history = _run(adaptation=adapt, env_mismatch_fn=env_mismatch_fn)
    _assert_fall_safety(history, 'humanoid_balance')
    _log_fig2_metric(history, 'humanoid_balance')
    _save_test_artifacts(history, 'humanoid_balance', 'test_humanoid_midswitch')
