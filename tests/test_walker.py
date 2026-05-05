"""Walker sanity tests: matched dynamics, sustained mismatch, mid-episode switch."""
import numpy as np

from agents.adaptation import TheoryStepAdaptation
from agents.mpc import make_mpc
from agents.mujoco_dynamics import WalkerDynamics
from simulations.simulation import run_simulation
from tests.shared import (
    _assert_fall_safety, _log_fig2_metric, _save_test_artifacts,
)


_H = 60
_R = 1
_N = 30
_N_STEPS = 400


def _make_env():
    env = WalkerDynamics(stateless=False, speed_goal=1.5)
    env.reset(env.get_default_initial_state())
    return env


def _run(H=_H, R=_R, N=_N, n_steps=_N_STEPS, mismatch_factor=1.0,
         adaptation=None, env_mismatch_fn=None, seed=42):
    np.random.seed(seed)
    agent = make_mpc('walker', H, R, N=N, mismatch_factor=mismatch_factor)
    if adaptation is not None:
        agent.adaptation = adaptation
    env = _make_env()
    _, _, history = run_simulation(agent, env, n_steps=n_steps,
                                   env_mismatch_fn=env_mismatch_fn,
                                   interval=None)
    return history


def test_walker_basic():
    history = _run()
    _assert_fall_safety(history, 'walker')
    _log_fig2_metric(history, 'walker')
    _save_test_artifacts(history, 'walker', 'test_walker_basic')


def test_walker_mismatch():
    history = _run(mismatch_factor=1.5)
    _assert_fall_safety(history, 'walker')
    _log_fig2_metric(history, 'walker')
    _save_test_artifacts(history, 'walker', 'test_walker_mismatch')


def test_walker_midswitch():
    switch_step = _N_STEPS // 2
    post_factor = 1.5

    switched = [False]
    def env_mismatch_fn(e, step_idx):
        if not switched[0] and step_idx >= switch_step:
            e.apply_mismatch(post_factor, kind='torso_mass')
            switched[0] = True

    adapt = TheoryStepAdaptation(adapt=('recompute', 'horizon'),
                                 warmup_replans=5, noise_floor_window=10,
                                 max_recompute=5,
                                 min_horizon=50, max_horizon=80)
    history = _run(H=60, R=3, adaptation=adapt, env_mismatch_fn=env_mismatch_fn)
    _assert_fall_safety(history, 'walker')
    _log_fig2_metric(history, 'walker')
    _save_test_artifacts(history, 'walker', 'test_walker_midswitch')
