"""Verify Cython cartpole rollout matches Python implementation."""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest

from configs import DT, PLOTS_DIR
from agents.dynamics import CartPoleDynamics
from agents.mpc_python import make_cartpole_mpc
# Skip the whole module when the cython extension is not built.
_cartpole_cy = pytest.importorskip('agents._cartpole_cy')
cartpole_rollout = _cartpole_cy.cartpole_rollout
from simulations.simulation import run_simulation
from visualization.plots_cartpole import plot_cartpole_history

os.makedirs(PLOTS_DIR, exist_ok=True)


"""Verify Cython pointmass rollout matches Python implementation."""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from configs import DT, PLOTS_DIR
from agents.dynamics import PointMass2D
from agents.utils import GPForceField, figure_eight
from agents.mpc_python import make_pointmass_mpc
# Skip the pointmass-cython tests when the extension is not built.
_pointmass_cy = pytest.importorskip('agents._pointmass_cy')
pointmass_rollout = _pointmass_cy.pointmass_rollout
from simulations.simulation import run_simulation
from visualization.plots_pointmass import plot_tracking_summary

os.makedirs(PLOTS_DIR, exist_ok=True)


def test_cartpole_rollout_correctness(n_samples=100, horizon=50, seed=42):
    """Compare Cython rollout output to Python _forward_stateless, element by element."""

    rng = np.random.RandomState(seed)

    # Physics parameters
    gravity, masscart, masspole, length, dt = 9.8, 1.0, 0.1, 0.5, 0.02

    # Create Python-only dynamics model
    py_model = CartPoleDynamics(
        gravity=gravity, masscart=masscart, masspole=masspole,
        length=length, dt=dt, stateless=True, use_cython=False,
    )

    # Generate test data
    initial_state = np.array([0.0, 0.0, 0.15, 0.0])
    actions = rng.uniform(-10.0, 10.0, size=(n_samples, 1, horizon))
    batched_states = np.tile(initial_state, (n_samples, 1))

    # Python rollout
    py_states, py_costs = py_model.query(initial_state, actions)

    # Cython rollout
    cy_states, cy_costs = cartpole_rollout(
        batched_states, actions, gravity, masscart, masspole, length, dt,
        *py_model.cost_weights,
    )

    # Compare state trajectories
    state_err = np.max(np.abs(py_states - cy_states))
    print(f"Max state error: {state_err:.2e}")
    assert state_err < 1e-12, f"State mismatch: max error = {state_err:.2e}"

    # Compare per-step costs
    cost_err = np.max(np.abs(py_costs - cy_costs))
    print(f"Max cost error: {cost_err:.2e}")
    assert cost_err < 1e-10, f"Cost mismatch: max error = {cost_err:.2e}"

    print(f"Correctness OK: {n_samples} samples x {horizon} steps")

    # Visual verification: run closed-loop MPC with Cython backend
    agent, env = make_cartpole_mpc(agent_args={'proposal_args': {'tsteps': 50}})
    env.reset(np.array([0.0, 0.0, 0.2, 0.0]))
    _, _, history = run_simulation(agent, env, n_steps=1000, interval=None)

    plot_cartpole_history(history)
    plt.savefig(PLOTS_DIR + 'test_cartpole_cython.svg')
    plt.close('all')


def test_cartpole_rollout_speed(n_samples=1000, horizon=200, n_reps=20, seed=42):
    """Benchmark Cython vs Python rollout speed."""

    rng = np.random.RandomState(seed)

    # Physics parameters
    gravity, masscart, masspole, length, dt = 9.8, 1.0, 0.1, 0.5, 0.02

    # Create Python-only model for benchmark baseline
    py_model = CartPoleDynamics(
        gravity=gravity, masscart=masscart, masspole=masspole,
        length=length, dt=dt, stateless=True, use_cython=False,
    )

    # Generate test data matching real sweep dimensions
    initial_state = np.array([0.0, 0.0, 0.1, 0.0])
    actions = rng.uniform(-10.0, 10.0, size=(n_samples, 1, horizon))
    batched_states = np.tile(initial_state, (n_samples, 1))

    # Warm up
    py_model.query(initial_state, actions)
    cartpole_rollout(batched_states, actions, gravity, masscart, masspole, length, dt, *py_model.cost_weights)

    # Benchmark Python
    t0 = time.perf_counter()
    for _ in range(n_reps):
        py_model.query(initial_state, actions)
    py_time = (time.perf_counter() - t0) / n_reps

    # Benchmark Cython
    t0 = time.perf_counter()
    for _ in range(n_reps):
        cartpole_rollout(batched_states, actions, gravity, masscart, masspole, length, dt, *py_model.cost_weights)
    cy_time = (time.perf_counter() - t0) / n_reps

    speedup = py_time / cy_time
    print(f"Python: {py_time*1000:.1f} ms   Cython: {cy_time*1000:.1f} ms   Speedup: {speedup:.1f}x")
    print(f"  ({n_samples} samples x {horizon} horizon, mean of {n_reps} reps)")


def test_pointmass_rollout_correctness(n_samples=10, tsteps=5, seed=42):
    """Compare Cython pointmass rollout to Python _forward_stateless."""

    rng = np.random.RandomState(seed)

    # Create Python-only model
    force_field = GPForceField(seed=seed)
    py_model = PointMass2D(
        force_field, figure_eight,
        stateless=True, dt=DT, use_cython=False,
    )

    # Generate test data
    initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    actions = rng.uniform(-20.0, 20.0, size=(n_samples, 2, tsteps))
    batched_states = np.tile(initial_state, (n_samples, 1))

    # Python rollout
    py_states, py_costs = py_model.query(initial_state, actions)

    # Cython rollout (uses lookup tables)
    ff = force_field
    cy_states, cy_costs = pointmass_rollout(
        batched_states, actions,
        ff._table_fx.ravel(), ff._table_fy.ravel(),
        ff._table_n, ff._table_extent,
        py_model._curve_table.ravel(),
        py_model._curve_n, py_model._curve_extent,
        py_model.mass, py_model.dt,
        py_model.tracking_speed, py_model.curve_scale,
        py_model.cost_weights['curve'], py_model.cost_weights['tracking'],
        py_model.cost_weights['control'],
    )

    # Compare state trajectories (lookup tables introduce small approximation)
    state_err = np.max(np.abs(py_states - cy_states))
    print(f"Max state error: {state_err:.2e}")
    assert state_err < 0.05, f"State mismatch: max error = {state_err:.2e}"

    # Compare per-step costs
    cost_err = np.max(np.abs(py_costs - cy_costs))
    print(f"Max cost error: {cost_err:.2e}")
    assert cost_err < 0.5, f"Cost mismatch: max error = {cost_err:.2e}"

    print(f"Correctness OK: {n_samples} samples x {tsteps} steps")

    # Visual verification: run closed-loop tracking MPC with Cython backend
    horizon_steps = int(0.1 / DT)
    n_steps = int(10.0 / DT)
    agent, env = make_pointmass_mpc(agent_args={'horizon_steps': horizon_steps})
    _, _, history = run_simulation(agent, env, n_steps=n_steps, interval=None)
    states = history.get_item_history('state')

    # Plot tracking summary
    fig = plot_tracking_summary(
        states, DT, env.force_field, figure_eight, 1.5,
        title='MPC tracking (Cython backend)',
    )
    fig.savefig(PLOTS_DIR + 'test_tracking_cython.svg')
    plt.close(fig)


def test_pointmass_rollout_speed(n_samples=10, tsteps=5, n_reps=200, seed=42):
    """Benchmark Cython vs Python rollout speed for point mass."""

    rng = np.random.RandomState(seed)

    # Create Python-only model for benchmark baseline
    force_field = GPForceField(seed=seed)
    py_model = PointMass2D(
        force_field, figure_eight,
        stateless=True, dt=DT, use_cython=False,
    )

    # Generate test data
    initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    actions = rng.uniform(-20.0, 20.0, size=(n_samples, 2, tsteps))
    batched_states = np.tile(initial_state, (n_samples, 1))

    # Warm up
    ff = force_field
    cy_args = (
        batched_states, actions,
        ff._table_fx.ravel(), ff._table_fy.ravel(),
        ff._table_n, ff._table_extent,
        py_model._curve_table.ravel(),
        py_model._curve_n, py_model._curve_extent,
        py_model.mass, py_model.dt,
        py_model.tracking_speed, py_model.curve_scale,
        py_model.cost_weights['curve'], py_model.cost_weights['tracking'],
        py_model.cost_weights['control'],
    )
    py_model.query(initial_state, actions)
    pointmass_rollout(*cy_args)

    # Benchmark Python
    t0 = time.perf_counter()
    for _ in range(n_reps):
        py_model.query(initial_state, actions)
    py_time = (time.perf_counter() - t0) / n_reps

    # Benchmark Cython
    t0 = time.perf_counter()
    for _ in range(n_reps):
        pointmass_rollout(*cy_args)
    cy_time = (time.perf_counter() - t0) / n_reps

    speedup = py_time / cy_time
    print(f"Python: {py_time*1000:.2f} ms   Cython: {cy_time*1000:.2f} ms   Speedup: {speedup:.1f}x")
    print(f"  ({n_samples} samples x {tsteps} horizon, mean of {n_reps} reps)")
