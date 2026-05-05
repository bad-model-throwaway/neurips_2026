import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from configs import DT, PLOTS_DIR
from agents.dynamics import PointMass2D
from agents.utils import GPForceField, figure_eight
from agents.mpc_python import make_pointmass_mpc
from simulations.simulation import run_simulation
from tests.shared import _pointmass_tracking_converged
from visualization.plots_pointmass import plot_force_field, add_plot_dynamics, plot_tracking_summary

# Create plot directory if it doesn't exist
os.makedirs(PLOTS_DIR, exist_ok=True)


def configurable_tracking_test(n_steps=None, agent_args=None, adapt_args=None, model_args=None, test_name='basic'):
    """Run a pointmass tracking MPC test with configurable parameters."""
    if n_steps is None:
        n_steps = int(50.0 / DT)

    # Default horizon if not specified in agent_args
    if agent_args is None:
        agent_args = {}
    agent_args.setdefault('horizon_steps', int(0.1 / DT))

    # Get agent and environment
    agent, env = make_pointmass_mpc(agent_args=agent_args, adapt_args=adapt_args, model_args=model_args)

    # Run the simulation
    agent, env, history = run_simulation(agent, env, n_steps=n_steps)
    states = history.get_item_history('state')

    # Plot and save
    fig = plot_tracking_summary(
        states, DT, env.force_field, figure_eight, 1.5,
        title=f'MPC tracking: {test_name}', history=history,
    )
    fig.savefig(PLOTS_DIR + f'test_tracking_{test_name}.svg')
    plt.close(fig)

    # Assert tracking converged
    _pointmass_tracking_converged(history)


## Tests that will be discovered and performed by pytest

### Autonomous dynamics (no agent)
def test_pointmass_autonomous():
    """Test autonomous dynamics by initializing masses and letting them evolve."""

    # Create force field and dynamics
    force_field = GPForceField(seed=42)
    env = PointMass2D(force_field, figure_eight, stateless=False)

    # Initialize grid of starting positions
    n_per_dim = 3
    extent = 1.5
    grid = np.linspace(-extent, extent, n_per_dim)
    start_positions = [(x, y) for x in grid for y in grid]

    # Run each point mass forward with zero control
    n_steps = 20
    zero_action = np.array([0.0, 0.0])
    histories = []

    for x0, y0 in start_positions:

        # Initial state: position, zero velocity, s=0
        initial_state = np.array([x0, y0, 0.0, 0.0, 0.0])
        env.reset(initial_state)

        # Collect trajectory
        history = [env.state.copy()]
        for _ in range(n_steps):
            env.step(zero_action)
            history.append(env.state.copy())
        histories.append(history)

    # Plot force field then overlay trajectories
    fig, ax = plot_force_field(force_field, figure_eight, curve_scale=1.5)
    add_plot_dynamics(ax, histories)
    ax.set_title('Autonomous point mass dynamics')
    fig.savefig(PLOTS_DIR + 'test_pointmass_autonomous.svg')
    plt.close(fig)

### Tracking tests
def test_tracking_basic():
    configurable_tracking_test(test_name='basic')

### Adaptation tests
#
# Pointmass baseline running_error is ~0.13 (median), so thresholds must
# exceed this to avoid perpetual tightening. Horizon and recompute bounds
# are scaled to the pointmass domain (horizon ~5 steps, not cartpole ~50).

def test_tracking_adaptation_recompute_ode():
    agent_args = {'recompute_interval': 5}
    adapt_args = {
        'adapt_class': 'ODEStepAdaptation',
        'adapt_params': ('recompute',),
        'adapt_kwargs': {'min_error_threshold': 0.20, 'max_recompute': 5},
    }
    configurable_tracking_test(agent_args=agent_args, adapt_args=adapt_args, test_name='adaptation_recompute_ode')

def test_tracking_adaptation_recompute_cost():
    agent_args = {'recompute_interval': 5}
    adapt_args = {
        'adapt_class': 'CostErrorAdaptation',
        'adapt_params': ('recompute',),
        'adapt_kwargs': {'max_recompute': 5, 'min_error_threshold': 0.01},
    }
    configurable_tracking_test(agent_args=agent_args, adapt_args=adapt_args, test_name='adaptation_recompute_cost')

def test_tracking_adaptation_horizon_ode():
    adapt_args = {
        'adapt_class': 'ODEStepAdaptation',
        'adapt_params': ('horizon',),
        'adapt_kwargs': {'min_error_threshold': 0.20, 'min_horizon': 3, 'max_horizon': 10},
    }
    configurable_tracking_test(adapt_args=adapt_args, test_name='adaptation_horizon_ode')

def test_tracking_adaptation_horizon_cost():
    adapt_args = {
        'adapt_class': 'CostErrorAdaptation',
        'adapt_params': ('horizon',),
        'adapt_kwargs': {'min_horizon': 3, 'max_horizon': 10, 'horizon_step': 1, 'min_error_threshold': 0.01},
    }
    configurable_tracking_test(adapt_args=adapt_args, test_name='adaptation_horizon_cost')
