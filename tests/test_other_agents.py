import os
import numpy as np
import matplotlib

from configs import PLOTS_DIR
from agents.base import Agent
from agents.lqr import LQRDynamics, LQRProposal, LQRDecision
from agents.mpc import MPCEvaluation
from simulations.simulation import run_simulation

matplotlib.use('Agg')
os.makedirs(PLOTS_DIR, exist_ok=True)

def test_lqr_agent():
    """Test LQR agent on 2D double integrator."""

    # Define simple 2D double integrator: state = [position, velocity]
    dt = 0.1
    A = np.array([[1, dt], [0, 1]])
    B = np.array([[0], [dt]])
    Q = np.eye(2)
    R = np.array([[0.1]])

    # Create environment (stateful) and world model (stateless)
    env   = LQRDynamics(A, B, Q, R, stateless=False)
    model = LQRDynamics(A, B, Q, R, stateless=True)

    # Create agent components
    proposal   = LQRProposal(A, B, Q, R)
    evaluation = MPCEvaluation()
    decision   = LQRDecision()

    # Build agent
    agent = Agent(proposal, model, evaluation, decision)

    # Initialize environment
    initial_state = np.array([1.0, 0.0])
    env.reset(initial_state)

    # Run simulation
    agent, env, history = run_simulation(agent, env, n_steps=20, interval=5)

    # Assert state converges toward zero
    final_norm = np.linalg.norm(env.state)
    initial_norm = np.linalg.norm(initial_state)
    assert final_norm < 0.5 * initial_norm, \
        f"LQR did not converge: final |state| = {final_norm:.4f}, initial = {initial_norm:.4f}"
