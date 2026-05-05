"""Subclasses of base components defined for LQR setting."""

import numpy as np

from scipy.linalg import solve_discrete_are
from agents.base import Dynamics, Proposal, Decision, Agent


class LQRDynamics(Dynamics):
    """Linear Quadratic Regulator world (environment and model).

    Optional Gaussian process noise (noise_std) is added to the next state
    after the deterministic step, so an env instance and a planner-model
    instance can carry independent noise levels.
    """

    def __init__(self, A, B, Q, R, stateless=False, noise_std=0.0):
        super().__init__(stateless)
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        self.noise_std = float(noise_std)

    def cost_function(self, state, ctrl=None):
        """Quadratic state cost. Supports 1D state or batched [n_samples, state_dim]."""
        if state.ndim == 1:
            return state.T @ self.Q @ state
        # Batched: per-row x^T Q x via element-wise sum
        return np.sum((state @ self.Q) * state, axis=1)

    def _step_stateless(self, state, action, params=None):
        """Linear dynamics with quadratic cost; optional additive process noise.

        Supports 1D state (action 1D) and batched state shape
        [n_samples, state_dim] (action shape [n_samples, action_dim]).
        """
        if state.ndim == 1:
            next_state = self.A @ state + self.B @ action
            cost = state.T @ self.Q @ state + action.T @ self.R @ action
        else:
            next_state = state @ self.A.T + action @ self.B.T
            cost = (np.sum((state @ self.Q) * state, axis=1)
                    + np.sum((action @ self.R) * action, axis=1))

        # Process noise on the next state. Independent draws per call mean each
        # planner-rollout candidate sees its own disturbance realization.
        if self.noise_std > 0.0:
            next_state = next_state + np.random.randn(*next_state.shape) * self.noise_std

        return next_state, cost

class LQRProposal(Proposal):
    def __init__(self, A, B, Q, R):
        # Solve for P, then compute K
        self.P = solve_discrete_are(A, B, Q, R)
        self.K = np.linalg.inv(R + B.T @ self.P @ B) @ (B.T @ self.P @ A)

    def __call__(self, state):
        # Return as [n_samples=1, action_dim, horizon=1]
        action = -self.K @ state
        return action.reshape(1, -1, 1)

class LQRDecision(Decision):
    """Single-proposal decision: queue n_actions from proposals[0] in queue order.

    LQR has a deterministic single plan (no minimum to find), so this just
    slices the requested number of actions from the only proposal.
    """
    def __call__(self, proposals, trajectories, evaluations, n_actions=1):
        # proposals shape: [1, action_dim, horizon]
        actions = [proposals[0, :, t] for t in range(n_actions)]
        return actions, 0


class FiniteHorizonLQRProposal(Proposal):
    """Receding-horizon LQR proposal under a misspecified planning model.

    Computes the time-0 greedy gain for an ell-step LQR problem with terminal
    cost K_ell = 0 using the misspecified (A_hat, B_hat) model, then rolls
    out an ell-length action sequence by applying that gain repeatedly under
    the misspecified dynamics. The agent's queue consumes the first tau
    actions before re-planning.

    Returned proposals shape: [1, action_dim, horizon] - a single deterministic
    plan, matching the (n_samples, action_dim, tsteps) layout used elsewhere.
    """

    def __init__(self, A_hat, B_hat, Q, R, horizon=10):
        self.A_hat = np.atleast_2d(np.asarray(A_hat, dtype=float))
        self.B_hat = np.atleast_2d(np.asarray(B_hat, dtype=float))
        self.Q = np.atleast_2d(np.asarray(Q, dtype=float))
        self.R = np.atleast_2d(np.asarray(R, dtype=float))
        self.horizon = int(horizon)
        self._update_gain()

    def update_parameters(self, parameters):
        """Sync horizon (ell) from agent parameters; recompute gain when changed."""
        h = int(parameters.get('horizon', self.horizon))
        if h != self.horizon:
            self.horizon = h
            self._update_gain()

    def _update_gain(self):
        """Backward Riccati for ell-1 iterations from K_ell = 0, then time-0 gain."""
        # Cost-to-go after ell-1 iterations of the wrong-model Riccati operator
        K = np.zeros_like(self.Q)
        for _ in range(max(self.horizon - 1, 0)):
            K = self._riccati_step(K)

        # Time-0 greedy gain: L = -(R + B^T K B)^{-1} B^T K A
        BTKB = self.B_hat.T @ K @ self.B_hat
        BTKA = self.B_hat.T @ K @ self.A_hat
        self.L = -np.linalg.solve(self.R + BTKB, BTKA)

    def _riccati_step(self, K):
        """One backward Riccati step: K_{t-1} = A^T K A - A^T K B (R + B^T K B)^{-1} B^T K A + Q."""
        A, B, Q, R = self.A_hat, self.B_hat, self.Q, self.R
        BTKB = B.T @ K @ B
        BTKA = B.T @ K @ A
        return A.T @ K @ A - A.T @ K @ B @ np.linalg.solve(R + BTKB, BTKA) + Q

    def __call__(self, state):
        """Roll out `horizon` actions under the misspecified model with constant gain L."""
        x = np.atleast_1d(np.asarray(state, dtype=float))
        actions = []
        for _ in range(self.horizon):
            u = self.L @ x
            actions.append(u)
            x = self.A_hat @ x + self.B_hat @ u

        # Pack as [n_samples=1, action_dim, horizon]
        actions = np.stack(actions, axis=-1)
        return actions[np.newaxis, ...]


def make_lqr_mpc(a=1.5, b=1.0, q=1.0, r=1.0,
                 a_hat=None, b_hat=None,
                 horizon=10, recompute_interval=1,
                 n_samples=30, P=3, interp='cubic',
                 sigma=0.5, dt=1.0,
                 env_noise_std=0.0, model_noise_std=0.0,
                 initial_state=None, seed=None):
    """Build a scalar-LQR MPC agent and environment using SplinePSProposal.

    Mirrors make_cartpole_mpc but with LQRDynamics in place of the cartpole
    physics. The planner samples N action sequences over a `horizon`-step
    lookahead, parameterized by a P-knot spline, evaluates each under the
    misspecified model, and selects argmin via SplinePSArgminDecision.

    a, b, q, r        : true scalar LQR parameters
    a_hat, b_hat      : planner's misspecified dynamics (defaults to true)
    P, interp         : spline parameterization of the action search space.
                        interp='cubic' with P=3 gives a low-dimensional smooth
                        search; interp='zero' with P=horizon gives independent
                        per-step samples (random-shooter equivalent).
    sigma             : per-knot Gaussian std for action sampling. With
                        ctrl_low/high=±1 and clip=False, sigma is the absolute
                        per-knot std rather than a fraction of the action range.
    env_noise_std     : process noise on the true environment
    model_noise_std   : process noise on the planner's lookahead model
    """
    from agents.mpc import SplinePSProposal, SplinePSArgminDecision, MPCEvaluation

    if a_hat is None: a_hat = a
    if b_hat is None: b_hat = b

    # Scalar 1x1 system matrices
    A_env = np.array([[float(a)]])
    B_env = np.array([[float(b)]])
    A_mod = np.array([[float(a_hat)]])
    B_mod = np.array([[float(b_hat)]])
    Q_mat = np.array([[float(q)]])
    R_mat = np.array([[float(r)]])

    env   = LQRDynamics(A_env, B_env, Q_mat, R_mat, stateless=False, noise_std=env_noise_std)
    model = LQRDynamics(A_mod, B_mod, Q_mat, R_mat, stateless=True,  noise_std=model_noise_std)

    # Initial state (scalar, packed as 1D array of shape (1,))
    if initial_state is None:
        rng = np.random.default_rng(seed)
        initial_state = np.array([rng.uniform(-0.1, 0.1)])
    else:
        initial_state = np.atleast_1d(np.asarray(initial_state, dtype=float))
    env.reset(initial_state)

    # Action bounds: ±1 with clip=False so sigma is the absolute per-knot std
    ctrl_low  = np.array([-1.0])
    ctrl_high = np.array([ 1.0])

    proposal = SplinePSProposal(
        action_dim=1, tsteps=horizon, n_samples=n_samples, dt=dt,
        ctrl_low=ctrl_low, ctrl_high=ctrl_high,
        P=P, sigma=sigma, interp=interp,
        include_nominal=True, clip=False,
    )
    decision = SplinePSArgminDecision(proposal=proposal)

    parameters = {'recompute_interval': recompute_interval, 'horizon': horizon}
    agent = Agent(proposal, model, MPCEvaluation(), decision, parameters=parameters)

    return agent, env
