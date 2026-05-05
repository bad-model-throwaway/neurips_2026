"""Subclasses of base components defined for CartPole MPC."""
import warnings
import numpy as np

from configs import DT, SEED
from agents.base import Proposal, Evaluation, Decision, Agent
from agents.adaptation import ODEStepAdaptation, CostErrorAdaptation, make_adapter
from agents.dynamics import CartPoleDynamics, PointMass2D
from agents.utils import GPForceField, figure_eight


class CartPoleMPCRandomShooterProposal(Proposal):
    """Random sampling proposal for MPC."""

    def __init__(self, tsteps=50, n_samples=500, action_low=-10.0, action_high=10.0):
        self.tsteps = tsteps
        self.n_samples = n_samples
        self.action_low = action_low
        self.action_high = action_high

    def update_parameters(self, parameters):
        if 'horizon' in parameters:
            self.tsteps = parameters['horizon']

    def __call__(self, state):
        """Generate random action sequences [n_samples, 1, tsteps]."""
        return np.random.uniform(
            self.action_low, self.action_high,
            size=(self.n_samples, 1, self.tsteps)
        )

class CartPoleMPCGPProposal(Proposal):
    """GP-correlated random sampling proposal for cartpole MPC.

    Generates temporally correlated force trajectories by sampling from a
    Gaussian process prior with RBF kernel. Drop-in replacement for
    CartPoleMPCRandomShooterProposal.
    """

    def __init__(self, tsteps=50, dt=DT, n_samples=100, action_scale=7.0, tau=None):
        """
        tsteps: MPC planning horizon [steps]
        dt: timestep duration [s]
        n_samples: number of random action sequences to evaluate
        action_scale: standard deviation of force magnitude [N]
        tau: GP kernel time constant [s]. If None, scales with horizon.
        """
        self.tsteps = tsteps
        self.dt = dt
        self.n_samples = n_samples
        self.action_scale = action_scale
        if tau is None:
            horizon_s = tsteps * dt
            tau = max(0.1, 0.2 * np.sqrt(horizon_s))
        self.tau = tau

        self._precompute_kernel()

    def _precompute_kernel(self):
        """Compute RBF kernel matrix and Cholesky factor over horizon timesteps."""
        t = np.arange(self.tsteps) * self.dt
        t1, t2 = np.meshgrid(t, t)
        K = np.exp(-0.5 * (t1 - t2)**2 / self.tau**2)
        K += 1e-6 * np.eye(self.tsteps)
        self.LT = np.linalg.cholesky(K).T

    def update_parameters(self, parameters):
        if 'horizon' in parameters and parameters['horizon'] != self.tsteps:
            self.tsteps = parameters['horizon']
            horizon_s = self.tsteps * self.dt
            self.tau = max(0.1, 0.2 * np.sqrt(horizon_s))
            self._precompute_kernel()

    def __call__(self, state):
        """Generate GP-correlated 1D action sequences [n_samples, 1, tsteps]."""
        z = np.random.randn(self.n_samples, self.tsteps)
        actions = z @ self.LT * self.action_scale
        return actions[:, np.newaxis, :]


class CartPoleMPCEvaluation(Evaluation):
    """Trajectory cost evaluation for MPC."""

    def __call__(self, trajectories, proposals):
        """Sum costs over time for each sample."""
        states, costs = trajectories
        return np.sum(costs, axis=0)

class CartPoleMPCDecision(Decision):
    """Select lowest-cost trajectory, return first n_actions actions."""

    def __call__(self, proposals, trajectories, evaluations, n_actions=1):
        best_idx = np.argmin(evaluations)
        actions = [proposals[best_idx, 0, t] for t in range(n_actions)]
        return actions, best_idx

class TrackMPCProposal(Proposal):
    """GP-correlated random sampling proposal for 2D tracking MPC.

    Generates temporally correlated force trajectories by sampling from a
    Gaussian process prior with RBF kernel. This produces smooth control
    signals that can sustain force in a direction.
    """

    def __init__(self, tsteps=10, dt=0.01, n_samples=500, action_scale=20.0, tau=None):
        """
        tsteps: MPC planning horizon [steps]
        dt: timestep duration [s]
        n_samples: number of random action sequences to evaluate
        action_scale: standard deviation of force magnitude [N]
        tau: GP kernel time constant [s]. If None, scales with horizon
             to keep effective dimensionality bounded.
        """
        self.tsteps = tsteps
        self.dt = dt
        self.n_samples = n_samples
        self.action_scale = action_scale  # [N]
        if tau is None:
            horizon_s = tsteps * dt
            tau = max(0.1, 0.3 * np.sqrt(horizon_s))
        self.tau = tau

        # Precompute GP kernel matrix and Cholesky factor for sampling
        self._precompute_kernel()

    def _precompute_kernel(self):
        """Compute RBF kernel matrix over horizon timesteps."""
        # Time indices [s]
        t = np.arange(self.tsteps) * self.dt

        # RBF kernel: k(t1, t2) = exp(-(t1-t2)^2 / (2*tau^2))
        t1, t2 = np.meshgrid(t, t)
        K = np.exp(-0.5 * (t1 - t2)**2 / self.tau**2)

        # Add small jitter for numerical stability
        K += 1e-6 * np.eye(self.tsteps)

        # Cholesky factor for sampling: samples = L @ z where z ~ N(0, I)
        self.L = np.linalg.cholesky(K)

    def update_parameters(self, parameters):
        if 'horizon' in parameters and parameters['horizon'] != self.tsteps:
            self.tsteps = parameters['horizon']
            horizon_s = self.tsteps * self.dt
            self.tau = max(0.1, 0.3 * np.sqrt(horizon_s))
            self._precompute_kernel()

    def __call__(self, state):
        """Generate GP-correlated 2D action sequences [n_samples, 2, tsteps]. Forces in N."""
        # Sample standard normal [n_samples, 2, tsteps]
        z = np.random.randn(self.n_samples, 2, self.tsteps)

        # Transform through Cholesky factor to get correlated samples
        # L is [tsteps, tsteps], z is [n_samples, 2, tsteps]
        # We want actions[i, d, :] = L @ z[i, d, :]
        actions = np.einsum('ij,ndj->ndi', self.L, z)

        # Scale to desired force magnitude
        actions *= self.action_scale

        return actions


class TrackMPCProposalConst(Proposal):
    """Constant-force proposal for 2D tracking MPC.

    Samples a single (fx, fy) force pair and repeats it for the entire horizon.
    This reduces the search space to 2D, making random sampling very effective.
    """

    def __init__(self, tsteps=10, n_samples=500, action_low=-20.0, action_high=20.0):
        """
        tsteps: MPC planning horizon [steps]
        n_samples: number of random action sequences to evaluate
        action_low, action_high: control force bounds [N]
        """
        self.tsteps = tsteps
        self.n_samples = n_samples
        self.action_low = action_low    # [N]
        self.action_high = action_high  # [N]

    def update_parameters(self, parameters):
        if 'horizon' in parameters:
            self.tsteps = parameters['horizon']

    def __call__(self, state):
        """Generate constant 2D action sequences [n_samples, 2, tsteps]. Forces in N."""
        # Sample single force pair for each sample [n_samples, 2]
        forces = np.random.uniform(
            self.action_low, self.action_high,
            size=(self.n_samples, 2)
        )

        # Repeat across time dimension [n_samples, 2, tsteps]
        actions = np.tile(forces[:, :, np.newaxis], (1, 1, self.tsteps))

        return actions


class TrackMPCEvaluation(Evaluation):
    """Trajectory cost evaluation for tracking MPC."""

    def __call__(self, trajectories, proposals):
        """Sum costs over time for each sample."""
        states, costs = trajectories
        return np.sum(costs, axis=0)


class TrackMPCDecision(Decision):
    """Select lowest-cost trajectory, return first n_actions actions."""

    def __call__(self, proposals, trajectories, evaluations, n_actions=1):
        best_idx = np.argmin(evaluations)
        actions = [proposals[best_idx, :, t] for t in range(n_actions)]
        return actions, best_idx


def _warn_noise_scale(noise_std, scales, label):
    """Warn if noise_std exceeds 50% of any named acceleration scale."""
    if noise_std <= 0:
        return
    for name, accel in scales.items():
        if noise_std > 0.5 * accel:
            warnings.warn(
                f"{label} noise_std={noise_std:.1f} is "
                f"{noise_std / accel:.0%} of {name}={accel:.1f}"
            )


def make_cartpole_mpc(
        agent_args=None,
        adapt_args=None,
        model_args=None,
        noise_std=1.0
        ):
    """Build cartpole MPC agent and environment pair.

    model_args: dict of CartPoleDynamics kwargs for the planning model.
        Omitted parameters use CartPoleDynamics defaults (no mismatch).
    """

    # Warn if noise approaches environmental or control scales
    total_mass = 1.0 + 0.1  # masscart + masspole
    _warn_noise_scale(noise_std, {
        'g/l': 9.8 / 0.5,
        'u_max/(ml)': 10.0 / (total_mass * 0.5),
    }, 'CartPole')

    # Create environment (stateful, with noise) and planning model (stateless, no noise)
    if model_args is None:
        model_args = {}
    env   = CartPoleDynamics(stateless=False, noise_std=noise_std)
    model = CartPoleDynamics(stateless=True, **model_args)

    # Environment initial state
    x0 = np.random.uniform(low=-0.25, high=0.25)    # cart position [m]
    v0 = np.random.uniform(low=-0.1 , high=0.1 )    # cart velocity [m/s]
    p0 = np.random.uniform(low=-0.1 , high=0.1 )    # pole angle [rad]
    q0 = np.random.uniform(low=-0.1 , high=0.1 )    # pole angular velocity [rad/s]

    # Assign to both env and agent model
    env.reset(  np.array([x0, v0, p0, q0]))
    model.reset(np.array([x0, v0, p0, q0]))

    # Proposal: select class via proposal_class key, default to random shooter
    proposal_kwargs = agent_args.get('proposal_args', {}) if agent_args else {}
    proposal_classes = {
        'random': CartPoleMPCRandomShooterProposal,
        'gp': CartPoleMPCGPProposal,
    }
    proposal_class_name = agent_args.get('proposal_class', 'random') if agent_args else 'random'
    proposal_class = proposal_classes[proposal_class_name]
    proposal = proposal_class(**proposal_kwargs)

    # Evaluation and decision
    evaluation = CartPoleMPCEvaluation()
    decision = CartPoleMPCDecision()

    # Inject model cost function for CostErrorAdaptation
    if adapt_args and adapt_args.get('adapt_class') == 'CostErrorAdaptation':
        adapt_args.setdefault('adapt_kwargs', {})['cost_function'] = model.cost_function

    # Set up adaptation
    adaptation = make_adapter(adapt_args)

    # Construct the agent
    recompute_interval = agent_args.get('recompute_interval', 1) if agent_args else 1
    parameters = {'recompute_interval': recompute_interval, 'horizon': proposal.tsteps}
    agent = Agent(proposal, model, evaluation, decision, adaptation=adaptation, parameters=parameters)

    return agent, env


def make_pointmass_mpc(
        agent_args=None,
        adapt_args=None,
        model_args=None,
        noise_std=5.0,
        seed=SEED,
        ):
    """Build pointmass tracking MPC agent and environment pair.

    model_args: dict of PointMass2D kwargs for the planning model (e.g. mass).
        Omitted parameters use PointMass2D defaults (no mismatch).
    """

    # Create force field and dynamics
    force_field = GPForceField(seed=seed)
    initial_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # Warn if noise approaches environmental or control scales
    max_field_force = np.max(np.abs(force_field._table_fx)) + np.max(np.abs(force_field._table_fy))
    _warn_noise_scale(noise_std, {
        'field accel': max_field_force / 0.5,
        'u_max/m': 20.0 / 0.5,
    }, 'PointMass')

    # Create environment (stateful, with noise) and planning model (stateless, no noise)
    if model_args is None:
        model_args = {}
    env = PointMass2D(
        force_field, figure_eight,
        stateless=False, dt=DT,
        noise_std=noise_std,
        initial_state=initial_state,
    )
    model = PointMass2D(
        force_field, figure_eight,
        stateless=True, dt=DT,
        **model_args,
    )

    # Components
    horizon_steps = agent_args.get('horizon_steps', 5) if agent_args else 5
    proposal = TrackMPCProposal(tsteps=horizon_steps, dt=DT)
    evaluation = TrackMPCEvaluation()
    decision = TrackMPCDecision()

    # Inject model cost function for CostErrorAdaptation
    if adapt_args and adapt_args.get('adapt_class') == 'CostErrorAdaptation':
        adapt_args.setdefault('adapt_kwargs', {})['cost_function'] = model.cost_function

    # Set up adaptation
    adaptation = make_adapter(adapt_args) if adapt_args else None

    # Construct the agent
    recompute_interval = agent_args.get('recompute_interval', 1) if agent_args else 1
    parameters = {'recompute_interval': recompute_interval, 'horizon': horizon_steps}

    agent = Agent(proposal, model, evaluation, decision,
                  adaptation=adaptation, parameters=parameters)

    return agent, env
