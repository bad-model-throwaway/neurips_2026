"""Define abstract base classes for agents and environments in order to standardize interfaces."""
from abc import ABC, abstractmethod

import numpy as np

class Dynamics(ABC):
    """Environment model and internal Dynamics model.

    Subclasses override cost_function and _step_stateless; step/forward
    are derived. Initialize with stateless=True to forbid stateful step.
    """

    def __init__(self, stateless):
        self.state = None
        self.cost  = None
        self.stateless = stateless

    @abstractmethod
    def cost_function(self, state, ctrl=None):
        """Return cost for given state (and optionally ctrl).

        State-only envs (CartPole, Hopper) ignore ctrl. MJPC-style
        envs (Walker, HumanoidStand) use it for control-effort residuals.
        Implement vectorized version if needed.
        """
        pass

    @abstractmethod
    def _step_stateless(self, state, action, params=None):
        """Take action, return (next_state, cost). Implement vectorized version if needed."""
        pass

    def snapshot(self):
        """Return a snapshot of the environment's current state."""
        return ['state', 'cost'], [self.state.copy(), self.cost]

    def reset(self, state):
        """Useful because includes cost reset."""
        self.state = state
        self.cost  = self.cost_function(state)

    def _forward_stateless(self, state, actions, params=None):
        """Forward pass without state updating.

        state: [state_dim] or [n_samples, state_dim]
        actions: [action_dim, time] or [n_samples, action_dim, time]
        """
        # Initialize storage
        states, costs = [state], [self.cost_function(state)]

        # Number of time steps
        tsteps = actions.shape[-1]

        # Iterate over time steps
        for t in range(tsteps):
            action = actions[..., t]
            state, cost = self._step_stateless(state, action, params)
            states.append(state)
            costs.append(cost)

        # Stack yields dimensions [..., tsteps + 1]
        return np.stack(states), np.stack(costs)

    def step(self, action, params=None):
        """Update internal state with a single action."""
        if self.stateless: raise RuntimeError("Trying to use stateful step on stateless model.")
        self.state, self.cost = self._step_stateless(self.state, action, params)
        return self.state, self.cost

    def forward(self, actions, params=None):
        """Update internal state over action sequence."""
        if self.stateless: raise RuntimeError("Trying to use stateful step on stateless model.")
        states, costs = self._forward_stateless(self.state, actions, params)
        self.state = states[..., -1]
        self.cost  = costs[..., -1]
        return self.state, self.cost

    def query(self, state, actions, params=None):
        """Exposed stateless forward method for external querying.

        Broadcasts state to match batch dimension of actions if needed.
        """
        # Broadcast state if actions has batch dimension that state lacks
        if actions.ndim == 3 and state.ndim == 1:
            n_samples = actions.shape[0]
            state = np.tile(state, (n_samples, 1))

        return self._forward_stateless(state, actions, params)

class Proposal(ABC):
    @abstractmethod
    def __call__(self, state):
        """Generate action proposals given current state."""
        pass

    def update_parameters(self, parameters):
        """Sync internal state from adaptive parameters. No-op by default."""
        pass

class Evaluation(ABC):
    @abstractmethod
    def __call__(self, states, actions):
        """Evaluate a trajectory, return cost or value."""
        pass

class Decision(ABC):
    @abstractmethod
    def __call__(self, proposals, trajectories, evaluations, n_actions=1):
        """Select action(s) based on proposals, trajectories, and their evaluations.

        Returns (actions_list, best_idx) where actions_list has n_actions entries.
        """
        pass

class Adaptation(ABC):
    """Class for monitoring performance over time and interfacing with adaptation."""
    def __init__(self, analytics=None):
        self.analytics = analytics if analytics is not None else {}

    def snapshot(self):
        """Return adaptation state for history recording. Override in subclasses."""
        return [], []

    @abstractmethod
    def update_monitor(self, state, cost, history, parameters):
        """Keep track of how things are going. Update concerns."""
        pass

    @abstractmethod
    def update_expectations(self, pred_states, pred_costs):
        """Receive information about the current plan."""
        pass

    @abstractmethod
    def adapt_parameters(self, parameters):
        """Adapt parameters based on the monitor"""
        pass

class Agent:
    """Generic agent with optional adaptive metaparameter introspection.

    Use interact() as the sole entry point.
    """
    def __init__(self, proposal, model, evaluation, decision, adaptation=None, parameters=None):
        self.proposal   = proposal
        self.model      = model
        self.evaluation = evaluation
        self.decision   = decision
        self.adaptation = adaptation
        self.parameters = parameters if parameters is not None else {}
        self.queue      = []
        self.replan     = lambda: len(self.queue) == 0
        self.action     = None
        self.history    = []        # For internal use, expose anything via snapshot()

    def interact(self, state, cost):
        """Interact with environment for a number of steps."""

        # Inform the internal tracker of the latest results
        if self.adaptation is not None:
            self.adaptation.update_monitor(state, cost, self.history, self.parameters)

        # Sometimes we replan, sometimes we don't
        if self.replan():
            # Sync any adaptive parameters to components
            self.proposal.update_parameters(self.parameters)

            # Generate set of actions to consider
            proposals = self.proposal(state)

            # Simulate state and cost over time
            trajectories = self.model.query(state, proposals)

            # Evaluate the resulting trajectories
            evaluations = self.evaluation(trajectories, proposals)

            # Make a decision about what action(s) to take
            select_n = self.parameters.get('recompute_interval', 1)
            self.queue, best_idx = self.decision(proposals, trajectories, evaluations, select_n)

            # Notify the tracker of the plan
            if self.adaptation is not None:
                self.adaptation.update_expectations(
                    trajectories[0][:, best_idx, :],
                    trajectories[1][:, best_idx],
                )

        # Pop next action from queue
        self.action = self.queue.pop(0)

        # Remember the state-action pair
        self.history.append((state, self.action, cost, self.parameters.copy()))

        # Update anything adaptive based on history
        if self.adaptation is not None:
            self.adaptation.adapt_parameters(self.parameters)

        return self.action
    
    def snapshot(self):
        """Return a snapshot of the agent's current state."""
        keys = ['action', *self.parameters.keys()]
        vals = [self.action, *self.parameters.values()]
        if self.adaptation is not None:
            adapt_keys, adapt_vals = self.adaptation.snapshot()
            keys.extend(adapt_keys)
            vals.extend(adapt_vals)
        return keys, vals