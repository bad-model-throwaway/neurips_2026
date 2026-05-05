"""SARSA agent components for discrete action spaces."""
import numpy as np

from agents.base import Proposal, Evaluation, Decision, Agent


class TabularQ:
    """Tabular Q-function for discrete state-action spaces."""

    def __init__(self, n_states, n_actions, initial_value=0.0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.table = np.full((n_states, n_actions), initial_value)

    def __call__(self, state, actions=None):
        """Return Q values for state and optionally specific actions."""
        if actions is None:
            return self.table[state]
        return self.table[state, actions]

    def update(self, state, action, target, alpha):
        """Update Q value toward target."""
        self.table[state, action] += alpha * (target - self.table[state, action])


class IdentityModel:
    """Pass-through model that returns state unchanged."""

    def query(self, state, actions):
        """Return state as trajectory (no dynamics)."""
        return state


class SARSAProposal(Proposal):
    """Propose all discrete actions for evaluation."""

    def __init__(self, n_actions):
        self.n_actions = n_actions

    def __call__(self, state):
        """Return all actions as proposals [n_actions, 1, 1]."""
        return np.arange(self.n_actions).reshape(-1, 1, 1)


class SARSAEvaluation(Evaluation):
    """Evaluate actions using Q function."""

    def __init__(self, Q):
        self.Q = Q

    def __call__(self, trajectories, proposals):
        """Return Q values for each proposed action.

        trajectories: state passed through from IdentityModel
        """
        state = trajectories
        actions = proposals[:, 0, 0]
        return self.Q(state, actions)


class SARSADecision(Decision):
    """Epsilon-greedy action selection."""

    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def __call__(self, proposals, trajectories, evaluations, n_actions=1):
        """Select action epsilon-greedy with respect to Q values (minimizing cost)."""
        actions = proposals[:, 0, 0]
        if np.random.random() < self.epsilon:
            best_idx = np.random.choice(len(actions))
        else:
            best_idx = np.argmin(evaluations)
        return [actions[best_idx]], best_idx


class SARSAAgent(Agent):
    """SARSA agent with TD learning."""

    def __init__(self, proposal, model, evaluation, decision, Q, alpha=0.1, gamma=0.99):
        super().__init__(proposal, model, evaluation, decision)
        self.Q = Q
        self.alpha = alpha
        self.gamma = gamma

    def update_parameters(self):
        """Perform SARSA update using history."""

        # Need at least two transitions to form (s, a, r, s', a')
        if len(self.history) < 2:
            return

        # Get previous and current transitions
        s, a, c = self.history[-2]
        s_next, a_next, _ = self.history[-1]

        # SARSA target: r + gamma * Q(s', a')
        target = c + self.gamma * self.Q(s_next, a_next)

        # Update Q
        self.Q.update(s, a, target, self.alpha)


def test_sarsa_gridworld():
    """Test SARSA on simple gridworld."""

    # Simple 1D gridworld: states 0-9, goal at state 9
    n_states = 10
    n_actions = 2  # 0=left, 1=right

    # Create components
    Q = TabularQ(n_states, n_actions, initial_value=0.0)
    proposal = SARSAProposal(n_actions)
    model = IdentityModel()
    evaluation = SARSAEvaluation(Q)
    decision = SARSADecision(epsilon=0.1)

    # Build agent
    agent = SARSAAgent(proposal, model, evaluation, decision, Q, alpha=0.1, gamma=0.99)

    # Training loop
    n_episodes = 100
    for episode in range(n_episodes):
        state = 0
        cost = 1
        agent.history = []
        steps = 0

        while state != n_states - 1 and steps < 50:
            action = agent.interact(state, cost)

            # Environment step
            if action == 0:
                state = max(0, state - 1)
            else:
                state = min(n_states - 1, state + 1)
            cost = 0 if state == n_states - 1 else 1
            steps += 1

        if episode % 20 == 0:
            print(f"Episode {episode}: steps={steps}")

    # Print learned Q values
    print("\nLearned Q values (state x action):")
    print("State | Left  | Right")
    for s in range(n_states):
        print(f"  {s}   | {Q.table[s, 0]:5.2f} | {Q.table[s, 1]:5.2f}")


if __name__ == "__main__":
    test_sarsa_gridworld()
