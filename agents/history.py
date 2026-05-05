import numpy as np

class History():
    """Snapshot-based history. Keys discovered once from agent and env."""
    def __init__(self, agent, env):
        env_keys, _ = env.snapshot()
        agent_keys, _ = agent.snapshot()
        self.keys = ['step'] + env_keys + agent_keys
        self.record = []

    def update(self, step, agent, env):
        _, env_vals = env.snapshot()
        _, agent_vals = agent.snapshot()
        self.record.append((step, *env_vals, *agent_vals))

    # Per-key format: (width, decimals) for floats, width for ints
    FMT = {
        'step':               '{key}={val:>5d}',
        'state':              '{key}=[{state}]',
        'cost':               '{key}={val:5.1f}',
        'action':             '{key}={val:6.2f}',
        'recompute_interval': '{key}={val:>3d}',
        'horizon':            '{key}={val:>3d}',
        'running_error':      '{key}={val:5.2f}',
        'error':              '{key}={val:5.2f}',
        'state_pe':           '{key}={val:5.2f}',
        'cost_pe':            '{key}={val:5.2f}',
        'cost_surprise':      '{key}={val:5.2f}',
        'threshold':          '{key}={val:5.2f}',
    }

    def _fmt(self, key, val):
        """Format a single key-value pair for printing."""
        if val is None:
            return f"{key}={'N/A':>5s}"
        fmt = self.FMT.get(key)
        if fmt is None:
            return f"{key}={val}"
        if key == 'state':
            state = ' '.join(f'{v:+6.2f}' for v in val)
            return fmt.format(key=key, state=state)
        if isinstance(val, np.ndarray):
            arr = ' '.join(f'{v:+6.2f}' for v in val)
            return f"{key}=[{arr}]"
        return fmt.format(key=key, val=val)

    def print_last(self):
        r = self.record[-1]
        parts = [self._fmt(k, r[i]) for i, k in enumerate(self.keys)]
        print(', '.join(parts))

    def get_item_history(self, key):
        """Extract a list of values for a given key from the history."""
        idx = self.keys.index(key)
        return np.array([r[idx] for r in self.record])

    def get_state_action_cost(self):
        """Extract states, actions, costs arrays from History tuples."""
        states  = self.get_item_history('state')
        actions = self.get_item_history('action')
        costs   = self.get_item_history('cost')
        return states, actions, costs

