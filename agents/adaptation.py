"""Adaptation classes for adaptive parameter control."""
import numpy as np

from agents.base import Adaptation


class CostErrorAdaptation(Adaptation):
    """Threshold-and-streak adaptation using cost-space prediction error.

    Computes signed error in cost space (cost(actual) - cost(predicted)),
    tracks the worst absolute error per recompute interval, and applies
    additive adjustments with asymmetric logic:
    - Sustained low error (streak): relax by +1
    - Error above dynamic threshold: tighten proportionally
    """

    def __init__(self,
            cost_function,
            adapt=(),
            threshold_factor=1.2,
            warmup_steps=20,
            error_window=10,
            min_recompute=1,
            max_recompute=15,
            min_horizon=40,
            max_horizon=60,
            horizon_step=2,
            min_error_threshold=0.01,
            verbose=0
            ):
        super().__init__()
        self.cost_function = cost_function
        self.adapt = set(adapt)
        self.threshold_factor = threshold_factor
        self.warmup_steps = warmup_steps
        self.error_window = error_window
        self.min_recompute = min_recompute
        self.max_recompute = max_recompute
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self.horizon_step = horizon_step
        self.min_error_threshold = min_error_threshold
        self.verbose = verbose

        # Monitor state
        self.predicted_trajectory = None
        self.steps_since_recompute = 0
        self.interval_max_abs_error = 0.0
        self.low_error_streak = 0

        # Sliding window error tracking
        self.error_history = []
        self.total_error_count = 0
        self.mean_error = None
        self.threshold = min_error_threshold

        # Analytics
        self.param_history = {name: [] for name in self.adapt}
        self.n_recomputations = 0
        self.ready_to_adapt = False

    def snapshot(self):
        """Return current adaptation state."""
        error = self.error_history[-1] if self.error_history else None
        return ['mean_error', 'prediction_error', 'threshold'], [self.mean_error, error, self.threshold]

    def update_monitor(self, state, cost, history, parameters):
        """Compute cost-space prediction error, update sliding window."""
        if not history or self.predicted_trajectory is None:
            return

        # Compute cost-space error at current step within interval
        pred_idx = self.steps_since_recompute + 1
        if pred_idx >= self.predicted_trajectory.shape[0]:
            self.steps_since_recompute += 1
            return

        predicted_state = self.predicted_trajectory[pred_idx]
        error = self.cost_function(state) - self.cost_function(predicted_state)

        # Track worst absolute error in this interval
        self.interval_max_abs_error = max(self.interval_max_abs_error, abs(error))

        # Update sliding window and running mean
        self.total_error_count += 1
        self.error_history.append(error)
        if len(self.error_history) > self.error_window:
            self.error_history.pop(0)
        if self.total_error_count >= self.warmup_steps:
            self.mean_error = np.mean(self.error_history)

        self.steps_since_recompute += 1

        # Flag adaptation at recompute boundary
        recompute_interval = parameters.get('recompute_interval', 1)
        at_boundary = self.steps_since_recompute >= recompute_interval
        self.ready_to_adapt = (
            bool(self.adapt) and at_boundary
            and self.mean_error is not None
        )

    def update_expectations(self, pred_states, pred_costs):
        """Receive predicted trajectory from selected plan."""
        self.predicted_trajectory = pred_states
        self.steps_since_recompute = 0
        self.interval_max_abs_error = 0.0
        self.n_recomputations += 1

    def adapt_parameters(self, parameters):
        """Apply threshold-and-streak adaptation at recompute boundaries."""

        # Record current values for analytics
        for name in self.adapt:
            key = 'recompute_interval' if name == 'recompute' else name
            self.param_history[name].append(parameters.get(key))

        if not self.ready_to_adapt:
            return
        self.ready_to_adapt = False

        # Dynamic threshold from running mean, floored at minimum
        worst = self.interval_max_abs_error
        self.threshold = max(
            self.threshold_factor * abs(self.mean_error),
            self.min_error_threshold,
        )

        # Sustained low error: relax by +1 after error_window consecutive low intervals
        if worst < self.min_error_threshold:
            self.low_error_streak += 1
            if self.low_error_streak >= self.error_window:
                self.low_error_streak = 0
                self._adjust(+1, parameters)
            return

        # Error above threshold: tighten proportionally
        self.low_error_streak = 0
        if worst > self.threshold:
            delta = -max(1, int(np.ceil(worst / self.threshold)))
            self._adjust(delta, parameters)

    def _adjust(self, delta, parameters):
        """Apply signed delta to adapted parameters (+relax, -tighten)."""

        # Infer max_recompute from horizon if not set
        max_recompute = self.max_recompute
        if max_recompute is None:
            max_recompute = parameters.get('horizon', 300)

        if 'recompute' in self.adapt:
            old = parameters['recompute_interval']
            new = int(np.clip(old + delta, self.min_recompute, max_recompute))
            parameters['recompute_interval'] = new
            if self.verbose > 0 and new != old:
                print(f"  [Adapt] recompute_interval {old} -> {new}")

        if 'horizon' in self.adapt:
            # Sign flips: relaxing = shorter horizon, tightening = longer
            old = parameters['horizon']
            new = int(np.clip(
                old - delta * self.horizon_step,
                self.min_horizon,
                self.max_horizon,
            ))
            parameters['horizon'] = new

            # Clamp recompute to not exceed horizon
            if parameters.get('recompute_interval', 1) > new:
                parameters['recompute_interval'] = new

            if self.verbose > 0 and new != old:
                print(f"  [Adapt] horizon {old} -> {new}")


class ODEStepAdaptation(Adaptation):
    """ODE step-size adaptation driven by a pluggable error estimate.

    Maintains a sliding window of error estimates and applies ODE step-size
    control at recompute boundaries. Subclasses override estimate_error to
    define the error signal; the default compares predicted vs actual states.

    Each adapted parameter is registered with a direction (+1 = multiply by
    factor when error is low, -1 = divide) and clipping bounds.
    """

    # Registry of adaptable parameters: (param_key, direction, min, max)
    # param_key: key in the agent's parameters dict
    # direction +1: low error -> scale up (e.g. recompute interval relaxes)
    # direction -1: low error -> scale down (e.g. horizon shrinks)
    PARAM_REGISTRY = {
        'recompute': ('recompute_interval', +1, 1 , 15),
        'horizon':   ('horizon',            -1, 30, 70),
    }

    def __init__(self,
            adapt=(),
            min_error_threshold=0.10,
            warmup_steps=50,
            safety=0.9,
            alpha=0.3,
            relax_step=0.2,
            verbose=0,
            timescale=250,
            **bounds_overrides
            ):
        super().__init__()
        self.min_error_threshold = min_error_threshold
        self.warmup_steps = warmup_steps
        self.timescale = timescale
        self.safety     = safety
        self.alpha      = alpha
        self.relax_step = relax_step
        self.verbose    = verbose

        # Build active parameter specs from registry, applying bound overrides
        self.adapt_specs = {}
        for name in adapt:
            param_key, direction, default_min, default_max = self.PARAM_REGISTRY[name]
            lo = bounds_overrides.get(f'min_{name}', default_min)
            hi = bounds_overrides.get(f'max_{name}', default_max)
            self.adapt_specs[name] = (param_key, direction, lo, hi)

        # Monitor state
        self.pred_states = None
        self.pred_costs = None
        self.threshold = min_error_threshold
        self.running_error = None
        self.running_cost = None
        self.error = None
        self.state_pe = None
        self.cost_pe = None
        self.cost_surprise = None
        self.ready_to_adapt = False
        self.steps_since_recompute = 0
        self.steps_since_adapt = 0
        self.total_steps = 0

        # Continuous-valued parameter tracking (initialized lazily)
        self.adapt_values = None

        # Analytics: one history list per adapted parameter
        self.param_history = {name: [] for name in self.adapt_specs}
        self.n_recomputations = 0

    def snapshot(self):
        """Return current adaptation state."""
        keys = ['running_error', 'error', 'state_pe', 'cost_pe', 'cost_surprise', 'threshold']
        vals = [self.running_error, self.error, self.state_pe, self.cost_pe, self.cost_surprise, self.threshold]
        return keys, vals

    def estimate_error(self, state, cost, history):
        """Return scalar error estimate, or None if unavailable.

        Combines state prediction error with log-relative cost surprise.
        State PE measures model accuracy; cost surprise measures plan quality
        degradation relative to running baseline.
        """
        # Index for prediction in plan
        pred_idx = self.steps_since_recompute + 1

        # No prediction implies no error estimate
        if (self.pred_states is None) or (pred_idx >= self.pred_states.shape[0]):
            return None

        # State prediction error
        pred_state = self.pred_states[pred_idx]
        self.state_pe = np.linalg.norm(state - pred_state)

        # Cost prediction error
        pred_cost = self.pred_costs[pred_idx]
        self.cost_pe = abs(cost - pred_cost)

        # Log-relative cost surprise: positive when cost exceeds baseline
        eps = 1e-6
        if self.running_cost is not None:
            self.cost_surprise = max(0.0, np.log(max(cost, eps) / max(self.running_cost, eps)))
        else:
            self.cost_surprise = 0.0

        return self.state_pe + 0.1 * self.cost_surprise

    def update_monitor(self, state, cost, history, parameters):
        """Collect error estimate, update exponential moving averages, flag adaptation."""
        if not history:
            return

        # Update running cost baseline during warmup, then freeze
        if self.running_cost is None:
            self.running_cost = cost
            self.cost_history = [cost]
        elif self.total_steps < self.warmup_steps:
            self.cost_history.append(cost)
            self.running_cost = np.percentile(self.cost_history, 75)

        # Get error from pluggable estimator
        self.error = self.estimate_error(state, cost, history)

        # Update running error average
        if self.error is not None:
            decay = 1.0 / self.timescale
            if self.running_error is None:
                self.running_error = self.error
            else:
                self.running_error += decay * (self.error - self.running_error)

        self.steps_since_recompute += 1
        self.steps_since_adapt += 1
        self.total_steps += 1

        # Flag adaptation at recompute boundary when ready
        recompute_interval = parameters.get('recompute_interval', 1)
        at_boundary = self.steps_since_recompute >= recompute_interval
        self.ready_to_adapt = (
            bool(self.adapt_specs) and at_boundary
            and self.running_error is not None
            and self.total_steps >= self.warmup_steps
            and self.steps_since_adapt >= 50
        )

    def update_expectations(self, pred_states, pred_costs):
        """Receive efference copy of the selected plan."""
        self.pred_states = pred_states
        self.pred_costs = pred_costs
        self.steps_since_recompute = 0
        self.n_recomputations += 1

    def adapt_parameters(self, parameters):
        """Apply ODE step-size control to each registered parameter.

        Maintains continuous float values internally to avoid integer rounding
        deadlock. Modifies parameters only at recompute boundaries when the
        monitor has flagged readiness.
        """
        # Lazy initialization of continuous values from current parameters
        if self.adapt_values is None:
            self.adapt_values = {}
            for name, (param_key, direction, lo, hi) in self.adapt_specs.items():
                self.adapt_values[name] = float(parameters[param_key])

        # Record current values for analytics
        for name, (param_key, direction, lo, hi) in self.adapt_specs.items():
            self.param_history[name].append(parameters.get(param_key))

        if not self.ready_to_adapt:
            return
        self.ready_to_adapt = False
        self.steps_since_adapt = 0

        # Asymmetric: multiplicative tightening, additive relaxation
        self.threshold = self.min_error_threshold
        ratio = self.threshold / max(self.running_error, 1e-12)
        tighten = ratio < 1.0

        for name, (param_key, direction, lo, hi) in self.adapt_specs.items():
            if tighten:
                scale = self.safety * ratio ** self.alpha
                scale = scale if direction > 0 else 1.0 / scale
                self.adapt_values[name] *= scale
            else:
                self.adapt_values[name] += direction * self.relax_step

            self.adapt_values[name] = np.clip(self.adapt_values[name], lo, hi)
            old = parameters[param_key]
            new = int(round(self.adapt_values[name]))
            parameters[param_key] = new
            if self.verbose > 0 and new != old:
                print(f"  [Adapt] {name} {old} -> {new}")


class TheoryStepAdaptation(Adaptation):
    """Closed-loop eigenvalue adaptation via the divergence-ratio estimator.

    Implements the trigger and update rule from supplement Section 12.
    The trigger is the ratio of consecutive end-of-window prediction-error
    magnitudes, with noise-floor subtraction (Eq. 12.8 floor subtraction)
    and log-domain EWMA smoothing. The update rule is AIMD: additive
    cost-favorable when |A_tau| <= epsilon, multiplicative recovery when
    the constraint is violated.
    """

    # (param_key, direction, min, max). Direction +1: cost decreases with the
    # parameter (recompute interval). Direction -1: cost increases with it
    # (horizon).
    PARAM_REGISTRY = {
        'recompute': ('recompute_interval', +1, 1 , 15),
        'horizon':   ('horizon',            -1, 30, 70),
    }

    def __init__(self,
            adapt=(),
            epsilon=0.95,
            c_recovery=0.5,
            a_step=1.0,
            c_gate=2.0,
            ewma_window=5,
            warmup_replans=10,
            noise_floor_window=50,
            noise_floor_pct=5.0,
            verbose=0,
            **bounds_overrides
            ):
        super().__init__()
        self.epsilon = float(epsilon)
        self.log_epsilon = float(np.log(self.epsilon))
        self.c_recovery = float(c_recovery)
        self.a_step = float(a_step)
        self.c_gate = float(c_gate)
        self.ewma_window = int(ewma_window)
        self.ewma_alpha = 2.0 / (self.ewma_window + 1.0)
        self.warmup_replans = int(warmup_replans)
        self.noise_floor_window = int(noise_floor_window)
        self.noise_floor_pct = float(noise_floor_pct)
        self.verbose = verbose

        self.adapt_specs = {}
        for name in adapt:
            param_key, direction, default_min, default_max = self.PARAM_REGISTRY[name]
            lo = bounds_overrides.get(f'min_{name}', default_min)
            hi = bounds_overrides.get(f'max_{name}', default_max)
            self.adapt_specs[name] = (param_key, direction, lo, hi)

        self.pred_states = None
        self.pred_costs = None
        self.steps_since_recompute = 0
        self.n_replans_completed = 0

        self.E_curr = None
        self.E_prev = None
        self.E_window = []
        self.eta = None
        self.sigma = None
        self.gated = False

        self.ready_to_adapt = False
        self.adapt_values = None

        self.param_history = {name: [] for name in self.adapt_specs}
        self.n_recomputations = 0

    def snapshot(self):
        keys = ['E_curr', 'E_prev', 'eta', 'sigma', 'log_epsilon', 'gated']
        vals = [self.E_curr, self.E_prev, self.eta, self.sigma,
                self.log_epsilon, self.gated]
        return keys, vals

    def update_monitor(self, state, cost, history, parameters):
        if not history:
            return

        recompute_interval = parameters.get('recompute_interval', 1)
        pred_idx = self.steps_since_recompute + 1

        # Capture end-of-window divergence E_n and update trigger state.
        end_of_window = (
            self.pred_states is not None
            and pred_idx >= recompute_interval
            and pred_idx < self.pred_states.shape[0]
        )

        if end_of_window:
            E_n = float(np.linalg.norm(state - self.pred_states[pred_idx]))
            self.E_prev = self.E_curr
            self.E_curr = E_n
            self.n_replans_completed += 1

            self._update_noise_floor(E_n)

            if (self.eta is not None and self.E_prev is not None):
                self._update_sigma()

            if (self.n_replans_completed >= self.warmup_replans
                    and self.eta is not None
                    and bool(self.adapt_specs)):
                self.ready_to_adapt = True

        self.steps_since_recompute += 1

    def _update_noise_floor(self, E_n):
        """Track running low-percentile of E_n as the noise floor estimate."""
        self.E_window.append(E_n)
        if len(self.E_window) > self.noise_floor_window:
            self.E_window.pop(0)
        if len(self.E_window) >= self.noise_floor_window:
            pct = float(np.percentile(self.E_window, self.noise_floor_pct))
            self.eta = max(pct, 1e-12)

    def _update_sigma(self):
        """Compute log-ratio sigma with floor subtraction and gating."""
        if max(self.E_curr, self.E_prev) > self.c_gate * self.eta:
            hat_curr = np.sqrt(max(self.E_curr**2 - self.eta**2, 0.0))
            hat_prev = np.sqrt(max(self.E_prev**2 - self.eta**2, 0.0))
            hat_curr = max(hat_curr, 1e-12)
            hat_prev = max(hat_prev, 1e-12)
            log_ratio = float(np.log(hat_curr / hat_prev))
            if self.sigma is None:
                self.sigma = log_ratio
            else:
                self.sigma = (self.ewma_alpha * log_ratio
                              + (1.0 - self.ewma_alpha) * self.sigma)
            self.gated = False
        else:
            self.gated = True

    def update_expectations(self, pred_states, pred_costs):
        self.pred_states = pred_states
        self.pred_costs = pred_costs
        self.steps_since_recompute = 0
        self.n_recomputations += 1

    def adapt_parameters(self, parameters):
        """Apply AIMD update at the replan boundary."""
        if self.adapt_values is None:
            self.adapt_values = {
                name: float(parameters[spec[0]])
                for name, spec in self.adapt_specs.items()
            }

        # Record current parameter values every step
        for name, (param_key, _, _, _) in self.adapt_specs.items():
            self.param_history[name].append(parameters.get(param_key))

        if not self.ready_to_adapt:
            return
        self.ready_to_adapt = False

        # Constraint is satisfied if signal is below threshold or we are
        # in the noise-dominated regime.
        constraint_satisfied = (
            self.gated
            or self.sigma is None
            or self.sigma <= self.log_epsilon
        )

        for name, (param_key, direction, lo, hi) in self.adapt_specs.items():
            if constraint_satisfied:
                self.adapt_values[name] += direction * self.a_step
            else:
                if direction > 0:
                    self.adapt_values[name] *= self.c_recovery
                else:
                    self.adapt_values[name] /= self.c_recovery

            self.adapt_values[name] = float(np.clip(self.adapt_values[name], lo, hi))
            old = parameters[param_key]
            new = int(round(self.adapt_values[name]))
            parameters[param_key] = new
            if self.verbose > 0 and new != old:
                print(f"  [Theory] {name} {old} -> {new}")


def make_adapter(adapt_args=None):
    """Factory for creating adaptation instances."""
    if adapt_args is None:
        return None

    adapt_type   = adapt_args.get('adapt_class')
    adapt_params = adapt_args.get('adapt_params')
    adapt_kwargs = adapt_args.get('adapt_kwargs', {})

    if adapt_type == 'CostErrorAdaptation':
        adapt_class = CostErrorAdaptation
    elif adapt_type == 'ODEStepAdaptation':
        adapt_class = ODEStepAdaptation
    elif adapt_type == 'TheoryStepAdaptation':
        adapt_class = TheoryStepAdaptation
    else:
        raise ValueError(f"Unknown adapt_type: {adapt_type}")

    return adapt_class(adapt=adapt_params, **adapt_kwargs)
