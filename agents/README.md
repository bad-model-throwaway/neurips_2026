# Modular Agent Framework

Defines composable agent architectures for MPC on top of a single **MuJoCo** dynamics backend.
- Separates agents into proposal, model, evaluation, and decision components
- Provides a generic `MuJoCoDynamics` base with three concrete subclasses: CartPole, Walker, HumanoidStand
- Enables batched trajectory rollouts for sampling-based methods
- Includes adaptive MPC that adjusts planning parameters based on prediction error
- Exposes a single `make_mpc(env_name, ...)` factory shared across all environments

## Module Structure

The framework provides abstract base classes that define component interfaces, with concrete implementations for specific control problems.

Files:
- `base.py`: abstract base classes for the agent framework
- `mujoco_dynamics.py`: MuJoCo-backed dynamics — generic `MuJoCoDynamics` base plus `MuJoCoCartPoleDynamics`, `WalkerDynamics`, `HumanoidStandDynamics`. Module-level `MISMATCH_FACTORS` dict holds the canonical 4-level sweep values per env.
- `dynamics.py`: small perturbation-array helper (`make_perturbation_cartpole`) used by the mismatch sweeps. No dynamics classes live here.
- `mpc.py`: MJPC-aligned spline Predictive Sampling (`SplinePSProposal` + `SplinePSArgminDecision`), plus the unified `make_mpc(env_name, H, R, ...)` factory. Per-env defaults live in `PROPOSAL_CONFIGS`.
- `spline.py`: `TimeSpline` utility — zero-order / linear / cubic-Hermite interpolation used by `SplinePSProposal`.
- `adaptation.py`: ODEStepAdaptation and CostErrorAdaptation strategies plus the `make_adapter` factory
- `rewards.py`: dm_control-style `tolerance()` primitive used by the MuJoCo cost functions
- `history.py`: snapshot-based simulation history recording

## Base Classes

### Dynamics

Environment and world model abstraction. Concrete subclasses implement `cost_function` and `_step_stateless`. The base class provides:
- `reset`: Initialize state and cost
- `step`: Stateful single-step update
- `forward`: Stateful multi-step rollout
- `query`: Stateless rollout for planning

### Agent Components

The agent decomposes decision-making into four components:
- **Proposal**: Generates candidate actions given current state
- **Model**: Predicts trajectories from state-action sequences
- **Evaluation**: Scores trajectories
- **Decision**: Selects action based on evaluations

### Adaptation

Optional fifth component that monitors prediction error during simulation and mutates the agent's `parameters` dict (e.g. `recompute_interval`, `horizon`) between replans. Three concrete strategies:
- `ODEStepAdaptation`: tracks per-step ODE residual.
- `CostErrorAdaptation`: tracks signed cost-space prediction error.
- `TheoryStepAdaptation`: tracks the closed-loop coarse eigenvalue via the divergence-ratio estimator from supplement Section 12, with noise-floor subtraction and log-domain EWMA. Update rule is AIMD (additive cost-favorable, multiplicative recovery), matching supplement Section 11's resource-rational adaptation.

All three are constructed via `make_adapter(adapt_args)` where `adapt_args` is a dict:
```python
adapt_args = {
    'adapt_class':  'ODEStepAdaptation',     # or 'CostErrorAdaptation' or 'TheoryStepAdaptation'
    'adapt_params': ('recompute',),          # which parameter(s) to adjust
    'adapt_kwargs': {'min_error_threshold': 0.08},
}
```

### Agent

Generic agent that composes the four mandatory components plus an optional adaptation component:
```python
agent = Agent(proposal, model, evaluation, decision,
              adaptation=make_adapter(adapt_args),
              parameters={'recompute_interval': 1, 'horizon': 50})
action = agent.interact(state, cost)
```

`interact()` runs propose → query → evaluate → decide on each replan, queues actions over `recompute_interval` steps, and (if `adaptation` is set) updates `parameters` based on prediction error.

## MuJoCo Dynamics

All three pipeline environments inherit from a single generic `MuJoCoDynamics` base. The base owns the `mjModel` / `mjData` lifecycle, the stateful/stateless rollout interface, and the `query` batched-rollout used by sampling-based proposals. Each subclass plugs in:
- the XML asset under `agents/xmls/` (loaded once per process)
- a `cost_function(state)` specific to the task
- an `apply_mismatch(factor: float)` method that mutates a single physical parameter on the planning model — this is the contract `make_mpc` and `simulations/sweep_grid.py` rely on to sweep mismatch generically.

### Stage costs (dm_control formulations)

All three environments use stage costs derived from the DeepMind Control Suite (Tassa et al. 2018). Each `cost_function` returns `1 - reward` so cost ∈ `[0, 1]` per step. The `cost_weights` constructor kwarg is **deprecated** and ignored (issues a `DeprecationWarning` if passed non-None).

**CartPole** (`cartpole.swingup`, Tassa 2018 §B.3):
```
upright        = (cos(theta) + 1) / 2
centered       = tolerance(x, margin=2.0)
small_velocity = (1 + tolerance(theta_dot, margin=5.0)) / 2
reward         = upright * centered * small_velocity
cost           = 1 - reward
```
The `small_control` factor is omitted; the stage cost is state-only.

**Walker** (MJPC walker task residuals, 4 terms, all quadratic):
```
cost = 10 * (torso_z - height_goal)^2
     +  3 * (torso_zaxis_z - 1)^2
     +  1 * (com_vx - speed_goal)^2
     +  0.1 * Σ ctrl_i^2
```

**HumanoidStand** (MJPC humanoid/stand residuals, 5 terms — see `stand.cc`): smooth-abs on head-height residual, smooth-abs on balance (capture point), quadratic on CoM velocity, quadratic on joint velocities, cosh on control.

The `tolerance()` primitive is implemented locally in `agents/rewards.py` (no `dm_control` dependency). It matches the dm_control API: `tolerance(x, bounds, margin, sigmoid, value_at_margin)`.

| Env class | XML | State / Action | Mismatch parameter | Direction |
|---|---|---|---|---|
| `MuJoCoCartPoleDynamics` | `cartpole.xml` | 4D / 1D | pole length (scale) | longer planning pole = harder at large H |
| `WalkerDynamics`         | `walker.xml`   | 18D / 6D | torso mass (scale) | heavier torso; clean monotone swing (#68) |
| `HumanoidStandDynamics`  | `humanoid/humanoid.xml` | 55D / 21D | torso mass (scale) | heavier torso; r=2+ exposes long-H breakdown (#74) |

The canonical 4-level sweep values per env (col 1 = exact match, cols 2–4 = increasing mismatch; used by Figure 3) live in `MISMATCH_FACTORS`:

```python
from agents.mujoco_dynamics import MISMATCH_FACTORS

MISMATCH_FACTORS == {
    'CartPole':      [1.0, 2.0, 3.0, 3.5],
    'Walker':        [1.0, 1.6, 2.0, 2.6],
    'HumanoidStand': [1.0, 1.5, 2.0, 2.5],
}
```

## MPC Components

All MPC pieces live in `mpc.py`. The single optimizer is **MJPC-aligned spline Predictive Sampling** (Howell et al. 2022), implemented as a cooperating `SplinePSProposal` / `SplinePSArgminDecision` pair.

- `SplinePSProposal(action_dim, tsteps, n_samples, dt, ctrl_low, ctrl_high, P, sigma, interp, sigma2, mix_prob, include_nominal, clip)`: maintains a nominal `TimeSpline` with `P` knots over the horizon. Each replan it samples `N` candidate splines by perturbing each knot with Gaussian noise scaled by the half-range of the actuator control limits (`sigma × (ctrl_high - ctrl_low) / 2`, matching MJPC `planner.cc:342-345`), renders each candidate on the per-step grid, optionally clips to `[ctrl_low, ctrl_high]`, and returns actions of shape `[N, action_dim, tsteps]`. An optional bimodal noise mix (`sigma2`, `mix_prob`) adds occasional wide perturbations for coverage.
- `SplinePSArgminDecision(proposal)`: picks the single minimum-cost rollout (MJPC `planner.cc:184-188,204`) and warm-starts the next replan via a non-sliding resample of the winning spline shifted forward by `R × dt` (MJPC `planner.cc:295-322`).
- `MPCEvaluation`: sum-of-costs across the horizon.

**Per-env defaults (`PROPOSAL_CONFIGS`)**

`mpc.py` exports a module-level dict that records the proposal type, sample count, and proposal kwargs used per env. All envs use `spline_ps` with `P=3` cubic knots and argmin decision; `sigma` and `N` vary per env (MJPC's shipped values for walker; tuned values for cartpole/humanoid_stand).

```python
from agents.mpc import PROPOSAL_CONFIGS

# See agents/mpc.py for the current values; cartpole uses sigma=0.3 N=30,
# walker uses sigma=0.5 N=30 with speed_goal=1.5,
# humanoid_stand uses sigma=0.25 N=30.
```

## Factory: `make_mpc`

`mpc.py` provides one factory that constructs an agent for any supported env:

```python
from agents.mpc import make_mpc

agent = make_mpc(
    env_name='cartpole',          # 'cartpole' | 'walker' | 'humanoid_stand'
    H=50, R=1,                    # planning horizon, recompute interval
    N=None,                       # samples per replan (defaults from PROPOSAL_CONFIGS)
    mismatch_factor=1.0,          # 1.0 = exact match; passed to model.apply_mismatch
    proposal=None,                # only 'spline_ps' is supported; None uses PROPOSAL_CONFIGS
    proposal_kwargs=None,         # passed through to SplinePSProposal
    decision=None,                # must be 'spline_ps_argmin' if set
)
```

Resolution order for `proposal` / `N` / `proposal_kwargs`:
1. **Explicit kwargs win** — `make_mpc('walker', H, R, N=400)` uses N=400 regardless of what the dict says.
2. Otherwise, defaults are read from `PROPOSAL_CONFIGS[env_name]`.
3. Unknown `env_name` raises `ValueError` — no silent fallback.

Examples:

```python
# CartPole, exact-match planning model, defaults from PROPOSAL_CONFIGS
agent = make_mpc('cartpole', H=50, R=1)

# Walker, custom sample budget
agent = make_mpc('walker', H=60, R=1, N=400)
```

## Test coverage

End-to-end coverage for each environment in `PROPOSAL_CONFIGS` lives under `tests/`, complemented by unit tests for the spline-PS primitives and adaptation state machines. Closed-loop tests use the default `PROPOSAL_CONFIGS` entry for their env and assert task-specific success predicates (kept in `tests/shared.py`). The spline-PS unit tests map one-to-one onto the four MJPC-alignment features from issue #60.

| Test file | What it covers |
|---|---|
| `tests/test_cartpole.py` | Cartpole matched-dynamics smoke + perturbation recovery + adaptation (`ODEStepAdaptation`, `CostErrorAdaptation`; `recompute` and `horizon` arms) |
| `tests/test_walker.py`   | Walker matched-dynamics smoke (standing + forward progress at `speed_goal=1.5`) |
| `tests/test_humanoid_stand.py` | Humanoid-stand matched-dynamics smoke (head stays above 1.3 m, tail-half mean \|r_height\| < 0.1 m) + `test_visual_rollout` (`visual` marker) |
| `tests/test_spline_ps.py` | MJPC-alignment unit tests: `include_nominal`, two-σ mix (`sigma2` / `mix_prob`), action clipping, plan-shift / warm-start, plus proposal shape and argmin decision |
| `tests/test_spline.py`    | Spline interpolation primitives (zero / linear / cubic-Hermite) |
| `tests/test_adaptation.py` | Adaptation state machines |
| `tests/test_rewards.py`    | `rewards.tolerance()` (imported by `mujoco_dynamics.py`) |

**Rollout videos (`visual` marker).** Each env smoke test file also ships a `test_visual_rollout` gated behind `@pytest.mark.visual`. The default `pytest tests/` run excludes them (via `addopts = ["-m", "not visual"]` in `pyproject.toml`), so the core suite stays fast and does not need a GL context. Opt in with `pytest -m visual tests/`, which re-runs the matched-dynamics rollout for cartpole and walker and writes `test_<env>_rollout.mp4` to `PLOTS_DIR` (`./data/plots/`) via the `record_rollout_video` helper in `tests/shared.py`. Offscreen rendering works out of the box on macOS; on headless Linux set `MUJOCO_GL=osmesa` before running.

## Usage Pattern

All implementations follow the same base pattern:

```python
# Create dynamics for environment and planning
env   = SomeDynamics(stateless=False)
model = SomeDynamics(stateless=True)

# Create agent components
proposal   = SomeProposal(...)
evaluation = SomeEvaluation(...)
decision   = SomeDecision(...)

# Build agent
agent = Agent(proposal, model, evaluation, decision)

# Interaction loop
env.reset(initial_state)
for _ in range(n_steps):
    action = agent.interact(env.state, env.cost)
    env.step(action)
```

The adaptive variant differs only by the `adaptation` argument:

```python
from agents.adaptation import make_adapter

agent = Agent(
    proposal, model, evaluation, decision,
    adaptation=make_adapter({
        'adapt_class':  'ODEStepAdaptation',
        'adapt_params': ('recompute',),
        'adapt_kwargs': {'min_error_threshold': 0.08},
    }),
    parameters={'recompute_interval': 4, 'horizon': 50},
)
```

## Design Notes

The framework uses costs rather than rewards. Evaluation and decision components minimize cost via argmin.

Proposal shape convention: `[n_samples, action_dim, tsteps]` for batched methods, `[action_dim, tsteps]` for single proposals.
