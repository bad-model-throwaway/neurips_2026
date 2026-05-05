# Continuous Predictive Control with Bad World Models

Simulation framework for studying how model predictive control tolerates model error through frequent replanning. Sweeps planning horizon and recompute interval jointly under parametric model mismatch, and develops adaptive controllers that adjust planning parameters via prediction error.

The project implements a modular agent architecture and runs parameter sweeps across three environments — **CartPole**, **Walker**, and **HumanoidStand** — all backed by a single **MuJoCo** dynamics layer.

- `agents/` implements the propose-evaluate-decide agent loop with adaptation, on top of MuJoCo dynamics
- `simulations/` runs parallel parameter sweeps producing pickle results
- `visualization/` generates manuscript figures using a panel-per-SVG architecture
- `tests/` verifies agent construction and simulation correctness
- `cluster/` launches parallel simulation jobs on HPC

The pipeline flow proceeds as:

1. Diagnostics: Probe per-env control quality, replan-time scaling, and mismatch sensitivity
2. Sweep: Run a unified MPC grid sweep over horizon, recompute interval, and model mismatch (Figure 2)
3. Adaptive: Run adaptive MPC sweeps across mismatch levels (Figure 3)
4. Compile: Assemble multi-panel SVG figures for publication

## Top-Level Files

Pipeline execution:

- `configs.py`: centralized configuration (paths, timestep, seed, plot settings, worker count, MuJoCo XML path)
- `analysis.py`: analysis utilities for sweep results (cost rates, landscapes, recompute traces)
- `run.py`: run the full pipeline (diagnostics, sweeps, then figures)

## Agents

Modular agent framework with abstract base classes and interchangeable components, built on a single MuJoCo dynamics backend. See [agents/README.md](agents/README.md).

- `base.py`: abstract base classes for Dynamics, Proposal, Evaluation, Decision, Adaptation, Agent
- `mujoco_dynamics.py`: MuJoCo-backed dynamics — generic `MuJoCoDynamics` base plus `MuJoCoCartPoleDynamics`, `WalkerDynamics`, and `HumanoidStandDynamics`. Each subclass provides an `apply_mismatch(factor)` method that mutates its planning model. `MISMATCH_FACTORS` exposes the canonical 4-level sweep values per env.
- `dynamics.py`: small perturbation-array helpers used by the mismatch sweeps
- `mpc.py`: MJPC-aligned spline Predictive Sampling (`SplinePSProposal` + `SplinePSArgminDecision`, Howell et al. 2022) as the single optimizer. The unified `make_mpc(env_name, H, R, ...)` factory and per-env defaults live in `PROPOSAL_CONFIGS`.
- `spline.py`: `TimeSpline` utility (zero-order / linear / cubic-Hermite) used by `SplinePSProposal`
- `adaptation.py`: ODEStepAdaptation and CostErrorAdaptation strategies plus `make_adapter`
- `rewards.py`: dm_control-style `tolerance()` primitive used by the MuJoCo cost functions
- `history.py`: snapshot-based simulation history recording

## Simulations

Runs simulations for manuscript and supplementary figures. This subpackage should not contain analysis or model logic.

- `simulation.py`: agent-environment loop, history unpacking, parallel execution
- `dataio.py`: result packing, saving, and loading utilities
- `diagnostics.py`: control-quality, timing-model, and mismatch-sensitivity probes per env. Runnable as `python -m simulations.diagnostics`
- `sweep_grid.py`: unified `(H, R, mismatch, rep)` grid sweep across all three envs (Figure 2 backbone). Exposes `DEFAULT_GRIDS` (publication) and `SMOKE_GRIDS` (3-CPU local). Runnable as `python -m simulations.sweep_grid --env {cartpole,walker,humanoid_stand} [--smoke]`
- `sweep_cartpole_adaptive.py`: adaptive MPC sweep across cartpole mismatch levels (Figure 3)
- `sweep_cartpole_summary.py`: cartpole cost-vs-mismatch summary panel (wraps `sweep_cartpole_adaptive` with four conditions)

## Visualization

Panel-per-SVG architecture for manuscript figures. This subpackage should not contain simulation, analysis, or content logic.

- `figures.py`: main figure panel functions (Figures 2--4)
- `supplement.py`: supplement figure panel functions
- `compile.py`: SVG grid composition with row/column labels
- `heatmaps.py`: 1×4 heatmap row panel (env × mismatch column) used by the new 3×4 Figure 3
- `plots_cartpole.py`: cartpole timeseries and adaptation panel functions
- `plots_sweep.py`: heatmaps, sweep plots, and error distributions
- `svgtools.py`: SVG manipulation and PDF conversion

## Tests

Run all tests with:

```bash
uv run pytest tests/
```

Do not run test files directly with `python tests/file.py` — the project root must be on `sys.path`, which pytest handles via the `pythonpath` setting in `pyproject.toml`.

- `shared.py`: convergence and stability check helpers
- `test_cartpole.py`: cartpole balance, perturbation, adaptation under mismatch (MuJoCo backend)
- `test_adaptation.py`: ODE and cost adaptation unit tests
- `test_spline.py`: `TimeSpline` interpolation unit tests
- `test_spline_ps.py`: `SplinePSProposal` / `SplinePSArgminDecision` unit tests (shape, nominal, noise scaling, clipping, warm-start, argmin)
- `test_rewards.py`: `tolerance()` primitive unit tests

## Setup

### 1. Python dependencies

```bash
uv pip install numpy scipy matplotlib pytest cairosvg mujoco imageio[ffmpeg]
```

`mujoco` is required by every dynamics class — there is no alternative backend.

### 2. MuJoCo XML assets (bundled)

MJCF model files for the MuJoCo dynamics wrappers are vendored in `agents/xmls/` (copied from [google-deepmind/mujoco_playground](https://github.com/google-deepmind/mujoco_playground) at commit `dd38c28`; see `agents/xmls/LICENSE`). No extra setup is needed.

To use a different set of XML models, override with the `MUJOCO_XML_DIR` environment variable:

```bash
MUJOCO_XML_DIR=/path/to/xmls python run.py
```

## Running the pipeline

```bash
python run.py
```

This invokes `main()` in [run.py](run.py), which orchestrates diagnostics, sweeps, and figure compilation, caches sweep results to `data/results/*.pkl`, and renders manuscript figures. `main()` accepts boolean flags to selectively enable each experiment:

| Flag | Default | Effect |
|---|---|---|
| `diagnostics_cartpole` | `True` | CartPole control-quality, timing, and mismatch-sensitivity probes |
| `metagrid_cartpole` | `True` | CartPole `(H, R, mismatch)` grid sweep + Figure 2 cartpole panel |
| `metagrid_walker` | `True` | Walker `(H, R, mismatch)` grid sweep + Figure 2 walker panel |
| `metagrid_humanoid_stand` | `True` | HumanoidStand `(H, R, mismatch)` grid sweep + Figure 2 humanoid panel |
| `adaptive_cartpole` | `False` | CartPole adaptive sweep + Figure 3 cartpole row + Supplement 3 + walker summary/perturbation (currently bundled with cartpole) |

Override individual flags by importing `main` and calling it directly:

```python
from run import main
main(adaptive_cartpole=True, metagrid_walker=False)
```

Set `SMOKE=1` to run the `(H, R)` grid sweeps on the smaller `SMOKE_GRIDS` (intended for a 3-CPU local check):

```bash
SMOKE=1 N_WORKERS=3 python run.py
```

Pipeline utilities exposed in [run.py](run.py):

- `cache(filename, func, **kwargs)`: load cached pickle from `data/results/`, or run `func` and save.
- `clean(results=False, figures=False, plots=False)`: clear the corresponding output directory.
- `clean_metagrid()`: remove `grid_*.pkl` caches.

## Parallelism

Sweep modules run jobs in parallel via `multiprocessing.Pool` (with the `spawn` start method, required by MuJoCo). The worker count is controlled by `N_WORKERS` in `configs.py` (default: **10**). Override via environment variable for cluster runs:

```bash
N_WORKERS=32 python run.py
```

The pool size is also clipped to `cpu_count - 2` so a high `N_WORKERS` value will not exceed available cores.

## Roadmap

- ~~**Single MuJoCo backend across CartPole, Hopper**~~ — landed in the `repo_restructure_mujoco` branch.
- ~~**Retire dense planners (MPPI / GP / random-shooter) and consolidate on spline-PS**~~ — landed in issue #76.
- ~~**Retire Hopper in favor of Walker + HumanoidStand**~~ — landed in this branch.

