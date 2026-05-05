# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv pip install numpy matplotlib scipy pytest

# Build Cython extensions (required for performance-optimized rollouts)
python3 setup.py build_ext --inplace

# Run all tests (always via pytest — not directly with python)
uv run pytest tests/

# Run a single test file
uv run pytest tests/test_cartpole.py

# Run the full simulation and figure pipeline
uv run python run.py
```

## Architecture

### Package dependency DAG

```
configs.py  ←  agents/  ←  simulations/  ←  visualization/
                                                     ↓
                                                   run.py
```

`configs.py` is star-imported by all packages. Nothing imports from `tests/` or `visualization/`. This DAG is strict: no skipping levels.

### Agent framework (`agents/`)

The core abstraction is a four-stage pipeline:

```
Proposal(state) → Model.query(state, proposals) → Evaluation(trajectories) → Decision(evaluations) → action
```

- **`base.py`**: Abstract base classes — `Dynamics`, `Proposal`, `Evaluation`, `Decision`, `Agent`
- **`dynamics.py`**: `CartPoleDynamics` and `PointMass2D`. The same class serves as both the real environment (`stateless=False`) and the planning model (`stateless=True`). Model mismatch is introduced via constructor parameters (e.g. `length` factor), not subclassing.
- **`mpc.py`**: Concrete MPC components and `AdaptiveMPCAgent`. Factory functions `make_cartpole_mpc` and `make_pointmass_mpc` are the public API for constructing agent-environment pairs.
- **`adaptation.py`**: `ODEStepAdaptation` and `CostErrorAdaptation` strategies.

`AdaptiveMPCAgent` is the uniform agent type — with defaults (`recompute_interval=1, adapt=()`), it behaves as standard MPC.

Proposal shape convention: `[n_samples, action_dim, tsteps]` for batched, `[action_dim, tsteps]` for single.

### Cython backends

`agents/_cartpole_cy.pyx` and `agents/_pointmass_cy.pyx` are optional performance optimizations. They must return per-step costs in the same shape as NumPy rollouts (cost aggregation belongs to the `Evaluation` component, not the dynamics backend). Backend choice is invisible to the agent framework.

### Simulation layer (`simulations/`)

- `simulation.py`: generic `run_simulation` loop, parallel `run_pool` with streaming `on_result` callback
- Each `sweep_*.py` defines module-level grid constants, a `_worker` pure function, and a single `run_*_sweep` entry point. Workers build their own agent-env pairs and return packed result dicts with full metadata.
- The MPC grid sweep streams per-grid-point pickles to disk via `on_result`; smaller sweeps return in-memory dicts.

### Pipeline execution (`run.py`)

`main()` orchestrates sweeps → figures with a `cache()` helper that loads from `RESULTS_DIR` if a pickle exists, otherwise runs and saves. Output directories: `data/results/`, `data/figures/`, `data/plots/`.

### Visualization (`visualization/`)

Three-layer hierarchy: reusable plot primitives (`plots_*.py`) → per-figure panel functions (`figures.py`, `supplement.py`) → SVG grid composition (`compile.py`). Visualization code never runs simulations. Final figures are publication-width SVGs composed programmatically via `svgtools.py`.

## Conventions

- `verbose > 0` gates print statements; use inline (`if verbose > 0: print(...)`) rather than block-initializing.
- Thread count is fixed at startup in `configs.py` (`OMP_NUM_THREADS=1`) to prevent thread/process contention with `multiprocessing`.
- Tests save diagnostic plots to `data/plots/` and are callable from a REPL as standalone functions.
