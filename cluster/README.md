# Cluster sweeps

SLURM driver scripts for the Figure 2 grid sweep on CCV. All commands below
assume you're on a CCV login node (e.g. `login009`) and `cd`'d into
`cluster/grid_sweep/`.

## Full sweep (all 6 envs)

Envs: `cartpole`, `cartpole_quadratic`, `walker`, `humanoid_stand`,
`humanoid_balance`, `humanoid_stand_gravity`.

1. **Submit all envs.** This is a QOS-aware submitter — it sbatches one env at
   a time and waits for the queue to drain below the CCV condo cap before
   submitting the next. Runs on a compute node so it doesn't tie up the login
   node:

   ```bash
   sbatch submit_all_envs.sh            # production (DEFAULT_GRIDS)
   sbatch submit_all_envs.sh --smoke    # smoke grids, reps=3 — run this first
   ```

   Monitor with `squeue -u $USER`. The submitter itself is the `sweep_submitter`
   job; per-env arrays show up as `grid_<env>` (or `smoke_<env>`).

2. **Aggregate once all per-cell jobs finish.** Writes one
   `results/grid_<env>.pkl` per env from the atomic per-cell pickles:

   ```bash
   bash aggregate_all_envs.sh           # matches the sbatch above
   bash aggregate_all_envs.sh --smoke   # matches the smoke sbatch
   ```

   This script activates the conda env itself, so it works from a bare login
   shell. It's pickle I/O only (seconds), so it stays on the login node —
   don't `sbatch` it.

   If any cells are missing, `aggregate.py` prints the exact
   `sbatch --array=<indices>` line to resubmit.

## Single env

Same pattern, one env at a time:

```bash
bash submit_all.sh --env cartpole [--smoke] [--n-steps N]
# ...wait for array to finish...
python aggregate.py --env cartpole [--smoke]
```

### CartPole quadratic-cost variant (issue #145)

`cartpole_quadratic` is a separate env entry that shares MuJoCoCartPoleDynamics
with `cartpole` but swaps the bounded tolerance-product cost for strict LQR-
balancing quadratic `g(x,u) = xᵀQx + uᵀRu` (Q=diag(1, 0.1, 3, 1), R=0.1; weights
in `PROPOSAL_CONFIGS['cartpole_quadratic'].env_kwargs`). Run it the same way:

```bash
bash submit_all.sh --env cartpole_quadratic [--smoke]
python aggregate.py --env cartpole_quadratic [--smoke]
```

It is included in `submit_all_envs.sh` so the full 6-env sweep covers it
automatically. Run it via the single-env path above when you only need
fresh data for #147's matrix-Riccati theory overlay without re-sweeping
everything else.

## Outputs

Per-cell pickles live in `grid_sweep/results/<env>_grid_cells/` (or
`<env>_smoke_cells/`). Aggregated results land at
`grid_sweep/results/grid_<env>.pkl`. To feed them into the figure code,
copy into `data/results/`.
