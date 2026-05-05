#!/bin/bash
#
# Run aggregate.py for every env — writes results/grid_<env>.pkl per env
# from the per-cell pickles produced by submit_all.sh.
#
# Usage:
#   cd cluster/grid_sweep/
#   bash aggregate_all_envs.sh [--smoke]

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# On CCV (or any host with Environment Modules) activate the same conda env the
# SLURM jobs use, so aggregate.py can import tqdm / mujoco / etc. No-op locally.
if command -v module >/dev/null 2>&1; then
    # Lmod's `module` function isn't auto-defined under bare `bash`; source
    # /etc/profile.d/zz_activate_lmod_user.sh first so this works in every
    # context (login shell, sbatch script, nested bash invocation).
    source /etc/profile.d/zz_activate_lmod_user.sh 2>/dev/null || true
    module load anaconda3/2023.09-0-aqbc
    source activate shitty_bird_env
fi

ENVS=(cartpole cartpole_quadratic walker humanoid_stand humanoid_balance humanoid_stand_gravity)

for env in "${ENVS[@]}"; do
    echo "=============================================="
    echo "Aggregating ${env}"
    echo "=============================================="
    python "${SCRIPT_DIR}/aggregate.py" --env "${env}" "$@" || {
        echo "  (aggregate.py failed for ${env} — continuing)"
    }
done

echo
echo "Aggregated ${#ENVS[@]} envs: ${ENVS[*]}"
echo "Output pickles in: ${SCRIPT_DIR}/results/grid_<env>.pkl"
