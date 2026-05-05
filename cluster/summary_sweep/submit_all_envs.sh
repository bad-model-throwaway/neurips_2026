#!/bin/bash
#SBATCH --time=0:10:00
#SBATCH --mem=256M
#SBATCH --cpus-per-task=1
#SBATCH --job-name=sum_submitter
#SBATCH -o log/submitter.%j.out
#SBATCH -e log/submitter.%j.err
#
# Submit Figure 3 summary sweep arrays for all three environments.
# Mirrors cluster/grid_sweep/submit_all_envs.sh.
#
# Total slots: cartpole(63) + walker(57) + humanoid_balance(54) = 174 —
# well under the 1200-task CCV QOS cap, so all three can be submitted at once
# without QOS gating. We still run on a compute node (via sbatch) to avoid
# login-node policy violations.
#
# Usage:
#   cd cluster/summary_sweep/
#   sbatch submit_all_envs.sh          # recommended
#
# After all per-cell jobs finish (for each env):
#   python aggregate.py --env cartpole
#   python aggregate.py --env walker
#   python aggregate.py --env humanoid_balance
#   cp results/summary_*.pkl ../../data/results/

set -eu

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${SCRIPT_DIR}"
mkdir -p log

# Lmod's `module` function isn't auto-defined under bare `bash`; source
# /etc/profile.d/zz_activate_lmod_user.sh first so this works in every
# context (login shell, sbatch script, nested bash invocation).
source /etc/profile.d/zz_activate_lmod_user.sh 2>/dev/null || true
module load anaconda3/2023.09-0-aqbc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate shitty_bird_env

ENVS=(cartpole walker humanoid_balance)

for env in "${ENVS[@]}"; do
    echo "$(date '+%F %T')  Submitting ${env}"
    bash "${SCRIPT_DIR}/submit_all.sh" --env "${env}"
done

echo
echo "$(date '+%F %T')  All ${#ENVS[@]} envs submitted: ${ENVS[*]}"
echo
echo "After all slots finish, aggregate each env:"
for env in "${ENVS[@]}"; do
    echo "  python aggregate.py --env ${env}"
done
echo "  cp results/summary_*.pkl ../../data/results/"
