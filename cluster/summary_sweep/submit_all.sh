#!/bin/bash
#
# Submit a SLURM array for one env's Figure 3 summary sweep.
# Mirrors cluster/grid_sweep/submit_all.sh.
#
# Index space: one slot per (condition_idx, mismatch_idx).
# Each slot runs N_EPISODES reps sequentially on 1 CPU.
#
#   Cartpole:         3 conditions × 11 mismatches =  33 slots
#   Walker:           3 conditions × 13 mismatches =  39 slots
#   Humanoid balance: 3 conditions × 14 mismatches =  42 slots
#
# Usage:
#   cd cluster/summary_sweep/
#   bash submit_all.sh --env cartpole
#   bash submit_all.sh --env walker
#   bash submit_all.sh --env humanoid_balance
#
# After all jobs finish:
#   python aggregate.py --env <env>
#   cp results/summary_<env>.pkl ../../data/results/

set -eu

ENV=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env) ENV="$2"; shift 2 ;;
        *)     echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -z "$ENV" ]] && { echo "usage: $0 --env <env>"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "${SCRIPT_DIR}/log" "${SCRIPT_DIR}/results"

# Load conda to query sweep dimensions.
# Lmod's `module` function isn't auto-defined under bare `bash`; source
# /etc/profile.d/zz_activate_lmod_user.sh first so this works in every
# context (login shell, sbatch script, nested bash invocation).
source /etc/profile.d/zz_activate_lmod_user.sh 2>/dev/null || true
module load anaconda3/2023.09-0-aqbc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate shitty_bird_env

N_JOBS=$(python -c "
import sys; sys.path.insert(0, '${REPO_ROOT}')
from simulations.sweep_cartpole_summary         import CONDITIONS as CC, MISMATCHES as CM
from simulations.sweep_walker_summary           import CONDITIONS as WC, MISMATCHES as WM
from simulations.sweep_humanoid_balance_summary import CONDITIONS as HC, MISMATCHES as HM
d = {'cartpole': (CC, CM), 'walker': (WC, WM), 'humanoid_balance': (HC, HM)}
c, m = d['${ENV}']
print(len(c) * len(m))
")

# Wall-time budget: reps are sequential in each slot at N_EPISODES=100.
# Cartpole: ~2 min/ep × 100 = ~3.3 h  → budget 6h
# Walker:   ~5 min/ep × 100 = ~8.3 h  → budget 16h
# Humanoid: ~5 min/ep × 100 = ~8.3 h  → budget 16h (5 min hard timeout
#                                       on each ep via SIGALRM in the worker)
# Memory: 4G covers cartpole/humanoid; walker's Fixed-frequent (R=1) cond
# allocates ~1000 plans × Spline-PS samples × MuJoCo state copies and OOMs
# at 4G — bump to 8G (matches grid_sweep/submit_all.sh).
case "$ENV" in
    cartpole)          SLURM_TIME="6:00:00"  ; SLURM_MEM="4G" ;;
    walker)            SLURM_TIME="16:00:00" ; SLURM_MEM="8G" ;;
    humanoid_balance)  SLURM_TIME="16:00:00" ; SLURM_MEM="4G" ;;
    *) echo "Unknown env: ${ENV}"; exit 1 ;;
esac

cd "${SCRIPT_DIR}"

sbatch \
    --job-name="sum_${ENV}" \
    --time="${SLURM_TIME}" \
    --mem="${SLURM_MEM}" \
    --cpus-per-task=1 \
    --nodes=1 \
    --array=0-$((N_JOBS - 1)) \
    -o "log/sum_${ENV}_%a.%j.out" \
    -e "log/sum_${ENV}_%a.%j.err" \
    run_one_cell.sh "$ENV"

echo "Submitted ${N_JOBS}-slot array for env=${ENV} (time=${SLURM_TIME})"
echo "After all slots finish:"
echo "  cd ${SCRIPT_DIR}"
echo "  python aggregate.py --env ${ENV}"
echo "  cp results/summary_${ENV}.pkl ${REPO_ROOT}/data/results/"
