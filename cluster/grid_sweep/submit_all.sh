#!/bin/bash
#
# Submit one SLURM job array for one env — one array slot per (H, R, factor)
# triple. Each slot runs reps episodes sequentially on one CPU, writing one
# atomic per-cell pickle per episode. Idempotent: existing pickles are
# skipped, so re-running this script after a partial batch only re-runs the
# missing slots.
#
# Usage:
#   cd cluster/grid_sweep/
#   bash submit_all.sh --env cartpole [--smoke] [--n-steps N] \
#                      [--chunk-index I --total-chunks K]
#
#   --smoke         Use SMOKE_GRIDS (fewer H values, reps=3) instead of DEFAULT_GRIDS.
#                   Run this first after a new deploy to verify the pipeline and
#                   get per-cell timing estimates before launching the full sweep.
#   --n-steps N     Override episode length (e.g. --n-steps 50 for a fast timing probe).
#   --total-chunks K  Total number of rep-chunks across which one cell is split.
#                     Default: 1. At reps=100 the worst cell (walker H=183 R=1,
#                     ~100 min/episode) needs K=4 to stay under 48 h; K=5 is safer.
#   --chunk-index I   Which chunk this submission covers (0 ≤ I < K). Default: 0.
#                     Submits a single SLURM array of n_cells tasks where each
#                     slot runs reps [I*chunk_size, (I+1)*chunk_size). Submitting
#                     all chunks at once for one env exceeds the 1200-task QOS
#                     cap, so submit_all_envs.sh iterates (env, chunk) with a
#                     queue gate between submissions.
#
# After all jobs finish:
#   python aggregate.py --env <env> [--smoke]

set -eu

ENV=""
SMOKE=""
N_STEPS=""
CHUNK_INDEX=0
TOTAL_CHUNKS=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)            ENV="$2";          shift 2 ;;
        --smoke)          SMOKE="--smoke";   shift ;;
        --n-steps)        N_STEPS="--n-steps $2"; shift 2 ;;
        --chunk-index)    CHUNK_INDEX="$2";  shift 2 ;;
        --total-chunks)   TOTAL_CHUNKS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[[ -z "$ENV" ]] && { echo "usage: $0 --env <env> [--smoke] [--n-steps N]"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p log

# Load conda to query grid dimensions. submit_all_envs.sh runs as an sbatch
# submitter on a compute node and invokes this script via plain `bash`, which
# is non-interactive and does NOT auto-source /etc/profile.d, so the `module`
# function isn't defined. Source Lmod's init explicitly so `module load`
# works whether this script is run from a login shell or from inside the
# submitter job.
source /etc/profile.d/zz_activate_lmod_user.sh 2>/dev/null || true
module load anaconda3/2023.09-0-aqbc
source activate shitty_bird_env

GRID_TYPE=$([ -n "$SMOKE" ] && echo "SMOKE_GRIDS" || echo "DEFAULT_GRIDS")
read -r N_JOBS N_REPS <<<"$(python -c "
import sys; sys.path.insert(0, '${REPO_ROOT}')
from simulations.sweep_grid import DEFAULT_GRIDS, SMOKE_GRIDS
grids = SMOKE_GRIDS if '${SMOKE}' else DEFAULT_GRIDS
g = grids['${ENV}']
print(len(g['H']) * len(g['R']) * len(g['mismatch']), g['reps'])
")"

if [[ -n "$SMOKE" ]]; then
    # Smoke runs: each job is 3 reps of one (H, R, factor) — well under
    # the 15-min cap even at H=170 R=1.
    SLURM_TIME="00:15:00"
    JOB_PREFIX="smoke"
    CELL_DIR="${ENV}_smoke_cells"
else
    # Each slot runs (n_reps / rep_chunks) episodes for one (H, R, factor).
    # Per-slot timing measured on the 2026-05-01 reps=100 sweep (sacct):
    #   walker            max slot @ 20 reps = 2.29 h  → ~6.9 min/episode
    #   humanoid_balance  max slot @ 20 reps = 1.79 h  → ~5.4 min/episode
    #   cartpole          max slot @ 20 reps = 0.75 h  → ~2.3 min/episode
    # An earlier (pre-fix) probe estimated ~100 min/episode for the worst
    # walker cell; the actual cost is ~14× lower, so the original 5-chunk
    # split was overprovisioned by the same factor. At reps=100 the worst
    # serial cell now lands at ~12 h, well under the 48 h CCV condo cap:
    #   rep_chunks=1 → 100 reps × 6.9 min = ~12 h per slot (works)
    #   rep_chunks=2 → 50 reps  × 6.9 min = ~6 h per slot  (default — half
    #                  the queue churn, halves crash-recovery cost)
    SLURM_TIME="48:00:00"
    JOB_PREFIX="grid"
    CELL_DIR="${ENV}_grid_cells"
fi

# Validate chunk inputs and compute this chunk's rep range. Last chunk picks
# up any remainder from non-divisible (n_reps, total_chunks).
if (( TOTAL_CHUNKS < 1 )); then
    echo "--total-chunks must be >= 1"; exit 1
fi
if (( CHUNK_INDEX < 0 || CHUNK_INDEX >= TOTAL_CHUNKS )); then
    echo "--chunk-index must satisfy 0 <= I < total_chunks"; exit 1
fi
CHUNK_SIZE=$(( (N_REPS + TOTAL_CHUNKS - 1) / TOTAL_CHUNKS ))
REP_START=$(( CHUNK_INDEX * CHUNK_SIZE ))
REP_END=$(( REP_START + CHUNK_SIZE ))
if (( REP_END > N_REPS )); then REP_END=$N_REPS; fi
if (( REP_START >= N_REPS )); then
    echo "chunk ${CHUNK_INDEX}/${TOTAL_CHUNKS} maps to empty rep range — nothing to submit"
    exit 0
fi

CHUNK_TAG=""
if (( TOTAL_CHUNKS > 1 )); then
    CHUNK_TAG="_c${CHUNK_INDEX}"
fi

sbatch \
    --job-name="${JOB_PREFIX}_${ENV}${CHUNK_TAG}" \
    --time="${SLURM_TIME}" \
    `# 8G covers walker H=183 R=1 (10 of 20 chunks OOMed at 4G in the` \
    `# 2026-04-30 run — Spline-PS samples × MuJoCo state copying × 800` \
    `# sim steps overflows 4G on the worst cell). Other envs comfortable.` \
    --mem="8G" \
    --cpus-per-task=1 \
    --nodes=1 \
    --array=0-$((N_JOBS - 1)) \
    -o "log/${JOB_PREFIX}_${ENV}${CHUNK_TAG}_%a.%j.out" \
    -e "log/${JOB_PREFIX}_${ENV}${CHUNK_TAG}_%a.%j.err" \
    run_one_cell.sh "$ENV" ${SMOKE} ${N_STEPS} \
        --rep-start "$REP_START" --rep-end "$REP_END"

echo "Submitted env=${ENV} chunk=${CHUNK_INDEX}/$((TOTAL_CHUNKS-1)): array of ${N_JOBS} jobs, reps [${REP_START}, ${REP_END})"
echo "Per-cell pickles -> results/${CELL_DIR}/"
if (( CHUNK_INDEX == TOTAL_CHUNKS - 1 || TOTAL_CHUNKS == 1 )); then
    echo "After all chunks finish, run: python aggregate.py --env ${ENV}$([ -n "$SMOKE" ] && echo " --smoke")"
fi
