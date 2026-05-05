#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --mem=256M
#SBATCH --cpus-per-task=1
#SBATCH --job-name=sweep_submitter
#SBATCH -o log/submitter.%j.out
#SBATCH -e log/submitter.%j.err
#
# Foolproof launcher for the full Figure 2 sweep across all DEFAULT_GRIDS envs.
#
# Submitting 6 envs × 400 tasks = 2400 array tasks exceeds the CCV per-user
# submit limit (~1500). This script works around that by calling submit_all.sh
# one env at a time and pausing between envs until the queue has drained below
# a safe threshold. The wait happens on a compute node (via sbatch), which is
# compliant with the cluster's login-node usage policy.
#
# At reps=100 a single cell at the slowest grid point fits comfortably in
# the 48 h SLURM cap, so the env is split into a small number of rep-chunks
# (default 2 chunks of 50 reps each). This script iterates (env × chunk)
# with the same queue gate. See submit_all.sh for the per-cell timing
# justification.
#
# Usage:
#   cd cluster/grid_sweep/
#   sbatch submit_all_envs.sh                       # default reps-chunks=2
#   sbatch submit_all_envs.sh --total-chunks 1      # one slot per cell
#   sbatch submit_all_envs.sh --smoke               # smoke-grid (reps=3, 1 chunk)
#
# After all per-cell jobs finish:
#   bash aggregate_all_envs.sh [--smoke]

set -eu

# Under sbatch, $0 points to SLURM's spool copy (non-writable), so dirname "$0"
# lands in a directory we can't mkdir in. Prefer SLURM_SUBMIT_DIR when set.
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "${SCRIPT_DIR}"
mkdir -p log

# Manuscript Figure 2 envs only. The auxiliary grids (humanoid_stand,
# humanoid_stand_gravity, cartpole_quadratic) are not used in the paper —
# add them back here if you need to refresh those caches.
ENVS=(cartpole humanoid_balance walker)

# Parse our own --total-chunks; pass everything else through to submit_all.sh.
TOTAL_CHUNKS=2
PASSTHRU=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --total-chunks) TOTAL_CHUNKS="$2"; shift 2 ;;
        --smoke)        TOTAL_CHUNKS=1; PASSTHRU+=("$1"); shift ;;
        *)              PASSTHRU+=("$1"); shift ;;
    esac
done

# Max queued tasks (running + pending) before we'll submit another 400-task
# array. 700 + 400 new + 1 (the submitter itself) = 1101, leaving ~100 margin
# below the confirmed 1200-job CCV condo cap (MaxSubmitJobsPerUser).
QUEUE_THRESHOLD=700
POLL_INTERVAL=300       # seconds between queue-depth polls
MAX_WAIT_PER_ENV=86400  # 24h max wait per (env, chunk) before giving up

submit_one_chunk() {
    local env="$1"
    local chunk_index="$2"
    shift 2
    local waited=0

    echo "=============================================="
    echo "$(date '+%F %T')  Submitting ${env} chunk ${chunk_index}/$((TOTAL_CHUNKS-1))"
    echo "=============================================="

    # Skip (env, chunk) that already has an active submission (running or
    # pending). Job names are set by submit_all.sh as
    # grid_<env>[_c<chunk>] / smoke_<env>[_c<chunk>], so we can tell one
    # from the other and avoid duplicate arrays on re-runs.
    local prefix="grid"
    [[ "$*" == *"--smoke"* ]] && prefix="smoke"
    local jobname="${prefix}_${env}"
    if (( TOTAL_CHUNKS > 1 )); then jobname="${jobname}_c${chunk_index}"; fi
    if squeue -u "$USER" -h -o "%j" | grep -qx "${jobname}"; then
        echo "$(date '+%F %T')  ${jobname} already has active jobs — skipping"
        return 0
    fi

    while true; do
        # -h strips header; -r forces one row per array element so pending
        # arrays (shown compressed as JobID_[i-j] without -r) count toward
        # the QOS cap correctly. The cap counts individual tasks, not rows.
        local n
        n=$(squeue -u "$USER" -h -r | wc -l)

        if [ "$n" -lt "$QUEUE_THRESHOLD" ]; then
            echo "$(date '+%F %T')  queued=${n} < ${QUEUE_THRESHOLD} — submitting ${jobname}"
            if bash "${SCRIPT_DIR}/submit_all.sh" --env "${env}" \
                    --chunk-index "${chunk_index}" --total-chunks "${TOTAL_CHUNKS}" \
                    "$@"; then
                return 0
            fi
            echo "$(date '+%F %T')  submission for ${jobname} failed; retrying in ${POLL_INTERVAL}s"
        else
            echo "$(date '+%F %T')  queued=${n} >= ${QUEUE_THRESHOLD} — waiting ${POLL_INTERVAL}s"
        fi

        sleep "${POLL_INTERVAL}"
        waited=$((waited + POLL_INTERVAL))
        if [ "$waited" -ge "$MAX_WAIT_PER_ENV" ]; then
            echo "ERROR: waited ${MAX_WAIT_PER_ENV}s for ${jobname} without submitting — aborting" >&2
            return 1
        fi
    done
}

for env in "${ENVS[@]}"; do
    for ((chunk=0; chunk<TOTAL_CHUNKS; chunk++)); do
        submit_one_chunk "${env}" "${chunk}" "${PASSTHRU[@]}"
    done
done

echo
echo "$(date '+%F %T')  All ${#ENVS[@]} envs × ${TOTAL_CHUNKS} chunks submitted: ${ENVS[*]}"
