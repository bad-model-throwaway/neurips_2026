#!/bin/bash
# SLURM array script — one (condition, mismatch) cell, all reps sequentially.
# Submitted by submit_all.sh as a job array.
#
# $1 = env (cartpole | walker | humanoid_balance)
# SLURM_ARRAY_TASK_ID encodes (condition_idx, mismatch_idx) via run_one_cell.py.

set -eu

# Lmod's `module` function isn't auto-defined under bare `bash`; source
# /etc/profile.d/zz_activate_lmod_user.sh first so this works in every
# context (login shell, sbatch script, nested bash invocation).
source /etc/profile.d/zz_activate_lmod_user.sh 2>/dev/null || true
module load anaconda3/2023.09-0-aqbc
source activate shitty_bird_env

python "${SLURM_SUBMIT_DIR}/run_one_cell.py" "$1" --task-id "${SLURM_ARRAY_TASK_ID}"
