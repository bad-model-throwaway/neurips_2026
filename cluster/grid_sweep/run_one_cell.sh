#!/bin/bash
# SLURM batch script for one (H, R) grid cell — all mismatch factors × reps.
#
# Submitted as a job array by submit_all.sh. Each array slot maps to one
# (H_i, R_i) pair via:
#   H_i = SLURM_ARRAY_TASK_ID // n_R
#   R_i = SLURM_ARRAY_TASK_ID  % n_R
# Episodes run sequentially on one CPU; per-cell pickles are written atomically.

set -eu

# Lmod's `module` function isn't auto-defined under bare `bash`; source
# /etc/profile.d/zz_activate_lmod_user.sh first so this works in every
# context (login shell, sbatch script, nested bash invocation).
source /etc/profile.d/zz_activate_lmod_user.sh 2>/dev/null || true
module load anaconda3/2023.09-0-aqbc
source activate shitty_bird_env

# $1 = env; remaining args (--smoke, --n-steps N) forwarded from submit_all.sh
python "${SLURM_SUBMIT_DIR}/run_one_cell.py" "$1" --task-id "${SLURM_ARRAY_TASK_ID}" "${@:2}"
