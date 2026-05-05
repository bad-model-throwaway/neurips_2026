#!/bin/bash
#SBATCH --job-name=cp_summary
#SBATCH -t 24:00:00
#SBATCH --mem=128G
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atsushi_kikumoto@brown.edu
#SBATCH -o cluster/log/cp_summary_%j.out
#SBATCH -e cluster/log/cp_summary_%j.err

set -eu

# Resolve repo root. Under interactive `bash cluster/<script>.sh` invocation,
# BASH_SOURCE points at the script on disk, so we self-locate regardless of
# cwd. Under `sbatch`, SLURM rewrites BASH_SOURCE to a spool-dir copy, so we
# fall back to SLURM_SUBMIT_DIR (the directory you ran sbatch from). The
# trailing sanity check fails fast with a clear message if neither path
# lands on the repo root.
if [[ -f "${BASH_SOURCE[0]}" && "${BASH_SOURCE[0]}" == */cluster/* ]]; then
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
else
    REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
fi
cd "${REPO_ROOT}"
if [[ ! -d simulations ]]; then
    echo "ERROR: '${REPO_ROOT}' is not the repo root (no simulations/ found)." >&2
    echo "       Submit this script from the repo root:" >&2
    echo "         cd <repo-root> && sbatch cluster/$(basename "$0")" >&2
    exit 1
fi
mkdir -p cluster/log

# Lmod's `module` function isn't auto-defined under bare `bash`; source
# /etc/profile.d/zz_activate_lmod_user.sh first so this works in every
# context (login shell, sbatch script, nested bash invocation).
source /etc/profile.d/zz_activate_lmod_user.sh 2>/dev/null || true
module load anaconda3/2023.09-0-aqbc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate shitty_bird_env

# Pin workers to allocated CPUs.
N_WORKERS=${SLURM_CPUS_PER_TASK} python -m simulations.sweep_cartpole_summary

echo "Done — summary_cartpole.pkl written to data/results/"
