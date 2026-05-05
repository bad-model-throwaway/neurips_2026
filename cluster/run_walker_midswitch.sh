#!/bin/bash
#SBATCH --job-name=wk_midswitch
#SBATCH -t 02:00:00
#SBATCH --mem=16G
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atsushi_kikumoto@brown.edu
#SBATCH -o cluster/log/wk_midswitch_%j.out
#SBATCH -e cluster/log/wk_midswitch_%j.err

set -eu

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p cluster/log

# Lmod's `module` function isn't auto-defined under bare `bash`; source
# /etc/profile.d/zz_activate_lmod_user.sh first so this works in every
# context (login shell, sbatch script, nested bash invocation).
source /etc/profile.d/zz_activate_lmod_user.sh 2>/dev/null || true
module load anaconda3/2023.09-0-aqbc
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate shitty_bird_env

N_WORKERS=${SLURM_CPUS_PER_TASK} python -m simulations.sweep_walker_midswitch

echo "Done — midswitch_walker.pkl written to data/results/"
