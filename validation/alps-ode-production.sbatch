#!/bin/bash -l
#SBATCH --job-name=ode-nside16384
#SBATCH --account=lp165
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --time=24:00:00
#SBATCH --array=0-5%2
#SBATCH --uenv=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --output=alps-validation-ode-production-%A_%a.out
set -eo pipefail
LC_ROOT=${SLURM_SUBMIT_DIR:?Submit from the source checkout}
source "$LC_ROOT/validation/alps-env.sh"
set -u
cd "$LC_ROOT"
# Review the pilot timings/memory before submitting. No automatic retries.
srun --cpu-bind=socket --kill-on-bad-exit=1 python3 -u validation/ode_comparison.py run-face --phase production "$@"
