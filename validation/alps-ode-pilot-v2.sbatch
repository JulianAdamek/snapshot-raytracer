#!/bin/bash -l
#SBATCH --job-name=ode-pilot-v2
#SBATCH --account=lp165
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --time=02:00:00
#SBATCH --uenv=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --output=alps-validation-ode-pilot-v2-%A_%a.out
set -eo pipefail
LC_ROOT=${SLURM_SUBMIT_DIR:?Submit from the source checkout}
source "$LC_ROOT/validation/alps-env.sh"
set -u
cd "$LC_ROOT"
srun --cpu-bind=socket --kill-on-bad-exit=1 python3 -u validation/campaign.py run --resume "$@"
