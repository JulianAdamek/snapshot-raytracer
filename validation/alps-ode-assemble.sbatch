#!/bin/bash -l
#SBATCH --job-name=ode-assemble
#SBATCH --account=lp165
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=0
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --uenv=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --output=alps-validation-ode-assemble-%A_%a.out
set -eo pipefail
LC_ROOT=${SLURM_SUBMIT_DIR:?Submit from the source checkout}
source "$LC_ROOT/validation/alps-env.sh"
set -u
cd "$LC_ROOT"
srun --ntasks=1 python3 -u validation/campaign.py assemble --resume "$@"
