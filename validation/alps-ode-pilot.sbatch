#!/bin/bash -l
#SBATCH --job-name=ode-pilot
#SBATCH --account=lp165
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --uenv=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --output=alps-validation-ode-pilot-%j.out
set -eo pipefail
LC_ROOT=${SLURM_SUBMIT_DIR:?Submit from the source checkout}
source "$LC_ROOT/validation/alps-env.sh"
set -u
cd "$LC_ROOT"
# One debug job avoids its two-submitted-jobs/user limit. Four GPUs work on
# distinct faces at each step; groups 0..2 corrected, 3..5 legacy.
for LC_GROUP in 0 1 2 3 4 5; do
    srun --cpu-bind=socket --kill-on-bad-exit=1 python3 -u validation/ode_comparison.py \
        run-face --phase pilot --group "$LC_GROUP" "$@"
done
