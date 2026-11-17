#!/bin/bash -l
#SBATCH --job-name=ode-prepare
#SBATCH --account=lp165
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --uenv=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --output=alps-validation-ode-prepare-%j.out
set -eo pipefail
LC_ROOT=${SLURM_SUBMIT_DIR:?Submit from the source checkout}
source "$LC_ROOT/validation/alps-env.sh"
set -u
cd "$LC_ROOT"
srun --ntasks=1 --gpus-per-task=1 bash -c '
    set -euo pipefail
    ./tests/lightcone-tests
    ./tests/lightcone-geometry-tests
    ./tests/lightcone-geometry-gpu
    ./tests/lightcone-storage-tests
    ./tests/lightcone-time-tests
    ./tests/lightcone-reference-tests
    ./tests/lightcone-bundle-tests
    for control in 0000 00 01 10 11; do ./.build/variant-$control; done
    python3 validation/test_workflow.py
    python3 validation/test_ode_comparison.py
    exec python3 -u validation/ode_comparison.py prepare "$@"
' ode-prepare "$@"
