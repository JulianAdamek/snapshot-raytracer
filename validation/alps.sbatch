#!/bin/bash -l
#SBATCH --job-name=raytrace-static
#SBATCH --account=lp165
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --uenv=prgenv-gnu/24.11:v1
#SBATCH --view=modules
#SBATCH --output=alps-validation-%j.out

set -eo pipefail
# Submit from the prepared source directory; source paths must be absolute after cd.
LC_ROOT=${SLURM_SUBMIT_DIR:?Submit from the raytracer source directory}
source "$LC_ROOT/validation/alps-env.sh"
set -u
cd "$LC_ROOT"
printf 'Job %s on %s; source %s\n' "$SLURM_JOB_ID" "$(hostname)" "$LC_ROOT"
module list
# One Slurm task sees one GPU. All validation executes on the allocated node.
srun --ntasks=1 --gpus-per-task=1 bash -c '
    set -euo pipefail
    nvidia-smi
    printf "CUDA_VISIBLE_DEVICES=%s\n" "${CUDA_VISIBLE_DEVICES:-unset}"
    ./tests/lightcone-time-tests
    ./tests/lightcone-tests
    ./tests/lightcone-geometry-tests
    ./tests/lightcone-geometry-gpu
    ./tests/lightcone-storage-tests
    ./tests/lightcone-reference-tests
    python3 validation/test_workflow.py
    exec python3 -u validation/run_static.py "$@"
' validation-task "$@"
