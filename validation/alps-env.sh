#!/bin/bash
# Source from a prgenv-gnu modules view, both while building and in the batch job.
module load cray-mpich cuda gcc gsl hdf5 python/3.12.5
# The login/compute-node system Python is 3.6, which lacks subprocess.run's
# capture_output/text arguments. Select the uenv interpreter explicitly above
# and fail early if a shell configuration overrides it.
python3 -c 'import sys; print("Validation interpreter:", sys.executable, sys.version.split()[0]); assert sys.version_info >= (3, 8), "Validation requires Python >= 3.8; load the uenv Python module"' || return 1
export LC_LOCAL_PREFIX=${LC_LOCAL_PREFIX:-/users/adamek/local_arm}
if [[ ! -f "$LC_LOCAL_PREFIX/include/chealpix.h" ]]; then
    printf 'Missing ARM HEALPix headers under %s\n' "$LC_LOCAL_PREFIX" >&2
    return 1
fi
export PKG_CONFIG_PATH="$LC_LOCAL_PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="$LC_LOCAL_PREFIX/lib:${LD_LIBRARY_PATH:-}"
# mpicxx supplies the Cray MPICH/GTL headers and libraries required by parallel HDF5.
# Restrict nvcc arguments to the supported include/library flags; runtime search
# paths are supplied here rather than passing raw -Wl flags through nvcc.
LC_MPI_CFLAGS=()
LC_MPI_LIBS=()
read -r -a LC_MPI_COMMAND <<< "$(mpicxx -show)"
for LC_FLAG in "${LC_MPI_COMMAND[@]:1}"; do
    case "$LC_FLAG" in
        -I*) LC_MPI_CFLAGS+=("$LC_FLAG") ;;
        -L*) LC_MPI_LIBS+=("$LC_FLAG"); export LD_LIBRARY_PATH="${LC_FLAG:2}:$LD_LIBRARY_PATH" ;;
        -l*) LC_MPI_LIBS+=("$LC_FLAG") ;;
    esac
done
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
