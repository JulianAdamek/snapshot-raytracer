#!/bin/bash
# Compilation only. Run scientific/CPU/GPU tests through alps.sbatch.
set -eo pipefail
LC_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$LC_ROOT/validation/alps-env.sh"
set -u
cd "$LC_ROOT"
LC_HDF_CFLAGS="$(pkg-config --cflags hdf5) ${LC_MPI_CFLAGS[*]}"
LC_HDF_LIBS="$(pkg-config --libs hdf5) -lhdf5_hl ${LC_MPI_LIBS[*]}"
module list
nvcc --version
g++ --version | head -1
make -j4 ARCH=grace_hopper CXX=mpicxx "NVCC=nvcc -ccbin $(command -v g++)" \
    "HDF5_CFLAGS=$LC_HDF_CFLAGS" "HDF5_LIBS=$LC_HDF_LIBS" \
    "INCLUDES=-I$LC_LOCAL_PREFIX/include" "LIBS=-L$LC_LOCAL_PREFIX/lib -lcfitsio" \
    snapshot-raytracer lightcone-raytracer validation/validation-tools \
    tests/lightcone-tests tests/lightcone-storage-tests tests/lightcone-time-tests tests/lightcone-reference-tests \
    tests/lightcone-geometry-tests tests/lightcone-geometry-gpu
file snapshot-raytracer lightcone-raytracer validation/validation-tools
printf 'Build complete. Tests have not run. Submit validation/alps-isw.sbatch for time-derivative validation.\n'
