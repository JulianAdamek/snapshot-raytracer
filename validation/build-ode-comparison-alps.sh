#!/bin/bash
# Compile on a login node; execute all tests through Slurm.
set -eo pipefail
LC_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$LC_ROOT/validation/alps-env.sh"
set -u
cd "$LC_ROOT"
LC_BUILD_ARGS=(ARCH=grace_hopper CXX=mpicxx "NVCC=nvcc -ccbin $(command -v g++)"
    "HDF5_CFLAGS=$(pkg-config --cflags hdf5) ${LC_MPI_CFLAGS[*]}"
    "HDF5_LIBS=$(pkg-config --libs hdf5) -lhdf5_hl ${LC_MPI_LIBS[*]}"
    "INCLUDES=-I$LC_LOCAL_PREFIX/include" "LIBS=-L$LC_LOCAL_PREFIX/lib -lcfitsio")
mkdir -p validation/bin
# Explicit target: TARGET=... would otherwise select the snapshot source rule.
make -j4 "${LC_BUILD_ARGS[@]}" CORRECT_SHEAR_TRANSPORT=0 CORRECT_SACHS_B=0 \
    NONLINEAR_SCALAR=0 CORRECT_OBSERVER_DISTANCE=0 lightcone-raytracer
cp lightcone-raytracer validation/bin/lightcone-legacy
make -j4 "${LC_BUILD_ARGS[@]}" CORRECT_SHEAR_TRANSPORT=1 CORRECT_SACHS_B=1 \
    NONLINEAR_SCALAR=1 CORRECT_OBSERVER_DISTANCE=1 lightcone-raytracer snapshot-raytracer \
    validation/validation-tools tests/lightcone-geometry-gpu tests/lightcone-tests \
    tests/lightcone-storage-tests tests/lightcone-time-tests tests/lightcone-reference-tests \
    tests/lightcone-geometry-tests tests/lightcone-bundle-tests \
    .build/variant-0000 .build/variant-00 .build/variant-01 .build/variant-10 .build/variant-11
cp lightcone-raytracer validation/bin/lightcone-corrected
cp validation/validation-tools validation/bin/validation-tools
python3 - <<'PY'
import datetime, json, subprocess
from pathlib import Path
root=Path.cwd()
result={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'tracked_diff':subprocess.check_output(['git','diff','HEAD'],text=True),
        'nvcc':subprocess.check_output(['nvcc','--version'],text=True),
        'compiler':subprocess.check_output(['g++','--version'],text=True),
        'architecture':'sm_90a', 'optimization':'-O3 --use_fast_math'}
(root/'validation/bin/build.json').write_text(json.dumps(result,indent=2)+'\n')
PY
printf 'Built both ODE variants, snapshot tracer and helpers. No tests or jobs were run.\n'
