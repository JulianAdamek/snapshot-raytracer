# GPU validation on Alps

Sources and GH200 binaries are prepared at:

```text
/ritom/scratch/cscs/adamek/raytracing/snapshot-raytracer-validation
```

Submit the time-derivative/ISW validation from a normal login shell:

```bash
ssh alps
cd /ritom/scratch/cscs/adamek/raytracing/snapshot-raytracer-validation
sbatch validation/alps-isw.sbatch
```

Preparation does not submit jobs. The script requests account `lp165`, one node
on `debug`, one task with one GPU, 16 CPU cores and 30 minutes. Slurm mounts
`prgenv-gnu/24.11:v1` with the modules view; the environment loads
`cray-mpich cuda gcc gsl hdf5 python/3.12.5` and ARM HEALPix/CFITSIO from
`/users/adamek/local_arm`. No preceding `uenv start` is needed. All scientific
CPU and GPU tests execute inside the allocation. Python >=3.8 is required; the
system Python 3.6 is insufficient.

The default ISW run uses the `Ngrid=576` final snapshot **snap011**, `Nside=16`
and source-plane target `z=0.15`. The snapshot remains frozen for every test:

1. Generate one full-sky snapshot reference (RING) and one full-volume light-cone
   reference (NESTED).
2. Trace all twelve true HEALPix base faces with preloaded ziggurat storage and
   compare each face with the corresponding part of the full-volume result.
3. Assemble all twelve faces and compare all eight complete light-cone maps.
4. Trace face 7 with optional RING output and verify ordering and values.
5. Require exactly zero ISW and compare the original seven light-cone maps
   with a freshly computed snapshot reference including the shear-sign correction.
6. Repeat the full-volume, twelve-face, assembly and RING tests with explicit
   `--time-derivatives zero`, comparing them with default finite differences.
7. Retain snapshot comparisons and allocation, read, host memory and timing
   measurements.

The CPU suites, including layout and FITS ordering controls, run first. To change
geometry or resolution, pass driver options after the job script:

```bash
sbatch validation/alps-ziggurat.sbatch --nside 32 --slab-cells 8 --padding-cells 12
```

`alps-isw.sbatch` uses `run_static.py --validate-isw`, which also enables the
ziggurat suite. It uses fresh references by default. Both tracers include the
negative static shear transport coefficient; earlier jobs 5275307 and 5276882
used the previous positive sign and are not unchanged-output references for
this correction. The default suite has 42 comparisons, retaining the established
snapshot endpoint budgets and strict light-cone storage/mode comparisons.

`--baseline-run PATH` optionally adds two strict saved-map comparisons for a
matching corrected run. Its manifest must contain `static_shear_transport:
negative`, and its resolution, snapshot, cosmology and targets must match.
A pre-correction baseline is rejected before payload preparation. Historical
results are preserved. The existing `alps-ziggurat.sbatch` runs the single-mode
storage suite and supports changing resolution independently.

Other options include `--snapshot`, `--targets`, `--max-lookback-cells`,
`--data-dir`, `--settings`, and `--output-root`. Start with the defaults on the
debug queue. The look-back cap is applied to every light-cone reference and slab
case; it must allow the target to be reached. The driver refuses to run outside
Slurm unless explicitly given `--allow-local` for a workstation GPU.

The earlier, smaller static-field workflow remains available:

```bash
sbatch validation/alps.sbatch
```

It defaults to the 288-grid `snap010`, `Nside=16`, targets `0.05, 0.15, 0.3`,
and full-volume face 7 at the deepest target. `--all-batches` adds all twelve
full-volume faces plus assembly; `--batch -1` skips the extra face. This workflow
also uses the new true NESTED face batching and full-sky snapshot references.

Monitor either job with:

```bash
squeue -u adamek
# Replace JOBID with the number printed by sbatch:
tail -f alps-validation-JOBID.out
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,MaxRSS
```

## Fixture and model

The driver reads settings and original HDF5 files under
`/ritom/scratch/cscs/adamek/raytracing` without modifying them. Defaults select
`unity6_4032box_576_settings_used.ini` for ziggurat validation and the corresponding
288 settings for the smaller workflow.

The resolved cosmology uses `h=0.67`, box length `4032/h = 6017.910447761194 Mpc`
(the snapshot CLI expects Mpc), and
`Omega_m = (omega_b + omega_cdm + sum(m_ncdm*deg_ncdm)/93.14)/h²`, approximately
`0.3204034678236443`. `Omega_lambda = 1 - Omega_m`. Radiation is omitted in both
tracers to match the snapshot implementation. The supplied simulation background
file is not used for this frozen-field equivalence test.

The observer is at the cube center with `a_obs=1`. Its synthetic conformal time
is the flat LCDM particle horizon, `tau_obs/L ~= 2.3925614609369177`. A fine-step
RK4 integration generates the tabulated background. The native-endian packed
int32/float64 sidecar uses the legacy thickness-3 shell formulas, with records
spaced by eight grid light-crossing times. This gives off-cone support for full
tricubic stencils, including staggered B coordinates. Support extends beyond
the observer and deepest target; a coverage preflight tests sampled background
rays before either GPU executable launches.

The job links all three phi slots to the same snapshot. It converts the packed
float64 array[3] B snapshot to three scalar float32 cubes once, then links all
slots to these components. Links avoid three physical copies. Both tracers thus
receive identical float32 values before the common scaled-half B encoding.
The light-cone derivatives are enabled by default and must evaluate to exactly
zero on these identical slot cubes. Both executables include the static shear
transport correction, retaining their existing ellipticity conventions.

## Results and acceptance

Every invocation creates a fresh directory:

```text
validation/results/slurm-JOBID-UTC_TIMESTAMP/
  manifest.json
  lc_info.bin
  fixture/
  prepare.log
  coverage.log
  references/z0.15/
    snapshot.log
    snapshot/*.fits
    lightcone/{lightcone.log,run-metrics.json,*.fits}
    comparison.json
  faces/face-00/ ... faces/face-11/
    lightcone.log
    run-metrics.json
    *.fits
    comparison.json
    snapshot-comparison/comparison.json
  faces/assembled/*.fits
  assembly-comparison/comparison.json
  ring-output/{lightcone.log,run-metrics.json,comparison.json,*.fits}
  zero-mode/                  # corresponding full/face/assembly/RING cases
  summary.json
```

`manifest.json` records cosmology, units, input identities, source hashes,
synthetic shells, geometry options, job and device information, including the
`static_shear_transport: negative` correction tag. `summary.json`
records combined success, individual comparisons and process/tracer metrics; it
is updated as cases finish. Each executable log has a `.resources` companion
from `/usr/bin/time -v`, recording peak host RSS. Tracer metrics contain full and
resident voxel counts, field/device/staging byte estimates, requested source
payload bytes, geometry/loading/tracing seconds, effective look-back and residency
misses. Source byte counts describe requested HDF5 element data, not physical
filesystem traffic or HDF5 cache overhead. Peak RSS also includes CUDA/runtime
allocations and host maps.

`compare-strict` and `assemble` require all eight light-cone maps. `compare`
checks seven snapshot observables with the established endpoint budgets and
requires exactly zero light-cone ISW. `compare-frozen` checks the original seven
maps with strict float32 budgets and uses an exact zero ISW reference; it needs
no ISW file in the saved seven-map run. Reports include `nonzero_isw_pixels`.
All strict and assembly calls in the frozen driver pass `--require-zero-isw`,
requiring exact zero even when both maps agree on a nonzero value. Generic
strict comparisons without that flag also support evolving ISW maps. Nonzero
or invalid ISW fails frozen validation, with CPU controls for these cases.

Comparisons read the FITS `ORDERING` keyword and canonicalize to NESTED. They
check all eight light-cone maps, reject NaN and HEALPix UNSEEN inside the selected face,
require matching validity masks and exact zeros outside a face, and check reached
redshift against the existing `1e-4` stopping tolerance. A face is compared against
the full-sky snapshot; the snapshot executable's historical contiguous RING
batch ranges are never used as face references. Full-sky maps retain the legacy
`_batch0.fits` suffix; explicit face 0 uses that suffix in its own directory.
No tangent-basis rotation accompanies a pixel-order permutation.

**Storage-path comparisons are strict.** Each pixel must satisfy
`abs(delta) <= 4*FLT_EPSILON*max(abs(reference),abs(candidate)) + 4*denorm_min`.
This permits float32 rounding without importing redshift-endpoint budgets from
snapshot comparisons. All selected pixels must be valid and every execution
must report zero residency misses. Twelve-face assembly and optional RING output
use the same strict criterion. Full and slab host integration checks additionally
require exact equality on synthetic fields.

Snapshot comparisons allow the pre-existing redshift stopping uncertainty and
background-table interpolation error. Their distance budget is
`2e-4 / [(cell_size_mpc*H0/c)*E(target_z)]` grid units. Shape/deflection screening
uses `rtol = 1e-4 + 4*distance_budget/background_distance_cells`, with absolute
floors `1e-7` ellipticity, `1e-10` rotation and `1e-8` deflection. Reports include
absolute/RMS differences, relative L2 differences, scales, ordering, masks and
applied tolerances. These budgets are screening criteria for frozen-field
endpoint agreement, not physical error estimates for evolving light cones.

Numerical comparison failures set the combined result false and exit code 2
after remaining cases finish. Command failures, missing maps or residency misses
stop immediately with logs retained. Residency failures suppress new FITS output
and never trigger a retry or full-volume fallback.

## Rebuild and local checks

```bash
uenv run --view=modules prgenv-gnu/24.11:v1 -- \
  bash validation/build-alps.sh 2>&1 | tee alps-build.log
```

This compiles both tracers, four CPU regression executables and the validation
helper for ARM64/CUDA `sm_90a`, without running scientific tests on the login node.
It discovers Cray MPICH/GTL flags and pins nvcc's host compiler to loaded GCC.
Local `make test` runs CPU numerical, FITS and workflow controls without launching
GPU kernels; CUDA compiles a wrapper around the snapshot source itself.
The previous storage and time-derivative suites passed in jobs 5275307 and
5276882. GPU validation of the static shear correction awaits the next job. CPU analytic tests
cover scalar time evolution, nonuniform finite differences, mixed B derivatives,
RHS tensor contractions, endpoint refinement and convergence across slope jumps.
Independent source-free shear tests check the sign in both implementations,
including scalar-gradient and divergence-free vector-gradient fixtures. In the
constant vector case, the weighted shear must evolve exponentially and
`ktilde0 * sigma` must be conserved. These tests do not rely on agreement between
the two codes. Genuinely evolving simulation light cones need subsequent validation.

Relevant CSCS guidance: [Slurm](https://docs.cscs.ch/running/slurm/),
[uenv](https://docs.cscs.ch/software/uenv/using/),
[Daint](https://docs.cscs.ch/clusters/daint/).
