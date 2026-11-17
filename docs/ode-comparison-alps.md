# Frozen-snapshot ODE comparison on Alps

This exercise measures the combined effect of **all four ODE corrections** at
source redshift 0.5, using the same frozen z=0.25 fields for both runs. It produces
maps for subsequent analysis; it does not yet implement the published B-mode
purification pipeline or quantify the individual corrections.

## Inputs and numerical controls

The defaults use the Alps checkout `/users/adamek/git/snapshot-raytracer` and:

```text
/ritom/scratch/cscs/adamek/raytracing/output2400/
  lc_1440box_2400_settings_used.ini
  lc_1440box_2400_snap004_phi.h5
  lc_1440box_2400_snap004_B.h5
```

The input headers specify 2400³ float32 cells, with three packed float32 B
components per cell. The box is 1440 Mpc/h; settings supply h=0.67 and
`Omega_m = Omega_b + Omega_cdm + sum(m_ncdm * deg_ncdm)/(93.14*h²)`.
The synthetic background is flat matter + Lambda with `Omega_Lambda=1-Omega_m`;
radiation is omitted to match the snapshot tracer's background model. The observer
is at `(0.5,0.5,0.5)`, a=1, independently of the epoch of the frozen snapshot.
The snapshot's own light-cone settings are not used.

Both executables in the comparison are builds of `lightcone-raytracer`:

| Variant | Shear transport | Antisymmetric Sachs B | Nonlinear scalar factors | Observer D' |
|---|---|---|---|---|
| `corrected` | 1 | 1 | 1 | 1 |
| `legacy` | 0 | 0 | 0 | 0 |

See [ODE controls](ode-controls.md) for the flag meanings. The legacy build
restores the historical ODE choices. This is not a bitwise recreation of the
2026 executable: both runs use today's spatial interpolation, redshift tolerance
`1e-8`, endpoint refinement, storage helpers and tabulated background. This
controls numerical differences while measuring the equation changes. Both use
raw snapshot phi as the single scalar, as in the old snapshot calculation;
no chi subtraction is introduced in this comparison.

A synthetic thickness-1 sidecar supplies a fine background table (two grid-cell
light-crossing times between knots). Every record has the same broad metric shell
covering the complete integration domain. Later records overwrite earlier ones;
this is intentional because the payload is static. `--time-derivatives zero`
avoids any need for temporal brackets and makes ISW identically zero. There is
no threefold field allocation. The table extends at least 64 cells beyond the
background distance to z=0.5, allowing perturbed rays to reach the source plane.
Both variants use exactly this same time cap and table.

The packed B cube is streamed in four-layer Z slabs into three scalar HDF5 files,
without changing its normalization. Conversion uses approximately 352 MiB of
payload buffers, plus HDF5 overhead, rather than a complete host cube. Phi is
symlinked read-only. Existing scaled-half B encoding and staggered positions
are preserved. Snapshot inputs are never modified.

## Build and submit

Compile from a normal Alps login shell. Scientific tests run only in Slurm:

```bash
ssh alps
cd /users/adamek/git/snapshot-raytracer
uenv run --view=modules prgenv-gnu/24.11:v1 -- \
  bash validation/build-ode-comparison-alps.sh > alps-build-ode.log 2>&1
```

The build produces separate corrected/legacy GH200 binaries under
`validation/bin`, the snapshot executable, regression tests, and I/O helpers.
The default root light-cone binary is left fully corrected. Modules and ARM
HEALPix/CFITSIO paths come from `validation/alps-env.sh`; the uenv also supplies
Python 3.12. No Python numerical packages are needed.

First submit preparation and its dependent pilot:

```bash
prep=$(sbatch --parsable validation/alps-ode-prepare.sbatch)
sbatch --dependency=afterok:"$prep" validation/alps-ode-pilot.sbatch
```

Preparation requests one normal node for up to two hours. It runs CPU regression
and GPU geometry checks, snapshots the two binaries and their hashes, prepares
the HDF5 components, and writes the manifest last as a completion marker.
The pilot requests one debug node for up to 30 minutes and traces all twelve
faces in both variants at Nside 256, four faces at a time. It is one debug job,
respecting the partition's limit of two submitted jobs per user.
If desired, request the normal queue and a longer pilot time when submitting:
`sbatch --partition=normal --time=02:00:00 ...`.

The default results root is:

```text
/ritom/scratch/cscs/adamek/raytracing/output2400/ode-comparison-z0.5/
```

Preparation requires a **new** root, and each face refuses to overwrite an
existing run. To change it, pass the same `--work-dir /absolute/new/path` to every
job and summary command. Other preparation options are `--data-dir`, `--nside`,
`--target-z`, `--slab-cells` and `--padding-cells`. The default padding is 16 cells
in addition to the fixed four-cell stencil halo. Later jobs read these settings
from the immutable manifest, rather than taking independent scientific options.

After the pilot finishes, inspect its report:

```bash
uenv run --view=modules prgenv-gnu/24.11:v1 -- bash -c '
  source validation/alps-env.sh
  python3 validation/ode_comparison.py summarize --phase pilot
'
```

This reads only small JSON files, so it can run on a login node. The report must
pass with all 24 faces present, all rays successful, zero residency misses, valid
FITS products, and zero ISW. Review device bytes, host RSS and timings. The report
also scales tracing and geometry times by the pixel-count ratio to estimate the
production cost; this is an estimate, not a runtime guarantee. Field loading and
large-map I/O also need time. Leave margin below the normal queue's 24-hour limit;
if a face will take longer, defer production and split the work further.
The initial pilot's field volume may slightly underestimate the high-resolution
footprint, because production launches more directions.

Only after reviewing that report, submit production:

```bash
sbatch validation/alps-ode-production.sbatch
```

Production requires a successful saved pilot summary. It uses account `lp165`,
one node per array element, four tasks with one GPU and 72 CPU cores each, and
24 hours on `normal`. There are six elements, at most two running concurrently:

| Array element | Variant | Faces on ranks 0,1,2,3 |
|---|---|---|
| 0 | corrected | 0–3 |
| 1 | corrected | 4–7 |
| 2 | corrected | 8–11 |
| 3 | legacy | 0–3 |
| 4 | legacy | 4–7 |
| 5 | legacy | 8–11 |

No jobs are submitted by the build or Python drivers. Slurm job/array IDs appear
in `alps-validation-ode-*.out` in the checkout. Use `squeue -u adamek` and
`sacct -j JOBID --format=JobID,State,ExitCode,Elapsed,MaxRSS` to monitor.
Each task writes a separate detailed log in its face directory.
After production, run the same summary command with `--phase production`.

## Products, resource budget and checks

```text
ode-comparison-z0.5/
  manifest.json              # settings, inputs, binary hashes, flags, model
  build.json                 # commit, tracked changes, compilers, architecture
  lc_info.bin
  fixture/{lc_phi.h5,lc_B1.h5,lc_B2.h5,lc_B3.h5,B1.h5,B2.h5,B3.h5}
  pilot/{corrected,legacy}/face-00/ ... face-11/
  production/{corrected,legacy}/face-00/ ... face-11/
    invocation.json          # exact CLI, job, rank, visible GPU
    lightcone.log{,.resources}
    run-metrics.json          # allocation, read bytes, times, ODE flags, ray counts
    fits-check.log
    success.json             # written only after all checks pass
    dAmap_batchN.fits
    ellipticity1map_batchN.fits
    ellipticity2map_batchN.fits
    rotationmap_batchN.fits
    deflection1map_batchN.fits
    deflection2map_batchN.fits
    zmap_batchN.fits
    iswmap_batchN.fits
  pilot-summary.json
  production-summary.json
```

All products use **NESTED**, Nside 16384 in production: 3,221,225,472 pixels per
full map. Face N fills exactly `[N*Nside²,(N+1)*Nside²)`, with zero exterior.
Eight maps per face are retained, including the exactly-zero ISW control. This
existing full-sized output format uses **96 GiB per face**, **2.25 TiB** for 24
faces, plus **154.5 GiB** for converted B files. Preparation checks filesystem free
space with 10% headroom; project quotas must also permit the allocation.
The pilot adds about 576 MiB. No full-size assembled duplicates are generated.

A production face's ray buffers alone need 13 GiB of device memory, plus its
single-slot slab field buffers. The tracer checks the actual device allocation
estimate before reading payloads. It does not automatically shrink the footprint,
retry, or switch storage modes. Padding or coverage failures remain visible.

Every completed run checks the four ODE flags, derivative/storage/output modes,
cap, endpoint tolerance and every ray's status. A bounded-memory FITS scan checks
all eight **active face intervals**, including finite values, reached redshift,
exactly zero ISW, and 64-bit pixel counts/order metadata. It does not reread the
zero exterior at production size; zero masking and face assembly are covered by
the existing small-map regression controls. Inputs and prepared binary hashes
are rechecked at each invocation. Large input payloads use size/mtime identities,
not expensive full-file checksums. Failed or killed runs have no success marker.

Corrected and legacy maps are expected to differ; this stage does not apply a
map-equality criterion between those two physical models. After both complete,
compare paired face maps and assess angular spectra using identical analysis
settings. The separate small-map regression tests check each implementation's
internal consistency.

## Analysis code available from Francesca

Inspected public repository revision
`4f287de7f843b55bc849b69621f684606a849572` (the only branch is `main`):
[relativistic_effects_weak_lensing](https://github.com/leporif/relativistic_effects_weak_lensing).
Its README and `Figures.ipynb` provide theory calculations, plotting and
**precomputed** measured spectra under `sim-data`. Their filenames include
`Spectra_nside16384_downgradeto8192_z0.5_full.dat`, supporting the stated
high-resolution-then-downgrade procedure. No map-to-spectrum, healpy, pymaster,
NaMaster or purification implementation was found in the tracked code/notebook
sources or environment specification.

Before reproducing the published purified B modes, obtain the map-analysis code
or its precise mask, apodization, pure-B choices, harmonic/downgrade method,
ell bins, pixel-window corrections and ellipticity/shear conventions. Do not use
the published spectra as a numerical equality reference for these different
snapshots. Assemble NESTED face intervals once per quantity; convert to RING only
as required by analysis. Changing ordering alone does not rotate the spin basis.
The tracer's native ellipticity has `epsilon ~= -4 gamma` at linear order; preserve
its documented screen conventions in the later spin-2 analysis.

CSCS guidance: [Daint partitions and hardware](https://docs.cscs.ch/clusters/daint/),
[Slurm and one rank per GPU](https://docs.cscs.ch/running/slurm/).
