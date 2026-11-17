# Tiled loading and resumable Alps campaigns

This workflow continues the frozen 2400³ experiment without modifying its prepared
fixture, original binaries, sidecar, manifests or pilot outputs. Final products are
eight full-sky NESTED FITS maps per equation variant. The optical equations and
sampling conventions are unchanged.

## Build and initialize

From `/users/adamek/git/snapshot-raytracer` on Alps:

```bash
uenv run --view=modules prgenv-gnu/24.11:v1 -- \
  bash validation/build-ode-comparison-alps.sh > alps-build-tiled.log 2>&1
```

This compiles both equation variants, both production-code GPU regression variants,
CPU controls, the snapshot executable and I/O helpers. It submits no jobs and runs
no scientific tests on login nodes.

Initialize a new campaign after compiling. The default existing work root is
`/ritom/scratch/cscs/adamek/raytracing/output2400/ode-comparison-z0.5`.

```bash
uenv start --view=modules prgenv-gnu/24.11:v1
source validation/alps-env.sh
python3 validation/campaign.py init
```

The default new campaign is `WORK_ROOT/campaigns/tiled-v1`. Pass the same
`--campaign /absolute/path` on every command to choose another directory.
Initialization requires a new directory; it copies small executables, pins their
hashes and the Python workflow, and references the original read-only fixture.
It does not reconvert or copy simulation cubes. `--level 1` is the default four-way
partition; a larger level requires a new campaign and fresh feasibility checks.
An implementation update requires a new campaign because Python and binary
identities are pinned. Old campaigns and their results remain available.

## Submit stages

Generate the diagnostic task table:

```bash
python3 validation/campaign.py tasks --phase diagnostic
```

The driver prints the exact `sbatch` command, including an immutable task table.
Copy and submit that command. No command is automatically submitted.

The diagnostic job requests a normal node with four GPUs and two hours. It first
runs pinned CPU and GPU controls, including actual production-loader runs in both
ODE variants. It imports and verifies the six successful saved Nside=256 faces.
Then four GPU tasks trace full corrected faces 4, 6, 0 and 5 using tiled loading.
Faces 0 and 5 are strictly compared against the original saved FITS products;
faces 4 and 6 exercise the orientation that previously timed out. All results
must have zero failed rays and exactly zero ISW.

After completion:

```bash
python3 validation/campaign.py summary --phase diagnostic
python3 validation/campaign.py tasks --phase pilot
```

Submit the printed pilot command. This schedules only missing Nside=256 coverage,
using four NESTED children per missing base face. Imported and diagnostic parents
satisfy their complete intervals and are not retraced. Each array element runs
four tasks/GPUs; pilot array concurrency is one, on normal with a two-hour limit.

After the pilot, generate its summary and the scaling probe:

```bash
python3 validation/campaign.py summary --phase pilot
python3 validation/campaign.py tasks --phase probe
```

Submit the printed probe command. One debug node runs part 0 of faces 0 and 4 at
Nside=1024 in both variants (four GPUs, 30 minutes). It measures actual field
sizes, I/O and kernel time for the smaller regions.

Inspect all reports before production:

```bash
python3 validation/campaign.py summary --phase probe
python3 validation/campaign.py tasks --phase production
```

Production is blocked unless diagnostics, pilot coverage, regression controls and
probe gates pass. The driver allows a 2% field-size margin, retains auxiliary
allocations, predicts production ray buffers, and reserves at least 2 GiB or 5%
of device memory, whichever is larger. Predicted time uses 125% of scaled tracing
time plus observed loading and a one-hour output/checking allowance; the result
must fit within 20 hours. These are planning estimates, not runtime guarantees.
The tracer also checks actual free device memory before reading payloads.

Submit the printed production command. At subdivision level 1 there are 96 tasks:
48 regions per variant, grouped into 24 four-GPU array elements. All four children
of a face normally share a node. Production uses normal, 24 hours, concurrency two.
No settings are silently reduced and there is no automatic retry or storage fallback.

After production:

```bash
python3 validation/campaign.py summary --phase production
sbatch validation/alps-ode-assemble.sbatch --campaign \
  /ritom/scratch/cscs/adamek/raytracing/output2400/ode-comparison-z0.5/campaigns/tiled-v1
```

Assembly runs inside a normal allocation (one CPU task, 16 cores, two hours),
validates every input and streams maps through CFITSIO with 64-bit indices. It
writes and verifies one full-sky quantity at a time without a full-map host array.
Stable final links are:

```text
CAMPAIGN/assembly/corrected/fullsky/*map_batch0.fits
CAMPAIGN/assembly/legacy/fullsky/*map_batch0.fits
```

Pixel ordering is NESTED; permutation to RING, if needed by later analysis, does
not rotate tangent bases. This workflow does not implement B-mode purification.

## Resume and provenance

Use `tasks --phase PHASE` again after an interruption. It generates a new table
containing only uncovered intervals. Existing tables remain unchanged and may
also be explicitly resubmitted; workers revalidate completed outputs before
skipping them. An interval completed at Nside=256 never satisfies production at
Nside=16384.

Each task has an advisory filesystem lock and a claim record. A live or
unverifiable claim is refused. A killed job's claim is archived only with explicit
resumption and after Slurm reports it terminal. Every launch writes a new attempt
directory, preserving prior logs and partial files. The scripts request resumption
explicitly. Interrupted kernels are rerun from the beginning of their region;
there is no GPU-state checkpoint.

Success markers are written last, after configuration checks, complete FITS/HDF5
validation and SHA256 hashes. Missing or modified completed artifacts are errors,
not permission to overwrite them. Parent/child intervals are selected without
overlap. Assembly refuses holes, duplicates, mismatched science IDs and failed
frozen rays. Input identities retain the original fixture's size/mtime checks;
sidecars and executables are hashed. Task planning and summaries read metadata
only and can run on login nodes; validation, import and assembly run under Slurm.

```text
CAMPAIGN/
  campaign.json, build.json, bin/
  tables/PHASE-TIMESTAMP.json + .sha256
  controls/attempt-*/ ...; controls.json
  imports/corrected/nside-256/face-XX/level-0-part-0/
  diagnostic/...  pilot/...  probe/...  production/...
    .lock, claim.json                 # claim exists only while running
    attempt-JOB-TIMESTAMP/
      invocation.json
      lightcone.log, lightcone.log.resources
      run-metrics.json
      ray-part.h5
      fits-check.log
    success.json
  PHASE-summary.json
  assembly/{corrected,legacy}/{attempt-*/fullsky/,fullsky,success.json}
```

At Nside=16384, compact intermediates include eight float32 maps and int32 ray
statuses: approximately 216 GiB across both variants. Final FITS maps add 192 GiB.
Initialization checks approximately 408 GiB plus 10% headroom of incremental free
space. Existing converted B components occupy about 155 GiB and are reused.
Intermediates are retained; there are no full-sized per-child or per-face map sets.

## Loader and executable interfaces

Existing full-face commands retain their filenames and behavior. New options:

| Option | Default | Meaning |
|---|---|---|
| `--io-mode auto\|regions\|tiles` | `auto` | Auto tiles x-oriented ziggurats; regional loading remains for y/z and full storage. |
| `--io-buffer-mib N` | `256` | Positive numeric staging budget for tiles, excluding HDF5/runtime overhead. |
| `--subdivision-level L` | `0` | Divide an explicit face into 4^L contiguous NESTED intervals. |
| `--part P` | `0` | Select one interval; L must be at most log2(Nside). |
| `--output-format fits\|compact-hdf5` | `fits` | Existing FITS maps or one compact region artifact. |
| `--run-id SHA256` | Required for compact output | Scientific configuration ID supplied by the workflow. |

Nonzero subdivision requires compact output and an explicit face. Compact output
requires NESTED ordering. The original Nside is used in `pix2ang_nest64`; this does
not lower angular resolution. Each ziggurat is conservatively built from the
actual launched directions, with the existing halo, padding and time cap.

Tiles read full-x rows in merged y bands and increasing z. A canonical source band
is read once per field/slot and copied to all required unwrapped periodic images.
Dataset handles remain open. Tile height accounts for float32 scalar storage,
float32 phi+chi combination, or float32 plus scaled-half B encoding. The loader
rejects a budget smaller than one XY plane. This is bounded staging rather than
a permanent repacked input or an on-demand cache. Regional staging retains its
existing slab-sized requirement, reported separately from the tile budget.

Compact HDF5 format version 1 has one contiguous 1D float32 dataset for each map
name and one int32 `status` dataset. Root scalar attributes are `FORMAT_VERSION`,
`NSIDE`, `FIRSTPIX`, `NPIX`, `FACE`, `LEVEL`, `PART`, `TARGET_Z`, `REDSHIFT_TOL`,
`ORDERING`, `RUN_ID` and canonical JSON `PHYSICS`. Pixels are in increasing global
NESTED order. Output uses 1,048,576-element blocks and is published only after
closing and validation. Failed rays retain tracer conventions; the frozen campaign
requires all rays successful.

New helper modes are `check-part FILE`, `compare-part FILE REFERENCE`,
`assemble-parts OUTDIR FILE...`, and `import-face FITS_DIR FILE NSIDE FACE Z TOL RUN_ID PHYSICS_JSON`.
Generic compact writing permits failed rays and evolving ISW; campaign checking
and assembly deliberately enforce successful frozen results.

## Metrics and checks

The tracer atomically updates stage metrics before loading, after every field,
before/after tracing and after output. A loading heartbeat reports field/slot/band
at least every 30 seconds, including while an HDF5 call blocks. Additional metrics
include effective I/O mode, requested HDF5 bytes/calls/time, B encoding and transfer
time, staging peak, free/total GPU memory, global ray interval and output time.
HDF5 requested bytes are not physical filesystem traffic; compare Slurm counters
separately. `/usr/bin/time -v` retains process RSS and wall time. Validation and
hashing durations are recorded by the driver.

`make test` includes CPU preload, region addressing, temporal jets, compact output,
assembly, resumption and CLI controls. `make test-gpu` additionally runs actual
production-code full/regional/tiled/subdivided comparisons for all twelve faces,
both B layouts, periodic crossings and both ODE variants. Grid copies must be bit
identical; GPU map comparisons use the existing strict float32 rounding criterion,
matching status masks and exact zero frozen ISW. GPU execution remains a Slurm task
on Alps; compilation alone is not GPU validation.
