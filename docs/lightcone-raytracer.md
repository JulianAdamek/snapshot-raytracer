# Light-cone raytracer

`lightcone-raytracer` traces rays through legacy gridded gevolution light cones,
interpolating the Weyl and frame-dragging potentials in space and conformal time.
The sidecar supplies the background expansion, so the executable does not require
an LCDM model or a box length in Mpc. `snapshot-raytracer` remains a separate tool.

## Build and run

Use CUDA, HDF5 with its high-level API, chealpix (plus CFITSIO for static
linking) and a host compiler with OpenMP:

```bash
make lightcone-raytracer ARCH=rtx3060
# Other targets: ARCH=a100 or ARCH=grace_hopper.
```

A full-sky example at `Nside=16`, with output in its own directory:

```bash
mkdir -p output/full
(
  cd output/full
  ../../lightcone-raytracer /absolute/path/lc 16 \
    --info /absolute/path/lc_info.bin --thickness 3 \
    --vertex 0.5 0.5 0.5 --observer-tau 2.3925614609369177 \
    --target-z 0.15
)
```

Replace the example vertex and observation time with those used to construct
**your** light cone. Positions and conformal times are normalized by the comoving
box length `L`; `--target-z` is the observed redshift at which each ray stops.
Nside must be a power of two. Files are written in the working directory.

## Input data

For thickness `T`, slots run from `0` to `T-1`:

```text
lc_phi_0.h5  ... lc_phi_<T-1>.h5
lc_B1_0.h5   ... lc_B1_<T-1>.h5
lc_B2_0.h5   ... lc_B2_<T-1>.h5
lc_B3_0.h5   ... lc_B3_<T-1>.h5
```

Each file contains a cubic scalar float32 or float64 `/data` dataset, with file
axes ZYX. The scalar input is the dimensionless Weyl potential
`W = (phi + psi)/2 = phi - chi/2`; filenames retain the legacy `phi` label.
B components must already have the legacy writer's normalization. They are
staggered by default; choose `--b-layout collocated` for `LIGHTCONE_INTERPOLATE`
outputs. The sidecar cannot identify that build option.

`--scalar phi` (the legacy default) means the `_phi` files already contain W.
`--scalar weyl` reads `_weyl` files instead; `--scalar phi-chi` reads raw `_phi`
and `_chi` files and forms `W = phi - 0.5f*chi` per region, before time sampling.
For single-precision gasgrd output this matches its stored Weyl field exactly.

The native-endian binary info file has no header. Each packed record is an int32
cycle followed by float64 `tau/L`, scale factor, particle inner/outer radii, and
`T+1` metric shell boundaries (radii in box units): `4 + 8*(T+5)` bytes.
Supply `--thickness` explicitly. A slot contains shells recorded at different
times, not a constant-time snapshot; later overlapping writes take precedence.

The sampler reconstructs a common-time value at every tricubic stencil node
before evaluating spatial gradients and Hessians. Time interpolation is linear;
time derivatives use fixed-node finite differences, including mixed B derivatives.
At an exact time knot, derivative estimation prefers the past interval and uses
the future interval if no past sample exists. The background uses a
shape-preserving cubic interpolation with no extrapolation.

Thickness 1 uses filenames `lc_phi.h5`, `lc_B1.h5`, etc., and requires explicit
`--time-derivatives zero`. This mode suppresses time-derivative terms and ISW
while retaining the available field-value sampling. Default finite differences
require at least two recorded time samples at every required stencil node.

## Shell geometry conventions

For new gasgrd cubes, supply the writer's additional binary geometry sidecar:

```bash
./lightcone-raytracer lc 16 --info lc_lccube_info.bin \
  --geometry lc_lccube_geometry.bin --target-z 0.15
```

This selects **gasgrd convention 1**. The exact vertex, direction, opening cosine,
radial limits, thickness, B layout and default observation time come from that
file. Vertex/direction/opening/radial CLI overrides are rejected; repeated
thickness/B-layout options must agree. An explicit `--observer-tau` may override
the recorded observation time, with a warning if it differs. HDF5 grid dimensions
must match the recorded grid size. Unsupported convention versions, corrupt files
and explicitly supplied missing files are errors; they never silently select
legacy arithmetic.

The scalar source defaults to `_weyl` if declared by the geometry, otherwise to
`phi-chi`. Gasgrd's raw `_phi` alone is not a Weyl input. These checks prevent
using raw phi inadvertently when importing geometry.

The vendored [lccube_geometry.h](../lccube_geometry.h) is copied unchanged from
gasgrd_dev commit `725898c`. Its convention-1 routine is also called by that
writer: integer unwrapped node coordinates, explicitly rounded division and
subtraction, then a specified fused multiply-add sequence. CPU and CUDA sampling
use the same convention; no gasgrd checkout is required to build this tracer.

Without `--geometry`, the tracer selects **legacy gevolution arithmetic** and
warns once that numerical consistency cannot be fully verified. For each node it
constructs the wrapped grid position and corresponding shifted vertex, matching
the old writer. Squared-radius and angular dot-product expressions use separately
rounded products and left-associated sums, with fusion disabled for those
operations. The physical periodic image is preserved; this is not a minimum-image
radius. B staggering is included in both modes.

Supply the original normalized geometry with full round-trip precision (17
significant decimal digits, or hexadecimal floats). Exact binary vertices such
as 0 and 0.5 avoid coordinate-rounding ambiguity, but legacy simulation builds
may have used a different contraction policy. The fallback therefore offers
compatibility with unfused legacy arithmetic, not a universal bitwise guarantee.
It neither adds an epsilon to shell boundaries nor discards samples near them.
Both modes retain half-open `[inner, outer)` shells and warn about possible
periodic overlap without rejecting the footprint. Overwritten data remain lost.

## Face batches and memory

Tracing and default FITS output use **NESTED** ordering. `--batch b` selects the
base face with NESTED indices `[b*Nside^2, (b+1)*Nside^2)`, for `b=0..11`.
These are different from the snapshot executable's historical RING batch ranges.

For one face, preload only its padded slab region with:

```bash
mkdir -p output/face-0
(
  cd output/face-0
  ../../lightcone-raytracer /absolute/path/lc 16 \
    --info /absolute/path/lc_info.bin --thickness 3 \
    --vertex 0.5 0.5 0.5 --observer-tau 2.3925614609369177 \
    --target-z 0.15 --batch 0 --storage ziggurat
)
```

| Option | Default | Meaning |
|---|---|---|
| `--storage full` | `full` | Load all cubes; supports full sky or one face. |
| `--storage ziggurat` | Opt-in | Load slabs for one explicit face, then launch one tracing kernel. |
| `--slab-cells N` | `16` | Positive slab thickness; ziggurat only. |
| `--padding-cells N` | `8` | Nonnegative ray-excursion allowance in cells, plus a fixed four-cell stencil halo; ziggurat only. |
| `--max-lookback-cells S` | Available background interval | Positive look-back cap in grid-cell light-crossing times, shared by both storage modes. |
| `--ordering nested\|ring` | `nested` | FITS output order; RING conversion changes pixel placement only. |
| `--time-derivatives finite-difference\|zero` | `finite-difference` | Select the time-derivative approximation. |

Slab selection uses unwrapped coordinates; periodic wrapping applies only when
reading the source arrays. There is no on-demand cache or automatic retry. A
residency miss reports the node, pixel, field, slot and time, suppresses FITS
output and exits unsuccessfully. Review the padding and look-back cap before
rerunning. Allocation estimates are checked against available GPU memory.

Optional `--direction dx dy dz --opening-deg angle` and
`--radial-range rmin rmax` declare the recorded footprint; the angle is an opening
half-angle in degrees and radii are in box units. Undeclared holes cannot be
detected. Possible periodic self-overlap produces warnings, not rejection;
overwritten simulation data cannot be recovered. Missing spatial or temporal
coverage anywhere in a full stencil invalidates that ray.

## Outputs and validation

All maps are float32, full-sized HEALPix FITS files. Face runs are zero outside
the selected face; twelve face maps in the same ordering can be added together.
Full-sky runs retain the `_batch0.fits` suffix, so use a separate directory for
explicit face 0.

| Filename prefix | Stored quantity |
|---|---|
| `dAmap` | Integrated rescaled area distance, in grid units. |
| `ellipticity1map`, `ellipticity2map` | Native complex ellipticity components; `epsilon ~= -4 gamma` at linear order. |
| `rotationmap` | Image rotation in radians. |
| `deflection1map`, `deflection2map` | Deflection projections onto `e_theta` and `-e_phi`, respectively. |
| `zmap` | Reached observed redshift. |
| `iswmap` | Dimensionless integral `integral(tau_source, tau_obs) (phi' + psi') d tau` along the traced path. |

ISW has the positive source-to-observer temperature convention and excludes
endpoint gravitational-redshift and Doppler terms. Frozen fields give exactly
zero ISW. Failed rays have NaN optical/ISW outputs and retain the last available
redshift in `zmap`. `run-metrics.json` records storage sizes, requested source
bytes, timings, derivative mode, scalar source, shell geometry convention and
residency misses.

`make test` runs the CPU suites, including analytic shear transport tests and
comparisons against the snapshot source; CUDA compilation is needed but these
tests launch no GPU kernels. `make test-gpu` separately verifies host/device
radius bits, angular decisions and shell ownership for both geometry modes.
Frozen-field GH200 job 5277317 passed after the static shear-sign correction;
GPU validation of the geometry changes remains pending. Historical maps from
the previous sign are not unchanged-output references. Genuinely evolving simulation light cones still
need validation. See [the validation guide](../validation/README.md) for the Alps
job and [the reference notes](../README.md#static-shear-transport-correction)
for the shear correction and conventions.
