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
bytes, timings, derivative mode and residency misses.

`make test` runs the CPU suites, including analytic shear transport tests and
comparisons against the snapshot source; CUDA compilation is needed but these
tests launch no GPU kernels. Frozen-field GH200 checks passed before the static
shear-sign correction. Both tracers now include that correction, whose GPU
regression is pending inspection. Historical maps from the previous sign are
not unchanged-output references. Genuinely evolving simulation light cones still
need validation. See [the validation guide](../validation/README.md) for the Alps
job and [the reference notes](../README.md#static-shear-transport-correction)
for the shear correction and conventions.
