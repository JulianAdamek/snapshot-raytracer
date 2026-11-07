# Snapshot Raytracer

CUDA raytracing tools for HDF5 snapshots and legacy gridded light cones, producing HEALPix maps from the Weyl and frame-dragging potentials.

For the new executable, start with the [light-cone user guide](docs/lightcone-raytracer.md).
The [validation guide](validation/README.md) describes the CPU checks and Alps GPU jobs.

**Dependencies**
- CUDA Toolkit (nvcc, CUDA headers)
- HDF5 with high-level API (`hdf5` and `hdf5_hl`)
- chealpix
- OpenMP (host compiler support)
- pkg-config (optional, for auto-detecting HDF5 flags)

**Build**
```bash
make ARCH=rtx3060
```

Other supported architectures:
- `ARCH=a100`
- `ARCH=grace_hopper`

If your HDF5/chealpix/NVTX installs aren’t in default paths, pass overrides:
```bash
make ARCH=a100 INCLUDES="-I/opt/hdf5/include -I/opt/chealpix/include" \
    LIBS="-L/opt/hdf5/lib -L/opt/chealpix/lib -L/opt/nvtx/lib"
```

**Usage**
```bash
./snapshot-raytracer <hdf5_file_potential> <hdf5_file_B> <Nside> \
  --target-z <z> --box-size-mpc <L> --omega-m <Om> --omega-lambda <Ol> --h0 <km/s/Mpc> [--batch <n>]
```

Arguments:
- `<hdf5_file_potential>`: HDF5 file with Weyl potential dataset at `/data`
- `<hdf5_file_B>`: HDF5 file with frame-dragging (shift) vector field at `/data`
- `<Nside>`: HEALPix Nside for output maps
- `--target-z`: observed source redshift used to stop each ray
- `--box-size-mpc`: isotropic comoving box size used to convert grid units to Mpc
- `--omega-m`: matter density parameter for the flat `Lambda`CDM background
- `--omega-lambda`: dark-energy density parameter for the flat `Lambda`CDM background
- `--h0`: Hubble parameter in km/s/Mpc
- `--batch`: optional batch index (0..11) for a contiguous range of RING pixels (these ranges are not HEALPix base faces)

Example:
```bash
./snapshot-raytracer phi.h5 B.h5 2048 --target-z 1.5 --box-size-mpc 2400.0 \
  --omega-m 0.312 --omega-lambda 0.688 --h0 67.556 --batch 0
```

**Output Maps**
Writes HEALPix FITS files (RING ordering) named `dAmap_batchN.fits`, `ellipticity1map_batchN.fits`, `ellipticity2map_batchN.fits`,
`rotationmap_batchN.fits`, `deflection1map_batchN.fits`, `deflection2map_batchN.fits`, and `zmap_batchN.fits`. For batched outputs, all pixel values are zero
outside the selected RING range, so the maps can simply be added together.

Map meanings:
- `dA`: in grid units.
- `rotation`: dimensionless (radians).
- `ellipticity1`, `ellipticity2`: complex ellipticity `e = e1 + i e2`. See [arXiv:2002.04024](https://arxiv.org/abs/2002.04024) for more details.
- `deflection1`, `deflection2`: components of the deflection vector in the local tangent basis at each pixel. `deflection1` is the
  +theta component and `deflection2` is the -phi component, computed by projecting the change in ray direction onto those basis vectors.
- `zmap`: the redshift reached by each ray. Rays that fail to hit `--target-z` return their last reached redshift in `zmap` and `NaN` in the lensing maps.


## Light-cone raytracer

See the [short user guide](docs/lightcone-raytracer.md) for build and run examples, inputs,
face batching, slab storage and output conventions. The details below describe the implementation.

Build the separate executable with `make lightcone-raytracer ARCH=rtx3060` (or
`ARCH=grace_hopper` on Alps). The snapshot CLI remains as documented above.

```bash
./lightcone-raytracer lc 16 --info lc_info.bin --thickness 3 \
  --vertex 0.5 0.5 0.5 --observer-tau 2.39256 --target-z 0.15 \
  --batch 0 --storage ziggurat --slab-cells 16 --padding-cells 8
```

The observation time above is illustrative; use the time and vertex of the
actual light-cone construction. Position and conformal time are box normalized.
The sidecar supplies the background scale factor; no analytic cosmology or box
length is required by this executable.

Inputs are scalar `/data` cubes named `<base>_phi_<slot>.h5` and
`<base>_B1_<slot>.h5` through `B3`, with zero-based slots. Thickness 1 omits the
slot suffix. Files use ZYX axis order. Potentials retain the snapshot convention;
B values must already have the legacy writer's normalization. The native-endian
binary sidecar has no header: each record contains int32 cycle followed by
float64 tau, scale factor, particle inner/outer radii, and `T+1` metric shell
boundaries, without struct padding (`4 + 8*(T+5)` bytes).

`--geometry <base>_lccube_geometry.bin` imports exact geometry and selects the
shared gasgrd convention-1 arithmetic. Without it, the legacy writer's wrapped
position/shifted vertex and unfused operation order are used, with a warning that
the original geometry/compiler arithmetic cannot be fully verified. Neither mode
discards samples near shell boundaries. Footprint overlap remains warnings-only.
See [geometry conventions](docs/lightcone-raytracer.md#shell-geometry-conventions)
for the sidecar format, precision limits and scalar-input choices.

Each slot is a collection of shells recorded at different times. The reader
replays shell writes with later records taking precedence. At each RK4 stage it
reconstructs a common-time value separately at every tricubic stencil node,
then evaluates spatial derivatives. Thickness 1 uses its sole recorded sample;
thickness >=2 requires a time bracket or an exact sample. Background evolution
uses a shape-preserving cubic table interpolation without extrapolation. Missing
coverage at any required node invalidates the ray; there is no spatial-order
reduction. Scalar ISW and mixed frame-dragging time derivatives are evaluated
from finite differences at fixed stencil nodes, followed by the same spatial
interpolation. Slopes are constant between each node's recording times. At an
exact knot the past interval is used, falling back to the future interval only
when no past sample exists. Both values are required even at an exact knot.
Time derivatives use forward cell time `eta = Ngrid * tau/L`; a backward tracing
step advances `s = eta_obs - eta`. No derivative cubes or larger halos are stored.

`--time-derivatives finite-difference` is the default and requires thickness >=2.
For thickness 1, explicitly pass `--time-derivatives zero`. Zero mode disables the
new derivative terms and ISW, while retaining the original temporal sampling of
field values. An unavailable derivative bracket invalidates a ray rather than
silently assuming no evolution.

Light-cone tracing always uses NESTED indexing and requires power-of-two Nside.
Its default FITS ordering is now **NESTED**. Explicit `--batch b` traces exactly
NESTED pixels `[b*Nside², (b+1)*Nside²)`, a true HEALPix base face. Maps remain
full sized and are zero outside that face. `--ordering ring` permutes one map at
a time on the host before writing, sets `ORDERING=RING`, and leaves tangent-basis
components unchanged. The original seven map names are preserved; full-sky
runs still use `_batch0.fits`. Use separate output directories for full sky and
explicit face 0.

The eighth map, `iswmap_batchN.fits`, is the dimensionless source-to-observer
integral `I_ISW = integral(tau_source, tau_obs) (phi' + psi') d tau`, evaluated
along the traced path to its accepted redshift endpoint. Positive values have
the usual positive ISW temperature sign. The scalar input is the Weyl potential
`W = (phi + psi)/2 = phi - chi/2`, so the integrand is `2 W'`. The map excludes
endpoint gravitational-redshift and Doppler terms. Failed rays have NaN ISW;
batch-exterior pixels are zero. Frozen fields produce exactly zero ISW.

Storage and footprint options:

| Option | Default | Meaning |
|---|---|---|
| `--storage full\|ziggurat` | `full` | Full periodic cube or preloaded slabs; ziggurat requires explicit `--batch`. |
| `--slab-cells N` | `16` | Positive core slab thickness; only valid in ziggurat mode. |
| `--padding-cells N` | `8` | Nonnegative physical ray excursion allowance, in addition to the fixed four-cell stencil halo; only valid in ziggurat mode. |
| `--max-lookback-cells S` | Available time interval | Positive conformal look-back cap in grid-cell light-crossing times, applied identically in both storage modes. |
| `--time-derivatives finite-difference\|zero` | `finite-difference` | Fixed-node finite differences; thickness 1 requires explicit `zero`. |
| `--b-layout staggered\|collocated` | `staggered` | Choose collocated for `LIGHTCONE_INTERPOLATE` outputs; the sidecar cannot identify this build option. |
| `--direction dx dy dz --opening-deg angle` | Full sky | Declare the recorded angular footprint (opening half-angle in degrees). |
| `--radial-range rmin rmax` | No radial clipping | Declare the legacy recording limits in box units. |

Ziggurat geometry bounds every launched straight ray over the effective available
time interval, expanded by four cells plus the excursion padding. Polar faces
use z slabs; faces 4/6 use x and 5/7 use y. Each slab has rectangular transverse
bounds reduced over the actual launch directions. One shared XYZ layout serves
all fields and slots, with X contiguous and no duplicate axial halos. Storage
selection uses unwrapped coordinates; HDF5 hyperslab reads apply periodic
wrapping. Questionable footprints and possible periodic self-overlap generate
warnings; overwritten simulation values cannot be recovered.

All slabs are loaded before a single tracing kernel launch. Device allocation
sizes are checked against available memory before reading payloads. One slab is
staged at a time, keeping float32 phi and scaled-half B storage. `run-metrics.json`
and stdout record the derivative mode and report resident/full voxel counts, payload and staging bytes, requested
source bytes, and geometry/loading/tracing times. A residency miss reports the
missing node, NESTED pixel, field, slot and time, suppresses FITS output, and exits
unsuccessfully. Increase excursion padding or review the look-back cap; there is
no automatic retry or fallback. This path implements no on-demand cache.

`make test` runs CPU reader, interpolation, time-derivative/ISW, layout, FITS ordering, orchestration
and snapshot-reference checks. CUDA compilation is required for the reference
wrapper, but these tests launch no GPU kernels. See
[validation/README.md](validation/README.md) for the prepared Alps GPU validation
job using the 576-grid final snapshot and all twelve slab batches in both
finite-difference and zero modes, comparing the corrected tracers with fresh references.

### Static shear transport correction

Both tracers now use the negative homogeneous optical-shear transport coefficient
required by eq. (30) of [the reference](https://arxiv.org/pdf/2002.04024v2#page=4).
For backward cell time `s`, define `F = d log(ktilde0)/ds`. The Jacobi equation
`J_ss + F J_s = T J` implies `H_s = -F H + D^2 T_TF` for the weighted trace-free
deformation `H = D^2 (J_s J^-1)_TF`. Thus the coefficient multiplying either
stored shear component is negative. Both scalar-gradient and vector-gradient
contributions have been corrected; their existing spatial prefactors, tidal
sources and ellipticity conventions are preserved. Light-cone time derivatives
already use the same negative transport sign.

The image-ellipticity convention is independent of this correction: changing an
overall sign of optical shear and ellipticity leaves the homogeneous transport
coefficient unchanged. The shear-normalized quantity is `-epsilon/4` at linear
order, as discussed in [the snapshot paper, section 3.4](https://arxiv.org/pdf/2603.24179v2#page=17).

Frozen maps from jobs 5275307 and 5276882 contain the previous positive static
coefficient and are historical results. They are not unchanged-output references
for the corrected physics. Validation recomputes both tracer references and
checks full/slab, finite-difference/zero and FITS ordering consistency. It also
checks each implementation independently using seeded optical shear, zero
anisotropic tidal forcing and conservation of `ktilde0 * sigma` in a constant
vector-transport fixture. Explicit saved baselines must carry the manifest tag
`static_shear_transport: negative`.
