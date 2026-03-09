# Snapshot Raytracer

CUDA raytracing tool that reads HDF5 snapshots of the Weyl potential and frame-dragging potential (shift), then produces HEALPix maps.

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
- `--batch`: optional batch index (0..11) for a subset of pixels covering one HEALPix base pixel per batch

Example:
```bash
./snapshot-raytracer phi.h5 B.h5 2048 --target-z 1.5 --box-size-mpc 2400.0 \
  --omega-m 0.312 --omega-lambda 0.688 --h0 67.556 --batch 0
```

**Output Maps**
Writes HEALPix FITS files (RING ordering) named `dAmap_batchN.fits`, `ellipticity1map_batchN.fits`, `ellipticity2map_batchN.fits`,
`rotationmap_batchN.fits`, `deflection1map_batchN.fits`, `deflection2map_batchN.fits`, and `zmap_batchN.fits`. For batched outputs, all pixel values are zero
outside the HEALPix base pixel of each given batch, so the maps can simply be added together.

Map meanings:
- `dA`: in grid units.
- `rotation`: dimensionless (radians).
- `ellipticity1`, `ellipticity2`: complex ellipticity `e = e1 + i e2`. See [arXiv:2002.04024](https://arxiv.org/abs/2002.04024) for more details.
- `deflection1`, `deflection2`: components of the deflection vector in the local tangent basis at each pixel. `deflection1` is the
  +theta component and `deflection2` is the +phi component, computed by projecting the change in ray direction onto those basis vectors.
- `zmap`: the redshift reached by each ray. Rays that fail to hit `--target-z` return their last reached redshift in `zmap` and `NaN` in the lensing maps.
