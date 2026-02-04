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
./snapshot-raytracer <hdf5_file_potential> <hdf5_file_B> <Nside> <n_steps> [<batch>]
```

Arguments:
- `<hdf5_file_potential>`: HDF5 file with Weyl potential dataset at `/data`
- `<hdf5_file_B>`: HDF5 file with frame-dragging (shift) vector field at `/data`
- `<Nside>`: HEALPix Nside for output maps
- `<n_steps>`: integration steps (1 step = 1 grid unit)
- `[<batch>]`: optional batch index (0..11) for a subset of pixels covering one HEALPix base pixel per batch

**Output Maps**
Writes HEALPix FITS files (RING ordering) named `dAmap_batchN.fits`, `ellipticity1map_batchN.fits`, `ellipticity2map_batchN.fits`,
`rotationmap_batchN.fits`, `deflection1map_batchN.fits`, and `deflection2map_batchN.fits`. For batched outputs, all pixel values are zero
outside the HEALPix base pixel of each given batch, so the maps can simply be added together.

Map meanings:
- `dA`: in grid units.
- `rotation`: dimensionless (radians).
- `ellipticity1`, `ellipticity2`: complex ellipticity `e = e1 + i e2`. See [arXiv:2002.04024](https://arxiv.org/abs/2002.04024) for more details.
- `deflection1`, `deflection2`: components of the deflection vector in the local tangent basis at each pixel. `deflection1` is the
  +theta component and `deflection2` is the +phi component, computed by projecting the change in ray direction onto those basis vectors.
