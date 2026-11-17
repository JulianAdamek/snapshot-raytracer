# Optical equations and historical error controls

Both executables now use the nonlinear scalar terms of Lepori et al. (2020),
the observer condition `D_prime = -exp(2*W_obs)`, negative shear transport and
the antisymmetric frame-dragging contribution to Sachs-basis transport.
Vector terms retain the paper's leading-order approximation. The Weyl-only
endpoint convention remains in place; this does not add a separate slip input.

Four independent compile-time controls are shared by both tracers:

| Make variable | C++ macro | Default | Meaning of 0 |
|---|---|---|---|
| `CORRECT_SHEAR_TRANSPORT` | `RT_CORRECT_SHEAR_TRANSPORT` | `1` | Restore the erroneous positive static shear-transport coefficient. |
| `CORRECT_SACHS_B` | `RT_CORRECT_SACHS_B` | `1` | Omit the antisymmetric B contribution to each screen vector. |
| `NONLINEAR_SCALAR` | `RT_NONLINEAR_SCALAR` | `1` | Restore the old mixed-order scalar gradients, tides and optical friction. |
| `CORRECT_OBSERVER_DISTANCE` | `RT_CORRECT_OBSERVER_DISTANCE` | `1` | Restore `D_prime = -1` at the observer. |

The shear switch changes only the static coefficient; the light-cone partial-time
transport terms remain enabled according to `--time-derivatives`. On a frozen
snapshot those time terms vanish in every variant.

The first two switches form the requested independent error matrix:

| Shear | Sachs B | Interpretation |
|---|---|---|
| 0 | 0 | Both transport errors present. |
| 1 | 0 | Correct shear sign only. |
| 0 | 1 | Restore antisymmetric B only. |
| 1 | 1 | Both corrected. |

Hold the other two controls fixed across this matrix. To reproduce the original
ODE approximation while isolating these two transport errors, set
`NONLINEAR_SCALAR=0 CORRECT_OBSERVER_DISTANCE=0` in all four builds. For example:

```bash
make TARGET=snapshot-ode-00 ARCH=rtx3060 \
  CORRECT_SHEAR_TRANSPORT=0 CORRECT_SACHS_B=0 \
  NONLINEAR_SCALAR=0 CORRECT_OBSERVER_DISTANCE=0
make TARGET=snapshot-ode-10 ARCH=rtx3060 \
  CORRECT_SHEAR_TRANSPORT=1 CORRECT_SACHS_B=0 \
  NONLINEAR_SCALAR=0 CORRECT_OBSERVER_DISTANCE=0
```

Use `ARCH=grace_hopper` on Alps. Give each build a distinct executable name and
each run a separate output directory. A build-configuration stamp forces
recompilation when these settings change, including when returning to defaults.
Each executable prints its configuration and writes it to `run-metrics.json`.
The JSON `physics` object records the four numeric 0/1 values.

These controls reproduce the selected ODE terms, not a bitwise reconstruction of
the executable used for the paper. Interpolation arithmetic, redshift stopping,
compiler options and input preparation must be controlled separately. In
particular, the snapshot sampler now subtracts coordinates in double precision
and wraps negative indices without the old twenty-box limit. Apply the same
numerical settings to every member of a transport-error comparison.

## Endpoint accuracy

Both CLIs accept `--redshift-tol`, a positive finite absolute tolerance, default
`1e-8`. `--redshift-tol 1e-4` restores the former stopping tolerance. After at most
24 refinement trials, inability to satisfy the requested tolerance is a ray
failure. The light-cone refinement remains transactional. FITS `zmap` values are
float32, so stored-map validation allows their rounding in addition to the
double-precision stopping tolerance.

## Validation

`make test` runs the existing CPU suites plus:

- an independent double-precision bundle of neighbouring geodesics, using only
  the geodesic equations to obtain distance, ellipticity and rotation;
- independent ISW quadrature along the central geodesic;
- a linear-growth plane-wave fixture constructed by writer replay, written to
  HDF5, read through the production reader and integrated through the production
  time/spatial sampler, with convergence at grids 24, 48 and 96 and both B layouts;
- all four shear/Sachs switch combinations in both tracer RHS implementations,
  and an additional build with all four corrections disabled;
- observer normalization, screen orthogonality, large translated coordinates,
  runtime tolerance and refinement-failure checks.

The analytic-field tests use a sampler-injection interface; production tracing
still uses the regular grid sampler. The snapshot reference wrapper compiles
the actual CUDA source for host execution. These tests require CUDA compilation
but do not launch GPU kernels. `make test-gpu` remains a separate GPU check.

The frozen Alps workflow now requires a fresh baseline with the same four
corrections and tolerance. Historical maps are deliberately not unchanged-output
references after these physics changes. Strict full/slab, batch assembly,
ordering and frozen finite-difference/zero-mode comparisons remain in force.
