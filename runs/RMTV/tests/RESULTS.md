# Validation record — 2026-09-09

The subsequent [mismatch investigation](MISMATCH_DIAGNOSIS.md) includes controlled
timestep, particle, scale, cap, and grid comparisons, plus an independent
continuum PDE residual and an initial DDMC face-flux diagnostic.

The implementation builds and runs in GNU Release, one radiation group, serial
and MPI. These checks used CPU transport and Open MPI P2P locally; no GPU,
RDMA, moving-mesh, AMR, or restart claim is made. The workspace contains other
ongoing changes; the RMTV implementation itself is confined to this run directory.

`validation_summary.json` retains the numerical measurements. Output and log
files from the larger exploratory runs remain under the ignored `verification/`
directory in this workspace. Focused scripts create fresh temporary directories
and print their locations. These results are not golden profile tolerances.

## Reference and initialization

- Pinned LANL reference: maximum pointwise table/upstream relative difference
  `4.62e-5` at the checked radii.
- Enclosed mass relative error against the analytic ambient-mass integral:
  `1.44e-8`.
- Shock-frame mass, momentum, and energy fluxes (including conductive flux)
  satisfy the reference jump. Temperature is continuous across the shock;
  density and velocity retain their one-sided jumps.
- Similarity energy scaling and the dimensional conductivity/opacity mapping
  pass independent algebraic checks.
- The actual initialized density, temperature, pressure, internal energy, and
  Cartesian velocity match the conservatively projected reference.
- Quadrature refinement and Cartesian reflection tests pass. Refinement around
  cells crossed by the fronts improves integration of the density jump.
- A deliberately perturbed tangent-cell center reproduced a `9.69e-5` density
  discrepancy before the geometric-tolerance correction, and `8.88e-16` after.
  This check is now included in `reference_test.cxx.in`.
- After that correction, serial/two-rank initial fields agree to about `1.3e-15`;
  after four hydro-only steps their maximum normalized field difference is
  `1.6e-13`. The comparison uses coordinates rather than process-local cell IDs.

The reference has an explicit central continuation below xi=0.02 and a tiny
front continuation at xi=2. These approximations and temperature/opacity
regularizations remain separate refinement parameters, not exact physics.

## Coupling and execution

`tests/verify.sh` and `tests/verify_mpi.sh` pass their focused checks:

- Positivity/finiteness of material fields and radiation energy.
- Correct physical start age and strictly advancing time.
- Material mass and gas-plus-packet energy conservation.
- Hydro-only control and coupled serial/two-rank execution.
- Actual DDMC activation, checked through event counts.
- An IMC/DDMC exercise with the actual default opacity cap and a short timestep.
  The temperature-field difference is about 3.04% for this low-particle-count
  exercise. It is reported, not treated as a quantitative equivalence proof.
- MPI `--help` exits normally and prints once, and invalid odd resolution is
  rejected before mesh construction.

The thermal-only radiation step conserves total energy to roundoff when the
run-local closure synchronizes total gas energy from internal energy plus
kinetic energy. Hydrodynamics remains fully enabled. No global energy
renormalization is used. Full radiation-momentum coupling is intentionally
outside this classical conduction benchmark.

## Physical limit and profile accuracy

The strengthened defaults are H=0.0256 s and K=937.5 K. On the reference over
xi in [0.02,1.99], the maximum radiation/material heat-capacity ratio is about
`1.28e-4`, maximum Knudsen ratio is below `0.01`, and equilibration time is below
`1.2e-10` times the physical age. The singular front itself is excluded from
these bounds. Run `reference/limit_report.py` to inspect them.

A full 16^3 octant calculation reached the requested final time in 201 steps:

| Quantity | Measurement |
|---|---:|
| Maximum relative energy drift | 1.39e-14 |
| Maximum relative mass drift | 1.04e-14 |
| Density L1 error | 14.0% |
| Temperature L1 error | 37.7% |
| Radial velocity L1 error | 43.2% |
| Maximum outer-boundary material temperature | 1.087 K (1 K floor) |

At the matching 100-step checkpoint (heat front approximately 0.63558 cm):

| Octant resolution | Density L1 | Temperature L1 | Radial velocity L1 |
|---|---:|---:|---:|
| 16^3 | 11.8% | 34.1% | 39.1% |
| 32^3 | 8.05% | 15.8% | 23.7% |

The 16^3 run used 8 new photons/cell and a population parameter of 16; the 32^3
run used 4 and 8, respectively. Both used quadrature order 8 with front-cell
refinement, seed 12345, and dt/age capped at 0.005. Thus these are useful
refinement diagnostics, **not a controlled measurement of spatial order**:
particle statistics also differ. The finer exploratory run was stopped after
collecting its 100-step comparison; no final-time result is claimed for it.
The tangent-cell correction was subsequently checked through the focused
serial/MPI tests; these profile studies are diagnostic observations, not exact
regression baselines for the final executable.

**Accuracy is not certified.** The coarse profile errors are substantial even
though the implementation's conservation and initialization checks pass.
Before claiming STORM passes RMTV quantitatively, use at least three spatial
resolutions and independently refine timestep, quadrature, particle count/seeds,
physical scales, opacity cap, temperature floor, and box size. Resolve the heat
front and shock, monitor outer-boundary contamination, and separate reference,
transport-limit, splitting, spatial, and statistical errors.

## Reproducing the focused tests

Build both executables as described in the parent README, then run:

```bash
tests/verify.sh
# Use your site's MPI transport settings. On the validation host:
OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader,tcp tests/verify_mpi.sh
```

A full coarse diagnostic run is:

```bash
./run.sh --octant --n 16 --quadrature 8 --photons 8 --initial-photons 4 --max-photons 16 --dump 100 --output output_full16
python3 analyze.py output_full16 --smoke --require-complete --plot
```

To reproduce the bounded finer comparison, use `--octant --n 32 --quadrature 8
--photons 4 --initial-photons 4 --max-photons 8 --max-steps 100` and analyze its
final snapshot at that actual age. It deliberately stops before the default
final heat-front position of 0.9 cm.
