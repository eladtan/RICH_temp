# RMTV: spherical hydro with nonlinear thermal transport

**Cluster execution:** submit simulations with Slurm, not directly on gateway
nodes. `sbatch runs/RMTV/submit_bigrun.sh` from the repository root submits the
32^3 octant DDMC convergence run on 16 cores in partition `bigrun`, then analyzes
the completed result. Output directories include the Slurm job ID.

This run evolves the strong-conduction Reinicke–Meyer-ter-Vehn similarity
solution using **RICH Euler hydrodynamics and STORM grey IMC/DDMC thermal
exchange**. The domain is a fixed Cartesian Voronoi mesh. `--octant` uses
reflecting symmetry planes to reduce the cost of a spherical calculation.

## Physics and the revised implementation decisions

The target equations are Euler mass/momentum conservation and

```text
rho De/Dt + p div(v) = div(chi grad(T)),
p = (gamma - 1) rho e,  e = Cv T,
gamma = 5/4,  chi = chi0 rho^-2 T^(13/2).
```

There is no radiation force in these equations. The implementation deliberately
uses `RadiationIMCParameters::withHydro=false` for transport: this disables
radiation momentum and moving-material corrections, **not thermal exchange**.
`noHydroFeedback` remains false. The independent `HydroStep` evolves density,
velocity, and pressure. After each MC step the driver synchronizes gas total
energy from material internal energy and the unchanged material momentum.
It never renormalizes the global energy budget or adjusts the solution to fit
reference data.

This choice revises the original plan's full radiation-momentum coupling.
Testing that path exposed accumulated radiation-step energy drift in a coarse
run; it is not exposed as an option in this benchmark. Thermal-only coupling
also matches the absence of radiative force in the reference equations.
STORM still has finite radiation storage and transport effects: **this is an
asymptotic conduction benchmark, not an exact identity between the transport
and conduction equations at arbitrary physical scales**. Radiation advection,
work, and force omitted here must remain negligible relative to gas evolution.

In LTE diffusion, `Er = a_rad T^4` and

```text
F = -c/(3 Sigma) grad(Er) = -[4 a_rad c T^3/(3 Sigma)] grad(T),
Sigma = 4 a_rad c/(3 chi0) rho^2 T^(-7/2).
```

`Sigma` is a macroscopic opacity in cm^-1. Absorption is grey and scattering is
zero. Planck, frequency-dependent, and temperature-evaluated opacity methods
all implement the same law. Compton, multigroup, polarization, planar momentum
projection, and slab transport are disabled. Build with exactly one group.

The default dimensional scales are

```text
length L = 1 cm, density D = 1 g/cm^3,
time H = 0.0256 s, temperature K = 937.5 kelvin,
velocity unit = L/H, specific energy unit = (L/H)^2,
Cv = (L/H)^2 / [(gamma-1) K],
chi0 = D^3 L^4 / [H^3 K^(15/2)].
```

The preflight report gives a maximum radiation/material heat-capacity ratio
of about 1.3e-4 and maximum mean-free-path/temperature-scale ratio below 0.01
on xi in [0.02, 1.99]. The final 0.5% of the sharp front is excluded explicitly;
this is not a pointwise guarantee at the singular front.

These are two applications of the limit sequence below to the initial
exploratory scales H=1e-4, K=3e4. They preserve the dimensionless reference, while reducing radiation energy
and heat capacity relative to matter. The default `Cv` is approximately
6.5104167 erg/(g K). `--time-unit` and `--temperature-unit` permit a physical
limit study, with coefficients rederived consistently. Light speed remains the
physical constant. One explicit limit sequence is `H -> 16 H` and
`K -> K/sqrt(32)`: at the same similarity coordinate it divides the radiation/
material heat-capacity ratio by four and multiplies optical depth by four. Do not replace these scales by raw ExactPack values in cgs.

## Initial conditions, geometry, boundaries, and time integration

- The initial **heat front** is at 0.45 cm and shock at 0.225 cm. The target
  final positions are 0.9 and 0.45 cm. Ambient density is `D (r/L)^(-19/9)`;
  the heated interior is the similarity solution, not the ambient power law.
- The physical clock starts at approximately `4.8208462656e-4 s`, and ends at
  `1.3120318781e-3 s` with default scales. The front evolves as `t^(9/13)`.
  No mesh-dependent energy bomb is deposited. No additional spherical source
  terms are needed on the actual three-dimensional mesh.
- Initial mass, Cartesian momentum, and **total material energy** are integrated
  over each cell. Temperature/pressure follow from the conservative projection,
  including unresolved kinetic energy. Midpoint tensor quadrature recursively
  refines cells cut by either spherical front twice. `--quadrature` controls the
  remaining integration error; test its convergence separately from grid error.
- The reference table stores separate shock branches. The origin has a regular
  continuation below 1% of the heat-front radius; see `reference/README.md` for
  the explicitly documented approximation and verification.
- Radiation starts in isotropic LTE with the represented cell temperature:
  `cell.Erad = a_rad T^4/rho`. Thus each finite-volume state starts in thermal
  equilibrium; no noisy radiation-temperature estimate determines its gas IC.
- The cube is `[-box,box]^3`, or `[0,box]^3` with `--octant`. Both hydro and
  radiation reflect at the walls. Outer walls are beyond the final heat front;
  they match the exact cold solution's zero flux while remaining isolated.
  At a nonzero numerical floor this is a closed approximation to the cold
  exterior. Monitor `max_boundary_T` and repeat with a larger box. A visibly
  heated outer boundary invalidates an accuracy claim against the unbounded
  reference even when global energy is conserved.
- The initial material temperature floor is 1 K. There is **no repeated hydro
  temperature clamp**. Opacity evaluation uses the same lower temperature bound
  and a finite upper opacity (`1e8 cm^-1` by default), including the cold region.
  Both regularizations require sensitivity checks; their use is reported.
- Hydro is second-order spatial reconstruction/HLLC and the existing two-stage
  hydro advance. Coupling is first-order Lie splitting, hydro then thermal
  transport. The same manually selected dt is used by both steps, and time is
  advanced exactly once. The last step is clipped to the requested final age.
- The timestep is bounded by a conservative sum of three directional acoustic
  rates (`--cfl`, default 0.2) and a fraction of current physical age
  (`--dt-fraction`, default 0.005). `--adaptive-radiation-dt` additionally honors
  STORM's suggestion, which can react strongly to single-cell MC noise.
- DDMC handles every cell with `sigma*chord >= --ddmc-min-tau` (default 3; STORM's own
  default is 15). The hot rarefied centre sits at `sigma*chord ~ 7.5`, falling to
  `~4.5` by `t_end` as the centre rarefies, and running it
  in IMC costs `~tau^2` effective scatters per crossing on a few cells that no
  load balancer can split; with the 15 threshold those cells took >90% of the
  MC wall time at n=48. `config.json` records `ddmc_min_cell_tau`.
- The mesh and MPI decomposition remain fixed. Remote material ghost states
  are refreshed after radiation before the next hydro reconstruction. The MC
  transport uses the RDMA (`--mc-manager`, default `rdma`, auto-selecting the
  one-sided backend) rather than the P2P
  backend, and the submit scripts no longer pin `OMPI_MCA_pml=ob1` or
  `OMPI_MCA_btl=self,vader,tcp`, which had confined traffic to TCP. Moving
  meshes, AMR, and restart are outside
  this first implementation.

## Build and run

From the repository root:

```bash
./build_rich.sh gnuRelease --test_name=RMTV --build-subdir=RMTV --energy_groups_num=1 --jobs=8
```

From this directory:

```bash
# Initialization only; output directories must be new or empty.
./run.sh --n 16 --quadrature 8 --init-only --output output_ic

# Short coupled smoke test, not an accuracy run.
./run.sh --n 8 --quadrature 16 --max-steps 4 --photons 8 --initial-photons 8 --max-photons 16 --output output_smoke
python3 analyze.py output_smoke --smoke

# An octant run offers twice the linear resolution at the same cell count
# as a full cube with the same --n. Defaults enable DDMC where eligible.
./run.sh --octant --n 32 --quadrature 8 --output output_octant
python3 analyze.py output_octant --require-complete --plot

# --imc disables DDMC for a transport comparison. Start with a short interval.
./run.sh --help
```

`--n` is cells per Cartesian axis, not radial shells. Serial runs may be costly
at meaningful radial resolution. For MPI, build and select its executable:

```bash
./build_rich.sh gnuReleaseMPI --test_name=RMTV --build-subdir=RMTV --energy_groups_num=1 --jobs=8
# From runs/RMTV, using your site's launcher/environment:
mpirun -np 4 ../../build/gnuReleaseMPI/RMTV/rich --octant --n 32 --output output_mpi
```

The executable resolves the reference path relative to the working directory;
`run.sh` changes to this directory automatically. A direct launch elsewhere
must pass `--reference /absolute/path/to/reference/shape.dat`.

## Outputs and verification

Every run saves `config.json`, the reference shape actually used, per-rank CSV
snapshots, and `diagnostics.csv`. `status.json` explicitly distinguishes reaching
the requested end time from stopping at `--max-steps`. `--init-only` writes just
the initial snapshot/diagnostics. `--hydro-only` is a debugging control with
thermal transport omitted and is not an RMTV solution.

`--radiation-only` instead disables hydro evolution while keeping thermal
exchange active. It writes `transport_ledger.csv`, which measures net energy
transfer out of fixed cell regions from their gas-plus-census energy balances.
It cannot be combined with `--hydro-only`. This is a transport diagnostic;
its evolved profiles are not a full RMTV solution. See
[tests/TRANSPORT_LEDGER_FINDINGS.md](tests/TRANSPORT_LEDGER_FINDINGS.md) for the
one-step IMC/DDMC comparisons with reference fluxes through identical mesh faces.

Diagnostics include mass, gas energy, packet energy, total energy drift, separate
hydro/radiation energy changes, radiation/material heat-capacity ratio, maximum
velocity/c, minimum cell optical depth, opacity-cap counts, temperature extrema,
outer-boundary temperature, initial floor energy, and DDMC step/leak counts.
The latter prevent mistaking a DDMC-enabled run for one that actually used it.
In octant runs extensive quantities refer to that octant, not the full sphere.

`analyze.py` compares with conservatively averaged reference fields at the
actual snapshot age, using the same temperature regularization as initialization.
It also reports the reference projection’s transverse component, so Cartesian
cell-averaging effects are not mistaken for physical asphericity. It reports volume-weighted L1 errors and transverse motion.
`--smoke` checks finiteness, positivity, increasing time, and conservation;
**it does not certify profile accuracy**. `--plot` writes `profiles.png`.

Run the focused checks with:

```bash
python3 reference/check.py
python3 reference/limit_report.py
tests/verify.sh
# After building the MPI executable, with site-appropriate MPI settings:
tests/verify_mpi.sh
```

The checks cover upstream reference agreement, mass/energy similarity laws,
isothermal shock fluxes (including conductive energy flux), units, conservative
projection, quadrature refinement, actual initialized fields, hydro-only and
coupled conservation, and an IMC/DDMC exercise that requires DDMC activation.
The test's C++ file uses `.cxx.in` because RICH recursively compiles every
`.cpp` under a selected run directory; it is compiled explicitly by the script.

For quantitative benchmark acceptance, independently refine spatial resolution,
quadrature, timestep, photon population/seeds, physical scales, floor/cap, and
box size. Track both front positions as well as profiles. Coarse full-interval
runs can conserve energy perfectly while transporting heat much too far; they
must not be presented as validated RMTV solutions. See [tests/RESULTS.md](tests/RESULTS.md) for
what was actually checked in this workspace.
