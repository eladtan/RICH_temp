# RMTV implementation plan for STORM + RICH

Prepared 2026-09-09 from the working tree at RICH `2c71b28f` and embedded STORM
`55ed49f`, including their current local changes. This is a plan; no solver or
benchmark implementation has been changed or run.

## Recommendation and scope

Implement the spherical strong-conduction Reinicke–Meyer-ter-Vehn problem as a
**coupled RICH hydrodynamics + STORM grey IMC/DDMC benchmark** under
`runs/rmtv/`. STORM supplies transport and material coupling; RICH supplies the
evolving density, velocity, pressure, and mesh. A standalone STORM example with
prescribed material profiles would exercise transport, but would not verify the
full RMTV problem.

The reference equations are Euler hydrodynamics with nonlinear **material heat
conduction**, not general radiation hydrodynamics. The proposed STORM test must
therefore demonstrate convergence toward the material-dominated, LTE diffusion
limit. Matching an opacity exponent alone does not establish equivalence.
If that limit cannot be reached economically, a direct nonlinear conduction
step in RICH is a separate fallback project; it would verify RICH conduction
and hydro, not STORM transport. [1]

## 1. Freeze the mathematical specification and units

Use the standard strong-conduction parameter set:

| Quantity | Value or definition |
|---|---|
| Geometry | Spherical, represented on a three-dimensional mesh |
| EOS | `p = (gamma-1) rho e = Gamma rho T`, constant specific `Cv` |
| `gamma` | `5/4` |
| Conductivity | `chi = chi0 rho^(-2) T^(13/2)` |
| Similarity coordinates | Heat front `xi_f = 2`, shock `xi_s = 1` |
| Reference defaults | `chi0 = 1`, `Gamma = 1`, `g0 = 1`, `beta0 = 7.197534e7`, in the reference conventions |
| Ambient state | Cold, stationary; density proportional to `r^(-19/9)` |
| Front motion | `r_f proportional to t^(9/13)` and `r_s = r_f/2` |
| Suggested interval | `r_f: 0.45 -> 0.9`, hence `r_s: 0.225 -> 0.45` |

The defaults come from ExactPack; the start/end front positions follow the CRASH
verification example. The exponents above follow by substituting the defaults
into the reference similarity formulas. [2,3,4]

Deliver `benchmark.json` with every physical parameter, scale, start/end time,
floor, seed, and numerical option. Convert the reference convention explicitly
to STORM's centimetres, grams, seconds, kelvin, and ergs. In particular, do not
use `Gamma = 1` as a cgs EOS parameter or copy `chi0 = 1` into a kelvin-based
conductivity without conversion. Derive `Cv = Gamma/(gamma-1)` in the chosen
temperature units, then use `IdealGas(gamma, Cv, 1, 0)`.

**Completion gate:** the converted reference satisfies the EOS, conductivity
units, front locations, and time scaling before any transport run.

## 2. Build a trustworthy reference generator

Add `reference/generate.py`, a pinned upstream revision/license record, and
checked-in profile tables with generation metadata. Prefer the existing
LANL ExactPack/Timmes implementation over writing a new similarity ODE solver.
Use a second published implementation/table to cross-check representative
points and shock states where available.

Important reference-interface details:

- ExactPack's `Rmtv._run(r,t)` ignores `t`; set `rf` separately for each time.
- Use the underlying similarity relation `t = (rf/(zeta*xif))^(1/alpha)`
  with `alpha = 9/13`; preserve the physical start-time offset when the
  simulation clock starts at zero.
- Audit actual output conversions: the underlying code multiplies velocity by
  `1e8`, specific energy and pressure by `1e16`, and temperature by `1e3`.
  Its comments/docstrings are not entirely consistent about the final units.
- Handle the origin by controlled integration/asymptotics. The upstream routine
  contains a small-radius integration cutoff; blindly evaluating `r=0` is not
  an adequate central-cell initialization. [2,3]

Generate profiles at the initial time, intermediate checkpoints, and final
time. Keep interpolation on each side of the shock separate. Integrate mass,
momentum, and total material energy over cells using quadrature that resolves
the shock and central region; derive primitives afterward. Use the same
averaging convention for comparisons. For shell diagnostics, use actual cell
volumes, not an equal-weight average of cell centers.

**Completion gate:** profiles pass EOS, front-ratio, similarity-collapse,
quadrature refinement, and conserved-integral consistency checks.

## 3. Establish the radiation-to-conduction mapping

For LTE grey diffusion, with macroscopic transport opacity `Sigma_tr`:

```text
Er = a_rad T^4
F  = -c/(3 Sigma_tr) grad(Er)
   = -[4 a_rad c T^3/(3 Sigma_tr)] grad(T).
```

Matching `F = -chi grad(T)` gives the proposed mapping:

```text
Sigma_tr = [4 a_rad c/(3 chi0)] rho^2 T^(-7/2)   [cm^-1].
```

This is a derivation for this implementation, not a claim that finite-opacity
IMC solves the exact RMTV equations. A mass opacity would have one fewer power
of density; STORM's transport event distances use the macroscopic opacity.
Start with pure grey absorption (`Sigma_a = Sigma_tr`, scattering zero), so
the Planck and Rosseland opacities coincide.

Add a run-local `RMTVOpacity.hpp` implementing Planck, frequency-independent
absorption, and temperature-evaluated methods consistently. Existing
`source/3D/radiation/PowerLawOpacity.cpp` shows the needed functional form, but
hardcodes a 1 K floor and throws in `CalcAbsorptionOpacity`. Avoid inheriting
those assumptions into the benchmark. No generic opacity API change appears
necessary; the RICH adapter already forwards temperature-dependent methods.

Initialize `cell.Erad = a_rad*T^4/rho` and generate an isotropic LTE photon
population, consistently in the frame expected by the existing hydro coupling.
Check material and radiation extensives against the initialized packet energy.

Before choosing production scales, evaluate over the evolving reference:

- Radiation energy/material energy `a_rad*T^4/(rho*Cv*T)`.
- Radiation/material heat capacity `4*a_rad*T^3/(rho*Cv)`.
- Radiation pressure/gas pressure and material speed/light speed.
- Mean free path/temperature-gradient scale and absorption equilibration
  time/evolution time (including the stiff numerical Fleck response).

Require these departures from the conduction limit to be smaller than the
intended benchmark error in the resolved heated region. Treat the cold front
and central region separately, since local ratios/scales can be singular.
Use a dimensional scaling study that preserves the dimensionless RMTV solution
and rederive all EOS/conductivity coefficients. Keep the physical light speed
initially. Changing light speed alone changes the mapped conductivity.

Use explicit, configurable temperature floors and finite opacity bounds;
record where they activate and their added energy. Demonstrate insensitivity
under tighter floors/bounds. Do not silently turn the cold region transparent.

**Completion gate:** a short run preserves the energy budget, remains finite,
and demonstrates a practical parameter regime approaching the target limit.

## 4. Assemble the coupled driver

Use these existing integration points:

| Existing code | Role |
|---|---|
| `runs/Elad_paper_mach2_lagrangian/test.cpp` | Example of `HydroStep`, `RadiationIMC`, `RadiationMCStep`, population control, and coupling |
| `source/newtonian/three_dimensional/simulation/steps/RadiationMCStep.*` | Material/packet evolution and mesh-change integration |
| `source/3D/radiation/RadiationIMC.hpp` | Adapter from RICH cells, EOS, opacity, and conserved state to STORM |
| `source/monte/radiation/RadiationIMCParameters.hpp` | Grey IMC, DDMC, random-walk, and hydro options |

Begin with a fixed, regular three-dimensional Voronoi mesh covering a full
cube centered on the explosion. This avoids importing planar geometry or
introducing a spherical transport grid API. Use analytic material ghost states
at outer boundaries beyond the final heat front. Implement radiation boundaries
consistent with the cold exterior/floor and tally escaped/injected energy.
Check the result with a larger box. An octant with reflecting symmetry planes
is a subsequent cost reduction after the full-domain smoke test.

Initialize at the finite reference time, rather than depositing a mesh-dependent
point-energy bomb. Set radial vector velocities from the analytic profile.
Use hydro feedback and full three-dimensional momentum; do not copy the Mach-2
example's `planarMomentumX` or slab transport settings. Disable Compton,
polarization, and multigroup transport. Preserve the existing conservative
radiation coupling and measure its asymptotically small pressure/work effects.

Start with ordinary IMC for a short, affordable baseline, with hydro CFL and a
temperature-change/time-step refinement study. Then enable DDMC in optically
thick cells and compare against IMC on an overlapping affordable case. Add
random walk only as a separately validated option. Once fixed-mesh results are
stable, test moving mesh and MPI particle migration; defer AMR.

**Completion gate:** the coupled evolution advances both fronts with the correct
ordering and produces repeatable radial profiles without conservation drift.

## 5. Diagnostics, convergence, and regression integration

Write snapshots and a compact diagnostic file containing density, radial and
transverse velocity, pressure, material temperature, radiation temperature,
specific internal energy, cell volumes, radiation energy, and limiter/floor
counts. Track total mass and material kinetic/internal plus packet radiation
energy, including boundary fluxes and population-control changes. In a full
sphere, also check net momentum and angular variation.

Add `plot.py` and `regression_tests/lib/check_rmtv.py` to report:

1. Volume-weighted normalized L1 errors against cell-averaged reference fields.
   Avoid pointwise relative temperature errors in the zero-temperature exterior.
2. Shock and thermal-front locations, their ratio, and their time scaling.
   Specify the numerical front estimator and test its threshold sensitivity.
3. Radial profiles and angular scatter, including error in transverse velocity.
4. Boundary-corrected conservation and radiation/conduction-limit diagnostics.
5. Separate mesh, time-step, photon-number/seed, and physical-limit studies.
   At least three mesh levels; several independent seeds for uncertainty.

Set regression thresholds from demonstrated convergence and statistical
uncertainty, not a single visually matching plot. Shock-containing global
errors should not be required to converge at second order. [4]

Register `rmtv_imc` and, after agreement, `rmtv_ddmc` with RICH THUNDER metadata
(`CATEGORY=physics`, serial/MPI variants as supported), extend
`regression_checks.sh`, and update the test catalog. Keep a cheap smoke case
separate from the full convergence study. Use statistical serial/MPI agreement,
not bitwise equality across decompositions.

Proposed deliverables:

```text
runs/rmtv/
  test.cpp                 coupled problem setup
  RMTVOpacity.hpp          grey opacity with explicit regularization
  benchmark.json           physical/numerical specification
  reference/generate.py    pinned reference generation and conversions
  reference/*.csv          initial/checkpoint/final profiles
  reference/README.md      provenance, units, reference checks
  plot.py
  README.md
regression_tests/cases/rmtv_imc/{test.cpp,REGRESSION_INFO}
regression_tests/cases/rmtv_ddmc/{test.cpp,REGRESSION_INFO}
regression_tests/lib/check_rmtv.py
```

## Implementation order and decision points

Implement in five reviewable increments: reference/units; opacity/limit study;
fixed-mesh coupled IMC; DDMC comparison and convergence; regression/MPI and
moving mesh. The first two increments are the feasibility gate. The principal
uncertainty is the cost of reaching the conduction limit near the thermal
front, not the availability of the hydro/transport interfaces. Do not promise
an exact RMTV match at arbitrary finite opacity or after grid refinement alone.

## Sources

1. [LANL ExactPack RMTV governing equations](https://github.com/lanl/ExactPack/blob/master/exactpack/solvers/rmtv/__init__.py).
2. [LANL ExactPack RMTV defaults and wrapper](https://github.com/lanl/ExactPack/blob/master/exactpack/solvers/rmtv/rmtv.py).
3. [LANL ExactPack Timmes implementation: scaling, integration, and conversions](https://github.com/lanl/ExactPack/blob/master/exactpack/solvers/rmtv/timmes.py).
4. [van der Holst et al., CRASH implementation and verification, section 4.3.2](https://arxiv.org/abs/1101.3758).

These implementations cite Reinicke & Meyer-ter-Vehn, *The Point Explosion with
Heat Conduction*, Physics of Fluids A 3, 1807 (1991), and Kamm,
*Investigation of the Reinicke & Meyer-ter-Vehn Equations: I. The Strong
Conduction Case*, LA-UR-00-4304 (2000).
