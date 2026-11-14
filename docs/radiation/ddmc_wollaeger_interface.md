# DDMC leakage and moving IMC-DDMC interfaces

## Scope

RICH uses the contiguous low-frequency grey-band DDMC construction of
Densmore, Thompson, and Urbatsch (2012).  The implementation accelerates
absorption/effective-scattering transport and isotropic elastic physical
scattering.  Physical elastic scattering enters the transport opacity and is
not sampled as an explicit DDMC event.

Compton redistribution is intentionally not supported with DDMC.  Requesting
`withCompton && withDDMC` is a hard configuration error.  This is deliberate:
sampling every Compton group change would reintroduce long histories in a
Compton-thick cell, while a correct batched in-cell frequency propagator needs
a separate derivation and validation campaign.

References:

* J. D. Densmore, K. G. Thompson, and T. J. Urbatsch, JCP 231,
  6924-6934 (2012), DOI 10.1016/j.jcp.2012.06.020.
* R. T. Wollaeger et al., ApJS 209, 36 (2013),
  DOI 10.1088/0067-0049/209/2/36, arXiv:1306.5700.

## Internal DDMC leakage

For a face shared by DDMC cells `i` and `j`, RICH uses the two-sided diffusion
resistance

```text
R_ij = d_iF / D_i + d_jF / D_j,
T_ij = A_F / R_ij,
lambda_i_to_j = T_ij / V_i.
```

This is not equivalent to dividing an unweighted harmonic mean by the
source-center-to-face distance.  On a uniform Voronoi mesh the old expression
was larger by a factor of two.  The new construction obeys

```text
V_i lambda_i_to_j = V_j lambda_j_to_i = T_ij.
```

The diffusion coefficient, cutoff and eligibility of MPI ghost cells are
exchanged before face rates are built, so a remote face never silently falls
back to the source-cell diffusion coefficient.

## Static transport-diffusion interface

If a packet frequency is not DDMC-eligible in the neighboring cell, the face
uses the asymptotic boundary leakage rate rather than the internal-cell rate.
For an IMC packet incident on a DDMC region, the static admission probability
is evaluated from the same boundary condition.  Rejected packets undergo a
diffuse-albedo reflection and stay in the IMC cell.

DDMC packets emitted into IMC use the angular density associated with detailed
balance of the static admission law.  The implementation therefore does not
reuse an internal-cell leakage rate at an IMC-DDMC boundary.

## Moving interface factor

The moving correction is

```text
G_U(mu) = 1 + 2 (U_n/c) K(mu).
```

`K(mu)` is evaluated from the nonsingular integral in Wollaeger et al.
Eq. (60).  RICH builds a high-resolution table using 192-point Gauss-Legendre
quadrature.  The fitted `a/mu-b*mu` expressions in the paper are not used in
the production path.

The static probability decides whether a packet is admitted.  Once admitted,
the packet weight is multiplied by `G_U`; `G_U` is not multiplied into a
probability that could exceed one.  Large corrected weights are split without
changing the total weight.  For an MPI-remote target the crossing keeps one
unbiased corrected-weight packet because newly created packets cannot be
inserted directly into a ghost cell.

The correction is first order in velocity and was derived for a Lagrangian,
homologous outflow.  RICH currently reconstructs the face material velocity as
a distance-weighted linear interpolation of the two adjacent cell velocities.
If `|U_n|/c` exceeds `ddmcMaxInterfaceVelocityOverC`, or if the exact
quadrature factor is nonfinite or nonpositive, the packet crosses as ordinary
IMC and DDMC entry is suppressed for that cell residence.  Replacing an
invalid moving factor by one would silently restore the known moving-interface
bias and is therefore not permitted.

## DDMC-to-DDMC motion

Ordinary DDMC leakage changes the resident cell but does not apply a Lorentz
transformation from the source-cell frame to the target-cell frame.  Spatial
diffusion and Doppler evolution are operator-split in the Wollaeger
formulation.  RICH retains its cell-divergence update

```text
d ln(nu) / dt = -(div U)/3
```

for a DDMC residence interval.  Frame transformations occur when the packet
changes representation between IMC and DDMC.

For DDMC-to-IMC leakage, the outgoing asymptotic direction is sampled in the
DDMC cell's comoving frame.  A resident DDMC packet has no meaningful
microscopic pre-leak direction, so RICH must not Lorentz-transform the stale
stored direction into an averaged face frame before sampling the outgoing ray.

If compression shifts a resident packet upward through the current
low-frequency DDMC cutoff before leakage or census, the analytically computed
cutoff-crossing time is included in the event competition.  The packet then
returns to IMC in the same cell with the remaining transport time; it is not
kept in DDMC until the next unrelated event.

## Advection with the gas

Packets keep lab positions on a mesh that is frozen during transport, in hydro
and post-process runs alike: after the hydro step the mesh moves but every
packet keeps its position and is only re-located into its new cell, and a DDMC
resident at census becomes an IMC packet at a random point of its cell. So the
gas never carries a DDMC packet with it. IMC gets that motion from the frame
transformations at each scattering; DDMC needs the advection term of the
comoving diffusion equation explicitly,

```text
dE0/dt + div(v E0) + (1/3) E0 div v + div F0 = 0.
```

With `ddmcAdvection` (default on; post-process option `transport.ddmc-advection`)
and the comoving frame active (hydro, or post-process with cell velocities):

* Every face between two DDMC cells gets a first-order upwind advection leak
  `ddmcFraction A max(u, 0) / V`, with `u` the mean of the two cells' gas
  velocities along the outward normal (the same face velocity as the
  divergence estimate, so the leak rates sum to the cell divergence). The
  packet moves to the downstream cell as a resident, with its comoving weight
  and frequency, and adds nothing to the face-flux moment (which estimates
  the diffusive flux `F0`). A face without advection draws its random numbers
  exactly as before.
* At an interface to IMC the gas carries radiation across a fixed face: the
  DDMC side must emit `cE0/4 + u E0/2` while the IMC side brings `cE0/4 - u E0/2`.
  The DDMC-to-IMC leak rate is multiplied by `1 + 2 u/c` (`|u|/c` bounded by
  `ddmcMaxInterfaceVelocityOverC`).
* At a CER flux-source face the IMC transport re-emits in the outer cell's
  comoving frame, so the work term there uses that cell's velocity alone, not
  the mean with the interior cell.
* The cell Peclet number `|v| (L/2) / D` is reported every step
  (`DDMC_ADVECTION` line: maximum and a histogram). `ddmcMaxCellAdvectionPeclet`
  (post-process `transport.ddmc-max-cell-advection-peclet`, default no limit)
  can return advection-dominated cells to IMC.

Validation (`examples/ddmc_advection_pulse`, `examples/ddmc_advection_interface`
in STORM, `regression_tests/cases/imc_postprocess_moving_frame` in RICH):

* A pulse in a uniformly moving thick scatterer moves with the gas: centroid
  displacement / `vt` = 1.006 (static mesh) and 1.004 (mesh moving with the
  gas), against 1.007 for IMC and -0.001 for DDMC without advection.
* A pure-scattering envelope moving at 0.02c around a CER source: DDMC
  reproduces the IMC energy change within 0.003 in outflow. In inflow it falls
  short by about 12% of the change on a 12^3 mesh and 6% on an 18^3 mesh:
  the first-order upwind smearing, about `1.5 beta tau_cell` of the diffusion.
* Known limit: a fixed IMC-DDMC interface with gas flowing through it keeps
  an `O(beta sigma h)` jump. The asymptotic interface condition and `G_U` are
  Lagrangian (interface moving with the gas); the leak factor above removes
  part of it (0.27 to 0.21 at `beta` = 0.03, `sigma h` = 8; 0.38 without
  advection). A fixed-interface moving condition is still to be derived.

## Compression in a frozen post-process flow

The `(1/3) E0 div v` term shifts a resident's comoving frequency and weight
by `exp(-div v dt / 3)` on every DDMC step, `dt` being the step's residence
time in the cell. When the shift carries a packet across the top of its
cell's diffusive band (`groupCutoff`), the packet becomes an IMC packet just
above the band edge with a random direction. In a hydro run the flow evolves
underneath; in a post-process the snapshot is frozen while a packet may live
for the whole transport time, so the shift compounds over far more
compression than the gas could sustain (snap_151: 59% of the multigroup light
came out just above 100 eV, the edge of the cool envelope's diffusive band).

Post-process option `transport.ddmc-compression-max-dynamical-times` (STORM
`postProcess.compressionMaxDynamicalTimes`; RICH default 1, STORM default
unlimited) bounds it. Each packet carries a clock that advances by
`rate dt` on every DDMC, IMC and random-walk step, with
`rate = max(|div v|, |v| / max(r, cell size))` of the current cell and `r`
measured from the observer center; once the clock reaches the limit, DDMC no
longer shifts the packet's frequency or weight (compression and expansion
alike), and the band top can no longer be reached by compression. The clock
is never reset per cell, travels with the packet across ranks and is
inherited by weight-window copies. 1 allows the compression of one local
dynamical time `min(1/|div v|, r/|v|)`; 0 turns the shift off; `inf` keeps
the unlimited shift. A post-process run with cell velocities and a finite
limit uses the host DDMC engine, not the shared kernel.

The work of the shift on packet weights is part of the energy budget
(`diagnostics/compression_work`, included in `transport_sink_residual`); each
generation prints `COMPRESSION` with the band-top conversions and their
energy, the DDMC steps in which a packet's clock ran out, and the work gained
and lost. Packets converted at the band top carry a flag: `ESCAPE_ORIGIN`
reports their escaping energy (`compression_converted`) and `HEAVY_PACKET`
lines mark them. Frame changes at DDMC exits and IMC scatterings also change
weights and remain outside the budget.

Validation (`regression_tests/cases/imc_postprocess_moving_frame`): no work
without DDMC or in a static envelope, negative work in outflow and positive
in inflow, zero work with a limit of 0, and an inflow work at a limit of 0.25
between zero and the unlimited value.

## Optical-depth threshold

The default threshold remains 15.  A larger threshold restricts DDMC to cells
that are deeper in the bulk diffusion limit, but reduces acceleration and can
move an uncorrected transport-diffusion interface into a regime where the
moving boundary-layer error is important.  After the static and moving
interface corrections are active, calibrate the threshold with

```text
3, 5, 10, 15, 30
```

and compare solution error, interface jump, runtime, variance and `G_U`
splitting diagnostics.

## Required validation

Before relying on the implementation for production moving-mesh calculations,
run all of the following in serial and MPI:

1. Uniform-grid pure-scattering diffusion with
   `<r^2> = 6 D t` and `D = c/(3 sigma_s)`.
2. Constant-field preservation on regular and distorted Voronoi meshes.
3. Local and cross-rank checks of
   `V_i lambda_i_to_j = V_j lambda_j_to_i`.
4. Static IMC-DDMC equilibrium-current balance.
5. Independent convergence tests of the Eq. (60) quadrature table.
6. Moving-interface comparisons with and without `G_U` at thresholds 3 and
   10, followed by the full threshold sweep.
7. Homologous-expansion redshift tests confirming that ordinary
   DDMC-to-DDMC leakage adds no extra frame shift.
8. MPI runs with a rank that owns zero cells.

Automated regression coverage includes:

* `ddmc_mpi_zero_cell`, which forces zero-owned-cell ranks and a cross-rank
  DDMC face, then checks remote resident leakage, face-flux reduction, packet
  serialization, weight conservation, and cross-rank leakage reciprocity.

The runtime diagnostics report interface incidence/admission/reflection,
moving-factor use and fallback, splitting, minimum incident cosine, maximum
`G_U`, invalid leakage geometry and the maximum local reciprocity residual.
