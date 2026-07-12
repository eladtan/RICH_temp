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
Eq. (60).  RICH builds a high-resolution table using 128-point Gauss-Legendre
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

For DDMC-to-IMC leakage, the outgoing asymptotic direction is sampled in the
DDMC cell's comoving frame.  A resident DDMC packet has no meaningful
microscopic pre-leak direction, so RICH must not Lorentz-transform the stale
stored direction into an averaged face frame before sampling the outgoing ray.

If compression shifts a resident packet upward through the current
low-frequency DDMC cutoff before leakage or census, the analytically computed
cutoff-crossing time is included in the event competition.  The packet then
returns to IMC in the same cell with the remaining transport time; it is not
kept in DDMC until the next unrelated event.
transformation from the source-cell frame to the target-cell frame.  Spatial
diffusion and Doppler evolution are operator-split in the Wollaeger
formulation.  RICH retains its cell-divergence update

```text
d ln(nu) / dt = -(div U)/3
```

for a DDMC residence interval.  Frame transformations occur when the packet
changes representation between IMC and DDMC.

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

The runtime diagnostics report interface incidence/admission/reflection,
moving-factor use and fallback, splitting, minimum incident cosine, maximum
`G_U`, invalid leakage geometry and the maximum local reciprocity residual.
