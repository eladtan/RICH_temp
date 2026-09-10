# Reference provenance

`vendor/timmes.py` is unmodified LANL ExactPack at commit
`9bacc477eb791e959c9621e771fc7a725db755fa`. Its BSD-3 license is retained as
`vendor/LICENSE.txt`. Source:
https://github.com/lanl/ExactPack/tree/9bacc477eb791e959c9621e771fc7a725db755fa/exactpack/solvers/rmtv

`generate.py` captures dense output from the two upstream integrations. The
upstream routine mutates its last pre-shock state when applying the shock jump;
the dense interpolants retain the separate one-sided solutions. `shape.dat`
therefore has two rows at xi=1; interpolating across these rows is forbidden.
The generator runs locally; no network access or ExactPack installation is
needed for generation or verification (NumPy/SciPy are required).

The table contains xi, xi^(kappa+sigma)*h, xi*u, xi^2*theta. The C++ reader
converts these dimensionless shapes using the documented physical scales.
There is no hidden eV/keV, jerk/erg, or shake/second conversion in the table.

The rounded published eigenvalue becomes ill-conditioned extremely close to
the origin. Below xi=0.02, the table uses a regular continuation: constant
central density and temperature, velocity proportional to radius. Its radius
is 1% of the heat-front radius. This is an explicit reference approximation,
not a new point-energy source. No pointwise evaluation is made at the singular
initial-time ambient profile. The small core's integral effect is included in
the enclosed-mass check; finer central-resolution studies must regenerate with
a smaller core and check sensitivity or refine the similarity eigenvalue.

The final interval from the upstream integration start (within approximately
1e-12 of the front) to xi=2 is continued to zero temperature. Beyond xi=2,
density follows the analytic ambient power law and velocity/temperature vanish.

Run `python3 reference/generate.py`, then `python3 reference/check.py` from the
run directory. `verification.json` records independent upstream point checks,
enclosed mass, similarity energy scaling, shock jump fluxes, and unit mapping.
The C++ quadrature checks in `tests/reference_test.cxx.in` additionally test the
actual reader, reflection symmetry, conserved projection, and grid/quadrature
refinement. The mass integrals are sensitive to under-resolving the density jump;
the main run exposes quadrature order for this reason.
