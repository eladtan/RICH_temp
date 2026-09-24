# Follow-up: temperature discrepancy and comparison correction

The original timing and probe measurements are real, but the first review missed
Branson's enabled synthetic scattering workload. The 3.654 ratio must not be
interpreted as a comparison of two implementations of the same gray transport
kernel. The recorded upstream code has `intensive_scatter_fraction = 0.1` in
`constants.h`. In the CPU SoA history loop, ten percent of effective scattering
events call `intensive_scatter` and replace the sampled direction with its result.
The sampler's own comment describes mimicking arithmetic and branching in other
transport codes. This executes even with `opacS=0`, since effective IMC scattering
has coefficient `(1-f)*opacA`. STORM's Crooked Pipe gray kernel does not contain
this additional workload. Its timing and physical effects require measurement.

Other verified differences:

- Coarse Cartesian spacings are 0.0625/0.08/0.08 cm; STORM explicitly refines the
  wall with first-layer width 0.01 cm. The total cell counts therefore disguise
  very different resolution of the energy deposition layer.
- STORM's driver sets cell Erad but never invokes `generateInitialParticles` or
  seeds the manager with those packets. The first transport starts from an empty
  census; Branson explicitly samples its 0.05-keV initial equilibrium field.
- Inlet staircase area and source/population allocation differ, as documented in
  review.md. Random walk is enabled only in STORM.
- The same thin-material, volume-weighted radius-0.1-cm probe definition is used.
  The final third-probe gap (~0.06935 keV) is much larger than the three-run
  observed spreads (STORM ~0.000426 keV, Branson ~0.000145 keV), so simple repeat
  noise is not a credible explanation of that gap. This is not a formal estimate
  of discretization or Monte Carlo bias.

Diagnostic job 10176489, on the same 128 ranks and eight nodes:

1. Branson coarse mesh, original radiation initialization, synthetic scattering
   disabled: isolates the extra workload against the original Branson runs.
2. Same Branson case with initial radiation temperature zero: isolates initialization.
3. Same empty-initial-radiation case with each Cartesian spacing halved (2.24M
   cells), regenerating the cylinder's material map: tests resolution/geometry
   sensitivity. The global photon budget stays fixed, but per-cell floors and
   the discretized source area change; this is a resolution sensitivity test,
   not a controlled runtime comparison.
4. The unchanged STORM executable with `--no-random-walk`: tests whether its
   acceleration explains the temperature gap.

No production source or frozen executable was modified. The no-synthetic-work
Branson binary is isolated under `build/branson_cp128/diagnosis/`.

## Completed diagnostic results

All four full diagnostics completed 120 cycles successfully in job 10176489.
These are one run per diagnostic, not repeated performance measurements.
All temperatures below are the same third probe at 81.5719 ns.

| Configuration | Probe 3 (keV) | Total time (s) |
|---|---:|---:|
| Original STORM, random walk on (three-run mean temperature; median time) | 0.124691 | 65.690 |
| Original Branson, synthetic work on (three-run mean temperature; median time) | 0.055338 | 240.023 |
| Branson coarse, synthetic work off | 0.055577 | 180.480 |
| Same, empty initial radiation | 0.055529 | 180.843 |
| Same, Cartesian spacing halved | 0.070792 | 134.096 |
| STORM, random walk off | 0.124851 | 88.941 |

Disabling synthetic work removes about 25% of the original Branson runtime,
but leaves the large temperature discrepancy. Emptying Branson's initial
radiation field changes probe 3 by only -0.0000476 keV. Turning STORM's random
walk off changes it by +0.0001597 keV relative to its original three-run mean;
this does not explain the discrepancy either.

Halving Branson's Cartesian spacings increases probe 3 by 0.0152633 keV
(27.49%). It closes
22.07% of the gap to STORM in this comparison. Probe 2 also rises
from 0.224865 to 0.257048 keV, toward STORM's original mean 0.283745 keV.
This is direct evidence of substantial mesh/interface sensitivity. It does not
prove that refinement alone explains the entire remaining gap, or that STORM
is spatially converged. Refinement also changes floors, populations and the
staircase material geometry, so this is not a pure cell-size-only error estimate.

The fine mesh has 224x100x100 cells (2.24M), spacings 0.03125/0.04/0.04 cm,
296064 thin cells, thin volume 14.8032 cm^3, and inlet area 0.7744 cm^2.
Its discrete inlet area is just 0.8333% larger than the coarse mesh's 0.768 cm^2.

A separate one-step source check, job 10176490, measured STORM's injection as
5.05023e11 erg. The coarse Branson input gives 4.93578e11 erg for that time step,
so the initial drive mismatch is about 2.32%, not a factor of two.
STORM's ledger had zero interior particle removals and normalized residual
1.41973e-11 in this check.

## Physical interpretation and remaining uncertainty

The best-supported explanation is unequal resolution of the absorbing wall
interfaces. Uniform-cell IMC deposits energy into cell-averaged material and
emits across the cell; unresolved wall heating and spatial transport errors can
remove too much energy from the transparent channel into the opaque material.
This is a mechanism consistent with the measured refinement trend, not a
measurement of each contribution to the remaining error.

Steinberg & Heizler's actual Crooked Pipe discussion identifies radiation
teleportation into the opaque material, uses logarithmically refined interfaces,
and compares smallest wall cells of 0.004 and 0.0015 cm. Both Branson grids here
are much coarser at the walls; STORM's 0.01-cm first-layer specification is also
coarser than those published interface scales. See section IV.3:
https://arxiv.org/html/2108.13453v2#S4.SS3

Before asserting agreement or an equal-accuracy speedup, the remaining work
would be a spatial convergence study with matched source/initialization and
adequate particle statistics, checking both codes. No such claim follows from
these diagnostics. The original 3.654 runtime ratio is an as-run result that
includes Branson's extra synthetic workload; it should not be reused as a
matched-gray-kernel speedup. The no-synthetic coarse control took 180.480 s,
and the finer empty-initial-radiation control took 134.096 s. Neither has been
shown to match STORM's accuracy.

Exact diagnostic measurements: `build/branson_cp128/diagnosis/run_10176489/summary.json`.
The accepted STORM source and binary remain unchanged. No RDMA or reallocation
protocol changes were made. All diagnostic and source-check jobs are complete.
