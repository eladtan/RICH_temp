# Heat-front mismatch investigation

**Follow-up:** [one-step transport isolation](TRANSPORT_LEDGER_FINDINGS.md)
now measures actual regional energy redistribution with hydro disabled.
It confirms a large IMC excess with adverse timestep dependence and a smaller
coarse-grid DDMC excess that approaches the reference under spatial refinement.
The discussion below records the earlier, less conclusive investigation.

The saved `verification/strong16/profiles_final.png` has an advanced heat front
and a broadened compressed shell. The strongest evidence from the controlled
checks below is for a large spatial discretization error in nonlinear heat
transport through the unresolved shell. Classical IMC teleportation remains a
possible additional contribution, but has not been isolated as the main cause.
Conservation alone does not validate these profiles.

## A concrete spatial error before evolution

Using the initialized, conservatively projected reference fields, evaluate the
LTE limit of the actual DDMC internal-face stencil. Compare its outward heat
flux with quadrature of the continuum reference flux on the same faces bounding
the set of cell centers with radius below 0.32 cm. This lies outside the initial
shock at 0.225 cm and inside the heat front at 0.45 cm. The face quadrature points
are checked to lie strictly in this smooth precursor. The opacity cap is
inactive at the evaluated face temperatures.

| Octant grid | Stencil / integrated reference heat flux |
|---|---:|
| 12^3 | 1.8591 |
| 24^3 | 1.0083 |

The approximately 86% excess on the coarse grid is present without photon
sampling, time advancement, or hydro evolution. It is a diagnostic of the
spatial operator acting on cell averages, not a measured MC energy tally and
not a quantitative attribution of the entire evolved profile error.

The stencil in `source/monte/radiation/ddmc/DDMCEngine.hpp` evaluates each side's
diffusion coefficient at a shared radiation temperature
`Tf = ((TL^4 + TR^4)/2)^(1/4)` and combines the two resistances. In this benchmark
the continuum material conductivity is `chi0 * rho^-2 * T^6.5`. Representing the
compressed shell and its temperature structure with a few cell averages can
therefore significantly change the heat flux. At initialization the shock
radius is only 2.25 cells on the 12^3 octant grid, and 4.5 cells on the 24^3 grid.

The face-temperature prescription is not automatically a factor-of-two bug:
at constant density, its hot-to-cold limiting flux differs from the exact
integral of this power-law conductivity by only about 2.2%. The much larger
measured discrepancy involves the spatially varying reference and projection.

## Controlled evolution checks

All comparisons below use the same physical endpoint within each table,
quadrature order 8, a fixed Cartesian octant mesh, and DDMC enabled. The base
particle settings are 8 new, 8 initial, and population parameter 16 per cell;
the base seed is 12345. All runs reached their requested endpoints.

At heat-front reference radius 0.5 cm (starting at 0.45 cm):

| Change from base | Temperature L1 |
|---|---:|
| Base: 12^3, dt/age = 0.005 | 23.12% |
| Four times smaller timestep | 22.58% |
| Four times larger timestep | 24.19% |
| Four times all particle population settings | 22.72% |
| Seed 67890 | 22.98% |
| 24^3, otherwise same settings | 13.36% |
| Four times optical depth, one-quarter radiation heat capacity ratio | 22.73% |

The last row scales H by 16 and K by 1/sqrt(32), scaling the temperature floor
by 1/sqrt(32) and opacity cap by 4 to preserve their dimensionless values.
Its CLI values are `--time-unit .4096 --temperature-unit 165.7281518405971
--floor .1767766952966369 --opacity-cap 4e8`. This weak response argues against
finite radiation heat capacity or departure from diffusion being the leading
error in this comparison. An earlier exploratory scale run retained the
original floor/cap; it is saved in the JSON but is not used in this table.

Doubling spatial resolution has a much larger effect than the sampled
timestep or particle variations. This is not a measured convergence order;
more resolutions and statistical realizations would be needed for that claim.
The short-interval density L1 does not improve in this comparison (about 6% on
both grids), so the hydro shock itself still needs a separate convergence study.

At the full reference heat-front radius 0.9 cm, all on 12^3:

| dt/age | Opacity cap [cm^-1] | Temperature L1 |
|---|---:|---:|
| 0.02 | 1e8 | 58.85% |
| 0.005 | 1e8 | 57.62% |
| 0.00125 | 1e8 | 58.09% |
| 0.005 | 1e12 | 58.17% |

The large error persists over a factor of 16 in timestep and a factor of
10,000 in opacity cap. Neither variation fixes it. The small nonmonotonic
differences are not interpreted as significant without repeated seeds.

## Reference and coupling checks

An independent finite-difference check evaluates

`dE/dt + div((E+p)v) = div(chi grad T)`

directly on interpolated dimensional reference fields at xi = 0.2, 0.6, 0.9,
1.1, 1.4, 1.7, and 1.9. The ratio of the two sides is between 0.99795 and
1.00047. This rules against a large conductivity normalization error at these
sampled smooth points; interpolation and differentiation limit this check's
accuracy near the front. Earlier reference jump, mass, and initialization
checks are recorded in `RESULTS.md`.

The thermal radiation update refreshes material internal energy, temperature,
and pressure; the driver then synchronizes total gas energy with kinetic
energy. No new double-counting or stale-pressure defect was identified in this
inspection. These runs retain energy conservation near roundoff.

## Teleportation interpretation

STORM's thermal source samples locations throughout a cell; it does not retain
subcell material absorption locations. Thus its IMC branch has the mechanism
for classical teleportation. See [Irvine, Boyd, and Gentile's description of
thermal-emission spatial error](https://www.osti.gov/servlets/purl/1759975).

However, these are DDMC-enabled calculations, with a different spatial
diffusion stencil in eligible cells. The observed front advance by itself is
not enough to attribute the error to IMC emission resampling. The timestep
sweep does not show a large worsening with timestep reduction, and the LTE
stencil already exhibits a substantial error outside the shock. Calling the
entire discrepancy teleportation would overstate the evidence.

The next corrective investigation should target resolution of the compressed
shell and nonlinear face heat flux. A comparison with deterministic diffusion
using the same hydro and cell averages, followed by a DDMC/IMC/source-treatment
comparison, would separate spatial diffusion error from emission teleportation.
No production solver changes were made during this diagnosis.

## Follow-up: IMC without DDMC

A matched four-step comparison on the 12^3 octant grid uses dt/age = 1e-5,
quadrature 8, and the default physical scales and opacity cap. The actual final
age is 0.00048210391022992014 s; the reference heat front advances only from
0.45 to approximately 0.45001246 cm. These runs deliberately stop at four
steps and do not validate the full benchmark.

| Method | New / initial / population parameter | Temperature L1 |
|---|---|---:|
| DDMC enabled | 8 / 8 / 16 | 0.0943% |
| Pure IMC (`--imc`) | 8 / 8 / 16 | 1.0007% |
| Pure IMC (`--imc`) | 32 / 32 / 64 | 0.9723% |

The base IMC and DDMC runs conserve energy to about 7e-15. DDMC event counts
are zero for IMC and positive for the accelerated comparison. The sum of
manager times for the base four-step runs was approximately 0.823 s for IMC
and 0.053 s for DDMC. These are local timings, with concurrent diagnostic
workloads, not a performance benchmark.

Thus switching off DDMC is supported, but it does not remove the short-time
profile error in this test. Increasing the IMC particle settings fourfold
barely changes its temperature L1. Uniform cell emission remains a plausible
source of this discrepancy; this experiment does not isolate that mechanism.

A second comparison uses the normal dt/age = 0.005. At the first-step age
0.00048449504968838317 s, pure IMC has temperature L1 4.4845%, versus 1.5875%
with DDMC. Its first radiation manager call took approximately 86.44 s,
compared with 0.0298 s for DDMC on this machine. The IMC run hit its
180-second timeout after completing two steps toward a reference heat front
of 0.5 cm; use only its completed checkpoints, not a claimed final-time result. Large effective
scattering counts in opaque material account for the severe runtime cost.
The matching DDMC output is `verification/diagnosis_ddmc_one` and the IMC
checkpoint is `verification/diagnosis_imc/step1_rank0.csv`.

Outputs are `verification/diagnosis_tiny_imc`, `diagnosis_tiny_ddmc`, and
`diagnosis_tiny_imc_more`. Reproduce with the base command below, replacing
`--rf-end .5 --dt-fraction .005` with `--max-steps 4 --dt-fraction 1e-5` and
adding `--imc` for the unaccelerated runs.

## Reproduction

From `runs/RMTV`, the base short diagnostic is:

```bash
./run.sh --octant --n 12 --quadrature 8 --rf-end .5 --dt-fraction .005 \
  --photons 8 --initial-photons 8 --max-photons 16 --dump 10000 \
  --output verification/diagnosis_base_dt
```

Use fresh output names, vary the corresponding option from the tables, and
use `--rf-end .9` for the full-interval cases. The higher particle case uses
`--photons 32 --initial-photons 32 --max-photons 64`. The finer spatial case
uses `--n 24`. Saved output is under `verification/diagnosis_*` in this workspace.

`python3 tests/diagnose_mismatch.py` reads those runs and recomputes the
continuum residual and initial face-flux comparison. The face-flux calculation
expects the named base and fine-grid output directories. Numerical measurements
are retained in `mismatch_diagnosis.json`.
