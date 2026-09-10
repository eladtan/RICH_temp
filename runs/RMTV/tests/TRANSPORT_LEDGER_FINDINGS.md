# One-step transport isolation

## Result

With hydro disabled, pure IMC transports far more energy than the reference
across all three measured surfaces. Refining space reduces the excess;
reducing the timestep increases the measured/reference transport ratio.
This is strong evidence of a teleportation-type spatial error in the IMC
material/emission representation. Hydro evolution and hydro/radiation
splitting cannot cause this particular discrepancy because no hydro step runs.

DDMC has a smaller, localized excess outside the compressed shell on the coarse
grid. Its measured transfer approaches the reference on the finer grid. This
supports the earlier spatial-stencil diagnosis for DDMC, now using actual
energy redistribution rather than just an LTE stencil estimate.

This does not demonstrate a single shared coding bug, nor does it attribute
every feature of the full coupled profile to these one-step errors. A source
representation change would still need its own verification.

## Measurement

`--radiation-only` skips HydroStep while retaining thermal exchange. Density,
velocity, mesh, and mass remain exactly fixed; the material temperature is
allowed to respond to radiation. Opacities and source coefficients follow the
normal IMC step treatment. We start from the same conservatively projected
reference state and LTE packet initialization for both methods.

For each region consisting of cells whose centers satisfy `r < R`, the run
records gas extensive energy plus the energy of census photons assigned to
those cells, before and after the step. The difference is the net outward
energy across the region's mesh-face boundary. Local absorption and emission
cancel in this balance. Population control preserves energy cell by cell.
The surfaces include symmetry-plane faces, whose normal flux is zero.
No temperature-derived flux or LTE transport approximation enters the measured
energy ledger. This is a net transfer measurement by regional conservation,
not a tally of the individual crossing histories or gross one-way fluxes.

The reference energy is an independent space/time quadrature of
`-chi0 * rho^-2 * T^6.5 * grad(T)` through the same mesh faces. The smooth
reference solution is used on those faces, not the cell-averaged stencil.
All sampled points are checked to lie outside the shock. The heat-front
continuation is treated as zero flux beyond the last reference table point.

Checks passed for all 16 one-step runs:

- Density, Cartesian velocity, mesh coordinates, and volume are bitwise
  unchanged; the hydro energy-change diagnostic is zero.
- Snapshot regional energies independently reproduce the extensive/packet ledger.
- Maximum whole-domain relative energy drift is 3.6e-15.
- The reference luminosity changes by at most 0.054% over any tested timestep.
- Increasing face quadrature from order 8 to 16 changes the reference integral
  by at most 0.121%, near the heat front; near the shell the difference is tiny.
- DDMC event counts are zero for pure IMC and positive for DDMC runs.

Thus neither finite reference evolution during the step nor the reference
quadrature explains the measured excesses. These are fixed-start, one-step
experiments, not a long-time temporal convergence study.

## Spatial comparison

Default physical scales, octant box 1.2 cm, initial heat front 0.45 cm, shock
0.225 cm, quadrature 8, and `dt/age = 1e-4` give
`dt = 4.820846265556052e-8 s`. Entries are measured/reference net outward energy
ratios. Ranges show the two seeds 12345 and 67890, not confidence intervals.

| Method / grid | R=0.32 cm | R=0.37 cm | R=0.42 cm |
|---|---:|---:|---:|
| IMC, 12^3 | 34.6–40.6 | 21.4–23.7 | 23.1–23.6 |
| IMC, 24^3 | 8.45–10.54 | 8.82–9.86 | 13.26–13.32 |
| DDMC, 12^3, higher statistics | 1.61–1.86 | 1.00–1.11 | 0.994–0.994 |
| DDMC, 24^3, higher statistics | 0.98–1.10 | 0.98–0.99 | 1.08–1.09 |

The IMC and original DDMC runs use 32 new and 32 initial photons per cell and
population parameter 64. Because the small DDMC net transfer had appreciable
sampling noise, DDMC was repeated with 256 new photons / population parameter
512 on 12^3, and 128 / 256 on 24^3; both keep 32 initial photons per cell.
These higher-statistics repetitions are used in the table. They change sampling
precision, not the deterministic face formula. Two seeds are insufficient for
a formal uncertainty estimate; the large IMC discrepancy is present in both.

For scale, at R=0.32 on 12^3 the reference transfer is about 1.125 erg; the
base IMC run transfers 38.97 erg. The whole-domain energy remains conserved.

## Timestep comparison

Pure IMC, 12^3, 32 new/initial photons, population parameter 64, seed 12345:

| dt/age | Measured energy at R=0.32 [erg] | Reference [erg] | Ratio |
|---|---:|---:|---:|
| 4e-4 | 85.94 | 4.498 | 19.10 |
| 1e-4 | 38.97 | 1.125 | 34.64 |
| 2.5e-5 | 16.82 | 0.281 | 59.83 |

The absolute energy decreases with the step duration, but much more slowly
than the physical transfer. This is the adverse timestep dependence that the
earlier full-profile timestep sweep did not isolate. The surfaces farther out
show the same pattern. Together with uniform within-cell emission in
`source/monte/radiation/source/IMCSourceProcess.hpp`, this supports the
teleportation interpretation. It does not distinguish initial cellwise
radiation-field discontinuities from source redistributions within that same
step. Both belong to the spatial representation being tested.

## Reproduction

Build the updated RMTV executable, then from `runs/RMTV`:

```bash
./run.sh --radiation-only --imc --octant --n 12 --quadrature 8 \
  --dt-fraction 1e-4 --max-steps 1 --photons 32 --initial-photons 32 \
  --max-photons 64 --seed 12345 --dump 1 --output verification/ledger_example
python3 tests/check_transport_ledger.py verification/ledger_example
```

Remove `--imc` for DDMC; vary resolution, timestep, particle settings, and seed
as listed above, always using a fresh output directory. Each run intentionally
stops after one step (`status.json` records that the full RMTV endpoint has not
been reached). Reference profiles at the new age are not an accuracy test of
the radiation-only evolution, since hydro is deliberately absent.

The data are in `verification/ledger_*`; the complete numerical report is
`tests/transport_ledger_results.json`. The updated executable builds in serial
and MPI; the diagnostic comparisons above use serial execution.
A normal coupled one-step run also reproduces the pre-change snapshot fields
bitwise, checking that the new diagnostic mode did not alter normal evolution.
