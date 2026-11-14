# Continuing a Monte Carlo post-process result

The IMC post-process (`source/3D/radiation/postprocess/`, driven by e.g.
`runs/paola/test.cpp`) averages each pass over independent final generations.
A finished result can be continued: a restart loads it, adds more final
generations to each pass, and writes one combined result, as if the first run
had asked for more generations.

```bash
rich ... --restart.from output/run_g75.h5 --transport.generations 50 \
         --output.stem output/run_g125
```

`runs/paola/submit.sh` has a `RESTART_FROM` setting for this.

## What a restart does

1. Checks that the settings that change the estimate match the file (list
   below) and that the recomputed CER-surface and FLD luminosities agree with
   the stored ones to 1e-6. Any difference stops the run and is listed.
   Files without `/comparison/fld_quadrature_version` (written before
   2026-11-14) took each FLD sample in the owning rank's nearest local cell,
   which is not always the true nearest cell, so their FLD luminosity changes
   by up to ~1e-3 with the MPI size; for them the FLD check uses 1e-2 and
   prints both values. Newer files take every sample in its nearest cell over
   all ranks, and the FLD luminosity does not depend on the decomposition.
2. Loads, for each pass, the running statistics (sample count, sums and sums
   of squares of every tallied quantity), the adaptive state (learned cell and
   group scores, observer deficits and running totals, group history) and the
   step/packet counts of the last final generation.
3. Balances the mesh on those counts, then runs the requested generations
   with the final Neyman allocation. **The burn-in and the learned-only probe
   are never rerun**; a file without this state cannot be continued.
4. Writes the combined result. `transport.generations 0` rewrites the result
   unchanged (the regression test uses this as a round trip).

Each generation's estimate is unbiased for the allocation it used, so the
equal-weight mean over all generations stays unbiased; the generation-to-
generation standard error is computed from the combined sums.

## Settings

Must match the file: snapshot and table paths, snapshot time and cycle,
observer geometry, `transport.source-dt`, `transport.duration`, the DDMC,
random-walk, cell-velocity and Compton settings, the polarization physics
settings, all `flux-source.*`, `volume-emission.enabled`, `.cutoff-fraction`,
`.gate-groups`, `.cooling-limit`, `.cooling-limit-cfl` and `.cooling-limit-dt`, `opacity-scaling.*`, `adaptive.source.enabled`,
`adaptive.group.quality`, `.source-cells` and `.frequency-sampling`, and the
build's `MONTECARLO_POLARIZATION`, DDMC and `ENERGY_GROUPS_NUM`. The DDMC
settings include `transport.ddmc-moving-interface-correction`,
`transport.ddmc-advection`, `transport.ddmc-max-cell-advection-peclet` and
`transport.ddmc-compression-max-dynamical-times`; files written before they
existed (that is, before the 2026-10-09 frame and DDMC fixes, whose results
with cell velocities are not comparable) are refused. Restart schema 2
(2026-10-11) adds the compression work and the split-aware polarization
statistics; files of schema 1 cannot be continued. The current schema is 3,
which adds the multigroup pass's per-group weight-window state; see
[Post-process band weight window](postprocess-band-weight-window.md#restart-schema-3).

May change: `transport.generations`, packet budgets and per-cell limits,
exploration-weight fractions, `volume-emission.cap-learned-weight`, score
power, adaptive targets, load balancing, communication engine, MPI size,
output.

The output may not replace the file being continued unless
`--restart.in-place true` is given; it is written through the usual scratch
file and renamed at the end.

## Random numbers

Fresh runs keep their seeds. A restart seeds generation *k* (counted over all
final generations of the result) from a hash of the pass, the rank and *k*, so
seeds never repeat across segments, and on one decomposition a chain of
restarts (A+1+2) reproduces one restart (A+3) up to floating-point summation
order. With MPI the restart's decomposition differs from the original run's,
so results are statistically equivalent rather than identical.

## File layout

Each pass group `/passes/{forward,grey}` gains:

| Group | Contents |
|---|---|
| `statistics_state/` | `samples` and every running sum, sum of squares, weight-squared sum, crossing count and maximum; arrays are flat with a `<name>_shape` dataset |
| `learned_state/` | learned scores (`adaptive/score_cell_ids`, `score_values`), observer deficits and running totals, group history and group scores, accumulated group-sampling diagnostics, `last_escaping_luminosity`, `completed_final_generations` |
| `load_balance/` | snapshot cell IDs, steps and source packets of the cells touched in the last final generation |

`/restart/segment_count` and `/restart/history/segment_NNNN/` record every
run that contributed: job id, MPI size, source file, generations added and
total per pass, build time and the full effective configuration.
`/metadata/build/energy_groups` records `ENERGY_GROUPS_NUM`.

Results written before restart support have only means and errors; they
cannot be continued.

## Test

`regression_tests/cases/imc_postprocess_restart` (serial; the same program
runs on 4 ranks as `imc_postprocess_restart_mpi`) runs a synthetic
core-envelope snapshot with the paola configuration and checks the round
trip, the chain equality, the generation counts and history, agreement of a
continued run with a fresh run of the same total length, and that a changed
setting or an unflagged overwrite is refused.
