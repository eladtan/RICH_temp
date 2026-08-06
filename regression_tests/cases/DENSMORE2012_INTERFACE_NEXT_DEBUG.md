# Densmore 2012 exposed-interface: next debugging stage

## Why this patch exists

The completed 300-cell comparison is not a generic bulk-DDMC failure.  Its
largest temperature error is in the final thin-side cell, immediately followed
by an opposite-sign error in the first thick-side cell.  At the same face two
changes occur together:

1. the low-frequency DDMC cutoff changes from group 17 to group 19, creating a
   mixed DDMC/transport leakage reaction; and
2. the final thin-side Voronoi generator is intentionally off-center so that
   the material face lies exactly at x=2 cm.

Changing the production formula before separating these effects would be
guesswork.  This patch therefore adds controlled ablations and an event ledger;
it does not change the default DDMC physics.

## Run matrix

The matrix contains four DDMC calculations and two matching pure-IMC controls:

| Geometry | DDMC cutoff | Purpose |
|---|---:|---|
| exposed 300-cell | natural | reproduce the current failure with event data |
| exposed 300-cell | capped at 17 | remove the 17-to-19 mixed channel while retaining the off-center geometry |
| centered-interface 315-cell | natural | remove the off-center geometry while retaining natural cutoff changes |
| centered-interface 315-cell | capped at 17 | remove both suspected mechanisms |

The cutoff cap is diagnostic-only.  Its default value is
`ENERGY_GROUPS_NUM`, which leaves existing calculations unchanged.

Run from the repository root:

```bash
chmod +x run_densmore2012_interface_debug_matrix.sh
RESUME=1 ./run_densmore2012_interface_debug_matrix.sh
```

Set `RESUME=0` to force all cases to rerun.  `NP`, `GROUPS`, `CONFIG`,
`PARTITION`, `SLURM_TIME`, and `TIMEOUT_SECONDS` can be overridden in the
environment.

## New event ledger

Each DDMC case writes one file per MPI rank and per kind of diagnostic:

```text
*_rankNNNN_ddmc_face_history.tsv
*_rankNNNN_ddmc_interface_events.tsv
```

The face history records the two directed representations of every DDMC face
in `1.90 <= x <= 2.10` at every time step.  It includes generator and centroid
locations, center-to-face distances, source/target cutoffs, diffusion data,
internal and asymptotic-boundary rates, the Planck common-band fraction, and
the final DDMC/transport channel rates.

The event ledger distinguishes:

- every IMC crossing candidate into a DDMC-eligible target;
- frequency rejects that cross the face while remaining IMC;
- angular incidents, admissions, reflections, and bypasses;
- DDMC-to-DDMC common-band leakage; and
- DDMC-to-IMC transport-channel leakage, including the sampled output group.

Counts and signed/absolute packet energies are aggregated by step, directed
face, group, and event kind.  This is essential: count balance alone can look
correct while unequal packet weights produce an energy bias.

The analysis script combines all MPI ranks and writes:

```text
*_interface_event_summary.tsv
*_interface_admission_balance.tsv
*_interface_net_energy.tsv
```

The net-energy file treats left-to-right crossings as positive and includes
admitted IMC packets, frequency-rejected IMC packets, bypasses, common-band
DDMC leakage, and DDMC-to-IMC leakage.  Reflections are intentionally excluded
because they do not cross the face.

## Decision table

Use the first-thin-cell and first-thick-cell temperature errors together with
the directional event ledger:

| Result | Interpretation | Next code inspection |
|---|---|---|
| cutoff-17 case removes most of the error; centered case does not | mixed-cutoff construction is primary | replace the band-averaged mixed transport reaction with a group-resolved detailed-balance derivation and test it independently |
| centered case removes most of the error; cutoff-17 case does not | off-center/noncentered face geometry is primary | audit the source control-volume geometry and build a convergence series with centered generators |
| both single ablations improve partially; combined case is good | interaction between cutoff mismatch and geometry | preserve both controls as regressions and repair the coupled face formula |
| combined case still has the sign-changing error | neither leading hypothesis is sufficient | inspect IMC admission/reflection normalization and DDMC absorption/Fleck coupling |
| event energy is balanced but temperatures diverge | interface transport is probably not the deposition source | compare per-cell material deposition and residence-time estimators |
| event energy is directionally biased from the first divergent step | face representation conversion is the source | identify the responsible group/event kind before changing coefficients |

## Required checks before a physics change

1. Confirm all four DDMC cases create exactly one event and face-history file
   per MPI rank.
2. Confirm the capped cases report source and target cutoff 17 at x=2.
3. Confirm the centered-interface mesh reports generators at 1.9975 and 2.0025 and both
   center-to-face distances equal 0.0025 cm.
4. Compare the first time step at which the profile difference grows with the
   first time step at which directional event energy becomes asymmetric.
5. Inspect groups 17 and 18 separately in `ddmc_to_imc`, `imc_admitted`, and
   `imc_frequency_reject` records.
6. Do not tune Monte Carlo packet counts or regression tolerances until the
   deterministic ablation identifies the mechanism.  Repeat seeds only after
   the systematic sign-changing bias is understood.

## Files to upload for the next review

Upload the four comparison `.txt` and `.tsv` pairs plus, for each DDMC case:

```text
*_interface_event_analysis.txt
*_interface_event_summary.tsv
*_interface_admission_balance.tsv
*_interface_net_energy.tsv
*_global_diagnostics.tsv
*_interface_history.tsv
```

Raw per-rank ledgers are only needed if the aggregate analysis points to a
specific rank, group, or directed face that requires event-level inspection.
