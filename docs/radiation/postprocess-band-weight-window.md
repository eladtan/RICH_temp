# Post-process per-group weight window and packet provenance

The IMC post-process (`source/3D/radiation/postprocess/`) can split heavy packets
with a weight window (`transport.split-observer-fraction`, `WeightSplit.hpp`). This
page describes the per-group extension of that window for the multigroup pass,
the packet-provenance diagnostic that tells whether the window can reach the
heavy packets of a band, the raw per-generation Stokes records, the replicate
seed offset, and the restart state they need.

## Why

The scalar window caps a packet at

    W = split-observer-fraction x (escaping energy of the last generation) / (observers)

divided by the cell importance `I_c = min(1, p_c / split-escape-reference)`
(`p_c` the cell's escaped over emitted energy). `W` is a fraction of the *total*
per-observer energy. A band holding 1% of the luminosity therefore lets through
packets ~10 times its whole per-observer energy per generation, and a few such
packets dominate its generation-to-generation variance (snap_429, run 28921407:
the 200-generation Kish count of the optical bands was 0.2-5% of 200 times the
per-generation count).

## Per-group window (`transport.split-per-group`)

For the multigroup pass only:

    W_g(n+1) = W(n+1) x split-group-scale x max(split-group-floor, f_g(n))

- `f_g(n)` is the share of group g in the raw outward-crossing energy of
  completed generation n (observer energies summed over observers, the
  all-reduced snapshot of `CollectGlobalObserverGroupQuality`). Probe, ramp and
  final generations count; burn-in generations do not. Until a spectrum exists
  the scalar window is used. A generation with no escaping energy keeps the
  previous spectrum; a negative or non-finite entry stops the run.
- A packet in lab-frame group g that starts an IMC step in cell c splits when
  `|w| > W_g / I_eff(c)`, `I_eff = max(I_c, split-importance-floor)` (the floor
  only for owned cells with an importance entry, and only once the full-emission
  importance statistics have started). The group is the packet's current lab
  frequency in the observer's convention (`[b_g, b_g+1)`, clamped).
- `W_g <= W` for every group: the threshold is never looser than the scalar
  window's. The copy budget and the copy cap are shared, so when they bind a
  packet can still receive fewer copies than its threshold asks for.
- Unchanged: equal copies with fresh RNG streams, `initialWeight` and
  `splitFactor` scaled, Stokes parameters copied, `k w^2` in every adaptive and
  observer second moment, DDMC residents and packets with a pending DDMC flux
  never split, no Russian roulette. The window is fixed within a generation and
  depends only on completed generations, so splitting stays unbiased.
- The single per-rank copy budget (`split-copy-budget`) is kept; a run that
  reduces or refuses any split is not an accepted setting (`WEIGHT_SPLIT
  ranks_with_refusals`).
- In the enabled mode `splitFactor x copies` must stay below 2^24 (exact float
  integers); otherwise the run stops with the packet's diagnostics.
- The split itself is `STORM::postprocess_split::apply`
  (`radiation/transport/PostProcessWeightSplit.hpp`), called by the transport
  before every step and exercised directly by the unit test.
- Before the multigroup pass the observer groups must be the transport's energy
  groups (same count and boundaries); otherwise the run stops.
- The grey pass keeps the scalar window. It inherits the multigroup decomposition
  and the multigroup pass's last escaping luminosity (first exploration cap and
  first window), so its results are not bit-identical between runs that differ
  only in the multigroup window.

Defaults: `split-per-group false`, `split-group-scale 1` (in (0, 1]),
`split-group-floor 1e-3` (in (0, 1]), `split-importance-floor 0` (in [0, 1];
requires `split-per-group`). All four, and `transport.seed-offset`, are restart
must-match options.

Reports (rank 0, every multigroup generation): `WEIGHT_WINDOW_GROUP` (spectrum
generation and validity, shares, active and candidate windows) and
`WEIGHT_SPLIT_GROUP` (events, copies, capped, refused and overweight steps in
cells without importance, per group). `WEIGHT_SPLIT` gains the per-rank maxima
of copies and refusals and the number of ranks with refusals.
`HEAVY_PACKETS_BY_GROUP` counts overweight escapes against `W_g`.

## Packet provenance (`diagnostics.packet-provenance`)

Diagnostic only: it never changes a transport decision and draws no random
number (checked by the weight-split regression test, run `Bd`). Each particle
carries three bytes in the padding after `on_track` (`sizeof` unchanged):

- `birthGroup`: lab group at the completed birth (255 = not a post-process
  source birth);
- `provenance0`: birth kind (volume/face x physical/learned group pdf),
  exploration-fallback cell, ever DDMC resident, ever carried a pending flux,
  descended from a budget-reduced/refused split, and for each of two candidate
  windows (`split-group-scale` 1 and 0.2) "heavier than the importance-one
  window in a zero-importance cell";
- `provenance1`: for each candidate, "had an overweight split opportunity" (IMC
  state, `I_eff > 0`, `|w| > W_c,g / I_eff`) and "a committed stochastic event
  followed it" (scattering, effective re-emission, source-face re-emission,
  random-walk step, DDMC step, every DDMC direction draw including conversions
  and fallbacks, and the reflection draw at a DDMC interface); and the
  channels that changed the lab group: moving-frame elastic scattering,
  effective thermal scattering, source-face re-emission; plus "went through DDMC
  frequency redistribution or compression" (history only; a resident has no lab
  group).

DDMC residence and pending flux are recorded where they are established (not
only at the next step), and the DDMC channel bit at every DDMC frequency
resampling (compression shift, band-top conversion, leakage and upscatter
resampling, face thermalization). The candidate windows are separate shadow arrays; the provenance hooks sit on
the host transport path and the run stops if a shared transport kernel would
skip them. Every outward crossing is tallied by escape group and
`x = w / e_g` (`e_g` the previous generation's escaping energy of the group per
observer), with count, sum w, raw sum w^2 and sum k w^2, crossed with the
mutually exclusive tail class per candidate (useful = opportunity followed by
stochastic transport, ballistic, zero importance, DDMC, none), the birth group,
the channel mask, the birth kind and the refused-ancestor flag
(`PacketProvenanceTally`).

The x bins are closed on the right, so the tail `x > 0.03` excludes x = 0.03.
Completed births are also histogrammed by weight (decades from 1e30 erg) with
their maximum, per birth class (kind + exploration cell): `BIRTH_WEIGHT_MAX`
per generation (with `diagnostics.band-reports`), `BIRTH_WEIGHT_FINAL` and
`/diagnostics/birth_weight` at the end of the pass. Without provenance the
histograms are neither filled nor reduced.

Output: `PROVENANCE_TAIL` per generation (tail `x > 0.03` sum w^2 per group and
the useful share per candidate), `PROVENANCE_GATE` at the end of the pass (per
candidate and group: `F` = useful tail w^2 / tail w^2 and the other class
shares, or `no_observed_tail`), and the tables under
`/passes/forward/diagnostics/packet_provenance` (flat arrays with `_shape`;
segment-local on a restart). The raw-tail shares are diagnostics: identical
ballistic copies lower the raw sum w^2 without lowering the variance of any
tally, so improvements are judged only from generation-to-generation errors.

## Band reports (`diagnostics.band-reports`)

Off by default. Every multigroup generation, rank 0 prints `BAND_BIRTH`
(completed births per birth group and kind), `BAND_ESCAPE` (escaping energy,
crossings and the medians of split-adjusted n_eff per group), `BAND_SCORE`
(each band's share of the combined source scores) and, with provenance,
`BIRTH_WEIGHT_MAX`. They need the global observer/group snapshot every
generation (three all-reduces, nine with polarization). Without them the
snapshot is collected only for the adaptive group quality; the per-group window
then reduces only the escaping energy it needs. Results do not depend on the
option.

## Generation Stokes records (`diagnostics.generation-stokes`)

The raw energy and absolute Stokes Q and U of every (observer, group) bin of
every final generation, both passes, in
`/passes/<pass>/diagnostics/generation_stokes` (`[generation][observer][group]`;
the grey pass stores the observer totals as one group), with the absolute
generation index. About 25 MB for 200 generations, 512 observers and ten groups.
They allow block/bootstrap analysis and influential-generation checks.

## Seed offset (`transport.seed-offset`)

0 keeps every random stream. Any other value R maps each seed `S` (fresh and
restart schedules, both passes, and the burn-in volume-emission cell subsample) to `SplitMix64(S XOR SplitMix64(R XOR D))` with a
fixed domain tag D, giving independent replicates; the unit test checks that no
seed collides within or between offsets 0, 1, 2 of a 1536-rank production
schedule.

## Absolute generation index

Generation records, the producing generation of the spectrum and the restart
chain use an absolute index: the first segment's burn-in, probe and ramp
generations keep their loop index, final generations count from an origin.
Every pass stores the index its next final generation takes
(`learned_state/next_generation_id`); a continuation starts from it whatever its
own ramp settings (files without it use the fresh-run layout).

## Restart (schema 3)

With `split-per-group` on, the multigroup pass saves its weight-split state in
`/passes/forward/learned_state/weight_split`: the escaped and emitted energy of
every cell that has either, by snapshot cell ID (gathered collectively before
rank 0 writes), the accumulated flag, and the lagged spectrum with its validity
and the absolute generation that produced it, and the group boundaries. A
restart restores it by cell ID before the initial restart rebalance and refuses,
on every rank, a file without it, with duplicate or unknown cell IDs, lengths
that do not match, negative or non-finite energies, flags other than 0/1,
accumulated energies under a cleared accumulated flag, an inconsistent spectrum,
or other group boundaries. Schema-2 files are still continued with the per-group
window off and seed offset 0, and refused explicitly otherwise. With the
window off nothing is saved and the scalar window restarts its statistics from
zero as before (a pre-existing behaviour, not changed here). The grey pass never
stores split state.

## Tests

- `imc_postprocess_band_window_unit`: windows, spectrum capture, provenance
  state machine, escape tallies, the splitter itself (thresholds and adjacent
  doubles, group edges, caps, partial and exhausted budgets, conservation of
  weight, initial weight and k w^2, Stokes copies, distinct RNG streams under
  repeated splits, DDMC/pending/zero-importance exclusions, importance floor,
  the 2^24 guard, candidate flags and fast-reject order), a ballistic fixture
  (identical copies cut the raw sum w^2 tenfold while the generation variance
  stays the same, and are classified ballistic), seed offsets (scheduled and
  downstream per-rank seeds) and the unsigned option parser.
- `imc_postprocess_weight_split` runs `Bd`, `E`, `V1B`, `V1E`: diagnostics leave
  the science identical, the per-group window splits more (also in the dimmest
  group) without refusals and agrees with the scalar window (paired
  per-generation differences of 12 predeclared endpoints, Bonferroni |t| < 4),
  and without scattering Q = U = 0 exactly and the box energy ledger closes.
- `imc_postprocess_restart(_mpi)` runs `GA`..`GZ`: round trip, restart chain of
  the multigroup pass (serial), absolute generation index across a continuation
  with other ramp settings, and every refusal listed above.
- `mpi_particle_dtype`: the provenance bytes survive the MPI datatype, the packed
  P2P wire and `AssignCold`.

The grey pass's restart chain is not reproduced when its scalar window is on,
because the scalar window's statistics restart from zero at every restart (the
behaviour before this change, kept on purpose).
