# Crooked Pipe 128-rank RDMA experiment — measured result

Subsequent CPU transport work and a new canonical executable are documented in
[the follow-up report](../rdma128_followup/results.md). The frozen binaries and
timings below describe the earlier RDMA-only experiment.

The requested 15–20% reduction in total runtime was **not achieved**. The final
implementation reduced median total elapsed time from **80.596 s to 78.510 s
(2.59%)** in the measured 120-cycle Crooked Pipe segment. It remains opt-in.

## Final controlled comparison

Slurm job **10176467**, eight exclusive nodes `d25g[133-140]`, 16 MPI ranks per node,
one core per rank, 128 ranks total. Actual provider: OFI `verbs/ibv`, MSG/RC.
Both binaries use the same toolchain, dependency source, physics options, mesh
parameters, photon budget, core binding and output configuration. Baseline source
was frozen from the existing working tree before this task's edits.

Bash's built-in timer measures the complete mpirun lifetime, including setup,
transport, rebalancing, probes, and teardown. Application cycle time excludes
initial setup; summed transport time sums the logged maximum-rank loop duration
for each step. It does not sum unrelated phase maxima.

| Run | Total elapsed (s) | Application cycles (s) | Sum of transport loops (s) |
|---|---:|---:|---:|
| baseline1 | 80.328 | 76.2778 | 64.2089 |
| candidate1 | 78.208 | 74.0844 | 62.8210 |
| candidate2 | 78.812 | 74.6703 | 62.5723 |
| default1 | 79.978 | 75.8621 | 63.9952 |
| baseline2 | 80.864 | 76.7510 | 63.6695 |

Baseline range: 80.328–80.864 s.
Opt-in range: 78.208–78.812 s.
Each group has two final runs, so these are descriptive measurements, not a
high-confidence lower bound on improvement. The default-off run is a single
compatibility measurement. The result covers 120 cycles ending at 81.5719 ns,
not the full 1000 ns physical endpoint.

## What changed

- Fixed receive allocations within an explicitly bracketed transport phase;
  descriptor and producer tail cached per phase, consumer head refreshed on
  credit shortage. Reallocation and reset are rejected during that phase.
- Prefix sends retain and account for unsent particles. Queue growth demand is
  exchanged between completed steps; existing pair-synchronized resizing remains.
- Incoming neighbors are discovered while other sources remain active. A source
  gets a bounded event slice, and unfinished histories retain their state.
- A maximum outgoing batch age is checked during cooperative progress. Final
  opt-in settings use 65,536 events per source slice and 5 ms batch age.
- Original payload completion and flushed tail publication are retained. Source
  registration cannot be released until the synchronous prefix transfer returns.

Default `MonteCarloConfig` leaves fixed-step queues and fair scheduling disabled,
with no age flush. Enable all three in Crooked Pipe using `--optimized-rdma`.
This choice avoids making a delicate protocol replacement the default for a
small measured gain well below the requested target.

## Experiments that changed the conclusion

The initial 8,192-event / 250 us configuration was consistently **10.54% slower**:
median 88.617 s versus 80.168 s over three runs each (job 10176463). Total event
counts were within 1% across those runs; more physics work does not explain the
regression. Relaxing both settings recovered it. Their individual contributions
were not isolated.

A separate ablation allocation (job 10176464) had baseline times 80.307 and
81.117 s (median 80.712 s):

| Variant | Total elapsed (s) | Evidence |
|---|---:|---|
| Fixed queues only; old history scheduling; no age flush | 79.456 | One run |
| Fair 65,536-event slices and 5 ms age, existing queue protocol | 80.757 | One run |
| Both changes, relaxed settings | 77.285, 79.832 | Two runs |

These ablations do not support a 15–20% whole-run forecast for these changes.
The next investigation should measure physics and imbalance on the ranks that
actually determine step completion. Existing `rma` phase maxima include waiting
and cannot establish the removable critical-path fraction.

## Correctness and review

The detailed invariant review is in [review.md](review.md). Executed checks:

- Two-rank MPI RMA and two-node native OFI queue tests: full rings, partial sends,
  wrap, delayed consumer, 12 grow/shrink epochs, source compaction/registration,
  forbidden-operation guards, and return to the legacy path.
- AddressSanitizer/UndefinedBehaviorSanitizer and container assertions on the
  queue test. Leak checking and vptr checking are excluded as documented in the
  review; prebuilt backend objects are not sanitizer-instrumented.
- MPI and native OFI full-manager tiny-ring tests; STORM_DEBUG resumption checks;
  P2P scheduling compatibility.
- Final four-rank native OFI assertion/conservation test, job **10176468**,
  completed 24 cycles with 7-entry initial rings and 4-event slices. Maximum
  absolute normalized energy residual was **6.01e-11**, matching the baseline
  validation's maximum. Exit status 0.
- All final production runs completed 120 cycles, including handler retirement
  and shrinking. Default-off behavior completed successfully as well.

Scheduling changes population ordering and floating-point accumulation. Probe
histories are not bitwise identical, and even baseline repeats vary. The small
number of low-photon benchmark runs does not prove ensemble or full-endpoint
physics equivalence. GPU, CXI and native IBV were not tested for this change.
No claim of universal or perfect correctness is made.

## Reproduce and inspect

Canonical rebuilt executable:
`/home/maorm/RICH/source/monte/examples/crooked_pipe/crooked_pipe`

```bash
mpirun --map-by ppr:16:node:PE=1 --bind-to core -np 128 \
  /home/maorm/RICH/source/monte/examples/crooked_pipe/crooked_pipe \
  20000 2 10 --boundary-photons 100 --manager rdma --rdma-engine ofi \
  --max-steps 120 --optimized-rdma --output-probes probes.txt
```

Use an eight-node allocation and the environment in `final.sbatch`. All logs,
source snapshots, build commands, binaries, and analysis scripts are under
`/home/maorm/RICH/build/rdma128_20260912/`. The exact final batch script and a
patch against the pre-task snapshot are copied beside this report.

Final executable SHA-256:
`6bbb61af01d585d6d1213c58ec9aaee40802c06fb42c05e39dcbc719d4a9c58e`

`final_source_sha256.json` and `external_sources_sha256.json` identify the
reviewed inputs. Runtime source and external dependency source hashes remained
unchanged through the final comparison. Existing unrelated workspace changes
were preserved.
