# STORM bug audit witnesses — 8 September 2026

These programs expose existing behaviors. They are not production fixes or regression tests that should all pass after correction. Run from the RICH repository root. The original execution used GCC/Open MPI on one machine; native InfiniBand and GPU hardware were not exercised.

Compile to a new temporary directory:

```bash
audit_build=$(mktemp -d /tmp/storm-bug-witnesses.XXXXXX)
audit_src=docs/STORM_bug_audit_2026-09-08_repro
g++ -std=c++17 -Isource/monte "$audit_src/tracker.cpp" -o "$audit_build/tracker"
g++ -std=c++17 -Isource/monte "$audit_src/nan_fleck.cpp" -o "$audit_build/nan_fleck"
mpicxx -std=c++17 -DSTORM_WITH_MPI -DFORCE_SYNCHRONOUS -Isource/monte -Isource/utils "$audit_src/counters.cpp" -o "$audit_build/counters"
mpicxx -std=c++17 -DSTORM_WITH_MPI -Isource/monte -Isource/utils "$audit_src/counters_eager.cpp" -o "$audit_build/counters_eager"
"$audit_build/tracker"
"$audit_build/nan_fleck"
timeout 8s mpirun --mca pml ob1 --mca btl self,vader,tcp --mca osc pt2pt --oversubscribe -np 2 "$audit_build/counters"
echo "forced-send witness exit: $?"
timeout 8s mpirun --mca pml ob1 --mca btl self,vader,tcp --mca osc pt2pt --oversubscribe -np 2 "$audit_build/counters" drain
timeout 8s mpirun --mca pml ob1 --mca btl self,vader,tcp --mca osc pt2pt --oversubscribe -np 2 "$audit_build/counters_eager"
```

The MPI MCA options are Open MPI-specific and select locally available transports. Adapt them for another MPI implementation. The first default-transport attempt emitted unavailable-InfiniBand warnings; supplied logs are from the explicit local-transport rerun.

Observed:

- tracker: `route size=2 origins=0,1`. Its zero exit status means the collision was reproduced, not that tracking is correct.
- nan_fleck: `error=0 weight=nan deposited=-nan integrated=1.8 events=1`.
- counters with FORCE_SYNCHRONOUS: timeout exit 124 after rank 0 finishes and rank 1 enters FinishCounters. The macro changes MPI_Isend to MPI_Issend only in this test translation unit; no production source was changed. This forces the need for a matching receive.
- counters with the drain argument: exit 0; both ranks finish because rank 0 supplies the missing receive. This control demonstrates the protocol dependency, not a complete proposed repair.
- counters_eager: exit 0 and `unreceived snapshot remains after FinishCounters`. This build uses the real MPI_Isend. Its blocking probe is deliberately a witness of the current queued message; after a fix, replace that probe with a bounded assertion that no stale message remains.

The NaN fixture is derived from `source/monte/tests/slab_transport_test.cpp`, with the Fleck factor replaced by quiet_NaN and a diagnostic main. It exercises the real shared transport header with valid geometry and otherwise finite opacity inputs.

To turn these witnesses into regressions, assert rejection and unchanged tallies for invalid Fleck factors, distinct identity-based routes for tracking, and successful diagnostic close with no queued prior-step messages for both ordinary and forced-synchronous sends. See the PDF for the full acceptance matrix.
