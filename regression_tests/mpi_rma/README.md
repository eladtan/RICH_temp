# STORM / EasyRMA MPI RMA regression

The production changes are EasyRMA's `MPIRemoteMemoryAgent.hpp` and two
explicitly approved lines in STORM's `ReallocationAgent.cpp` destructor.
`RankHandler2` and `MonteCarloManager` were not edited.
This directory builds the existing Crooked Pipe source into the build directory.

For the five-configuration performance comparison (Open MPI OFI/P2P/MPI RMA
and MPICH P2P/MPI RMA), see [the timing report](timing/README.md).

## Bug fixed

`FetchAndAdd(..., false)` gave MPI the address of a stack-local result, then
returned before the operation completed. It could return garbage and let a
later MPI flush overwrite a stack frame that had already been reused.
On Open MPI 4.1.6 with its `pt2pt` RMA component across two nodes, the original
code returned `4246976` instead of `123456`, then segfaulted at address
`0x1e240` (123456). The same reproducer passes with the fix.

The method now calls `MPI_Win_flush_local` before returning when `flush=false`.
This completes access to the local argument/result buffers while retaining the
option to defer remote completion. `flush=true` still uses `MPI_Win_flush`.
Errors from this atomic operation and its completion calls are checked.
Both calls are standard MPI-3; no vendor, operating-system, or CPU-specific
code was introduced. See the [MPI completion rules](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node331.htm).

The historical STORM benchmark crash was **not** reproduced. The current
Crooked Pipe source already passed the preliminary MPI RMA runs before this
fix, and its current queue path does not call `FetchAndAdd`. The isolated
crash above is therefore a confirmed EasyRMA bug, not proof of the cause of
the earlier STORM failure.

## Reproduce

Use matching MPI compiler wrappers and launcher from the same installation.
Run on allocated compute resources, as appropriate for the machine:

```bash
cmake -S regression_tests/mpi_rma -B build/mpi_rma_validation/local \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER="$(command -v mpicxx)" \
  -DCMAKE_C_COMPILER="$(command -v mpicc)"
cmake --build build/mpi_rma_validation/local -j 2
mpiexec -n 2 build/mpi_rma_validation/local/mpi_rma_agent
mpiexec -n 2 build/mpi_rma_validation/local/reallocation_cleanup
python3 regression_tests/mpi_rma/run_benchmark.py \
  --binary build/mpi_rma_validation/local/crooked_pipe_mpi_rma \
  --launcher 'mpiexec -n 4' \
  --output regression_tests/mpi_rma/results/local
```

`mpi_rma_agent` exercises Put/Get with deferred completion, element offsets,
scatter/batch writes, grow/shrink/zero-sized replacement, borrowed storage,
repeated deferred fetch-and-add, compare-and-swap, and repeated Free. It uses
a communicator with reversed ranks to check communicator-local addressing.
Both agent and shutdown regressions are registered with CTest; use
`MPIEXEC_PREFLAGS` when the launcher needs host or scheduler options.

The benchmark runner uses 3,600 cells (400 background points plus one random
wall-refinement layer), 2 new / 10 minimum packets per cell, and 20 boundary
packets per face. It runs to the actual 1000 ns endpoint with three modes:

- P2P control;
- MPI RMA with dynamic queues, initially 50 packets, exercising collective resizing;
- MPI RMA with fixed queues capped at 50 packets, exercising wrap and backpressure.

The runner requires successful termination, a complete finite temperature
history, energy residual below `1e-8`, and probe differences from P2P below
`5e-4 keV` (0.1% of the drive temperature). PMPI counters independently
verify that the RMA cases actually call MPI Put/Get and create windows.
IBV and OFI backends are disabled in this regression build. This is a
communication regression with reduced mesh/statistics, not a claim of
convergence to the published Crooked Pipe solution.

## Validation, 16 September 2026

Tests used four ranks on two allocated nodes (`d25g133`, `d25g135`), one
thread per rank, with Open MPI 4.1.6 (default RMA and `osc=pt2pt`) and MPICH
4.2.0 (`ch4:ucx`). The agent tests use one rank per node.

After both fixes, all nine full benchmark runs completed **1,039 cycles to 1000 ns**. The
largest relative energy residual was `6.43324e-9` in every run.

| MPI configuration | Dynamic-queue MPI puts | Fixed-queue MPI puts | Largest probe difference from P2P |
| --- | ---: | ---: | ---: |
| Open MPI, default | 47,415 | 45,341 | 0 at saved precision |
| Open MPI, software `pt2pt` RMA | 58,061 | 48,588 | 0 at saved precision |
| MPICH, `ch4:ucx` | 48,491 | 39,259 | `1e-05 keV` |

Dynamic-queue cases created 86 windows in total, versus 48 for fixed queues,
confirming that the run exercised window recreation. Each RMA run transferred
over 33 MB through MPI Put. The P2P controls recorded zero RMA operations.

`validation.json` records the measured results and exact benchmark commands.
Full final logs and probe histories are under the ignored `results/post_cleanup/` directory. Earlier evidence remains under `results/`.

The source snapshot was RICH `65bc7333`, STORM `39eadb4c`, and EasyRMA
`a23bb82c`, including the user's existing STORM working-tree changes.
SHA-256 comparisons against the beginning of the task confirmed that the
only changed tracked STORM file is `manager/parallel/ReallocationAgent.cpp`,
whose diff contains exactly the two approved lines.

## STORM shutdown defect — fixed with explicit permission

MPICH reported two pending communicator references per rank when the RDMA
manager shut down, both before and after the EasyRMA fix. Inspection found
that `ReallocationAgent::~ReallocationAgent` cancels its two posted receives
without completing them. Before the fix, the `reallocation_cleanup` diagnostic
confirmed four unretired receive requests on two ranks after the destructor
returned. After the fix it reports **zero** on Open MPI (default and software
RMA) and MPICH. Both regressions pass on all three configurations, and the
MPICH benchmark no longer reports pending communicator references.

`storm_cleanup.patch` records exactly two approved additions in
`source/monte/manager/parallel/ReallocationAgent.cpp`: `MPI_Wait` immediately
after each existing `MPI_Cancel`. The waits complete cancellation before the
receive buffers are destroyed, as required by the
[MPI cancellation rules](https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node72.htm).
Neither `RankHandler2` nor `MonteCarloManager` was changed.

The benchmark also passed before this patch, but the shutdown request lifetime
was incorrect. EasyRMA cannot complete requests owned privately by STORM.
MPICH also reports two leaked datatype handles per rank in the P2P and RMA
benchmark modes; these do not appear in the standalone EasyRMA agent test
and are outside these two fixes.
