# MadVoro correctness audit: integration and load-balancing contracts

Audit date: 8 September 2026. Root: `/home/maorm/RICH`. This subtask inspected current files and created only audit evidence. No production source was modified. The six findings below were reproduced with bounded one/two-rank programs. The source paths/line numbers refer to the current working tree; the parent report supplies repository/submodule fingerprints. No P0 issue is claimed.

Severity interpretation here: P1 is a valid configuration/input causing memory-safety failure, distributed hang, or silent association of physical data with the wrong cell; P2 is a supported API/configuration transition that reliably fails without such demonstrated corruption. Distinguish MadVoro-owned code, MeshDecomposer3D dependency code, and RICH integration code when assigning fixes.

## Findings at a glance

| ID | Scope | Severity | Concrete failure | Evidence level |
| --- | --- | --- | --- | --- |
| INT-01 | MadVoro core preparation | P1 | Mixed per-rank suppression flags substitute a too-short cached weight vector despite valid caller weights | Reproduced assertion / exit 134; uniform-suppression control succeeds |
| INT-02 | MeshDecomposer3D, exposed by MadVoro SetKernel | P2 | Changing an initialized Hilbert indexing kernel makes the next update reject its null converter | Reproduced caught exception / exit 3 |
| INT-03 | MeshDecomposer3D communicator contract | P1 | A manager on MPI_COMM_SELF uses a WORLD-sized Hilbert owner rank to index a SELF-sized transport array | Reproduced out-of-bounds assertion / exit 134 |
| INT-04 | MeshDecomposer3D OneDimensionalLoadBalancer | P2 | A successful single-rank rebalance creates zero bins, but intersection query requires one bin | Reproduced caught exception / exit 3 |
| INT-05 | RICH ExchangeChain integration | P1 | Field transfer ignores destination indices, so a valid self permutation associates fields with the wrong cell | Reproduced expected {101,100}, actual {100,101} / exit 3 |
| INT-06 | RICH ExchangeChain integration | P1 | An originally empty rank skips the collective even when receiving a cell through the chain | Reproduced rank-local early return and peer hang / timeout 124 |

INT-05/06 belong in a clearly marked RICH integration appendix or integration section, not in a list claiming all defects reside in the MadVoro submodule. INT-02/03/04 require MeshDecomposer3D changes, followed by a submodule-pointer update in the consuming repository.

## INT-01: globally suppressed exchange selects stale weights on nonsuppressing ranks

**Location and source chain**

- `source/3D/tessellation/voronoi/Voronoi3D.hpp:2019-2021`: combines every rank's exchange permission with MPI_LAND, producing a global `allowExchange`.
- Same file `2023-2024`: resizes `allPointsWeights` only if this rank's original `suppressExchange` argument is true.
- Same file `2030`: when global `allowExchange` is false, passes `this->allPointsWeights` to PointsManager on every rank, even ranks whose local flag was false.
- `source/3D/tessellation/MeshDecomposer3D/points_manager/PointsManager.hpp:290-296`: selected original indices index `allWeights[pointIdx]` without a bounds check.

**Trigger.** Start with an initialized MPI mesh. Increase local input length on a rank whose local suppressExchange=false, while another rank requests true. Pass a correctly sized weights vector on every rank. Global suppression takes effect but the nonsuppressing rank retains the previous shorter cached weights.

**Actual reproducer.** `docs/madvoro_bugs/evidence/integration_mixed_suppression.cpp` builds 128 deterministic random points with two ranks, yielding 64 points each. Each rank then adds one point and supplies 65 valid weights. Rank 0 requests suppressExchange=true and rank 1 requests false. Rebalance is explicitly suppressed on both ranks to isolate the exchange-flag path. The second call fails on rank 1:

```text
rank=0 count_before=64 input_count=65 weight_count=65 suppressExchange=1 entering_second_build
rank=1 count_before=64 input_count=65 weight_count=65 suppressExchange=0 entering_second_build
std::vector<double>::operator[] ... Assertion '__n < this->size()' failed.
```

Raw log: `evidence/integration_mixed_suppression.log`. Exit: 134. `_GLIBCXX_ASSERTIONS` catches the invalid access; unchecked production builds have undefined behavior and can silently import an invalid point weight. The optimized debug stack symbolizer points around the adjacent payload access at PM:296, but the assertion message explicitly names vector<double>, and the source-selected stale vector explains that exact access. Symbolized frames are in `integration_mixed_suppression_frames.txt`.

The exact same executable with argument `all_suppress` requests suppression on both ranks and succeeds with 65 returned points each: `integration_all_suppression_control.log`, exit 0. This excludes invalid point input, insufficient caller weights, and lack of ordinary suppressed-exchange support as the explanation.

**Fix contract.** Base all state preparation decisions on the globally agreed `allowExchange`, not the local argument. If retaining cached weights is intentional when exchange is suppressed, initialize/resize/remap them on **every** globally suppressed rank before use. At minimum the existing resize condition must follow `!allowExchange`; more broadly define how caller-supplied weights relate to cached weights after insertion/removal/permutation. Do not silently reinterpret those semantics as part of the safety patch. Collectively validate the actual selected weight vector and selected indices before starting communication; one rank throwing before other ranks enter collectives requires a consistent error policy.

**Regression tests.** Mixed local flags with input length growth on the nonsuppressing rank; shrink; unchanged length; full permutation after preparing identity metadata; all flags true/false; empty rank; different ranks choosing the suppressor; caller weights valid but distinctive. Require no out-of-bounds access under libstdc++ assertions/ASan, globally matching control flow, correct stored weight values under the documented policy, and unchanged point/payload mappings. Do not fix only the helper's subscript with `.at()`; that merely converts corruption to rank-local exception/deadlock.

## INT-02: changing an existing Hilbert kernel leaves a converter that cannot reinitialize

**Location and source chain**

- `source/3D/tessellation/voronoi/Voronoi3D.hpp:5510-5520`: public `SetKernel` delegates to `HilbertPointsManager::setIndexing`.
- `source/3D/tessellation/MeshDecomposer3D/points_manager/HilbertPointsManager.hpp:260-270`: an existing Hilbert balancer receives setIndexing, and the environment agent is cleared.
- `source/3D/tessellation/MeshDecomposer3D/load_balancing/HilbertLoadBalancer.hpp:290-294`: stores new indexing and sets `convertor=nullptr`.
- `HilbertPointsManager.hpp:283-291`: initialization constructs a new balancer only if the balancer pointer is null. Here the pointer remains nonnull, so initialization immediately calls rebalance on the converter-less balancer.
- `HilbertLoadBalancer.hpp:160-163`: rebalance throws if converter is null.

**Trigger.** Successfully initialize the normal Hilbert manager, then set a valid kernel and update again. Even replacing the kernel with a fresh Identity object fails; this is not a malformed transform.

**Actual reproducer.** `evidence/integration_load_balancer_lifecycle.cpp`, mode `kernel`, one MPI rank:

```text
first_update_succeeded changing_to_identity_kernel
caught_exception=HilbertLoadBalancer::rebalance: convertor was not initialized yet
```

Log: `integration_kernel_change.log`. Exit: 3. The reproducer calls the dependency methods directly to isolate the lifecycle, not a complete RICH simulation. The MadVoro SetKernel delegation is source-verified; no separate core SetKernel runtime was needed for this claim. Setting the kernel before initial construction follows the pendingIndexing path and is a distinct case that may work.

**Impact.** Valid dynamic kernel replacement cannot be followed by a normal rebuild. An implementation might work around it by re-creating the entire grid, but the public setter is currently an invalidating operation without a viable reconstruction path. This is P2; no physical corruption was demonstrated for this case because the next rebalance throws explicitly.

**Fix contract.** Preserve the new kernel as pending state and rebuild the converter from current bounds/points before using it. Either recreate the Hilbert balancer on the next initialization, or explicitly reconstruct its converter and recompute cuts. Old cuts in an old key space must not be reused blindly. Preserve the communicator and documented imbalance settings when recreating state; invalidate/rebuild the environment agent and routing caches consistently. Handle identical kernel objects as valid too. Do not simply remove the null-converter check.

**Regression tests.** Set Identity before initial build; replace with Identity after first build; replace with a valid nonidentity transform; repeat twice; clone after replacement; SetBox plus kernel transition; multiple ranks with synchronized setter/update; confirm final owner assignment and geometry against a fresh grid configured with the target kernel.

## INT-03: HilbertPointsManager's advertised communicator is not passed to its load balancer

**Location and source chain**

- `source/3D/tessellation/MeshDecomposer3D/points_manager/HilbertPointsManager.hpp:88-90` accepts and forwards a caller MPI communicator.
- `source/3D/tessellation/MeshDecomposer3D/points_manager/PointsManager.hpp:88-93` sets local rank/size from that communicator.
- `HilbertPointsManager.hpp:289` constructs `HilbertLoadBalancer` without this communicator.
- `source/3D/tessellation/MeshDecomposer3D/load_balancing/HilbertLoadBalancer.hpp:47-60` has no communicator parameter in the used constructors and forwards only boundaries.
- `source/3D/tessellation/MeshDecomposer3D/load_balancing/CurveLoadBalancer.hpp:15-16` calls `LoadBalancer<PointT>()`, selecting its default.
- `source/3D/tessellation/MeshDecomposer3D/load_balancing/LoadBalancer.hpp:19-23` defaults to MPI_COMM_WORLD and records world rank/size.
- Owner lookup clamps against the balancer's world size (`CurveLoadBalancer.hpp:38`), while `source/utils/mpi_utils/exchange.hpp:30,39` allocates destination arrays for the manager's communicator and indexes by that owner.

**Trigger.** In a two-rank world, create an independent HilbertPointsManager on MPI_COMM_SELF on each rank, then update valid points. Each manager's communicator has one rank, but the shared-world balancer can return rank 1. If only one subgroup enters the operation, unintended WORLD collectives can hang even before routing; the reproduced case has both ranks entering to reach the invalid route deterministically.

**Actual reproducer.** `evidence/integration_points_manager_comm.cpp`, two ranks. World rank 0 returns its two points; rank 1 aborts in a vector-of-destination-payload-vectors subscript:

```text
world_rank=1 entering MPI_COMM_SELF manager update
world_rank=0 entering MPI_COMM_SELF manager update
Weighted borders determined in 1 pass.
std::vector<std::vector<ExchangePoint<...>>>::operator[] ... Assertion '__n < this->size()' failed.
world_rank=0 result_count=2
```

Log: `integration_points_manager_comm.log`. Exit: 134. This reproduces the dependency's explicit communicator contract. **Do not describe it as proof that arbitrary subcommunicators are supported by the entire current MadVoro engine**: the core has many explicit WORLD callsites and no general communicator constructor. The error is valid in the independently usable dependency and can also affect custom manager injection if a caller reasonably expects its supplied communicator to be honored.

**Fix contract.** Thread the communicator through CurveLoadBalancer and both HilbertLoadBalancer constructors; pass it from HPM initialization; preserve it through HilbertLoadBalancer::clone and any factory/restoration path. Audit every collective and returned owner for the same local rank namespace. Add a cheap assertion/checked error that destination owner lies in `[0,manager_size)` before indexing, but communicator correctness is the real fix. Never fix by clamping world owner indices into subgroup size: that discards decomposition semantics and can hide world-collective hangs.

**Regression tests.** MPI_COMM_SELF on every world rank, disjoint split communicators of different sizes, reordered communicator rank order, a subgroup performing updates while other world ranks do unrelated work, clone inside subgroup, empty local ranks and supplied/custom load balancers. Check that no WORLD collective is invoked by the dependency in subgroup tests and that all returned ownership values belong to the subgroup.

## INT-04: OneDimensionalLoadBalancer's single-rank rebalance violates its own bin-count invariant

**Location and source chain**

- `source/3D/tessellation/MeshDecomposer3D/load_balancing/OneDimensionalLoadBalancer.hpp:58-67`: sets `bins_` to the result of `getWeightedBorders3<double>`.
- `source/3D/tessellation/MeshDecomposer3D/balance/weightedBalance3.hpp:218-220`: direct route returns an empty list for communicator size <=1. Root fallback has the same return at 121-123.
- `OneDimensionalLoadBalancer.hpp:87-94`: `getIntersectingRanks` throws unless bins_.size()==MPI size.
- `getOwner` at 72-75 already accepts empty bins and returns owner 0, so the two query APIs disagree on the state produced by rebalance.

**Trigger and reproducer.** On one rank, construct a valid unit-box OneDimensionalLoadBalancer, call rebalance on two ordinary points with unit weights, then query the box's center with radius 0.1. `integration_load_balancer_lifecycle.cpp onedim` produces:

```text
bins_after_rebalance=0
caught_exception=OneDimensionalLoadBalancer: bin count does not match MPI size
  Bin count: 0
  MPI size: 1
```

Log: `integration_onedim_single_rank.log`. Exit: 3. This is a dependency API failure after a successful normal setup; a full MadVoro geometry build was not needed. The generic environment adapter forwards intersection queries to this API when providesIntersectingRanks() is true (`environment/PlainDistributedOctEnvAgent.hpp:86-90`). The actual single-rank core search may bypass remote routing, so do not claim every MPI1 MadVoro build crashes; the intersection API does.

**Fix contract.** Define the bin/sentinel convention for one rank and keep constructors, rebalance, getOwner, getIntersectingRanks and restart IO consistent. Options: store one valid upper-domain sentinel during single-rank rebalance, or explicitly handle size==1 in the intersection query using the real domain box. Preserve conservative box/sphere intersection and empty-domain semantics. Do not invent P-1 or P requirements inconsistently: the existing multi-rank helper appends an extra sentinel.

**Regression tests.** One rank, each axis, inner/intersecting/outside spheres, explicit and generated bins, clone and IO roundtrip; two/three ranks with repeated cuts; all-empty or all-zero-weight policy as separately specified. Query owners and candidate-rank sets should remain consistent after rebalance and changeBox.

## INT-05: RICH ExchangeChain field transfer discards target indices

**Scope.** RICH integration code outside MadVoro. This can corrupt application buffers that are transferred through a chain even if MadVoro geometry and each ownership mapping are correct.

**Location and source chain**

- `source/mpi/ExchangeChain.hpp:34-35`: explicitly documents the mapping as original index -> (current rank,current index).
- `source/mpi/ExchangeChain.cpp:52-55` correctly assigns a retained original point the current position given by localIndices order; chain composition retains target indices.
- `source/mpi/ExchangeChain.hpp:56-60` groups values by target.first but never sends target.second.
- Same file `64-70`: reconstructs output by self-origin values then other source ranks; this is not the chain's specified final index order.
- `source/newtonian/three_dimensional/simulation/Simulation.hpp:174` registers this helper as `MigrationBuffer::transferChain`.
- `source/newtonian/three_dimensional/simulation/Simulation.cpp:298` applies physical steps' chains to registered buffers.

**Trigger.** Any valid chain in which final local order differs from the accidental self-first/original-index/source-rank concatenation. A self-only permutation is the smallest example. Multi-stage migrations can also reorder origins within a final rank; original indices alone cannot reconstruct that order.

**Actual reproducer.** `evidence/integration_exchange_chain.cpp`, mode `permutation`, one rank:

```text
rank=0 targets=1,0
rank=0 expected=101,100 actual=100,101 pass=0
```

It calls Reset(2), Exchange with retained local indices `{1,0}`, then transfers `{100,101}`. No geometry, remote messaging or floating point is involved. Exit: 3. Raw log `integration_exchange_chain_permutation.log`; clean-transport rerun `integration_exchange_chain_permutation_clean.log`. This proves silent field misassociation, not just a bookkeeping disagreement.

**Fix contract.** Send/receive enough metadata to scatter each value into the chain's declared target index, e.g. `{newIndex,value}` records with explicit serialization, or a communication plan derived from the reverse map. Allocate output to the final local map size; enforce one assignment per final slot; reject duplicate/missing/out-of-range target slots. Preserve variable-size serializable fields and all origin/rank namespaces. Do not sort values by source rank as a substitute for the declared target index. Old input storage must remain valid until all sends complete.

**Regression tests.** Single-rank swaps and random permutations; multiple ranks and nonmonotone receive-peer orders; two/three stage chains; reverse chain; distinct heterogeneous per-ID sentinel fields; registered RICH buffers after moving-mesh and remesh stages. Compare every final slot against `GetReversedTranslationMap()` and ensure preservation of all point-associated state. The ordinary tessellation-based MPI_exchange_data helper already compacts retained data in supplied exact order; do not regress that correct behavior while fixing the chain helper.

## INT-06: original local point count is not permission to skip chain transfer

**Scope.** RICH integration code outside MadVoro.

**Location and source chain**

- `source/mpi/ExchangeChain.hpp:23` defines GetNorg() as the size of the original->final map. It is an original local population count, not a global indication that the step did not migrate.
- `source/mpi/ExchangeChain.hpp:49-52` returns locally when that count is zero, before the required all-to-all at 62.
- `source/newtonian/three_dimensional/simulation/Simulation.cpp:133-134` duplicates the same local skip before calling any registered transferChain function. Fixing only the lower helper leaves the upper deadlock path.
- `ExchangeChain.cpp:11-24` correctly supports Reset(0); subsequent Exchange can populate globalTransferOrigins with received points while globalTransfer remains empty.

**Trigger.** An originally empty rank participates in a mesh migration and receives points. Its original map remains empty, but it must receive their physical fields. A nonempty peer enters the collective field transfer while it skips directly out.

**Actual reproducer.** `evidence/integration_exchange_chain.cpp empty`, exactly two ranks. Rank 0 initially owns a single value 1234 and the chain sends its point to rank 1, which initially owns nothing. Chain construction itself completes. Immediately before field transfer:

```text
rank=1 original_count=0 final_count=1 entering_field_transfer
rank=0 original_count=1 final_count=0 entering_field_transfer
rank=1 returned_field_transfer size=0
```

Rank 0 never prints returned_field_transfer. The bounded MPI job exits via timeout code 124 after 8 seconds. Log: `integration_exchange_chain_empty.log`. The logged completed chain and rank 1's early-return marker isolate the mismatch from earlier MPI setup. A clean transport rerun is also retained if present.

**Fix contract.** Distinguish an absent/uninitialized chain from a valid initialized chain whose local original population happens to be zero. Add explicit chain state or equivalent globally agreed operation state. All ranks enter a valid transfer even when local send count is zero; the reverse map determines how much arrives. Preserve a true no-movement/default-chain operation as a no-op on every rank. **Do not simply remove both early returns and clear every data vector unconditionally:** `RadiationMCStep.cpp:325-328` returns a default empty ExchangeChain to mean no point movement; a default inactive chain must not erase valid physical buffers. Make Reset(0) a valid initialized empty-source chain, and carry the state through copies/Reverse. Audit the helper's hardcoded WORLD communicator separately from operation participation.

**Regression tests.** Initially empty rank receiving first cell; initially nonempty rank becoming empty; several empty ranks; valid globally empty transfer; default inactive chain with nonempty data preserved; multiple registered buffer types; exchange chains across repeated builds; two communicators. Use a watchdog and explicit completion markers rather than treating timeout alone as evidence. Verify final sentinel 1234 reaches rank 1 and rank 0 returns normally with zero entries.

## Additional inspected facts and handoffs (not duplicate new findings)

- MadVoro's copy constructor at V:4984-4999 and HilbertPointsManager::clone at HPM:120 have separate lifecycle defects. Never-built HPM clone dereferences a null loadBalancer; populated copies omit some trees/state; base balance statistics/tolerance are not copied. The geometry agent owns reproduction and report wording for copies to avoid duplicating findings here.
- `ResolvePeriodicImageIndex` at V:3028-3061 accepts an unverified closest **owned** point for a periodic image whose physical source may be remote. However MockMesh's `resolveOldPreImage` at V:2539-2556 already checks coordinate distance, and actual RICH/MC users guard recognized MPI ghosts before calling local resolution (`LinearGauss3D.cpp:49-51`, `MonteCarloTransport.hpp:258-261`). The parent owns a direct API reproducer. Do not claim the current MockMesh definitely misroutes because of this helper without bypassing its added guard.
- `SetBox` at V:5531-5539 replaces PointsManager, which drops manager-specific settings/custom decomposition unless restored. This is a source-level state-policy concern; no separate behavior failure was reproduced here. Treat as follow-up, not another confirmed physical error.
- Many naked getters require a successfully built grid. Null access on a never-built grid is not automatically a defect unless the API promises otherwise. Copying a valid constructed object and setting a valid kernel are stronger lifecycle contracts than an arbitrary geometric query before build.
- The empty-weight and invalid-weight behavior of weightedBalance3 needs a documented contract and collectively safe validation. This subtask did not claim negative/NaN weights as a valid input or report unbounded root-memory/performance limits as independently reproduced correctness failures.

## Evidence build/run commands

All commands execute from `/home/maorm/RICH`; binaries go to `/tmp`, sources/logs stay under `docs/madvoro_bugs/evidence`. GCC 15.1.0/Open MPI 4.1.6 were in the inherited environment. First chain runs inherited UCX settings advertising unavailable `ib`; MPI fell back and reached the explicitly logged test stages. Clean reruns use `UCX_TLS=tcp,self,sm`. These are correctness tests, not timed performance comparisons.

```bash
mpicxx -std=c++17 -O1 -g1 -DRICH_MPI -DMADVORO_WITH_MPI \
  -DSPATIAL_DS_WITH_MPI -D__WITH_MPI -I. -Isource -Isource/utils \
  -Isource/3D/tessellation \
  docs/madvoro_bugs/evidence/integration_exchange_chain.cpp \
  source/mpi/ExchangeChain.cpp -o /tmp/integration_exchange_chain

mpicxx -std=c++17 -O1 -g1 -D_GLIBCXX_ASSERTIONS -fopenmp \
  -DMADVORO_WITH_MPI -DSPATIAL_DS_WITH_MPI -DRICH_MPI -D__WITH_MPI \
  -Isource/3D/tessellation/voronoi -Isource/3D/tessellation \
  -Isource/utils -I/software/x86_64/5.14.0/boost/1.78.0/include \
  docs/madvoro_bugs/evidence/integration_mixed_suppression.cpp \
  source/3D/tessellation/voronoi/exception/MadVoroException.cpp \
  source/3D/tessellation/voronoi/exception/InvalidArgumentException.cpp \
  source/3D/tessellation/voronoi/exception/SizeException.cpp \
  source/utils/mpi_utils/AmountManager.cpp -o /tmp/integration_mixed_suppression

mpicxx -std=c++17 -O1 -g1 -D_GLIBCXX_ASSERTIONS -fopenmp \
  -DMADVORO_WITH_MPI -DSPATIAL_DS_WITH_MPI -DRICH_MPI -D__WITH_MPI \
  -Isource/3D/tessellation/voronoi -Isource/3D/tessellation \
  -Isource/utils -I/software/x86_64/5.14.0/boost/1.78.0/include \
  docs/madvoro_bugs/evidence/integration_points_manager_comm.cpp \
  -o /tmp/integration_points_manager_comm

mpicxx -std=c++17 -O1 -g1 -D_GLIBCXX_ASSERTIONS -fopenmp \
  -DMADVORO_WITH_MPI -DSPATIAL_DS_WITH_MPI -DRICH_MPI -D__WITH_MPI \
  -Isource/3D/tessellation/voronoi -Isource/3D/tessellation \
  -Isource/utils -I/software/x86_64/5.14.0/boost/1.78.0/include \
  docs/madvoro_bugs/evidence/integration_load_balancer_lifecycle.cpp \
  -o /tmp/integration_load_balancer_lifecycle

# Run these sequentially; expected current-source failure codes are noted above.
UCX_TLS=tcp,self,sm timeout 5s mpirun --bind-to none -np 1 /tmp/integration_exchange_chain permutation
UCX_TLS=tcp,self,sm timeout 8s mpirun --bind-to none -np 2 /tmp/integration_exchange_chain empty
UCX_TLS=tcp,self,sm OMP_NUM_THREADS=1 timeout 12s mpirun --bind-to none -np 2 /tmp/integration_mixed_suppression mixed
UCX_TLS=tcp,self,sm OMP_NUM_THREADS=1 timeout 12s mpirun --bind-to none -np 2 /tmp/integration_mixed_suppression all_suppress
UCX_TLS=tcp,self,sm timeout 8s mpirun --bind-to none -np 2 /tmp/integration_points_manager_comm
UCX_TLS=tcp,self,sm timeout 8s mpirun --bind-to none -np 1 /tmp/integration_load_balancer_lifecycle kernel
UCX_TLS=tcp,self,sm timeout 8s mpirun --bind-to none -np 1 /tmp/integration_load_balancer_lifecycle onedim
```

The compiler logs are retained even where empty (successful compilation). The fix acceptance suite should reuse these as regression tests with expected exit code 0 and corrected results, then add the stated edge cases. No claimed finding depends on a costly production simulation.
