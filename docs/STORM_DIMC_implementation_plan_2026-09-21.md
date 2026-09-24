# STORM: DIMC implementation plan



---



01  /  DIMC in STORM

A separate physics choice, built from the same IMC engine.

**Implementation plan • 21 September 2026 • CPU transport**

> <b>Recommendation:</b> introduce <b>RadiationDIMC</b> through a material policy in the existing radiation implementation. Keep the transport equations and event tracking shared. Add a small, persistent bank of material-energy markers to each owned cell.

What DIMC changes

DIMC uses the IMC linearization, but represents material energy with position–energy particles. Thermal emission starts at their locations; continuous absorption creates new material particles. This preserves information inside a cell that ordinary cell-wide emission loses. The frequency-dependent extension keeps the same idea. [2, §II]

In plain words: remember **where the heat is**, so that the next photon starts there. This targets the spatial “teleportation” error. It does not remove statistical noise or the need to check time-step, mesh and material-particle convergence.

[Architecture: two public physics types select two material policies in one shared radiation implementation.]

Solutions to the two challenges

- <b>Code reuse:</b> two distinct C++ physics types, one implementation. The material policy chooses thermal birth sites and records the location of absorbed energy. IMC keeps its current behavior through the default policy.

- <b>Speed:</b> store compact arrays, collect deposits cheaply, then reduce each cell to a small marker population. The population reduction resembles clustering; DIMC itself is a radiation-transport method.

Start with a stationary, gray, absorption/scattering problem. Add MPI, frequency groups and RICH coupling through explicit validation stages. GPU work is outside this plan. All names proposed below are design names, not existing APIs.



---



02  /  Share the implementation

A compile-time policy makes DIMC a distinct physics type without copying IMC.

Generalize the current **RadiationIMC** owner into a shared **RadiationTransport&lt;…, MaterialModel&gt;**. Preserve the current IMC template arguments and constructor through an alias; add the DIMC alias with a different material model. These aliases denote different concrete types accepted by the existing manager.

```text
RadiationIMC<...>  = RadiationTransport<..., CellMaterial>
RadiationDIMC<...> = RadiationTransport<..., DiscreteMaterial>
```

The current class is **final**; subclassing it is not the intended extension point. Its lifecycle, source and transport components are already templated on their owner. Keep those components and make the owner generic. Audit forward declarations, traits and out-of-class definitions during the mechanical refactor. [C1]

| Shared code stays responsible for | Material policy supplies |
| --- | --- |

| Opacity tables, Fleck factor, EOS and source budgets | Persistent marker state and initialization |

| Photon direction, birth time and frequency sampling | Thermal birth location and matching marker debit |

| Geometry, event selection and packet attenuation | Deposit position and per-history accumulation |

| Cell energy application, census and diagnostics | End-step merging and consistency checks |

| Existing communication engines and photon routing | Marker migration/checkpoint payloads |

Use small hooks at the actual execution points

Proposed hooks: **beginStep**, **emitThermal**, **recordSegment**, **recordPointDeposit**, **finishCellVisit** and **endStep**. The source hook returns donor location and actual emitted energy; the common packet builder supplies the other properties. The transport hook receives the energy already calculated by the common kernel.

Give ordinary IMC empty deposit/history hooks. Template dispatch removes the unused work and storage. Keep method selection out of the inner event loop; select the concrete physics once when creating the manager.

> <b>Two necessary seams:</b> the fast thermal emitter calls its own geometry sampler, bypassing PositionSamplerT. The shared AdvanceIMC kernel updates only scalar material tallies. Both need a policy hook; changing only the public position sampler is insufficient. [C2–C4]



---



03  /  One time step and one ledger

Material markers describe the same energy as the cell; they are not an extra energy reservoir.

```text
At step boundaries:  sum(marker energies in cell i) = U_i
During transport:    marker sum + unflushed deposits
                    = U_i after emission + pending absorption
```

Here **U_i** is the cell’s extensive internal energy. The EOS still uses the cell state; individual markers do not receive independent temperatures. Include buffered records and history accumulators when checking the running ledger.

| Order | Action |
| --- | --- |

| 1. Initialize / resume | Seed markers only on a new calculation, using the existing cell-volume sampler and U_i/K. On restart, restore them. Never reseed a cell each step. |

| 2. Prepare | Check marker ownership and energy. Compute the usual IMC opacities, Fleck factor and thermal source budget from the start-of-step cell state. |

| 3. Emit | Choose a material donor, place the thermal photon at its position and debit that donor. Debit the cell once through the shared source accounting. Boundary photons keep their boundary source. |

| 4. Transport | Run the shared IMC kernel. Record spatial absorption in the material policy. Effective scattering stays a radiation event; it is not an additional material debit. |

| 5. Finish | Flush deposits, apply existing cell tallies once, reduce the marker count, verify the ledger and run the existing EOS update. These markers become next step’s source support. |

Keep source sampling positive and inexpensive

For the initial implementation, select donors from their **current positive energies**. A short cumulative scan is adequate for tens of markers and automatically reflects each withdrawal. A fixed alias table would become stale as donors are depleted. The paper’s emission changes only the material location/accounting part of the IMC source. [1, Algorithm 1]

**Explicit engineering extension:** if a selected donor cannot fund a nominal photon, split that source slot into smaller packets at the donors’ own positions. Each withdrawal is at most the donor energy; continue until the slot is funded. With K donors, at most K−1 extra fragments are needed beyond the nominal source count. Test the resulting source statistics; do not assume this unspecified edge case is automatically unbiased at finite packet count.

If the whole-cell source budget exceeds U_i, reject that step and reduce/subcycle the time step through the caller. Do not hide an inconsistent budget by clipping energies. New absorption remains unavailable for new thermal emission until the next time step.



---



04  /  Remember absorption locations

Add information to the existing attenuation calculation; do not calculate or deposit the energy twice.

For a straight segment in a stationary cell, let x₀ be its starting point, Ω its direction, ℓ its length, W the initial packet energy and k = f σₐ the effective absorption coefficient. The deposition rule can be evaluated stably as follows. [1, Algorithm 2]

```text
tau = k * length
q   = -expm1(-tau)
dE  = W * q
s   = -log1p(-u * q) / k        # 0 < u < 1
x_deposit = x_start + s * direction
```

The host adapter must capture the segment start before the kernel changes the particle. Reuse its attenuation result. In the kℓ → 0 limit, sample s ≈ uℓ when a nonzero deposit exists; if the deposit is zero, skip the marker work and random draw. Use the paper’s stochastic location for the reference implementation, rather than replacing it with the segment midpoint.

Accumulate one material record per cell visit

Within one photon’s visit to a cell, accumulate deposited energy A and position moment Q = Σ(dE · x_deposit), including successive scattering segments. On exit or census, append one marker (energy A, position Q/A). This follows the paper’s cell-visit construction. [1, Algorithm 2]

Use a small CPU **history context** outside the communicated photon payload. The shared manager creates/resumes it and passes it to a context-aware step overload; ordinary physics uses an empty context. Flush before a cell change, rank transfer, escape, death or census. A reflected photon may keep its context in the same cell.

> <b>Scheduling detail:</b> STORM can suspend a photon when its local event budget runs out. Preserve its context in a sidecar attached to the deferred work item; restore it on resumption. Do not keep unlabelled state in a single physics member, or mix deposits from different photons. [C6]

At a low-weight cutoff, record the residual packet energy as a **point deposit at the terminal location**, then flush. The shared kernel already puts that residual into the cell tally. Include all early returns in the hook audit. In folded slab transport, map the sampled deposit through the same physical coordinate folding; a straight line to the folded endpoint is incorrect. [C4]

Keep DIMC sampling in a separate random-number stream derived from packet identity, step and segment index. Extra marker draws must not shift the existing transport stream.



---



05  /  The merging algorithm

Small, local population reduction is sufficient. No global clustering service is needed.

Use the paper’s end-of-step reducer as the reference: select K distinct survivors with energy-weighted sampling, assign each other marker to its nearest current survivor, and update that survivor’s energy-weighted position. The published examples use roughly 10–30 survivors per cell. [1, Algorithm 3]

[Diagram: multiple material deposits within one cell reduce to a few energy-weighted representatives within that same cell.]

Concrete CPU implementation

- <b>Inputs:</b> residual old markers and all newly deposited records from one owned cell. Remove exact zero-energy entries. If their count M is at most K, keep all of them.

- <b>Choose seeds:</b> use weighted sampling without replacement. One proposed implementation gives each positive-energy marker an independent key −log(u)/E and keeps the K smallest keys in a small max-heap. This costs O(M log K). Fix this sampling convention in the tests.

- <b>Assign:</b> visit each non-seed once and scan the current K survivors using squared distances. No square root or spatial tree is required. Apply each merge immediately; do not freeze the survivor coordinates during the scan.

```text
E_new = E_survivor + E_incoming
x_new = x_survivor + (E_incoming / E_new)
                      * (x_incoming - x_survivor)
```

**Cost:** O(MK) distance comparisons plus seed selection; O(K) temporary survivor storage. With a fixed K such as 20, the work grows approximately linearly with the number of input records. Process different cells independently; keep the within-cell order fixed for reproducibility.

What is preserved, and what needs testing

Each merge preserves total energy and its first spatial moment, up to floating-point error. It does not preserve the full spatial distribution. Never merge across cell or material-interface boundaries. Positive weighted centroids stay inside convex Cartesian/Voronoi cells; other cell geometries need their own containment rule.

> <b>Answer to “is it clustering?”</b> The reducer is a single-pass, energy-weighted nearest-representative merge with moving centers. It resembles clustering, but it does not run iterative k-means. K is an accuracy parameter as well as a memory setting; sweep it before choosing a default.



---



06  /  Make the CPU version efficient

First remove allocation and unnecessary work. Then evaluate changes to the reduction schedule.

| Area | Proposed implementation |
| --- | --- |

| Persistent material state | Use flat per-cell arrays of positions and energies with offsets/counts. Prototype compact records; benchmark split arrays if nearest-neighbor scans dominate. No Particle object, velocity, frequency or MPI header per marker. |

| Transport output | Append one record per completed cell visit into reusable arenas. Group by local cell index using counting/prefix sums. Avoid a heap allocation or a cell-vector growth at every collision. |

| Source generation | Build budgets once per step; keep the existing direction/time/frequency routines. Scan the small mutable donor bank, and reserve room for source fragmentation. |

| Concurrency | Parallelize independent cell initialization, emission and merging where useful. Use worker-owned deposit buffers if transport becomes threaded; do not assume current scalar tally writes are thread-safe. |

| Instrumentation | Measure source, record creation, grouping and merging separately. Count peak records, mean/max M, fragments, merges, buffer flushes and bytes per owned cell. |

Account for peak memory, not only retained markers

Four doubles per 3-D marker cost 32 bytes before metadata. At K = 20, persistent marker payload is about **640 bytes per cell** (640 MB for one million owned cells). The reference implementation also needs O(D) temporary storage for D completed cell visits. A small final K alone does not bound this peak.

Optional bounded-memory mode — a separately validated variant

After establishing the end-step reference, prototype a per-cell cap B, initially 4K–8K. When a deposit batch would exceed B, apply the same reducer and retain K markers. Scan cells in small batches so both per-cell storage and the pending arena remain bounded. Reuse capacity between steps.

This repeated reduction costs roughly O(DK) overall for fixed B/K and bounds marker storage by O(CB), plus worker buffers, for C owned cells. It changes seed selection and merge order compared with end-step reduction. Compare it against the reference at equal K, source count, mesh and time step before enabling it by default.

> <b>No unsupported speed promise:</b> DIMC adds sampling and material bookkeeping. Judge it by error at a fixed CPU budget, as well as raw time per step. Benchmark direct scans before trying a kd-tree: maintaining a tree of moving centers may cost more than scanning 20 entries.



---



07  /  MPI, restart and RICH

Persistent material positions must survive ownership changes and hydrodynamic updates.

MPI ownership and checkpoints

Keep markers on the owner of their cell. Flush a photon’s cell-visit accumulator before it is routed to another rank, so the absorbed energy remains on the old cell owner. Use the existing P2P/RDMA engines for photons; material markers are not added to photon completion counts or photon population control.

On repartition, pack each cell’s markers with its stable global cell ID, then rebuild local offsets. A local cell index is not a persistent identity. Include marker positions, energies, reducer configuration and RNG state/counters in checkpoints. Initially support checkpoints only at completed step boundaries, with no unflushed histories.

Reuse the RICH wrapper too

The RICH RadiationIMC wrapper currently hardcodes STORM::RadiationIMC as Impl. Generalize the adapter over its implementation type and expose IMC and DIMC choices through thin aliases/wrappers. Reuse the existing opacity, EOS, boundary and diagnostic adapters. Avoid copying the large wrapper for DIMC. [C7]

Moving material is a separate integration milestone

Markers remain stationary during the first fixed-material transport release. For coupled RICH runs, advect positions with material motion and integrate marker transfer/splitting with the same conservative cell remap used by hydro. Apply local heating/cooling consistently to marker weights. Rescaling a cell’s weights can represent a local energy adjustment; it cannot substitute for advecting energy between cells.

After every hydro/remap stage, enforce nonnegative marker energy, cell containment, correct ownership and agreement with the hydro internal-energy ledger. Newly created cells need a conservative transfer of the parent material distribution. Uniform reseeding would discard the information DIMC was added to retain.

| Feature | Release decision |
| --- | --- |

| Gray, fixed mesh, absorption + elastic scattering | First validated mode. Cell material energy evolves; noHydroFeedback must not disable this ledger. |

| Multigroup | Next: use one material bank per cell, with group-dependent transport and the existing opacity-weighted emission/re-emission law. Verify both emission paths. [2, §II] |

| DDMC / random walk | Reject these combinations initially: their condensed paths need a derived spatial deposition rule. Reusing their scalar tallies is insufficient. |

| Compton, signed sources, diagnostic frozen material | Gate until their extra material transfers and source semantics are explicitly supported. Reject unsupported requests clearly. |



---



08  /  Implementation sequence

Each stage ends with a result that can be reviewed and tested independently.

| Stage | Work | Exit condition |
| --- | --- | --- |

| 1. Share IMC | Introduce the generic owner and cell-material policy. Keep aliases, APIs and source/transport math unchanged. | Existing IMC checks pass. Fixed-seed CPU histories agree where the refactor preserves operation order. No material storage or extra random draws in IMC. |

| 2. Add gray DIMC | Implement marker bank, conservative source withdrawal, segment hooks, history contexts and end-step reducer. | Serial fixed-mesh conservation, positivity, location and source tests pass; coarse Marshak comparison improves the targeted error. |

| 3. Bound and profile | Measure allocations and phase time. Add optional bounded-buffer reduction, compare data layouts and K/B sweeps. | Peak memory is measured and bounded in the optimized mode; its added error is within the chosen tolerance. |

| 4. MPI + restart | Add cell-state migration, context suspension/resumption and completed-step checkpoint I/O. | Cross-rank absorption, repartition and restart preserve material state and total energy. |

| 5. Groups + RICH | Reuse grouped transport/source code and generalize the RICH adapter. Implement conservative material advection/remap. | Frequency-dependent benchmarks and coupled moving-mesh tests pass. Unsupported accelerations remain gated. |

Keep the implementation focused

New DIMC files should contain only material storage, donor selection, deposit accumulation, reduction and configuration. Shared files receive the policy seams. Do not add a second AdvanceDIMC transport loop, a DIMC opacity solver, a second source direction sampler, or a copy of the manager.

Proposed initial controls

**materialParticlesPerCell** (start experiments at 20); **materialReductionMode** (end-step reference or bounded); **materialBufferCapacity** for the bounded mode; and a separate material RNG seed. Keep photon count and material-particle count independent. Report all of them with benchmark results.

> <b>Deliverable at stage 2:</b> a selectable DIMC physics implementation using the same CPU transport as IMC, with a documented validity range. Later stages expand that range; they should not silently inherit unsupported IMC features.

This is a design and inspection result, not an implemented or benchmarked feature. No repository source changes or simulation runs were performed for this plan.



---



09  /  Validation that answers the goal

Check conservation first, then the position of the heat front, then error per CPU time.

| Test | What it should establish |
| --- | --- |

| Single segment / cutoff | Total energy matches analytic attenuation, sampled deposit positions follow the truncated exponential, and cutoff residual energy appears exactly once. Cover zero, tiny and large optical depth. |

| Emission / equilibrium cell | Material debit equals emitted energy; markers remain nonnegative; thermal births are at donor positions. Test tiny donors, small photon counts, budget exhaustion and equilibrium over many steps. |

| Reducer invariants | Energy and first moment survive merging to floating-point tolerance. All markers remain in the same convex cell. Test unequal energies, zero entries, tied positions and M ≤ K. |

| Lifecycle / MPI | Scattering, reflection, census, escape and scheduling suspension lose no deposits. Remote transfer, repartition and restart retain the same ledger. Compare distributions across MPI layouts; do not require cross-layout bitwise identity. |

| Marshak / Densmore | At a deliberately coarse grid, measure heat-front position and material-temperature profile against an independently converged reference. Scan space and time separately. |

| Hohlraum / crooked pipe | Check multidimensional leakage, shadowing and interface behavior. Reuse existing STORM setups when their parameters match the intended benchmark. |

| Multigroup / hydro | Use Olson or an equivalent spectral benchmark; later use radiative-shock/moving-mesh cases for conservative marker advection and energy coupling. |

Measure two kinds of convergence separately

For material representation, test K = 10, 20, 30 and 60; compare end-step and bounded reduction with B = 4K and 8K. For Monte Carlo uncertainty, use multiple independent seeds and increase the photon budget. Do not mistake a quieter temperature curve for a more accurate heat-front position.

Report equal-photon-count comparisons to expose overhead, and **equal-wall-time comparisons** to judge usefulness. Record temperature error, front-position error, energy residual, peak memory and source/transport/merge times. Numerical acceptance limits should match each benchmark’s reference accuracy and statistical uncertainty.

> <b>Success means:</b> IMC still works through the shared implementation; DIMC preserves its material ledger; and the targeted coarse-mesh transport error decreases at an acceptable measured CPU and memory cost. No universal runtime ratio is assumed.



---



10  /  Evidence and review map

The plan is based on the current working tree and the two primary DIMC papers.

**Workspace:** /home/maorm/RICH
**Inspected revisions:** RICH 65bc7333; STORM 39eadb4.
Both working trees contain pre-existing edits. File references describe the inspected working-tree content, not clean commit snapshots.

| Ref. | Inspected code and why it matters |
| --- | --- |

| C1 | <b>source/monte/radiation/RadiationIMC.hpp:131</b><br/>Final physics owner; IMCState and owner-templated components. RadiationIMCFacade.hpp constructs and forwards to the components. |

| C2 | <b>source/monte/radiation/source/IMCSourceProcess.hpp:538, 565, 958</b><br/>Cell emission debit; fast-path selection; fallback particle construction. Both source paths need the same material policy. |

| C3 | <b>source/monte/radiation/source/SourceCore.hpp:72, 239, 249</b><br/>Shared source budget and packet builder; direct geometry sampling explains why a sampler-only substitution misses the fast path. |

| C4 | <b>source/monte/radiation/transport/AdvanceIMC.hpp:224, 346, 359</b><br/>Shared event kernel, scalar material deposit and cutoff residual. IMCTransportProcess.hpp:469, 482 calls it on the CPU. |

| C5 | <b>source/monte/radiation/imc/IMCLifecycleProcess.hpp:426, 805, 970</b><br/>Tally application, Fleck factor and preStep. IMCTransportProcess.hpp:916–994 applies tallies and synchronizes the cell. |

| C6 | <b>source/monte/manager/MonteCarloTransport.hpp:818, 851, 1012</b><br/>Per-photon processing, event-budget suspension and physics step call. MonteCarloPhysics.hpp:34–40 defines the public lifecycle. |

| C7 | <b>source/3D/radiation/RadiationIMC.hpp:351, 358, 418</b><br/>RICH wrapper, hardcoded implementation type and adapter construction. |

Primary references

**[1]** E. Steinberg &amp; S. I. Heizler. <i>A New Discrete Implicit Monte Carlo Scheme for Simulating Radiative Transfer Problems.</i> Astrophysical Journal Supplement Series 258, 14 (2022). §3 and Algorithms 1–3 are the algorithmic baseline.
<link href="https://arxiv.org/abs/2108.13453" color="#007f7b">arxiv.org/abs/2108.13453</link> • <link href="https://arxiv.org/pdf/2108.13453" color="#007f7b">Full paper</link>

**[2]** E. Steinberg &amp; S. I. Heizler. <i>Frequency-Dependent Discrete Implicit Monte Carlo Scheme for the Radiative Transfer Equation.</i> Nuclear Science and Engineering 197, 2343–2355 (2023). §II gives the grouped/frequency-dependent extension.
<link href="https://arxiv.org/abs/2303.06634" color="#007f7b">arxiv.org/abs/2303.06634</link> • <link href="https://doi.org/10.1080/00295639.2023.2190728" color="#007f7b">Published article</link>

The policy architecture, CPU data structures, source-fragment handling, bounded-buffer variant and rollout gates are engineering proposals for STORM. They are not claimed as additional results established by the papers.
