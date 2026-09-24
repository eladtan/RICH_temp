# Final skeptical review: MPI and integration findings

Reviewed on 2026-09-08. Checked `mpi.md`, `integration.md`, cited production statements, reproduction sources, and their retained output. No production changes or additional findings were introduced.

## Primary MPI specification check

Authoritative PDF: [MPI Forum, MPI 4.1 standard](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report.pdf). The PDF identifies **§3.7.5, Multiple Completions**; the Waitsome/Testsome descriptions are on printed pages88–89 (PDF pages130–131, counted from1). It defines the returned completed indices and associated statuses, and gives no ascending-index requirement. The immediate-return rule of MPI_Testsome does not add an ordering requirement. This supports the request-compaction portability argument.

Convenient hosted HTML: [Multiple Completions](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node76.htm). That HTML labels the section4.7.5 and identifies itself as an unofficial rendering. For the final report prefer the official PDF plus section title, or keep the HTML number explicitly tied to the HTML. Do not cite the “arbitrary order” prose about MPI_Testany as if it explicitly described MPI_Testsome; the relevant conclusion for Testsome is the absence of an ascending-order guarantee.

## MPI findings: limits that must survive consolidation

* **MPI-01, compaction:** the production statement sequence at `BuffersManager.hpp:365-383` matches the failing fixture. The fixture injects post-completion metadata; it does **not** reproduce Open MPI naturally returning `[2,0]`, nor a production Voronoi crash. Report as a proved bookkeeping failure under an API-permitted completion ordering. P1 is defensible as conditional failure impact in an active transport; if the report uses P1 to mean immediately reproduced failures on the deployed MPI, use P2 portability priority instead. Never place it in an unqualified “all these end-to-end MadVoro failures were reproduced” count.
* **MPI-02, one-way ghosts:** the exact routine removes receive-only peers, and partial active sets provide a coherent source-level trigger. But the fixture supplies the internal state directly and does not prove a complete current partial build reaches that state before another defect fails. Retain P1 **conditional partial-build impact**, clearly separated from end-to-end reproduced cases. Fixing only reciprocity can expose the separate all-owned-index versus active-volume-array problem. No wrong completed physical simulation was measured.
* **MPI-03, mixed periodicity:** production initialization creates repeated zero shifts; the verbatim helper fixture verifies factors9 and3. This is a definite active query-duplication/overfetch defect. P2 performance priority is reasonable. It is **not established numerical corruption**, and the factors apply to duplicate shifted queries, not total mesh runtime or every query record.
* **DEP-01, truncation:** real MPI tests of unchanged production header show two individually fitting serialized records exceeding the packet capacity and delivering incomplete data. Silent partial callbacks require the fixture's `MPI_ERRORS_RETURN` setting plus NDEBUG; with usual fatal MPI error handling, the likely symptom is termination instead. No such overflow was established for MadVoro's present buffer margins and16/1 point answer caps. P2 shared-library/configuration scope is appropriate.
* **DEP-02, eight-byte stall:** unchanged header plus real two-rank MPI verifies the stalled pending buffer. This is a supported generic scalar type, but not the serialized shape of MadVoro's active query records. P2 shared-library scope is appropriate.
* **DEP-03, self flattening:** real one-rank test proves inconsistency between per-query/by-rank and flattened results with sendToSelf=true. Current MadVoro explicitly uses false. Keep as P2 generic API/result consistency, not lost MadVoro self ghosts. Source comparison strengthens the intended consistency: `ThreePhasesQueryAgent.hpp:117-133` includes the self source in flattened results, while this backend excludes it. No need to claim this alternative backend was runtime-tested.

No current ghost-completion race was demonstrated merely from CountOutcoming omitting packed pending buffers; the existing logical response accounting supplies additional constraints. The final report should retain this rejected-overclaim explanation.

## Integration findings: severity and source correspondence

The six integration mechanisms correspond to the inspected statements and harnesses. Their stated severities are supported with the following boundaries:

| Finding | Review decision and caveat |
| --- | --- |
| INT-01 mixed suppression | P1 retained. V:2023 prepares cache using the local flag, while V:2030 chooses it using global permission; PM:295 reads the undersized vector. The assertion explicitly names vector<double>, and the uniform-suppression control succeeds. Valid caller weights are present; do not blame their size. Undefined unchecked behavior is inferred, not a recorded silent corrupted run. |
| INT-02 SetKernel lifecycle | P2 retained. HPM:264 nullifies the converter through HLB:293 and clears the environment; HPM:283-291 leaves the balancer pointer alive and rebalances without rebuilding its converter. The dependency method is directly tested and MadVoro's setter delegation is source-verified. Do not describe the test as a full MadVoro SetKernel run. |
| INT-03 communicator propagation | P1 retained for the dependency's explicit communicator constructor. HPM:289 omits the communicator, CurveLoadBalancer:16 takes the base default, and LoadBalancer:19 defaults to WORLD. The SELF manager has smaller destination arrays; the recorded assertion is consistent. This does not establish that the whole MadVoro engine advertises arbitrary subcommunicator support. |
| INT-04 single-rank bins | P2 retained. The normal one-rank weighted helper returns no cuts, while intersection queries require one bin. The component test proves the API failure after valid rebalance; it does not show every MPI1 MadVoro build fails. |
| INT-05 chain destination order | P1 retained. `ExchangeChain.hpp:59` drops target.second; `:65-69` rebuilds in source order. The self-only permutation test isolates a valid mapping and proves wrong field-to-index association without geometry or MPI network ambiguity. This is RICH integration, outside MadVoro. |
| INT-06 empty rank skips transfer | P1 retained. Header:49 and Simulation.cpp:133 independently gate a collective path on a local original count. The log reaches completed chain construction and shows the empty-origin receiver returning early, while its peer cannot finish. The timeout reinforces an already concrete collective mismatch; UCX warnings are not its proof. Preserve the default inactive-chain contract when fixing it. This is RICH integration, outside MadVoro. |

The integration notes appropriately reject several additional unproven scope leaps (e.g. periodic local-resolution errors automatically implying current MockMesh misrouting). Keep these boundaries in the report. Do not count RICH and MeshDecomposer3D findings as defects physically located in the MadVoro submodule.

## Reporting recommendation

Use separate evidence labels: **real API reproduction**, **production method with injected MPI completion order**, **verbatim helper fixture**, and **source-level partial-build consequence**. A priority is not a confidence level. Keep conditional P1 findings distinct from the directly reproduced P1 assertions, wrong permutations and collective mismatch. No P0 conclusion is supported. Reproduction exit codes deliberately signal detected defects; launcher nonzero-exit boilerplate alone is not evidence of a separate MPI failure.
