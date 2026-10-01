Review complete. The round-4 objections are resolved.

- [S3] Concrete repartition trigger in `Simulation::stepIndividual`, existing rebalance machinery, 5 s candidate threshold, 700–800 s measured saving basis, and fail-closed gates.
- [S4/S5] Concrete switches, expected savings, instrumentation, and correctness/speed gates. Rejecting higher `partial_build_fraction` is sound: the cited 366 s vs 146 s whole-mesh A/B and 8–28 s partial targets support it.
- [S6] FMM reranking is sound: 2.4% rebuild frequency makes interval widening only ~0.02 s/event, while evolved-state growth dominates. Reset threshold, expected 1.2 → ≤0.3 s/solve, accuracy, and zero-error gates are concrete.
- Section 0 normalization is corrected: D is 2108 retries/unit-time versus global 2229, zero invalid-energy failures versus four, so R2’s robustness gate is met without R1. The remaining pre-pericentre run is correctly retained.
- Dropping full-source-sweep widening is sound: 223 sweeps × 0.06 s cannot explain the 699 s scheduler residual.
- Spot checks confirmed the cited `RICH_STEP` fields, 223 sweep records, 2117 FMM traces, median `total_mean` ≈0.171 s, and 51 rebuild events.

[NIT] The FMM log’s maximum `total_mean` is about 1.06 s, while the plan says rebuilds reach 0.87 s. Clarify that 0.87 s is the rebuild-conditioned value, not the overall trace maximum. This does not gate approval.

VERDICT: APPROVE