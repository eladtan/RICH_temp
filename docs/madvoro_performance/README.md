# MadVoro performance plan

Start with [the 36-page PDF](MadVoro_Performance_Plan.pdf) or the identical [editable Markdown](MadVoro_Performance_Plan.md).

The plan contains 24 numbered work items (P00-P23), conditional runtime contribution estimates for each, source targets, implementation contracts, tests, rollout order, and an adversarial review guide. The estimates are not measured optimization speedups. Current-source local smoke results are recorded separately.

- `audit_notes/`: independent MPI, geometry, and RICH integration source audits. The consolidated plan incorporates corrections from a second review; its priorities and estimates take precedence over brainstorm ranges in these notes.
- `evidence/audit_smoke.cpp`: diagnostic harness compiled from the inspected current code.
- `evidence/baseline_*.log` and `baseline_summary.json`: sequential 1/2/4-rank and periodic baseline observations, with 14 successful count/volume smoke checks.
- `evidence/source_sha256.txt`, revision/compiler records, and build-configuration patch: provenance.
- `evidence/artifact_checks.json`: PDF text/page-boundary and proposal coverage checks.
- `render_plan.py` and `.tex`: reproducible local PDF generation using system LaTeX.

To regenerate the PDF on this host:

```bash
cd /home/maorm/RICH
python3 docs/madvoro_performance/render_plan.py
```

The earlier `smoke_*` logs include exploratory launches and are excluded from the report's baseline figures. No production optimization was implemented by this audit. Multi-node and full-RICH before/after measurements remain implementation acceptance work defined in the plan.
