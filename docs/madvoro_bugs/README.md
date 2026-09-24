# MadVoro correctness audit

[Read the 30-page PDF](MadVoro_Bug_Audit.pdf) or [the editable report](MadVoro_Bug_Audit.md).

The report contains 24 findings with exact locations, severity, scope, evidence, proposed fixes and regression gates. Twenty have direct production API/component witnesses, two use extracted algorithm fixtures, one injects a permitted MPI completion order, and one is source-only. Core, dependency and RICH integration findings are distinguished. Production source was not changed.

`evidence/` contains the small diagnostic sources, raw logs and source fingerprints. These deliberately record baseline defects; a fixture's exit code zero does not necessarily mean a correct result. The PDF explains each outcome. `audit_notes/` preserves the independent source reviews and qualifications.

From the RICH repository root:

```bash
python3 docs/madvoro_bugs/reproduce.py --list
python3 docs/madvoro_bugs/reproduce.py --build --run
```

The recipes assume the audited GCC/Open MPI environment. See the report and script for environment overrides and selection of individual cases. The reproduction driver was smoke-checked by rebuilding and running the centroid, extracted-helper and chain-permutation cases; their expected baseline defects remained visible. Other individual source programs were built and executed during the audit as recorded in their original logs.

To regenerate the document using the local system LaTeX installation:

```bash
python3 docs/madvoro_bugs/assemble_audit.py
python3 docs/madvoro_bugs/render_audit.py
```

Edit the corresponding source note before assembling; `assemble_audit.py` regenerates the main Markdown. `render_audit.py` emits LaTeX and performs three PDF passes. `evidence/artifact_checks.json` records the final document and source checks. The bundle ZIP contains the report, document scripts, source notes and text/code evidence; it excludes compiled binaries and preview images.
