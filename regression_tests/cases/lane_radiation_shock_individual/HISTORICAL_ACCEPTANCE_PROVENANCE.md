# Historical MG acceptance provenance

Pre-edit audit recorded 2026-08-12 before the shared historical-acceptance
implementation.

- Branch: `codex/individual-timesteps`
- Git revision: `9dba0e18e842ac897a4622a01873dcbedc0209cb`
- Tracked dirty-diff SHA-256: `390f205e077bbacf4b7643e77b81ee78a48dc66da9103442408bb7dd46068237`
- Full porcelain-status SHA-256: `d3fe89eaa311f0bf705680598fba8d6d97801d15fe439d7d4162a3befe3451b8`
- Compiler: Intel oneAPI DPC++/C++ 2024.2.1
- MPI: Open MPI 4.1.6, Intel/OneAPI/2024.2.1 module build
- Build lane: `intelReleaseMPI`, runtime energy-group count 16 for the coupled campaign

Retained executables and logs are diagnostic evidence only:

- Original same-cell block-Jacobi campaign binary:
  `5e2158fdf92559e720a88e6f9e6c9668503c57ce77dfb981b06f0838ac228d5f`
- Block-Jacobi plus GMRES(32) experiment binary:
  `3e902e2583d059995f40d03292d4e25ef79004fb9cc79c0e196be738cef200be`
- Rank-local ILU(0) plus GMRES(16) experiment/current pre-edit build:
  `76c921840dfd34c5c5647d09352956b29884e0336f44406954b73e8cd1ac3b22`
- Rank-local ILU(0) plus GMRES(32) experiment binary:
  `acb7931ef0129b9da2402c5a614809c59293606c4636a844f6f2eabc9cd8f107`

No retained source patch or source snapshot was found in the experiment result
roots. No behavior is inferred by reverse engineering a binary. The reviewed
workspace source is the only implementation input. The ILU/GMRES experiments
are put on hold; the production solver policy is restored to the runtime-group,
same-cell block-Jacobi preconditioner.

Old campaign timings remain diagnostic history and are not mixed with the new
historical-acceptance campaign.
