# IMC Compton Subprojects

This directory splits the root implementation plan into small projects with clear endpoints.

Canonical references:

- `../imc_compton_implementation_plan.md`
- `../imc_compton_implementation_plan.tex`
- `../source/3D/radiation/RadiationIMC.hpp`
- `../source/3D/radiation/RadiationIMC.cpp`
- `../source/Radiation/MultigroupDiffusion.cpp`

The Markdown plan is the source of truth. The TeX file is a formatted duplicate reference.

## Dependency Order

1. `01_signed_packet_foundation.md`
2. `02_compton_parameters_and_precompute_skeleton.md`
3. `03_compton_matrix_port_and_fleck.md`
4. `04_signed_group_source_generation.md`
5. `05_residual_branch_transport.md`
6. `06_deterministic_parity_harness.md`
7. `07_signed_tallies_mpi_boundaries.md`
8. `08_optional_analog_direct_compton.md`
9. `09_variance_controls_and_diagnostics.md`

Projects 01-07 implement the residual-only direct Compton path. Project 08 is optional and adds physical analog Compton events. Project 09 is follow-up variance and diagnostics work after correctness is established.

## Global Rules

- Keep `withCompton=false` behavior unchanged.
- Freeze Compton coefficients in `preStep`; do not update them during particle transport.
- Start with Mode 1: residual-only direct Compton.
- Keep source code changes inside the current IMC/Monte Carlo radiation architecture unless a project explicitly says otherwise.
- Every signed packet birth or death must update material energy conservatively.
- Do not use random walk when Compton residual sources are active unless random walk is extended to emit equivalent residual children.

## Acceptance Chain

- Projects 01-02: no-Compton regression and signed packet smoke checks pass.
- Project 03: zero-Compton coefficients reproduce the old Fleck factor and matrix behavior.
- Project 04: signed source packets conserve total material plus radiation energy at birth.
- Project 05: residual child expectation matches the residual matrix for positive and negative parents.
- Project 06: debug parity reconstructs the deterministic linearized matrix/source.
- Project 07: signed packet tallies, MPI movement, boundaries, and random-walk disabling are correct.
- Project 08: analog direct Compton events reproduce the intended group transfer in expectation.
- Project 09: variance controls are unbiased and diagnostics are actionable.
