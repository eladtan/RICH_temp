# 01 Signed Packet Foundation

## Purpose

Make the existing IMC transport safe for signed packet weights without enabling Compton. This is the foundation for anti-packets and signed residual source packets.

## Depends On

None.

## Code Areas

- `source/3D/radiation/RadiationIMC.cpp`
- `source/monte/MonteCarloParticle.hpp`
- Any MPI serialization or communication code that assumes positive packet weights
- Existing no-Compton regression cases

## Implementation Tasks

- Audit all `RadiationIMC` packet weight checks that compare against a low-weight threshold.
- Replace raw low-weight comparisons with `std::abs(particle.weight)` where removal depends on packet magnitude.
- Set `particle.initialWeight = std::abs(particle.weight)` everywhere an IMC particle is created, including material source particles, initial particles, boundary particles, and later child-packet helper paths.
- Confirm continuous absorption, census, boundary movement, and post-step tallies use signed weight where physics requires signed energy.
- Add a small helper for frequency selection by group, but keep ordinary thermal frequency generation behavior unchanged when no group is requested.
- Do not add `withCompton` yet.

## Endpoint

The IMC code can transport positive and negative packets without immediately deleting negative packets because of raw weight comparisons. With only positive packets, no-Compton behavior is unchanged.

## Correctness Tests

- Run a no-Compton regression with the same random seed as the baseline. Positive-packet histories should be identical or statistically unchanged.
- Add or run a signed packet smoke test:
  - One positive packet absorption increases material energy.
  - One negative packet absorption decreases material energy.
  - `postStep` uses the signed sum for radiation tallies.
  - Low-weight removal uses magnitude, not raw signed value.
- Build with MPI enabled if available and confirm signed packet fields serialize without sign loss.

## Stop / Do Not Include Yet

- Do not add Compton parameters.
- Do not add Compton data structures.
- Do not change Fleck factor logic.
- Do not change source generation except for signed-weight safety.
