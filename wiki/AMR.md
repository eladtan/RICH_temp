# Adaptive Mesh Refinement (AMR)

RICH supports adaptive mesh refinement in 3D, allowing cells to be refined (split) or removed (merged) during the simulation to concentrate resolution where it is needed.

## Overview

AMR in RICH operates on the Voronoi mesh by:

- **Refinement:** Splitting a cell into multiple smaller cells by inserting new mesh points
- **Removal (coarsening):** Merging small or unnecessary cells by removing mesh points

Both operations preserve conserved quantities (mass, momentum, energy) to high precision.

## Key Classes

| Class | Purpose |
|-------|---------|
| `AMR3D` | Main AMR driver that orchestrates refinement and removal |
| `CellsToRefine3D` | Abstract base: identifies which cells to refine |
| `CellsToRemove3D` | Abstract base: identifies which cells to remove |

## Refinement Criteria

Implement `CellsToRefine3D` to define when cells should be refined:

```cpp
class MassRefine : public CellsToRefine3D {
public:
    std::vector<size_t> operator()(
        const Tessellation3D& tess,
        const std::vector<ComputationalCell3D>& cells,
        double time) const override
    {
        std::vector<size_t> to_refine;
        for (size_t i = 0; i < cells.size(); ++i) {
            double mass = cells[i].density * tess.GetVolume(i);
            if (mass > max_mass_)
                to_refine.push_back(i);
        }
        return to_refine;
    }
};
```

Common refinement strategies:

| Strategy | Criterion |
|----------|-----------|
| Mass-based | Refine cells exceeding a mass threshold |
| Gradient-based | Refine near strong gradients (density, pressure) |
| Region-based | Refine within a geometric region of interest |
| Jeans-based | Refine to resolve the Jeans length (self-gravitating flows) |

## Removal Criteria

Implement `CellsToRemove3D` to define when cells should be removed:

```cpp
class RemoveBig : public CellsToRemove3D {
public:
    std::vector<size_t> operator()(
        const Tessellation3D& tess,
        const std::vector<ComputationalCell3D>& cells,
        double time) const override
    {
        std::vector<size_t> to_remove;
        for (size_t i = 0; i < cells.size(); ++i) {
            if (tess.GetVolume(i) > max_volume_)
                to_remove.push_back(i);
        }
        return to_remove;
    }
};
```

## Using AMR in a Simulation

```cpp
#include "source/newtonian/three_dimensional/AMR3D.hpp"

// Define criteria
MassRefine refiner(max_mass);
RemoveBig remover(max_volume);

// Create AMR object (needs EOS, refinement criteria, removal criteria, interpolation)
AMR3D amr(eos, refiner, remover, interp);

// In time loop: apply AMR periodically
if (sim.getCycle() % 10 == 0)
    amr(sim);
```

## Conservation Properties

AMR operations in RICH are designed to conserve:

- **Mass**: Total mass before and after refinement/removal matches to machine precision
- **Momentum**: Total momentum is conserved
- **Energy**: Total energy (kinetic + internal) is conserved

The `amr_random` regression test verifies this property, requiring conservation drift below 1e-8 (serial) or 1e-6 (MPI).

## Dynamic Domain Box

For simulations where the region of interest changes over time (e.g., expanding blast waves, disrupted stars), the domain box can be expanded dynamically:

```cpp
#include "source/3D/GeometryCommon/UpdateBox.hpp"

// In the time loop (e.g. every 7 cycles):
UpdateBox(tess, sim,
    0.5,               // min_velocity -- only track cells moving faster than this
    1e-5,              // volume_fraction -- controls density of new cells in expanded region
    reference_cell     // initial conditions for new cells
);

// After box update, recalculate volumes for AMR
auto box = sim.getTesselation().GetBoxCoordinates();
double newvol = (box.second.x - box.first.x) * (box.second.y - box.first.y)
              * (box.second.z - box.first.z);
refiner.SetSize(newvol);
remover.SetSize(newvol);
```

`UpdateBox` detects mesh points approaching the domain boundary, expands the box by ~5x the maximum cell width, fills the new volume with low-density reference cells, and rebuilds the tessellation.

## Performance Notes

- AMR adds overhead for mesh rebuilding; balance refinement frequency against accuracy needs.
- Over-refinement can dramatically increase cell count and slow the simulation.
- MPI AMR requires communication to maintain domain decomposition consistency.
- Typical AMR interval: every 10 cycles for production runs.
