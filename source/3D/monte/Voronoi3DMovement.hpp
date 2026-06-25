#ifndef MONTE_CARLO_VORONOI3D_MOVEMENT
#define MONTE_CARLO_VORONOI3D_MOVEMENT

#include "3D/elementary/Vector3D.hpp"
#include "3D/tessellation/Voronoi3D.hpp"
#include "monte/MonteCarloParticle.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include <spatial_ds/OctTree/OctTree.hpp>
#include "utils/debug/SmartTimer.hpp"
#include "MonteCarlo3D.hpp"
#ifdef RICH_MPI
    #include "mpi/mpi_commands.hpp"
    #include "3D/range/finders/utils/RankedIndexedVector.hpp"
    #include "mpi/ExchangeChain.hpp"
#else // RICH_MPI
    #include "3D/range/finders/utils/IndexedVector.hpp"
#endif // RICH_MPI

using Particle3D = MonteCarloParticle<Vector3D, Tessellation3D>;

#ifdef RICH_MPI
    #include "mpi/MPI_Particle3D_dtype.hpp"
#endif // RICH_MPI

void AssertLocations(const Tessellation3D &tess, const std::vector<Particle3D> &particles);

void UpdateNewCells(const Tessellation3D &tess, std::vector<Particle3D> &particles, const std::vector<size_t> &cellIDs);

void UpdateNewCells(const Tessellation3D &tess, std::vector<Particle3D> &particles, const std::vector<ComputationalCell3D> &cells);

#ifdef RICH_MPI
void UpdateNewCellsAfterExchange(const Tessellation3D &tess, std::vector<Particle3D> &particles, const ExchangeChain &chain);

void UpdateNewCellsAfterExchange(const Tessellation3D &tess, std::vector<Particle3D> &particles);
#endif // RICH_MPI

#endif // MONTE_CARLO_VORONOI3D_MOVEMENT