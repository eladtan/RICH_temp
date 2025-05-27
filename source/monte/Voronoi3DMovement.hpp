#ifndef MONTE_CARLO_VORONOI3D_MOVEMENT
#define MONTE_CARLO_VORONOI3D_MOVEMENT

#include "3D/elementary/Vector3D.hpp"
#include "3D/tesselation/voronoi/Voronoi3D.hpp"
#include "MonteCarloParticle.hpp"
#include "ds/OctTree/OctTree.hpp"
#ifdef RICH_MPI
    #include "mpi/serialize/mpi_commands.hpp"
    #include "3D/range/finders/utils/RankedIndexedVector.hpp"
#else // RICH_MPI
    #include "3D/range/finders/utils/IndexedVector.hpp"
#endif // RICH_MPI

using Particle3D = MonteCarloParticle<Vector3D, Tessellation3D>;

void AssertLocations(const Tessellation3D &tess, const std::vector<Particle3D> &particles);

void UpdateNewCells(const Tessellation3D &tess, std::vector<Particle3D> &particles);

std::vector<Particle3D> MoveParticlesAlongMesh(const Tessellation3D &tess, const std::vector<Particle3D> &particles);

#endif // MONTE_CARLO_VORONOI3D_MOVEMENT