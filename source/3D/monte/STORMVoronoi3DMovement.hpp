#ifndef RICH_STORM_VORONOI3D_MOVEMENT_HPP
#define RICH_STORM_VORONOI3D_MOVEMENT_HPP

#include "3D/monte/MonteCarlo3D.hpp"
#include "3D/tessellation/Voronoi3D.hpp"
#include "monte/mesh_movement/VoronoiMeshMovement.hpp"
#ifdef RICH_MPI
#include "mpi/MPI_Particle3D_dtype.hpp"
#endif
#include "misc/universal_error.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"

namespace rich_storm_movement_detail
{

inline const Voronoi3D &RequireVoronoi(const Tessellation3D &tess)
{
    const Voronoi3D *voronoi = dynamic_cast<const Voronoi3D *>(&tess);
    if(voronoi == nullptr)
    {
        throw UniversalError("UpdateNewCells: Tessellation is not a Voronoi3D");
    }
    return *voronoi;
}

} // namespace rich_storm_movement_detail

inline void AssertLocations(const Tessellation3D &tess, const std::vector<Particle3D> &particles)
{
    rich_storm_movement_detail::RequireVoronoi(tess);
    STORM::VoronoiMeshMovement<Vector3D, Tessellation3D>::AssertLocations(tess, particles);
}

inline void UpdateNewCells(const Tessellation3D &tess, std::vector<Particle3D> &particles,
                           const std::vector<size_t> &cellIDs)
{
    rich_storm_movement_detail::RequireVoronoi(tess);
    STORM::VoronoiMeshMovement<Vector3D, Tessellation3D>::UpdateNewCells(tess, particles, cellIDs);
}

inline void UpdateNewCells(const Tessellation3D &tess, std::vector<Particle3D> &particles,
                           const std::vector<ComputationalCell3D> &cells)
{
    const size_t N = tess.GetPointNo();
    if(cells.size() < N)
    {
        UniversalError eo("UpdateNewCells: cells.size() is smaller than the tessellation");
        eo.addEntry("cells.size()", cells.size());
        eo.addEntry("tess.GetPointNo()", N);
        throw eo;
    }

    std::vector<size_t> cellIDs;
    cellIDs.reserve(N);
    for(size_t i = 0; i < N; ++i)
    {
        cellIDs.push_back(cells[i].ID);
    }

    UpdateNewCells(tess, particles, cellIDs);

    for(Particle3D &particle : particles)
    {
        if(particle.cellIndex < N)
        {
            particle.cellID = cells[particle.cellIndex].ID;
        }
    }
}

#ifdef RICH_MPI
inline void UpdateNewCellsAfterExchange(const Tessellation3D &tess, std::vector<Particle3D> &particles)
{
    rich_storm_movement_detail::RequireVoronoi(tess);
    STORM::VoronoiMeshMovement<Vector3D, Tessellation3D>::UpdateNewCellsAfterExchange(tess, particles);
}
#endif // RICH_MPI

#endif // RICH_STORM_VORONOI3D_MOVEMENT_HPP
