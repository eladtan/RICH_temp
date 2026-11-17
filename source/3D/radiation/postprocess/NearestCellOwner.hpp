#ifndef IMC_POSTPROCESS_NEAREST_CELL_OWNER_HPP
#define IMC_POSTPROCESS_NEAREST_CELL_OWNER_HPP

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "source/3D/tessellation/Voronoi3D.hpp"
#include "source/misc/universal_error.hpp"

#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace imc_postprocess_tde {

// For each point, the local cell that contains it, on the one rank that owns
// that cell; localCell[i] is npos on every other rank. Points outside the box
// get npos everywhere unless includeOutsideBox, which assigns them to the
// nearest cell like any other point.
//
// A cell contains the points nearest to its generator, and that generator can
// belong to a rank other than the one whose domain region holds the point,
// while GetContainingCell only searches the calling rank's own generators.
// Every rank therefore offers its nearest local generator and the closest one
// over all ranks takes the point (ties go to the lowest rank). Collective.
inline std::vector<std::size_t> NearestLocalCells(
    Voronoi3D const& tess, std::vector<Vector3D> const& points, int rank,
    bool includeOutsideBox = false)
{
    constexpr std::size_t npos = std::numeric_limits<std::size_t>::max();
    std::size_t const nCells = tess.GetPointNo();
    std::size_t const nPoints = points.size();
    std::vector<std::size_t> localCell(nPoints, npos);
#ifdef RICH_MPI
    if (nPoints > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw UniversalError("Too many points for the nearest-cell MPI reduction");
    struct DistanceRank {
        double distance2;
        int rank;
    };
    std::vector<DistanceRank> nearest(
        nPoints, DistanceRank{std::numeric_limits<double>::infinity(), rank});
#endif
    // A failed lookup is reported on every rank together: a throw on one rank
    // alone would leave the others waiting in the reduction below.
    std::size_t badCell = npos;
    std::size_t badPoint = npos;
    for (std::size_t i = 0; i < nPoints; ++i) {
        if (nCells == 0 || (!includeOutsideBox && tess.IsPointOutsideBox(points[i])))
            continue;
        std::size_t const cell = tess.GetContainingCell(points[i]);
        if (cell >= nCells) {
            if (badPoint == npos) {
                badCell = cell;
                badPoint = i;
            }
            continue;
        }
        localCell[i] = cell;
#ifdef RICH_MPI
        double const distance = fastabs(tess.GetMeshPoint(cell) - points[i]);
        nearest[i].distance2 = distance * distance;
#endif
    }
    int failed = badPoint != npos ? 1 : 0;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &failed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
    if (failed != 0) {
        UniversalError eo("Nearest-cell lookup returned a non-local cell");
        if (badPoint != npos) {
            eo.addEntry("Rank", rank);
            eo.addEntry("Point index", badPoint);
            eo.addEntry("Cell index", badCell);
            eo.addEntry("Local cells", nCells);
        } else {
            eo.addEntry("Failed on", std::string("another rank"));
        }
        throw eo;
    }
#ifdef RICH_MPI
    if (nPoints > 0)
        MPI_Allreduce(MPI_IN_PLACE, nearest.data(), static_cast<int>(nPoints),
                      MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
    for (std::size_t i = 0; i < nPoints; ++i)
        if (nearest[i].rank != rank)
            localCell[i] = npos;
#endif
    return localCell;
}

} // namespace imc_postprocess_tde

#endif // IMC_POSTPROCESS_NEAREST_CELL_OWNER_HPP
