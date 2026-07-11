#include "newtonian/three_dimensional/FastMultipoleAcceleration3D.hpp"

#include <cmath>

#include "misc/memory_profile.hpp"
#include "misc/universal_error.hpp"

FastMultipoleAcceleration3D::FastMultipoleAcceleration3D(FmmGravityOptions options, double G):
    G_(G),
    calculator_(options)
{
#ifdef RICH_MPI
    throw UniversalError("FastMultipoleAcceleration3D: distributed sources are not implemented; serial FMM is disabled in MPI builds");
#else
    if(!std::isfinite(G_))
        throw UniversalError("FastMultipoleAcceleration3D: G must be finite");
#endif
}

void FastMultipoleAcceleration3D::operator()(const Tessellation3D& tess,
                                             const vector<ComputationalCell3D>& cells,
                                             const vector<Conserved3D>& fluxes,
                                             const double time,
                                             vector<Vector3D>& acc) const
{
#ifdef RICH_MPI
    (void) tess;
    (void) cells;
    (void) fluxes;
    (void) time;
    (void) acc;
    throw UniversalError("FastMultipoleAcceleration3D: distributed sources are not implemented; serial FMM is disabled in MPI builds");
#else
    (void) fluxes;
    (void) time;
    MEMORY_PROFILE_SCOPE("fmm gravity source");

    const std::size_t N = tess.GetPointNo();
    if(cells.size() < N)
        throw UniversalError("FastMultipoleAcceleration3D: cell array is smaller than owned tessellation");
    points_.resize(N);
    masses_.resize(N);
    for(std::size_t cellIdx = 0; cellIdx < N; ++cellIdx)
    {
        points_[cellIdx] = tess.GetCellCM(cellIdx);
        masses_[cellIdx] = cells[cellIdx].density * tess.GetVolume(cellIdx);
        if(!std::isfinite(masses_[cellIdx]))
        {
            UniversalError error("FastMultipoleAcceleration3D: non-finite cell mass");
            error.addEntry("cell", cellIdx);
            throw error;
        }
    }

    const std::pair<Vector3D, Vector3D> boundaries = tess.GetBoxCoordinates();
    calculator_.solve(points_, masses_, boundaries.first, boundaries.second, acc);

    for(std::size_t i = 0; i < acc.size(); ++i)
    {
        acc[i] *= G_;
        if(!std::isfinite(acc[i].x) || !std::isfinite(acc[i].y) ||
           !std::isfinite(acc[i].z))
        {
            UniversalError error("FastMultipoleAcceleration3D: non-finite acceleration after G scaling");
            error.addEntry("cell", i);
            throw error;
        }
    }
#endif
}

const FmmSolveStats& FastMultipoleAcceleration3D::getLastStats() const noexcept
{
    return calculator_.stats();
}
