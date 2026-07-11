#include "newtonian/three_dimensional/FastMultipoleAcceleration3D.hpp"

#include <cmath>

#include "misc/memory_profile.hpp"
#include "misc/universal_error.hpp"

namespace
{
FmmGravityOptions validateAccelerationOptions(FmmGravityOptions options)
{
#ifdef RICH_MPI
    int localSupported = options.computePotential ? 0 : 1;
    int globallySupported = 0;
    MPI_Allreduce(&localSupported, &globallySupported, 1, MPI_INT, MPI_LAND,
                  MPI_COMM_WORLD);
    if(globallySupported == 0)
        throw UniversalError(
            "FastMultipoleAcceleration3D: computePotential is unsupported by the acceleration adapter");
#else
    if(options.computePotential)
        throw UniversalError(
            "FastMultipoleAcceleration3D: computePotential is unsupported by the acceleration adapter");
#endif
    return options;
}

#ifdef RICH_MPI
double validateDistributedGravityConstant(double value)
{
    int initialized = 0;
    MPI_Initialized(&initialized);
    if(initialized == 0)
        throw UniversalError(
            "FastMultipoleAcceleration3D: MPI must be initialized before construction");
    int localFinite = std::isfinite(value) ? 1 : 0;
    int globalFinite = 0;
    MPI_Allreduce(&localFinite, &globalFinite, 1, MPI_INT, MPI_LAND,
                  MPI_COMM_WORLD);
    if(globalFinite == 0)
        throw UniversalError(
            "FastMultipoleAcceleration3D: G must be finite on every MPI rank");
    double minimum = 0.0;
    double maximum = 0.0;
    MPI_Allreduce(&value, &minimum, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&value, &maximum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if(minimum != maximum)
        throw UniversalError(
            "FastMultipoleAcceleration3D: G differs across MPI ranks");
    return value;
}

void requireOnEveryRank(bool localCondition, const char* message)
{
    int local = localCondition ? 1 : 0;
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if(global == 0)
        throw UniversalError(message);
}
#endif
}

FastMultipoleAcceleration3D::FastMultipoleAcceleration3D(FmmGravityOptions options,
                                                         double G):
#ifdef RICH_MPI
    G_(validateDistributedGravityConstant(G)),
#else
    G_(G),
#endif
    calculator_(validateAccelerationOptions(options))
{
#ifndef RICH_MPI
    if(!std::isfinite(G_))
        throw UniversalError("FastMultipoleAcceleration3D: G must be finite");
#endif
}

#ifdef RICH_MPI
FastMultipoleAcceleration3D::FastMultipoleAcceleration3D(
    FmmGravityOptions options,
    FmmDistributedOptions distributedOptions,
    double G):
    G_(validateDistributedGravityConstant(G)),
    calculator_(validateAccelerationOptions(options), distributedOptions)
{}
#endif

void FastMultipoleAcceleration3D::operator()(const Tessellation3D& tess,
                                             const vector<ComputationalCell3D>& cells,
                                             const vector<Conserved3D>& fluxes,
                                             const double time,
                                             vector<Vector3D>& acc) const
{
    (void) fluxes;
    (void) time;
    MEMORY_PROFILE_SCOPE("fmm gravity source");

    const std::size_t N = tess.GetPointNo();
#ifdef RICH_MPI
    requireOnEveryRank(cells.size() >= N,
        "FastMultipoleAcceleration3D: cell array is smaller than owned tessellation on an MPI rank");
#else
    if(cells.size() < N)
        throw UniversalError("FastMultipoleAcceleration3D: cell array is smaller than owned tessellation");
#endif
    points_.resize(N);
    masses_.resize(N);
#ifdef RICH_MPI
    cellIds_.resize(N);
#endif
    for(std::size_t cellIdx = 0; cellIdx < N; ++cellIdx)
    {
        points_[cellIdx] = tess.GetCellCM(cellIdx);
        masses_[cellIdx] = cells[cellIdx].density * tess.GetVolume(cellIdx);
#ifdef RICH_MPI
        cellIds_[cellIdx] = static_cast<std::uint64_t>(cells[cellIdx].ID);
#else
        if(!std::isfinite(masses_[cellIdx]))
        {
            UniversalError error("FastMultipoleAcceleration3D: non-finite cell mass");
            error.addEntry("cell", cellIdx);
            throw error;
        }
#endif
    }

    const std::pair<Vector3D, Vector3D> boundaries = tess.GetBoxCoordinates();
#ifdef RICH_MPI
    calculator_.solve(points_, masses_, cellIds_, boundaries.first,
                      boundaries.second, acc);
#else
    calculator_.solve(points_, masses_, boundaries.first, boundaries.second, acc);
#endif

    bool finiteAcceleration = true;
#ifndef RICH_MPI
    std::size_t firstInvalid = 0;
#endif
    for(std::size_t i = 0; i < acc.size(); ++i)
    {
        acc[i] *= G_;
        if(finiteAcceleration &&
           (!std::isfinite(acc[i].x) || !std::isfinite(acc[i].y) ||
            !std::isfinite(acc[i].z)))
        {
            finiteAcceleration = false;
#ifndef RICH_MPI
            firstInvalid = i;
#endif
        }
    }
#ifdef RICH_MPI
    requireOnEveryRank(finiteAcceleration,
        "FastMultipoleAcceleration3D: non-finite acceleration after G scaling on an MPI rank");
#else
    if(!finiteAcceleration)
    {
        UniversalError error("FastMultipoleAcceleration3D: non-finite acceleration after G scaling");
        error.addEntry("cell", firstInvalid);
        throw error;
    }
#endif
}

const FmmSolveStats& FastMultipoleAcceleration3D::getLastStats() const noexcept
{
    return calculator_.stats();
}
