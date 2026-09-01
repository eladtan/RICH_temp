#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <vector>

#include <mpi.h>

#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/misc/universal_error.hpp"
#include "source/mpi/mpi_commands.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/Ghost3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"

namespace
{

std::size_t CellID(const Vector3D &point)
{
    const std::size_t ix = static_cast<std::size_t>(std::llround(4.0 * point.x - 0.5));
    const std::size_t iy = static_cast<std::size_t>(std::llround(4.0 * point.y - 0.5));
    const std::size_t iz = static_cast<std::size_t>(std::llround(4.0 * point.z - 0.5));
    return 1 + 16 * ix + 4 * iy + iz;
}

double Distance(const Vector3D &a, const Vector3D &b)
{
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

std::vector<bool> BuildGhostMask(const Tessellation3D &tess)
{
    std::vector<bool> result(tess.GetTotalPointNumber(), false);
    const std::vector<std::vector<std::size_t>> &ghostIndices = tess.GetGhostIndeces();
    for(const std::vector<std::size_t> &indices : ghostIndices)
    {
        for(std::size_t index : indices)
        {
            if(index < result.size())
            {
                result[index] = true;
            }
        }
    }
    return result;
}

// A periodic image must carry the centroid of its pre-image shifted by the periodic translation.
// On a uniform Cartesian mesh every centroid coincides with its generating point, so an image whose
// centroid drifts by more than a fraction of a cell was never filled in from its pre-image.
double InferShift(double coordinate, double lower, double upper, bool periodic)
{
    if(!periodic)
    {
        return 0.0;
    }
    if(coordinate < lower)
    {
        return -(upper - lower);
    }
    if(coordinate >= upper)
    {
        return upper - lower;
    }
    return 0.0;
}

void AuditPeriodicImageCentroids(const Tessellation3D &tess, const Vector3D &lowerLeft,
                                 const Vector3D &upperRight, bool periodicX, bool periodicY,
                                 bool periodicZ, double tolerance, unsigned long long &checks,
                                 int &failures, int &resolutionFailures)
{
    const std::size_t ownedCells = tess.GetPointNo();
    const std::vector<bool> ghostMask = BuildGhostMask(tess);
    for(std::size_t faceIndex = 0; faceIndex < tess.GetTotalFacesNumber(); ++faceIndex)
    {
        const std::pair<std::size_t, std::size_t> &neighbors = tess.GetFaceNeighbors(faceIndex);
        const bool firstOwned = neighbors.first < ownedCells;
        const bool secondOwned = neighbors.second < ownedCells;
        if(firstOwned == secondOwned)
        {
            continue;
        }
        const std::size_t extended = firstOwned ? neighbors.second : neighbors.first;
        if(!tess.IsPeriodicImage(extended))
        {
            continue;
        }
        ++checks;
        if(!(Distance(tess.GetCellCM(extended), tess.GetMeshPoint(extended)) <= tolerance))
        {
            ++failures;
        }

        // Images that arrive as MPI ghosts carry their own exchanged state, and neither LinearGauss3D
        // nor the centroid update consults the resolver for them. Only images that have to be traced
        // back to a local pre-image matter here: if that pre-image is the wrong cell, both the state
        // and the centroid of the image come from an unrelated cell.
        if(extended < ghostMask.size() && ghostMask[extended])
        {
            continue;
        }
        const std::size_t physical = tess.ResolvePeriodicImageIndex(extended);
        if(physical < ownedCells)
        {
            const Vector3D image = tess.GetMeshPoint(extended);
            Vector3D untranslated = image;
            untranslated.x -= InferShift(image.x, lowerLeft.x, upperRight.x, periodicX);
            untranslated.y -= InferShift(image.y, lowerLeft.y, upperRight.y, periodicY);
            untranslated.z -= InferShift(image.z, lowerLeft.z, upperRight.z, periodicZ);
            if(!(Distance(tess.GetMeshPoint(physical), untranslated) <= tolerance))
            {
                ++resolutionFailures;
            }
        }
    }
}

std::size_t ExpectedStateIndex(const Tessellation3D &tess, std::size_t index,
                               const std::vector<bool> &ghostMask)
{
    const std::size_t ownedCells = tess.GetPointNo();
    if(index < ownedCells || (index < ghostMask.size() && ghostMask[index]))
    {
        return index;
    }
    if(tess.IsPeriodicImage(index))
    {
        const std::size_t physical = tess.ResolvePeriodicImageIndex(index);
        if(physical < ownedCells)
        {
            return physical;
        }
    }
    return index;
}

} // namespace

int main()
{
    MPI_Init(nullptr, nullptr);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int localFailures = 0;
    int localCentroidFailures = 0;
    int localResolutionFailures = 0;
    unsigned long long localExtendedFaces = 0;
    unsigned long long localMPIGhostFaces = 0;
    unsigned long long localPeriodicAliasFaces = 0;
    unsigned long long localCentroidChecks = 0;

    try
    {
        const Vector3D lowerLeft(0.0, 0.0, 0.0);
        const Vector3D upperRight(1.0, 1.0, 1.0);
        const double cellSpacing = 0.25;
        std::vector<Vector3D> points;
        if(rank == 0)
        {
            points = CartesianMesh(4, 4, 4, lowerLeft, upperRight);
        }
        points = MPI_Spread(points, 0, MPI_COMM_WORLD);

        Voronoi3D tess(lowerLeft, upperRight);
        tess.SetPeriodic(true, true, true);
        tess.BuildParallel(points);

        const std::size_t ownedCells = tess.GetPointNo();
        std::vector<ComputationalCell3D> cells(ownedCells);
        for(std::size_t i = 0; i < ownedCells; ++i)
        {
            cells[i].density = 1.0 + 0.01 * static_cast<double>(CellID(tess.GetMeshPoint(i)));
            cells[i].pressure = 2.0;
            cells[i].internal_energy = 3.0;
            cells[i].velocity = Vector3D();
            cells[i].ID = CellID(tess.GetMeshPoint(i));
        }
        MPI_exchange_data(tess, cells, true);

        const std::vector<bool> ghostMask = BuildGhostMask(tess);
        IdealGas eos(5.0 / 3.0);
        RigidWallGenerator3D ghost;
        LinearGauss3D interpolation(eos, ghost, false, 0.2, 0.5, 0.7, false,
                                    std::vector<std::string>(), "", false);
        std::vector<std::pair<ComputationalCell3D, ComputationalCell3D>> faceValues;
        interpolation(tess, cells, 0.0, faceValues);

        for(std::size_t faceIndex = 0; faceIndex < tess.GetTotalFacesNumber(); ++faceIndex)
        {
            const std::pair<std::size_t, std::size_t> &neighbors =
                tess.GetFaceNeighbors(faceIndex);
            const bool firstOwned = neighbors.first < ownedCells;
            const bool secondOwned = neighbors.second < ownedCells;
            if(firstOwned == secondOwned)
            {
                continue;
            }

            ++localExtendedFaces;
            const std::size_t extended = firstOwned ? neighbors.second : neighbors.first;
            if(tess.BoundaryFace(faceIndex))
            {
                ++localFailures;
                continue;
            }

            if(extended < ghostMask.size() && ghostMask[extended])
            {
                ++localMPIGhostFaces;
            }
            else if(tess.IsPeriodicImage(extended))
            {
                ++localPeriodicAliasFaces;
            }

            const std::size_t expectedIndex = ExpectedStateIndex(tess, extended, ghostMask);
            if(expectedIndex >= cells.size())
            {
                ++localFailures;
                continue;
            }
            const ComputationalCell3D &extendedValue =
                firstOwned ? faceValues[faceIndex].second : faceValues[faceIndex].first;
            if(extendedValue.ID != cells[expectedIndex].ID)
            {
                ++localFailures;
            }
        }

        AuditPeriodicImageCentroids(tess, lowerLeft, upperRight, true, true, true,
                                    0.5 * cellSpacing, localCentroidChecks, localCentroidFailures,
                                    localResolutionFailures);

        // KRTI-like slab: periodic in x and y, rigid in z, and only one cell across y, so that a cell
        // is its own periodic neighbour. The slab is additionally rebalanced, because repartitioning
        // rebuilds local periodic images through a different path than the initial build.
        const double slabLength = 0.5 * M_PI;
        const Vector3D slabLowerLeft(0.0, 0.0, -1.0);
        const Vector3D slabUpperRight(slabLength, slabLength, 1.0);
        const std::size_t slabNx = 16;
        const std::size_t slabNz = 20;
        std::vector<Vector3D> slabPoints;
        if(rank == 0)
        {
            slabPoints = CartesianMesh(slabNx, 1, slabNz, slabLowerLeft, slabUpperRight);
        }
        slabPoints = MPI_Spread(slabPoints, 0, MPI_COMM_WORLD);

        Voronoi3D slabTess(slabLowerLeft, slabUpperRight);
        slabTess.SetPeriodic(true, true, false);
        slabTess.BuildParallel(slabPoints);

        const double slabSpacing = std::min(slabLength / static_cast<double>(slabNx),
                                            2.0 / static_cast<double>(slabNz));
        AuditPeriodicImageCentroids(slabTess, slabLowerLeft, slabUpperRight, true, true, false,
                                    0.5 * slabSpacing, localCentroidChecks, localCentroidFailures,
                                    localResolutionFailures);

        slabTess.Rebalance(std::vector<double>(slabTess.GetAllPointsNo(), 1.0));
        AuditPeriodicImageCentroids(slabTess, slabLowerLeft, slabUpperRight, true, true, false,
                                    0.5 * slabSpacing, localCentroidChecks, localCentroidFailures,
                                    localResolutionFailures);
    }
    catch(const UniversalError &error)
    {
        reportError(error);
        ++localFailures;
    }
    catch(const std::exception &error)
    {
        std::cerr << "linear_gauss_periodic_mpi exception: " << error.what() << std::endl;
        ++localFailures;
    }

    int globalFailures = 0;
    int globalCentroidFailures = 0;
    int globalResolutionFailures = 0;
    unsigned long long globalExtendedFaces = 0;
    unsigned long long globalMPIGhostFaces = 0;
    unsigned long long globalPeriodicAliasFaces = 0;
    unsigned long long globalCentroidChecks = 0;
    MPI_Allreduce(&localFailures, &globalFailures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localCentroidFailures, &globalCentroidFailures, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&localResolutionFailures, &globalResolutionFailures, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&localExtendedFaces, &globalExtendedFaces, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localMPIGhostFaces, &globalMPIGhostFaces, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localPeriodicAliasFaces, &globalPeriodicAliasFaces, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localCentroidChecks, &globalCentroidChecks, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);

    const bool passed = globalFailures == 0 && globalCentroidFailures == 0 &&
                        globalResolutionFailures == 0 && globalExtendedFaces > 0 &&
                        globalMPIGhostFaces > 0 && globalPeriodicAliasFaces > 0 &&
                        globalCentroidChecks > 0;
    if(rank == 0)
    {
        std::cout << "linear_gauss_periodic_mpi PASS=" << (passed ? 1 : 0)
                  << " failures=" << globalFailures
                  << " centroid_failures=" << globalCentroidFailures
                  << " resolution_failures=" << globalResolutionFailures
                  << " extended_faces=" << globalExtendedFaces
                  << " mpi_ghost_faces=" << globalMPIGhostFaces
                  << " periodic_alias_faces=" << globalPeriodicAliasFaces
                  << " centroid_checks=" << globalCentroidChecks
                  << std::endl;
        std::ofstream metrics("linear_gauss_periodic_mpi_metrics.txt");
        metrics << "pass " << (passed ? 1 : 0) << '\n'
                << "failures " << globalFailures << '\n'
                << "centroid_failures " << globalCentroidFailures << '\n'
                << "resolution_failures " << globalResolutionFailures << '\n'
                << "extended_faces " << globalExtendedFaces << '\n'
                << "mpi_ghost_faces " << globalMPIGhostFaces << '\n'
                << "periodic_alias_faces " << globalPeriodicAliasFaces << '\n'
                << "centroid_checks " << globalCentroidChecks << '\n';
    }

    MPI_Finalize();
    return passed ? 0 : 1;
}
