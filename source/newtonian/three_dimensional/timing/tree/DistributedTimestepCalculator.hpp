#ifndef DISTRIBUTED_TIMESTEP_CALCULATOR_HPP
#define DISTRIBUTED_TIMESTEP_CALCULATOR_HPP

#ifdef RICH_MPI

#include <mpi.h>
#include "mpi/mpi_commands.hpp"
#include <spatial_ds/DistributedOctTree/DistributedOctTree.hpp>
#include "TimeRequestData.hpp"

#define TIME_POINTS_REQUEST_TAG 606
#define TIME_POINTS_SEND_TAG 607

class DistributedTimestepCalculator
{
public:
    using NodeData = TimingTree<Vector3D>::NodeData;

public:
    DistributedTimestepCalculator(const Vector3D &ll, const Vector3D &ur, const Tessellation3D &tess, std::vector<ComputationalCell3D> &cells, const std::vector<Vector3D> &faceVelocities, const MPI_Comm &comm = MPI_COMM_WORLD);

    DistributedTimestepCalculator(const Vector3D &ll, const Vector3D &ur, const Tessellation3D &tess, std::vector<ComputationalCell3D> &cells, const std::vector<Vector3D> &faceVelocities, TimingTree<Vector3D> *timingTree, const MPI_Comm &comm = MPI_COMM_WORLD);
   
    ~DistributedTimestepCalculator();

    inline std::vector<dt_t> calculateTimes() const
    {
        std::vector<size_t> allPoints(this->tess.GetPointNo());
        std::iota(allPoints.begin(), allPoints.end(), 0);
        return this->calculateTimesForIndices(allPoints);
    }

    std::vector<dt_t> calculateTimesForIndices(const std::vector<size_t> &cellsIndices) const;

private:
    MPI_Comm comm;
    int rank, size;
    TimingTree<Vector3D> *timingTree;
    bool createdTimingTree; // if the timing tree should be deleted at the end (avoid memory leaks)
    const Tessellation3D &tess; // if the timing tree should be deleted at the end
    const std::vector<Vector3D> &faceVelocities;
    std::vector<ComputationalCell3D> &cells;

    std::vector<std::vector<NodeData>> exchangeImportedValues(void) const;
};

#endif // RICH_MPI

#endif // DISTRIBUTED_TIMESTEP_CALCULATOR_HPP