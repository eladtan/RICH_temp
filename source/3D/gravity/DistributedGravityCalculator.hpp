#ifndef DISTRIUBTED_GRAVITY_CALCULATOR_HPP
#define DISTRIUBTED_GRAVITY_CALCULATOR_HPP

#include "3D/tesselation/Tessellation3D.hpp"
#include "DistributedGravityTree.hpp"
#include "mpi/mpi_commands.hpp"
#include "GravityTree.hpp"
#include "GravityTypes.h"

class DistributedGravityCalculator
{
public:
    DistributedGravityCalculator(const Tessellation3D &tess_, const std::vector<gravity_result_t> &masses_, double theta_, bool quadrupole_ = false, const MPI_Comm &comm_ = MPI_COMM_WORLD);

    std::vector<Vector3D> getAcceleration(const std::vector<Vector3D> &points, const std::vector<gravity_result_t> &masses) const;

    inline ~DistributedGravityCalculator()
    {
        delete this->gravityTree;
        delete this->distributedGravityTree;
    }

private:

    MPI_Comm comm;
    int rank, size;
    const Tessellation3D &tess;
    double theta;
    double thetaSquared;
    bool quadrupole;
    GravityTree<Vector3D> *gravityTree;
    const DistributedGravityTree *distributedGravityTree;
};

DistributedGravityCalculator::DistributedGravityCalculator(const Tessellation3D &tess_, const std::vector<gravity_result_t> &masses_, double theta_, bool quadrupole_, const MPI_Comm &comm_):
    tess(tess_), theta(theta_), thetaSquared(theta_ * theta_), quadrupole(quadrupole_), comm(comm_), distributedGravityTree(nullptr)
{
    MPI_Comm_size(this->comm, &this->size);
    MPI_Comm_rank(this->comm, &this->rank);
    auto [ll, ur] = this->tess.GetBoxCoordinates();
    GravityTree<Vector3D> *gravTree = new GravityTree<Vector3D>(ll, ur, this->theta, this->quadrupole);
    std::vector<MassedPoint<Vector3D>> massedPoints;
    size_t N = this->tess.GetPointNo();
    massedPoints.reserve(N);
    for(size_t pointIdx = 0; pointIdx < N; pointIdx++)
    {
        massedPoints.emplace_back(MassedPoint<Vector3D>(this->tess.GetCellCM(pointIdx), masses_[pointIdx]));
    }
    gravTree->build(massedPoints);
    this->gravityTree = gravTree;
    
    this->distributedGravityTree = new DistributedGravityTree(this->gravityTree, tess_, this->theta, this->quadrupole, DEFAULT_OWNER_SPLIT, this->comm);
}


std::vector<Vector3D> DistributedGravityCalculator::getAcceleration(const std::vector<Vector3D> &points, const std::vector<gravity_result_t> &masses) const
{
    // calculate the results, locally
    std::vector<Vector3D> results;
    for(const Vector3D &point : points)
    {
        results.emplace_back(this->gravityTree->gravity(point));
    }

    // now, exchange necessary points with other processes

    std::vector<std::vector<size_t>> indicesToRanks(this->size);
    std::vector<std::vector<Vector3D>> pointsToRanks(this->size);

    size_t N = points.size();
    for(size_t pointIdx = 0; pointIdx < N; pointIdx++)
    {
        const Vector3D &point = points[pointIdx];
        auto [gravityResult, ranksToRequest] = this->distributedGravityTree->gravity(point);
        for(int _rank : ranksToRequest)
        {
            if(this->rank != _rank)
            {
                indicesToRanks[_rank].push_back(pointIdx);
                pointsToRanks[_rank].push_back(point);
            }
        }
        results[pointIdx] += gravityResult;
    }

    // exchange the list
    std::vector<std::vector<Vector3D>> incomingPoints = MPI_Exchange_all_to_all(pointsToRanks, this->comm);

    // for each rank, we caluclate the gravity of the points we received from it
    std::vector<std::vector<Vector3D>> resultsForRanks;
    for(int _rank = 0; _rank < this->size; _rank++)
    {
        resultsForRanks.emplace_back();
        std::vector<Vector3D> &res = resultsForRanks.back();
        for(const Vector3D &point : incomingPoints[_rank])
        {
            res.push_back(this->gravityTree->gravity(point));
        } 
    }

    // exchange back the results
    std::vector<std::vector<Vector3D>> incomingResults = MPI_Exchange_all_to_all(resultsForRanks, this->comm);
    for(int _rank = 0; _rank < this->size; _rank++)
    {
        const std::vector<size_t> &indices = indicesToRanks[_rank];
        const std::vector<Vector3D> &rankResult = incomingResults[_rank];
        size_t N = indices.size();
        for(size_t i = 0; i < N; i++)
        {
            const size_t &pointIdx = indices[i];
            results[pointIdx] += rankResult[i];
        }
    }
    return results;
}

#endif // DISTRIUBTED_GRAVITY_CALCULATOR_HPP