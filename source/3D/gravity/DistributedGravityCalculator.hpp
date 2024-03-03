#ifndef DISTRIUBTED_GRAVITY_CALCULATOR_HPP
#define DISTRIUBTED_GRAVITY_CALCULATOR_HPP

#include "3D/tesselation/Tessellation3D.hpp"
#include "3D/environment/hilbert/DistributedOctEnvAgent.hpp"
#include "3D/hilbert/rectangular/HilbertTree3D.hpp"
#include "ds/DistributedOctTree/DistributedOctTree.hpp"
#include "DistributedGravityTree.hpp"
#include "mpi/mpi_commands.hpp"
#include "GravityTree.hpp"
#include "GravityTypes.h"

class DistributedGravityCalculator
{
public:
    using DistributedOctTree_Type = DistributedOctEnvironmentAgent::DistributedOctTree_Type;
    using HilbertTree_Type = HilbertTreeEnvironmentAgent::HilbertTree_Type;
    using DistributedOctTreeNode_Type = DistributedOctTree_Type::DistributedOctTreeNode;
    using HilbertTreeNode_Type = HilbertTree_Type::Node;

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
        massedPoints.emplace_back(MassedPoint<Vector3D>(this->tess.GetMeshPoint(pointIdx), masses_[pointIdx]));
    }
    gravTree->build(massedPoints);
    this->gravityTree = gravTree;
    
    this->distributedGravityTree = new DistributedGravityTree(this->gravityTree, tess_, this->theta, this->quadrupole, DEFAULT_OWNER_SPLIT, this->comm);

    // const EnvironmentAgent *envAgent = tess_.GetEnvironmentAgent();
    // const DistributedOctEnvironmentAgent *distributedOctEnvAgent = dynamic_cast<const DistributedOctEnvironmentAgent*>(const_cast<EnvironmentAgent*>(envAgent));
    // if(distributedOctEnvAgent != nullptr)
    // {
    //     this->distributedOctTree = distributedOctEnvAgent->getOctTree();
    //     this->boundingBoxesOfRanks = this->distributedOctTree->getBoundingBoxesOfRanks();
    // }

    // const HilbertTreeEnvironmentAgent *hilbertEnvAgent = dynamic_cast<const HilbertTreeEnvironmentAgent*>(const_cast<EnvironmentAgent*>(envAgent));
    // if(hilbertEnvAgent != nullptr)
    // {
    //     this->hilbertTree = hilbertEnvAgent->getHilbertTree();
    //     this->boundingBoxesOfRanks = this->hilbertTree->getBoundingBoxesOfRanks();
    // }

}

// void DistributedGravityCalculator::calculateExchangeListHelper(const GravityTree<Vector3D>::Node *node, boost::container::flat_set<int> &relevantRanks, std::vector<std::vector<MassedValue>> &list) const
// {
//     if(node == nullptr)
//     {
//         return;
//     }

//     const Vector3D &CM = node->value.CM; 
//     // check who from the relevant ranks doesn't want the node
    
//     // TODO: use the bounding boxes CM of the other remote box
//     auto shouldOpen = [&CM, &localBoundingBox=node->boundingBox, isLeaf = node->isLeaf, theta2 = this->thetaSquared](const _BoundingBox<Vector3D> &remoteBox)
//                       {
//                           if(isLeaf) return false;
//                           Vector3D closestPoint = remoteBox.closestPoint(CM);
//                           return ((closestPoint == CM) /* inside the box */ or ShouldOpenBox(closestPoint, localBoundingBox, CM, theta2) /* outside the box, yet should be opened */);
//                       };

//     boost::container::flat_set<int> newRelevantRanks;

//     for(int _rank : relevantRanks)
//     {
//         bool wantToOpen = std::any_of(this->boundingBoxesOfRanks[_rank].begin(), this->boundingBoxesOfRanks[_rank].end(), shouldOpen);

//         // if doesn't want to open, send the current node value to the rank and remove it from the relevant ranks list
//         if(wantToOpen)
//         {
//             newRelevantRanks.insert(_rank);
//         }
//         else
//         {
//             list[_rank].push_back(node->value);
//         }
//     }

//     newRelevantRanks.swap(relevantRanks);
//     // recursive call to children
//     if(not relevantRanks.empty())
//     {
//         for(const GravityTree<Vector3D>::Node *child : node->children)
//         {
//             this->calculateExchangeListHelper(child, relevantRanks, list);
//         }
//     }
//     newRelevantRanks.swap(relevantRanks);
// }

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

    // multiply with the mass of each point
    for(size_t pointIdx = 0; pointIdx < N; pointIdx++)
    {
        results[pointIdx] *= masses[pointIdx];
    }
    return results;
}

#endif // DISTRIUBTED_GRAVITY_CALCULATOR_HPP