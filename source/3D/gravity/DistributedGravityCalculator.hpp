#ifndef DISTRIUBTED_GRAVITY_CALCULATOR_HPP
#define DISTRIUBTED_GRAVITY_CALCULATOR_HPP

#include "3D/tesselation/Tessellation3D.hpp"
#include "3D/environment/hilbert/DistributedOctEnvAgent.hpp"
#include "3D/hilbert/rectangular/HilbertTree3D.hpp"
#include "ds/DistributedOctTree/DistributedOctTree.hpp"
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

    using MassedValue = GravityTree<Vector3D>::MassedValue;

    DistributedGravityCalculator(const Tessellation3D &tess_, const std::vector<gravity_result_t> &masses_, double theta_, bool quadrupole_ = false, const MPI_Comm &comm_ = MPI_COMM_WORLD);

    std::vector<Vector3D> getAcceleration(const std::vector<Vector3D> &points) const;

    inline ~DistributedGravityCalculator()
    {
        delete this->gravityTree;
    }

private:
    MPI_Comm comm;
    int rank, size;
    const Tessellation3D &tess;
    double theta;
    double thetaSquared;
    bool quadrupole;
    GravityTree<Vector3D> *gravityTree;
    std::vector<std::vector<_BoundingBox<Vector3D>>> boundingBoxesOfRanks;
    const DistributedOctTree_Type *distributedOctTree;
    const HilbertTree_Type *hilbertTree;
    
    // void calculateExchangeListHelper(const GravityTree<Vector3D>::Node *node, boost::container::flat_set<int> &relevantRanks, std::vector<std::vector<MassedValue>> &list) const;
    
    // void octTreeCalculateExchangeList(const DistributedOctTreeNode_Type *node, const std::vector<Vector3D> &points, std::vector<bool> &consider, std::vector<std::vector<size_t>> &pointsIndicesToRanks) const;

    void octTreeCalculateExchangeList(const DistributedOctTreeNode_Type *node, const Vector3D &point, size_t pointIdx, bool containsPoint, std::vector<boost::container::flat_set<size_t>> &pointsIndicesToRanks) const;

    inline std::vector<boost::container::flat_set<size_t>> calculateExchangeList(const std::vector<Vector3D> &points) const
    {
        std::vector<boost::container::flat_set<size_t>> pointsIndicesToRanks(this->size);
        size_t N = points.size();
        for(size_t pointIdx = 0; pointIdx < N; pointIdx++)
        {
            const Vector3D &point = points[pointIdx];
            if(this->distributedOctTree != nullptr)
            {
                this->octTreeCalculateExchangeList(this->distributedOctTree->getOctTree()->getRoot(), point, pointIdx, true, pointsIndicesToRanks);
            }
            else if(this->hilbertTree != nullptr)
            {
                // TODO
            }
        }
        return pointsIndicesToRanks;
    }
};

void DistributedGravityCalculator::octTreeCalculateExchangeList(const DistributedOctTreeNode_Type *node, const Vector3D &point, size_t pointIdx, bool containsPoint, std::vector<boost::container::flat_set<size_t>> &pointsIndicesToRanks) const
{
    if(node == nullptr)
    {
        return;
    }

    if(node->isLeaf)
    {
        for(int owner : node->value.owners)
        {
            if(owner != this->rank)
            {
                pointsIndicesToRanks[owner].insert(pointIdx);
            }
        }
        return;
    }

    const _BoundingBox<Vector3D> &box = node->boundingBox;
    const Vector3D &CM = (box.getLL() + box.getUR()) / 2; // TODO: incorrect!!!!!

    if(containsPoint or ShouldOpenBox(point, box, CM, this->thetaSquared))
    {
        int childContainsPoint = -1;
        if(containsPoint)
        {
            childContainsPoint = node->getChildNumberContaining(point);
        }
        for(size_t i = 0; i < CHILDREN; i++)
        {
            const DistributedOctTreeNode_Type *child = node->children[i];
            if(child == nullptr)
            {
                continue;
            }
            this->octTreeCalculateExchangeList(child, point, pointIdx, i == static_cast<size_t>(childContainsPoint), pointsIndicesToRanks);
        }
    }
}

DistributedGravityCalculator::DistributedGravityCalculator(const Tessellation3D &tess_, const std::vector<gravity_result_t> &masses_, double theta_, bool quadrupole_, const MPI_Comm &comm_):
    tess(tess_), theta(theta_), thetaSquared(theta_ * theta_), quadrupole(quadrupole_), comm(comm_), distributedOctTree(nullptr), hilbertTree(nullptr)
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

    const EnvironmentAgent *envAgent = tess_.GetEnvironmentAgent();
    const DistributedOctEnvironmentAgent *distributedOctEnvAgent = dynamic_cast<const DistributedOctEnvironmentAgent*>(const_cast<EnvironmentAgent*>(envAgent));
    if(distributedOctEnvAgent != nullptr)
    {
        this->distributedOctTree = distributedOctEnvAgent->getOctTree();
        this->boundingBoxesOfRanks = this->distributedOctTree->getBoundingBoxesOfRanks();
    }

    const HilbertTreeEnvironmentAgent *hilbertEnvAgent = dynamic_cast<const HilbertTreeEnvironmentAgent*>(const_cast<EnvironmentAgent*>(envAgent));
    if(hilbertEnvAgent != nullptr)
    {
        this->hilbertTree = hilbertEnvAgent->getHilbertTree();
        this->boundingBoxesOfRanks = this->hilbertTree->getBoundingBoxesOfRanks();
    }
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

std::vector<Vector3D> DistributedGravityCalculator::getAcceleration(const std::vector<Vector3D> &points) const
{
    // calculate the results, locally
    std::vector<Vector3D> results;
    for(const Vector3D &point : points)
    {
        results.emplace_back(this->gravityTree->gravity(point));
    }

    // now, exchange necessary points with other processes
    // first, get a list of what points to send to each rank
    std::vector<boost::container::flat_set<size_t>> pointsIndicesToRanks = this->calculateExchangeList(points);

    // we used the 'set' to avoid duplicates, but now we need to convert it to a vector
    std::vector<std::vector<size_t>> indicesToRanks(this->size);
    for(int _rank = 0; _rank < this->size; _rank++)
    {
        indicesToRanks[_rank].insert(indicesToRanks[_rank].end(), pointsIndicesToRanks[_rank].begin(), pointsIndicesToRanks[_rank].end());
    }

    std::vector<std::vector<Vector3D>> pointsToRanks;
    for(int _rank = 0; _rank < this->size; _rank++)
    {
        pointsToRanks.emplace_back();
        std::transform(indicesToRanks[_rank].cbegin(), indicesToRanks[_rank].cend(),
                        std::back_inserter(pointsToRanks[_rank]), [&points](size_t idx){return points[idx];});
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