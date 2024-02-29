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
    
    void calculateExchangeListHelper(const GravityTree<Vector3D>::Node *node, boost::container::flat_set<int> &relevantRanks, std::vector<std::vector<MassedValue>> &list) const;
    
    inline std::vector<std::vector<MassedValue>> calculateExchangeList(void) const
    {
        std::vector<std::vector<MassedValue>> list(this->size);
        boost::container::flat_set<int> relevantRanks;
        for(int _rank = 0; _rank < this->size; _rank++)
        {
            if(_rank != this->rank)
            {
                relevantRanks.insert(_rank);
            }
        }
        this->calculateExchangeListHelper(this->gravityTree->getOctTree()->getRoot(), relevantRanks, list);
        return list;
    }

    std::vector<std::vector<MassedValue>> exchangeImportedValues(const std::vector<Vector3D> &points) const;
};

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

void DistributedGravityCalculator::calculateExchangeListHelper(const GravityTree<Vector3D>::Node *node, boost::container::flat_set<int> &relevantRanks, std::vector<std::vector<MassedValue>> &list) const
{
    if(node == nullptr)
    {
        return;
    }

    const Vector3D &CM = node->value.CM; 
    // check who from the relevant ranks doesn't want the node
    auto shouldOpen = [&CM, &localBoundingBox=node->boundingBox, isLeaf = node->isLeaf, theta2 = this->thetaSquared](const _BoundingBox<Vector3D> &box)
                      {
                          if(isLeaf) return false;
                          Vector3D closestPoint = box.closestPoint(CM);
                          return ((closestPoint == CM) /* inside the box */ or ShouldOpenBox(closestPoint, localBoundingBox, CM, theta2) /* outside the box, yet should be opened */);
                      };

    boost::container::flat_set<int> newRelevantRanks;

    for(int _rank : relevantRanks)
    {
        bool wantToOpen = std::any_of(this->boundingBoxesOfRanks[_rank].begin(), this->boundingBoxesOfRanks[_rank].end(), shouldOpen);

        // if doesn't want to open, send the current node value to the rank and remove it from the relevant ranks list
        if(wantToOpen)
        {
            newRelevantRanks.insert(_rank);
        }
        else
        {
            list[_rank].push_back(node->value);
        }
    }

    newRelevantRanks.swap(relevantRanks);
    // recursive call to children
    if(not relevantRanks.empty())
    {
        for(const GravityTree<Vector3D>::Node *child : node->children)
        {
            this->calculateExchangeListHelper(child, relevantRanks, list);
        }
    }
    newRelevantRanks.swap(relevantRanks);
}

std::vector<std::vector<typename DistributedGravityCalculator::MassedValue>> DistributedGravityCalculator::exchangeImportedValues(const std::vector<Vector3D> &points) const
{
    std::vector<std::vector<MassedValue>> list = this->calculateExchangeList();
    std::vector<std::vector<MassedValue>> incomingValues = MPI_Exchange_all_to_all(list, this->comm);
    return incomingValues;
}

std::vector<Vector3D> DistributedGravityCalculator::getAcceleration(const std::vector<Vector3D> &points) const
{
    // if(this->rank == 0)
    // {
    //     this->gravityTree->print();
    //     std::cout << "**********************************************" << std::endl;
    //     this->distributedOctTree->print();
    // }
    // add the external values, to make the tree 'global'
    
    size_t totalAdded = 0;
    for(const std::vector<MassedValue> &values : this->exchangeImportedValues(points))
    {
        totalAdded += values.size();
        // std::cout << "rank " << this->rank << " is inserting " << values.size() << " values: " << values << std::endl;
        this->gravityTree->addExternalValues(values);
    }
    std::cout << "rank " << this->rank << " added " << totalAdded << " values" << std::endl;

    // calculate the results
    std::vector<Vector3D> results;
    for(const Vector3D &point : points)
    {
        results.emplace_back(this->gravityTree->gravity(point));
    }

    // if(this->rank == 0)
    // {
    //     this->gravityTree->print();
    // }
    return results;
}

#endif // DISTRIUBTED_GRAVITY_CALCULATOR_HPP