#ifndef DISTRIUBTED_GRAVITY_CALCULATOR_HPP
#define DISTRIUBTED_GRAVITY_CALCULATOR_HPP

#ifdef RICH_MPI
#include "3D/tessellation/Tessellation3D.hpp"
#include "DistributedGravityTree.hpp"
#include "mpi/mpi_commands.hpp"
#include "GravityTree.hpp"
#include "GravityTypes.h"

class DistributedGravityCalculator
{
public:
    DistributedGravityCalculator(const Tessellation3D &tess_, const std::vector<gravity_result_t> &masses_, double theta_, bool quadrupole_ = false, const MPI_Comm &comm_ = MPI_COMM_WORLD);

    std::vector<Vector3D> getAcceleration(const std::vector<Vector3D> &points) const;

    inline ~DistributedGravityCalculator()
    {
        delete this->gravityTree;
    }

private:
    using LocalNode = typename GravityTree<Vector3D>::Node;

    MPI_Comm comm;
    int rank, size;
    const Tessellation3D &tess;
    double theta;
    double thetaSquared;
    bool quadrupole;
    GravityTree<Vector3D> *gravityTree;
    std::vector<std::vector<GravityNodeData>> boundingBoxesOfRanks;
    mutable std::vector<boost::container::flat_set<int>> relevantRanksByDepths; // used in the recursion. in the i`th vector, saves the relevant ranks for level i in the tree. 
    mutable EnvironmentAgent<Vector3D>::RanksSet tempRanks;
    const GravityTree<Vector3D>::Node *realRootOfGravityTree;

    std::vector<std::vector<GravityNodeData>> calculateBoundingBoxesOfRanks(const Tessellation3D &tess) const;

    void getSendListHelper(const LocalNode *localNode, std::vector<std::vector<MassedValue<Vector3D>>> &result, int depth) const;
    
    inline std::vector<std::vector<MassedValue<Vector3D>>> getSendList() const
    {
        std::vector<std::vector<MassedValue<Vector3D>>> result(this->size);
        this->getSendListHelper(this->realRootOfGravityTree, result, 0);
        return result;
    }
};

DistributedGravityCalculator::DistributedGravityCalculator(const Tessellation3D &tess_, const std::vector<gravity_result_t> &masses_, double theta_, bool quadrupole_, const MPI_Comm &comm_):
    tess(tess_), theta(theta_), thetaSquared(theta_ * theta_), quadrupole(quadrupole_), comm(comm_)
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
    
    this->relevantRanksByDepths.resize(this->gravityTree->getOctTree()->getDepth() + 1);
    for(int _rank = 0; _rank < this->size; _rank++)
    {
        if(_rank != this->rank)
        {
            this->relevantRanksByDepths[0].insert(_rank);
        }
    }

    // Currently, the gravity tree is a oct-tree of the entire space. However, this processor only controls over a subspace, so most of the children are null.
    // We need to find the "real root", which is the first node that has more than 1 child.
    this->realRootOfGravityTree = this->gravityTree->getOctTree()->getRoot();   
    while(true)
    {
        const LocalNode *nonNullChild = nullptr;
        bool severalChildren = false;
        for(const LocalNode *child : this->realRootOfGravityTree->children)
        {
            if(child == nullptr)
            {
                continue;
            }
            if(nonNullChild == nullptr)
            {
                nonNullChild = child;
            }
            else
            {
                // found!
                severalChildren = true;
                break;
            }

        }
        if(severalChildren or (nonNullChild == nullptr))
        {
            break;
        }
        this->realRootOfGravityTree = nonNullChild;
    }
    this->boundingBoxesOfRanks = this->calculateBoundingBoxesOfRanks(this->tess);
}

std::vector<std::vector<GravityNodeData>> DistributedGravityCalculator::calculateBoundingBoxesOfRanks(const Tessellation3D &tess) const
{
    // first, find my LL and UR
    Vector3D myLL(std::numeric_limits<double>::max()), myUR(std::numeric_limits<double>::lowest());
    size_t N = tess.GetPointNo();
    for(size_t pointIdx = 0; pointIdx < N; pointIdx++)
    {
        Vector3D point = tess.GetMeshPoint(pointIdx);
        for(int dim = 0; dim < 3; dim++)
        {
            myLL[dim] = std::min(myLL[dim], point[dim]);
            myUR[dim] = std::max(myUR[dim], point[dim]);
        }
    }

    const LocalNode *gravityTreeRoot = this->realRootOfGravityTree;
    GravityNodeData data;
    data.boundingBox = gravityTreeRoot->boundingBox;
    data.CM = gravityTreeRoot->value.CM;
    data.mass = gravityTreeRoot->value.mass;
    data.Q = gravityTreeRoot->value.Q;

    std::vector<GravityNodeData> myData = {data};
    return MPI_All_cast_by_ranks(myData, this->comm);
}

std::vector<Vector3D> DistributedGravityCalculator::getAcceleration(const std::vector<Vector3D> &points) const
{
    // insert relevant nodes
    {
        std::vector<std::vector<MassedValue<Vector3D>>> insertToTreeByRanks;
        {
            std::vector<std::vector<MassedValue<Vector3D>>> sendList = this->getSendList();
            insertToTreeByRanks = MPI_Exchange_all_to_all(sendList, this->comm);
        }

        size_t totalArrived = 0;
        for(int _rank = 0; _rank < this->size; _rank++)
        {
            std::vector<MassedValue<Vector3D>> &rankData = insertToTreeByRanks[_rank];
            totalArrived += rankData.size();
            if(_rank != this->rank)
            {
                this->gravityTree->addExternalValues(rankData);
            }

            // erase memory of `rankData`
            rankData.clear();
            rankData.shrink_to_fit();
        }
        this->gravityTree->calculateMasses(); // recalculate masses of tree (also recalculates the CMs)
    }

    // calculate the results, locally
    std::vector<Vector3D> results;
    for(const Vector3D &point : points)
    {
        results.emplace_back(this->gravityTree->gravity(point));
    }
    return results;
}

void DistributedGravityCalculator::getSendListHelper(const LocalNode *localNode, std::vector<std::vector<MassedValue<Vector3D>>> &result, int depth) const
{
    if(localNode == nullptr)
    {
        return;
    }

    boost::container::flat_set<int> &relevantRanks = this->relevantRanksByDepths[depth];
    
    if(localNode->isLeaf)
    {
        // we need to send this node to the ranks that are relevant
        for(int _rank : relevantRanks)
        {
            result[_rank].emplace_back(localNode->value.CM, localNode->value.mass, localNode->value.Q);
        }
        return;
    }
    else
    {
        this->relevantRanksByDepths[depth + 1].clear();
        bool someoneWantsToOpen = false;

        for(int _rank : relevantRanks)
        {
            // check whether or not the rank `_rank` has a bounding box contained in `localNode`
            bool contained = std::any_of(this->boundingBoxesOfRanks[_rank].begin(), this->boundingBoxesOfRanks[_rank].end(),
                                        [localNode](const GravityNodeData &remote)
                                        {
                                            return remote.boundingBox.contains(localNode->boundingBox); // if local node is contained in `remote`'s bounding box
                                        });
            bool shouldOpen = false;
            if(contained)
            {
                // check if sphere of radius `w/theta` centered at `localNode->value.CM` intersects with rank `_rank`
                if(this->tempRanks.empty()) // may have been already calculated for another `_rank` value
                {
                    double radius = localNode->boundingBox.getWidth() / this->theta;
                    this->tempRanks = std::move(this->tess.GetEnvironmentAgent()->getIntersectingRanks(localNode->value.CM, radius));
                }
                shouldOpen = (this->tempRanks.find(_rank) != this->tempRanks.end()); // if contained in a bounding box of remote, check if the remote really intersects with me
            }
            else
            {
                // if not contained in any bounding box, check if should be opened by the theta rule
                shouldOpen = std::any_of(this->boundingBoxesOfRanks[_rank].begin(), this->boundingBoxesOfRanks[_rank].end(),
                                        [localNode, this](const GravityNodeData &remote)
                                        {
                                            return ShouldOpenBox(localNode->value.CM, localNode->boundingBox, remote.boundingBox.closestPoint(localNode->value.CM), this->thetaSquared);
                                        });
            }

            if(shouldOpen)
            {
                someoneWantsToOpen = true;
                // we should open, add the rank to the recursive list
                this->relevantRanksByDepths[depth + 1].insert(_rank);
            }
            else
            {
                // we need to send this node to this rank
                result[_rank].emplace_back(localNode->value.CM, localNode->value.mass, localNode->value.Q);
            }
        }

        this->tempRanks.clear();

        if(not someoneWantsToOpen)
        {
            return;
        }
        // else, call recursively to each one of my children
        for(size_t i = 0; i < CHILDREN; i++)
        {
            if(localNode->children[i] != nullptr)
            {
                this->getSendListHelper(localNode->children[i], result, depth + 1);
            }
        }
    }
}

#endif // RICH_MPI

#endif // DISTRIUBTED_GRAVITY_CALCULATOR_HPP