#ifndef DISTRIUBTED_GRAVITY_CALCULATOR_HPP
#define DISTRIUBTED_GRAVITY_CALCULATOR_HPP

#ifdef RICH_MPI
#include <limits>
#include <algorithm>
#include <cmath>
#include <memory>
#include "3D/tessellation/Tessellation3D.hpp"
#include "DistributedGravityTree.hpp"
#include "mpi/mpi_commands.hpp"
#include "GravityTree.hpp"
#include "FlatGravityTree.hpp"
#include "GravityTypes.h"

#ifndef SKELETON_DEPTH
#define SKELETON_DEPTH 3
#endif

class DistributedGravityCalculator
{
public:
    DistributedGravityCalculator(const Tessellation3D &tess_, const std::vector<gravity_result_t> &masses_, double theta_, bool quadrupole_ = false, const MPI_Comm &comm_ = MPI_COMM_WORLD);

    DistributedGravityCalculator(const std::vector<Vector3D> &sourcePoints,
                                 const std::vector<gravity_result_t> &masses,
                                 const Vector3D &domainLower,
                                 const Vector3D &domainUpper,
                                 double theta,
                                 bool quadrupole = false,
                                 const MPI_Comm &comm = MPI_COMM_WORLD);

    std::vector<Vector3D> getAcceleration(const std::vector<Vector3D> &points) const;

    double getWalkTime() const { return walkTime_; }

    inline ~DistributedGravityCalculator()
    {
        delete this->gravityTree;
    }

private:
    mutable double walkTime_ = 0;
    using LocalNode = typename GravityTree<Vector3D>::Node;

    MPI_Comm comm;
    int rank, size;
    Vector3D domainLower;
    Vector3D domainUpper;
    std::shared_ptr<EnvironmentAgent> environmentAgent;
    double theta;
    double thetaSquared;
    bool quadrupole;
    GravityTree<Vector3D> *gravityTree;
    std::vector<std::vector<GravityNodeData>> topNodesOfRanks;
    mutable std::vector<boost::container::flat_set<int>> relevantRanksByDepths;
    const GravityTree<Vector3D>::Node *realRootOfGravityTree;
    std::vector<int> prunedRanks;
    boost::container::flat_set<int> prunedSet;
    std::vector<MassedValue<Vector3D>> prunedSummaries;
    std::vector<int> exchangeNeighbors;

    void initialize(const std::vector<MassedPoint<Vector3D>> &massedPoints);
    void validateCollective(bool localInputValid) const;

    void getSendListHelper(const LocalNode *localNode,
                           std::vector<std::vector<MassedValue<Vector3D>>> &result,
                           const boost::container::flat_set<int> &relevantRanks) const;

    std::vector<int> getActiveExchangeNeighbors(const std::vector<std::vector<MassedValue<Vector3D>>> &sendList) const;

    inline std::vector<std::vector<MassedValue<Vector3D>>> getSendList() const
    {
        std::vector<std::vector<MassedValue<Vector3D>>> result(this->size);
        this->getSendListHelper(this->realRootOfGravityTree, result, this->relevantRanksByDepths[0]);
        return result;
    }
};

DistributedGravityCalculator::DistributedGravityCalculator(const Tessellation3D &tess_, const std::vector<gravity_result_t> &masses_, double theta_, bool quadrupole_, const MPI_Comm &comm_):
    comm(comm_), rank(0), size(1), domainLower(), domainUpper(),
    environmentAgent(tess_.GetEnvironmentAgent()), theta(theta_),
    thetaSquared(theta_ * theta_), quadrupole(quadrupole_),
    gravityTree(nullptr), realRootOfGravityTree(nullptr)
{
    MPI_Comm_size(this->comm, &this->size);
    MPI_Comm_rank(this->comm, &this->rank);
    const std::pair<Vector3D, Vector3D> box = tess_.GetBoxCoordinates();
    this->domainLower = box.first;
    this->domainUpper = box.second;
    bool localInputValid = tess_.GetPointNo() == masses_.size() &&
                           tess_.GetPointNo() > 0 &&
                           std::isfinite(theta_) && theta_ > 0.0;
    if(tess_.GetPointNo() == masses_.size())
    {
        for(size_t pointIdx = 0; pointIdx < tess_.GetPointNo(); ++pointIdx)
        {
            const Vector3D &point = tess_.GetCellCM(pointIdx);
            localInputValid = localInputValid &&
                std::isfinite(point.x) && std::isfinite(point.y) &&
                std::isfinite(point.z) && std::isfinite(masses_[pointIdx]) &&
                point.x >= this->domainLower.x &&
                point.x <= this->domainUpper.x &&
                point.y >= this->domainLower.y &&
                point.y <= this->domainUpper.y &&
                point.z >= this->domainLower.z &&
                point.z <= this->domainUpper.z;
        }
    }
    this->validateCollective(localInputValid);

    std::vector<MassedPoint<Vector3D>> massedPoints;
    massedPoints.reserve(tess_.GetPointNo());
    for(size_t pointIdx = 0; pointIdx < tess_.GetPointNo(); ++pointIdx)
        massedPoints.emplace_back(tess_.GetCellCM(pointIdx), masses_[pointIdx]);
    this->initialize(massedPoints);
}

DistributedGravityCalculator::DistributedGravityCalculator(
    const std::vector<Vector3D> &sourcePoints,
    const std::vector<gravity_result_t> &masses,
    const Vector3D &domainLower_,
    const Vector3D &domainUpper_,
    double theta_,
    bool quadrupole_,
    const MPI_Comm &comm_):
    comm(comm_), rank(0), size(1), domainLower(domainLower_),
    domainUpper(domainUpper_), environmentAgent(), theta(theta_),
    thetaSquared(theta_ * theta_), quadrupole(quadrupole_),
    gravityTree(nullptr), realRootOfGravityTree(nullptr)
{
    MPI_Comm_size(this->comm, &this->size);
    MPI_Comm_rank(this->comm, &this->rank);
    bool localInputValid = sourcePoints.size() == masses.size() &&
                           !sourcePoints.empty() &&
                           std::isfinite(theta_) && theta_ > 0.0;
    if(sourcePoints.size() == masses.size())
    {
        for(size_t pointIdx = 0; pointIdx < sourcePoints.size(); ++pointIdx)
        {
            const Vector3D &point = sourcePoints[pointIdx];
            localInputValid = localInputValid &&
                std::isfinite(point.x) && std::isfinite(point.y) &&
                std::isfinite(point.z) && std::isfinite(masses[pointIdx]) &&
                point.x >= this->domainLower.x &&
                point.x <= this->domainUpper.x &&
                point.y >= this->domainLower.y &&
                point.y <= this->domainUpper.y &&
                point.z >= this->domainLower.z &&
                point.z <= this->domainUpper.z;
        }
    }
    this->validateCollective(localInputValid);

    std::vector<MassedPoint<Vector3D>> massedPoints;
    massedPoints.reserve(sourcePoints.size());
    for(size_t pointIdx = 0; pointIdx < sourcePoints.size(); ++pointIdx)
        massedPoints.emplace_back(sourcePoints[pointIdx], masses[pointIdx]);
    this->initialize(massedPoints);
}

void DistributedGravityCalculator::validateCollective(
    bool localInputValid) const
{
    const bool localDomainValid =
        std::isfinite(this->domainLower.x) &&
        std::isfinite(this->domainLower.y) &&
        std::isfinite(this->domainLower.z) &&
        std::isfinite(this->domainUpper.x) &&
        std::isfinite(this->domainUpper.y) &&
        std::isfinite(this->domainUpper.z) &&
        this->domainLower.x < this->domainUpper.x &&
        this->domainLower.y < this->domainUpper.y &&
        this->domainLower.z < this->domainUpper.z;
    int localValidInt = localInputValid && localDomainValid ? 1 : 0;
    int globalValidInt = 0;
    MPI_Allreduce(&localValidInt, &globalValidInt, 1, MPI_INT, MPI_LAND,
                  this->comm);
    if(globalValidInt == 0)
        throw UniversalError(
            "DistributedGravityCalculator: invalid input on at least one rank");

    const double localDomain[6] = {
        this->domainLower.x, this->domainLower.y, this->domainLower.z,
        this->domainUpper.x, this->domainUpper.y, this->domainUpper.z};
    double minimumDomain[6] = {};
    double maximumDomain[6] = {};
    MPI_Allreduce(localDomain, minimumDomain, 6, MPI_DOUBLE, MPI_MIN,
                  this->comm);
    MPI_Allreduce(localDomain, maximumDomain, 6, MPI_DOUBLE, MPI_MAX,
                  this->comm);
    for(int i = 0; i < 6; ++i)
    {
        if(minimumDomain[i] != maximumDomain[i])
            throw UniversalError(
                "DistributedGravityCalculator: domain differs across MPI ranks");
    }
}

void DistributedGravityCalculator::initialize(
    const std::vector<MassedPoint<Vector3D>> &massedPoints)
{
    std::unique_ptr<GravityTree<Vector3D>> gravTree;
    int localBuildSucceeded = 1;
    try
    {
        gravTree.reset(new GravityTree<Vector3D>(
            this->domainLower, this->domainUpper,
            this->theta, this->quadrupole));
        gravTree->build(massedPoints);
    }
    catch(...)
    {
        localBuildSucceeded = 0;
    }
    int globalBuildSucceeded = 0;
    MPI_Allreduce(&localBuildSucceeded, &globalBuildSucceeded, 1, MPI_INT,
                  MPI_LAND, this->comm);
    if(globalBuildSucceeded == 0)
        throw UniversalError(
            "DistributedGravityCalculator: local tree construction failed on at least one rank");

    this->gravityTree = gravTree.get();

    // Find the "real root": the first node with more than 1 child
    this->realRootOfGravityTree = this->gravityTree->getOctTree()->getRoot();   
    while(true)
    {
        const LocalNode *nonNullChild = nullptr;
        bool severalChildren = false;
        for(const LocalNode *child : this->realRootOfGravityTree->children)
        {
            if(child == nullptr)
                continue;
            if(nonNullChild == nullptr)
                nonNullChild = child;
            else
            {
                severalChildren = true;
                break;
            }
        }
        if(severalChildren or (nonNullChild == nullptr))
            break;
        this->realRootOfGravityTree = nonNullChild;
    }

    // Phase 2: Collect real root + children, Allgather across all ranks.
    // Entry [0] is the root; entries [1..] are its non-null children.
    std::vector<GravityNodeData> myTopNodes;
    {
        GravityNodeData rootData;
        rootData.boundingBox = this->realRootOfGravityTree->boundingBox;
        rootData.CM = this->realRootOfGravityTree->value.CM;
        rootData.mass = this->realRootOfGravityTree->value.mass;
        rootData.Q = this->realRootOfGravityTree->value.Q;
        myTopNodes.push_back(rootData);

        if(!this->realRootOfGravityTree->isLeaf)
        {
            for(size_t i = 0; i < CHILDREN; i++)
            {
                const LocalNode *child = this->realRootOfGravityTree->children[i];
                if(child == nullptr)
                    continue;
                GravityNodeData childData;
                childData.boundingBox = child->boundingBox;
                childData.CM = child->value.CM;
                childData.mass = child->value.mass;
                childData.Q = child->value.Q;
                myTopNodes.push_back(childData);
            }
        }
    }
    this->topNodesOfRanks = MPI_All_cast_by_ranks(myTopNodes, this->comm);

    // Phase 3: Pruning decision. Walk local tree top-down to SKELETON_DEPTH
    // against each remote rank's allgathered top nodes.
    this->relevantRanksByDepths.resize(this->gravityTree->getOctTree()->getDepth() + 1);

    for(int _rank = 0; _rank < this->size; _rank++)
    {
        if(_rank == this->rank)
            continue;

        bool needsExchange = false;
        std::vector<std::pair<const LocalNode*, int>> nodeStack;
        nodeStack.push_back({this->realRootOfGravityTree, 0});

        while(!nodeStack.empty() && !needsExchange)
        {
            auto [node, depth] = nodeStack.back();
            nodeStack.pop_back();
            if(node == nullptr)
                continue;

            bool anyOpens = false;
            const std::vector<GravityNodeData> &remoteNodes = this->topNodesOfRanks[_rank];
            size_t firstRemoteNode = remoteNodes.size() > 1 ? 1 : 0;
            for(size_t ri = firstRemoteNode; ri < remoteNodes.size(); ri++)
            {
                const GravityNodeData &remote = remoteNodes[ri];
                if(remote.boundingBox.contains(node->boundingBox))
                {
                    anyOpens = true;
                    break;
                }
                Vector3D cp = remote.boundingBox.closestPoint(node->value.CM);
                if(ShouldOpenBox(node->value.CM, node->boundingBox, cp, this->thetaSquared))
                {
                    anyOpens = true;
                    break;
                }
            }

            if(!anyOpens)
                continue;

            if(depth >= SKELETON_DEPTH || node->isLeaf)
            {
                needsExchange = true;
            }
            else
            {
                for(size_t i = 0; i < CHILDREN; i++)
                    nodeStack.push_back({node->children[i], depth + 1});
            }
        }

        if(needsExchange)
        {
            this->relevantRanksByDepths[0].insert(_rank);
        }
        else
        {
            this->prunedRanks.push_back(_rank);
            const std::vector<GravityNodeData> &remoteNodes = this->topNodesOfRanks[_rank];
            if(remoteNodes.size() > 1)
            {
                for(size_t i = 1; i < remoteNodes.size(); i++)
                    this->prunedSummaries.emplace_back(remoteNodes[i].CM, remoteNodes[i].mass, remoteNodes[i].Q);
            }
            else if(!remoteNodes.empty())
            {
                this->prunedSummaries.emplace_back(remoteNodes[0].CM, remoteNodes[0].mass, remoteNodes[0].Q);
            }
        }
    }

    // Phase 4: Pruning flag exchange. Tell each rank whether we pruned it
    // so it can stop sending us detailed data and fall back to allgathered
    // coarse data for our contribution instead.
    {
        std::vector<int> sendFlags(this->size, 0);
        for(int r : this->prunedRanks)
            sendFlags[r] = 1;

        std::vector<int> recvFlags(this->size, 0);
        MPI_Alltoall(sendFlags.data(), 1, MPI_INT,
                     recvFlags.data(), 1, MPI_INT, this->comm);

        boost::container::flat_set<int> alreadyPruned(this->prunedRanks.begin(),
                                                       this->prunedRanks.end());
        for(int r = 0; r < this->size; r++)
        {
            if(recvFlags[r] == 0 || r == this->rank)
                continue;
            if(alreadyPruned.find(r) != alreadyPruned.end())
                continue;

            this->relevantRanksByDepths[0].erase(r);
            this->prunedRanks.push_back(r);

            const std::vector<GravityNodeData> &remoteNodes = this->topNodesOfRanks[r];
            if(remoteNodes.size() > 1)
            {
                for(size_t i = 1; i < remoteNodes.size(); i++)
                    this->prunedSummaries.emplace_back(remoteNodes[i].CM, remoteNodes[i].mass, remoteNodes[i].Q);
            }
            else if(!remoteNodes.empty())
            {
                this->prunedSummaries.emplace_back(remoteNodes[0].CM, remoteNodes[0].mass, remoteNodes[0].Q);
            }
        }

        this->prunedSet = boost::container::flat_set<int>(this->prunedRanks.begin(),
                                                           this->prunedRanks.end());
    }

    this->exchangeNeighbors.assign(this->relevantRanksByDepths[0].begin(),
                                    this->relevantRanksByDepths[0].end());
    gravTree.release();
}


std::vector<Vector3D> DistributedGravityCalculator::getAcceleration(const std::vector<Vector3D> &points) const
{
    std::vector<std::vector<MassedValue<Vector3D>>> sendList = this->getSendList();
    std::vector<int> activeExchangeNeighbors = this->getActiveExchangeNeighbors(sendList);

    // Stage 1: Flat-pack + post Isend/Irecv for counts (returns immediately)
    FlatSparseHandle flatHandle = MPI_flat_sparse_pack_and_post_counts(sendList, activeExchangeNeighbors, this->comm);

    sendList.clear();
    sendList.shrink_to_fit();

    // Stage 2 (overlapped with count exchange):
    // addPruned + calculateMasses, then compile + local walk
    this->gravityTree->addExternalValues(this->prunedSummaries);
    this->gravityTree->calculateMasses();

    double walk_t0 = MPI_Wtime();
    FlatGravityTree localFlat(*this->gravityTree);
    std::vector<Vector3D> results;
    results.reserve(points.size());
    for(size_t i = 0; i < points.size(); i++)
    {
        results.emplace_back(localFlat.gravity(points[i], true));
    }
    this->walkTime_ = MPI_Wtime() - walk_t0;

    // Stage 3: Waitall on counts + post payload sends/recvs
    // Stage 4: Waitall on payload + flat-unpack
    MPI_flat_sparse_post_payload(flatHandle, this->comm);
    std::vector<std::vector<MassedValue<Vector3D>>> insertToTreeByRanks =
        MPI_flat_sparse_wait<MassedValue<Vector3D>>(flatHandle);

    // Build temporary tree from received data (non-pruned ranks only)
    GravityTree<Vector3D> remoteTree(this->domainLower, this->domainUpper,
                                     this->theta, this->quadrupole);

    for(int _rank = 0; _rank < this->size; _rank++)
    {
        std::vector<MassedValue<Vector3D>> &rankData = insertToTreeByRanks[_rank];
        if(_rank != this->rank && this->prunedSet.find(_rank) == this->prunedSet.end())
        {
            remoteTree.addExternalValues(rankData);
        }
        rankData.clear();
    }
    insertToTreeByRanks.clear();
    insertToTreeByRanks.shrink_to_fit();
    remoteTree.calculateMasses();

    // Compile remote tree to flat array and walk
    double rwalk_t0 = MPI_Wtime();
    FlatGravityTree remoteFlat(remoteTree);
    for(size_t i = 0; i < points.size(); i++)
    {
        results[i] += remoteFlat.gravity(points[i], true);
    }
    this->walkTime_ += MPI_Wtime() - rwalk_t0;

    return results;
}

std::vector<int> DistributedGravityCalculator::getActiveExchangeNeighbors(
    const std::vector<std::vector<MassedValue<Vector3D>>> &sendList) const
{
    std::vector<int> sendFlags(this->size, 0);
    std::vector<int> recvFlags(this->size, 0);

    for(int r : this->exchangeNeighbors)
    {
        if(r != this->rank && !sendList[r].empty())
            sendFlags[r] = 1;
    }

    MPI_Alltoall(sendFlags.data(), 1, MPI_INT,
                 recvFlags.data(), 1, MPI_INT, this->comm);

    std::vector<int> result;
    result.reserve(this->exchangeNeighbors.size());
    for(int r : this->exchangeNeighbors)
    {
        if(r != this->rank && (sendFlags[r] != 0 || recvFlags[r] != 0))
            result.push_back(r);
    }
    return result;
}

void DistributedGravityCalculator::getSendListHelper(const LocalNode *localNode,
    std::vector<std::vector<MassedValue<Vector3D>>> &result,
    const boost::container::flat_set<int> &relevantRanks) const
{
    if(localNode == nullptr)
        return;
    if(relevantRanks.empty())
    {
        return;
    }

    if(localNode->isLeaf)
    {
        for(int _rank : relevantRanks)
            result[_rank].emplace_back(localNode->value.CM, localNode->value.mass, localNode->value.Q);
        return;
    }

    boost::container::flat_set<int> ranksToOpen;
    EnvironmentAgent::RanksSet tempRanks;
    bool tempRanksReady = false;

    for(int _rank : relevantRanks)
    {
        bool contained = this->topNodesOfRanks[_rank][0].boundingBox.contains(localNode->boundingBox);
        bool shouldOpen = false;
        if(contained)
        {
            if(!this->environmentAgent)
            {
                // Vector-only users do not have a domain-intersection agent.
                // Open conservatively; the geometric test below may still
                // collapse the node to a single multipole summary.
                shouldOpen = true;
            }
            else
            {
                if(!tempRanksReady)
                {
                    double radius = localNode->boundingBox.getWidth() /
                                    this->theta;
                    tempRanks = std::move(this->environmentAgent->
                        getIntersectingRanks(localNode->value.CM, radius));
                    tempRanksReady = true;
                }
                shouldOpen = (tempRanks.find(_rank) != tempRanks.end());
            }
        }
        else
        {
            const std::vector<GravityNodeData> &remoteNodes = this->topNodesOfRanks[_rank];
            size_t firstRemoteNode = remoteNodes.size() > 1 ? 1 : 0;
            Vector3D closestRemotePoint;
            double minDist2 = std::numeric_limits<double>::max();
            for(size_t ri = firstRemoteNode; ri < remoteNodes.size(); ri++)
            {
                Vector3D cp = remoteNodes[ri].boundingBox.closestPoint(localNode->value.CM);
                Vector3D diff = cp - localNode->value.CM;
                double d2 = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                if(d2 < minDist2)
                {
                    minDist2 = d2;
                    closestRemotePoint = cp;
                }
            }
            shouldOpen = minDist2 < std::numeric_limits<double>::max() &&
                ShouldOpenBox(localNode->value.CM, localNode->boundingBox,
                    closestRemotePoint, this->thetaSquared);
        }

        if(shouldOpen)
        {
            if(contained)
            {
                const std::vector<GravityNodeData> &remoteNodes = this->topNodesOfRanks[_rank];
                size_t firstRemoteNode = remoteNodes.size() > 1 ? 1 : 0;
                double minDist2 = std::numeric_limits<double>::max();
                for(size_t ri = firstRemoteNode; ri < remoteNodes.size(); ri++)
                {
                    double d2 = remoteNodes[ri].boundingBox.distanceSquared(localNode->value.CM);
                    if(d2 == 0) { minDist2 = 0; break; }
                    minDist2 = std::min(minDist2, d2);
                }
                if(minDist2 > 0 && localNode->boundingBox.getWidthSquared() < minDist2 * this->thetaSquared)
                {
                    result[_rank].emplace_back(localNode->value.CM, localNode->value.mass, localNode->value.Q);
                }
                else
                {
                    ranksToOpen.insert(_rank);
                }
            }
            else
            {
                ranksToOpen.insert(_rank);
            }
        }
        else
        {
            result[_rank].emplace_back(localNode->value.CM, localNode->value.mass, localNode->value.Q);
        }
    }

    if(ranksToOpen.empty())
        return;

    for(size_t i = 0; i < CHILDREN; i++)
    {
        if(localNode->children[i] != nullptr)
            this->getSendListHelper(localNode->children[i], result, ranksToOpen);
    }
}

#endif // RICH_MPI

#endif // DISTRIUBTED_GRAVITY_CALCULATOR_HPP