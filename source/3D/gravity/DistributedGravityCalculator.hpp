#ifndef DISTRIUBTED_GRAVITY_CALCULATOR_HPP
#define DISTRIUBTED_GRAVITY_CALCULATOR_HPP

#ifdef RICH_MPI
#include "3D/tessellation/Tessellation3D.hpp"
#include "DistributedGravityTree.hpp"
#include "mpi/mpi_commands.hpp"
#include "GravityTree.hpp"
#include "FlatGravityTree.hpp"
#include "GravityTypes.h"

#ifndef SKELETON_DEPTH
#define SKELETON_DEPTH 2
#endif

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
    std::vector<std::vector<GravityNodeData>> topNodesOfRanks;
    mutable std::vector<boost::container::flat_set<int>> relevantRanksByDepths;
    mutable EnvironmentAgent::RanksSet tempRanks;
    const GravityTree<Vector3D>::Node *realRootOfGravityTree;
    std::vector<int> prunedRanks;
    boost::container::flat_set<int> prunedSet;
    std::vector<MassedValue<Vector3D>> prunedSummaries;
    std::vector<int> exchangeNeighbors;

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
            for(const GravityNodeData &remote : this->topNodesOfRanks[_rank])
            {
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
}

std::vector<Vector3D> DistributedGravityCalculator::getAcceleration(const std::vector<Vector3D> &points) const
{
#ifdef GRAVITY_PROFILE
    double prof_t0 = MPI_Wtime();
    size_t prof_totalSent = 0;
    int prof_actualNeighbors = 0;
#endif

    // 1. Build send list
    std::vector<std::vector<MassedValue<Vector3D>>> sendList = this->getSendList();

#ifdef GRAVITY_PROFILE
    double prof_t_sendList = MPI_Wtime();
    for(int r = 0; r < this->size; r++)
    {
        prof_totalSent += sendList[r].size();
        if(!sendList[r].empty())
            prof_actualNeighbors++;
    }
#endif

    // 2. Insert pruned summaries into the local tree AFTER sendList is built
    //    (so pruned pseudo-particles don't leak into the send lists) but
    //    BEFORE the local walk (so the walk includes their contribution).
    this->gravityTree->addExternalValues(this->prunedSummaries);
    this->gravityTree->calculateMasses();

    // 3. Start non-blocking sparse exchange (using precomputed neighbor list)
    SparseExchangeHandle exchangeHandle = MPI_Iexchange_sparse_start(sendList, this->exchangeNeighbors, this->comm);
    sendList.clear();
    sendList.shrink_to_fit();

#ifdef GRAVITY_PROFILE
    double prof_t_istart = MPI_Wtime();
#endif

    // 4. Compile local tree to flat array and walk while exchange is in flight
    FlatGravityTree localFlat(*this->gravityTree);
    std::vector<Vector3D> results;
    results.reserve(points.size());
    for(const Vector3D &point : points)
    {
        results.emplace_back(localFlat.gravity(point, true));
    }

#ifdef GRAVITY_PROFILE
    double prof_t_localWalk = MPI_Wtime();
#endif

    // 5. Wait for exchange to complete (overlapped with step 4)
    std::vector<std::vector<MassedValue<Vector3D>>> insertToTreeByRanks =
        MPI_Iexchange_sparse_wait<MassedValue<Vector3D>>(exchangeHandle);

#ifdef GRAVITY_PROFILE
    double prof_t_wait = MPI_Wtime();
    size_t totalArrived = 0;
#endif

    // 6. Build temporary tree from received data (non-pruned ranks only)
    auto [ll, ur] = this->tess.GetBoxCoordinates();
    GravityTree<Vector3D> remoteTree(ll, ur, this->theta, this->quadrupole);

    for(int _rank = 0; _rank < this->size; _rank++)
    {
        std::vector<MassedValue<Vector3D>> &rankData = insertToTreeByRanks[_rank];
#ifdef GRAVITY_PROFILE
        totalArrived += rankData.size();
#endif
        if(_rank != this->rank && this->prunedSet.find(_rank) == this->prunedSet.end())
        {
            remoteTree.addExternalValues(rankData);
        }
        rankData.clear();
    }
    insertToTreeByRanks.clear();
    insertToTreeByRanks.shrink_to_fit();
    remoteTree.calculateMasses();

#ifdef GRAVITY_PROFILE
    double prof_t_remoteInsert = MPI_Wtime();
#endif

    // 7. Compile remote tree to flat array and walk
    FlatGravityTree remoteFlat(remoteTree);
    for(size_t i = 0; i < points.size(); i++)
    {
        results[i] += remoteFlat.gravity(points[i], true);
    }

#ifdef GRAVITY_PROFILE
    double prof_t_remoteWalk = MPI_Wtime();

    double phases[6] = {
        prof_t_sendList - prof_t0,
        prof_t_istart - prof_t_sendList,
        prof_t_localWalk - prof_t_istart,
        prof_t_wait - prof_t_localWalk,
        prof_t_remoteInsert - prof_t_wait,
        prof_t_remoteWalk - prof_t_remoteInsert
    };
    double phases_max[6], phases_min[6], phases_sum[6];
    MPI_Allreduce(phases, phases_max, 6, MPI_DOUBLE, MPI_MAX, this->comm);
    MPI_Allreduce(phases, phases_min, 6, MPI_DOUBLE, MPI_MIN, this->comm);
    MPI_Allreduce(phases, phases_sum, 6, MPI_DOUBLE, MPI_SUM, this->comm);

    unsigned long long sent_recv[2] = {
        static_cast<unsigned long long>(prof_totalSent),
        static_cast<unsigned long long>(totalArrived)
    };
    unsigned long long sent_recv_max[2], sent_recv_sum[2];
    MPI_Allreduce(sent_recv, sent_recv_max, 2, MPI_UNSIGNED_LONG_LONG, MPI_MAX, this->comm);
    MPI_Allreduce(sent_recv, sent_recv_sum, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, this->comm);

    int neighbors_max = 0, neighbors_min = 0;
    MPI_Allreduce(&prof_actualNeighbors, &neighbors_max, 1, MPI_INT, MPI_MAX, this->comm);
    MPI_Allreduce(&prof_actualNeighbors, &neighbors_min, 1, MPI_INT, MPI_MIN, this->comm);

    int pruned = static_cast<int>(this->prunedRanks.size());
    int pruned_max = 0, pruned_min = 0;
    double pruned_avg = 0;
    MPI_Allreduce(&pruned, &pruned_max, 1, MPI_INT, MPI_MAX, this->comm);
    MPI_Allreduce(&pruned, &pruned_min, 1, MPI_INT, MPI_MIN, this->comm);
    {
        double dp = static_cast<double>(pruned);
        MPI_Allreduce(&dp, &pruned_avg, 1, MPI_DOUBLE, MPI_SUM, this->comm);
        pruned_avg /= this->size;
    }

    if(this->rank == 0)
    {
        double invP = 1.0 / this->size;
        std::cout << "[GRAVITY_PROFILE] sendList:     min=" << phases_min[0] << " max=" << phases_max[0] << " avg=" << phases_sum[0] * invP << "\n"
                  << "[GRAVITY_PROFILE] istart:       min=" << phases_min[1] << " max=" << phases_max[1] << " avg=" << phases_sum[1] * invP << "\n"
                  << "[GRAVITY_PROFILE] local_walk:   min=" << phases_min[2] << " max=" << phases_max[2] << " avg=" << phases_sum[2] * invP << "\n"
                  << "[GRAVITY_PROFILE] wait:         min=" << phases_min[3] << " max=" << phases_max[3] << " avg=" << phases_sum[3] * invP << "\n"
                  << "[GRAVITY_PROFILE] remote_ins:   min=" << phases_min[4] << " max=" << phases_max[4] << " avg=" << phases_sum[4] * invP << "\n"
                  << "[GRAVITY_PROFILE] remote_walk:  min=" << phases_min[5] << " max=" << phases_max[5] << " avg=" << phases_sum[5] * invP << "\n"
                  << "[GRAVITY_PROFILE] sent_nodes: max=" << sent_recv_max[0] << " avg=" << sent_recv_sum[0] * invP
                  << "  recv_nodes: max=" << sent_recv_max[1] << " avg=" << sent_recv_sum[1] * invP
                  << "  max_neighbors=" << neighbors_max << " min_neighbors=" << neighbors_min << "/" << this->size << "\n"
                  << "[GRAVITY_PROFILE] pruned_ranks: min=" << pruned_min << " max=" << pruned_max << " avg=" << pruned_avg << "/" << (this->size - 1) << "\n"
                  << "[GRAVITY_PROFILE] prune_depth=" << SKELETON_DEPTH << "  pruned_tree_nodes=" << this->prunedSummaries.size() << "\n"
                  << "[GRAVITY_PROFILE] local_cells=" << points.size() << std::endl;
    }
#endif

    return results;
}

void DistributedGravityCalculator::getSendListHelper(const LocalNode *localNode, std::vector<std::vector<MassedValue<Vector3D>>> &result, int depth) const
{
    if(localNode == nullptr)
        return;
    if(static_cast<size_t>(depth) >= this->relevantRanksByDepths.size())
    {
        return;
    }

    boost::container::flat_set<int> &relevantRanks = this->relevantRanksByDepths[depth];

    if(localNode->isLeaf)
    {
        for(int _rank : relevantRanks)
            result[_rank].emplace_back(localNode->value.CM, localNode->value.mass, localNode->value.Q);
        return;
    }

    if(static_cast<size_t>(depth + 1) >= this->relevantRanksByDepths.size())
    {
        this->relevantRanksByDepths.resize(static_cast<size_t>(depth + 2));
    }
    this->relevantRanksByDepths[depth + 1].clear();
    bool someoneWantsToOpen = false;

    for(int _rank : relevantRanks)
    {
        bool contained = std::any_of(this->topNodesOfRanks[_rank].begin(), this->topNodesOfRanks[_rank].end(),
                                    [localNode](const GravityNodeData &remote)
                                    {
                                        return remote.boundingBox.contains(localNode->boundingBox);
                                    });
        bool shouldOpen = false;
        if(contained)
        {
            if(this->tempRanks.empty())
            {
                double radius = localNode->boundingBox.getWidth() / this->theta;
                this->tempRanks = std::move(this->tess.GetEnvironmentAgent()->getIntersectingRanks(localNode->value.CM, radius));
            }
            shouldOpen = (this->tempRanks.find(_rank) != this->tempRanks.end());
        }
        else
        {
            shouldOpen = std::any_of(this->topNodesOfRanks[_rank].begin(), this->topNodesOfRanks[_rank].end(),
                                    [localNode, this](const GravityNodeData &remote)
                                    {
                                        return ShouldOpenBox(localNode->value.CM, localNode->boundingBox, remote.boundingBox.closestPoint(localNode->value.CM), this->thetaSquared);
                                    });
        }

        if(shouldOpen)
        {
            someoneWantsToOpen = true;
            this->relevantRanksByDepths[depth + 1].insert(_rank);
        }
        else
        {
            result[_rank].emplace_back(localNode->value.CM, localNode->value.mass, localNode->value.Q);
        }
    }

    this->tempRanks.clear();

    if(not someoneWantsToOpen)
        return;

    for(size_t i = 0; i < CHILDREN; i++)
    {
        if(localNode->children[i] != nullptr)
            this->getSendListHelper(localNode->children[i], result, depth + 1);
    }
}

#endif // RICH_MPI

#endif // DISTRIUBTED_GRAVITY_CALCULATOR_HPP