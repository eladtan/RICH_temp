#ifndef FMM_LET_PLAN_HPP
#define FMM_LET_PLAN_HPP

#ifdef RICH_MPI

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <mpi.h>

#include "3D/gravity/fmm/FmmDiagnostics.hpp"
#include "3D/gravity/fmm/FmmM2LOperatorCache.hpp"
#include "3D/gravity/fmm/FmmTaylorExpansion.hpp"
#include "3D/gravity/fmm/FmmTree.hpp"
#include "3D/gravity/fmm/mpi/FmmPackets.hpp"
#include "3D/gravity/fmm/mpi/FmmPeerExchange.hpp"
#include "3D/gravity/fmm/mpi/FmmProcessTraversal.hpp"

struct FmmLetM2LInteraction
{
    std::size_t targetNode = 0;
    int sourceRank = -1;
    std::uint64_t sourceKey = 0;
};

struct FmmLetP2PInteraction
{
    std::size_t targetNode = 0;
    int sourceRank = -1;
    std::uint64_t sourceKey = 0;
};

class FmmLetPlan
{
public:
    FmmLetPlan();

    void build(const FmmTree& localTree,
               const std::vector<FmmRankRootDescriptor>& rootDescriptors,
               const FmmProcessPairPlan& processPlan,
               double thetaCritical,
               std::uint64_t topologyEpoch,
               const MPI_Comm& comm,
               FmmSolveStats& stats);

    void execute(const FmmTree& localTree,
                 const std::vector<Vector3D>& positions,
                 const std::vector<double>& masses,
                 const std::vector<std::uint64_t>& cellIds,
                 const FmmTaylorExpansion& layout,
                 const std::vector<double>& localMultipoles,
                 std::vector<double>& localLocals,
                 std::vector<Vector3D>& acceleration,
                 std::vector<double>* positiveKernelPotential,
                 std::size_t maxRemoteBytes,
                 std::size_t maxOperatorCacheBytes,
                 FmmSolveStats& stats) const;

    const std::vector<FmmLetM2LInteraction>& m2lInteractions() const
    {
        return m2lInteractions_;
    }

    std::size_t bytesOwned() const;

    const std::vector<FmmLetP2PInteraction>& p2pInteractions() const
    {
        return p2pInteractions_;
    }

private:
    struct PendingPair
    {
        std::size_t targetNode = 0;
        int sourceRank = -1;
        std::uint64_t sourceKey = 0;
    };

    static FmmRemoteNodeDescriptor descriptorForNode(const FmmNode& node,
                                                       int sourceRank,
                                                       std::uint64_t topologyEpoch);
    static bool admissible(const FmmNode& target,
                           const FmmRemoteNodeDescriptor& source,
                           double thetaCritical);

    std::unordered_map<std::uint64_t, std::size_t> localNodeByKey_;
    std::unordered_map<int,
        std::unordered_map<std::uint64_t, FmmRemoteNodeDescriptor>> remoteDescriptors_;
    std::vector<FmmLetM2LInteraction> m2lInteractions_;
    std::vector<FmmLetP2PInteraction> p2pInteractions_;
    std::unordered_map<int, std::vector<FmmSubscription>> subscriptionsToSend_;
    std::unordered_map<int, std::vector<FmmSubscription>> subscriptionsReceived_;
    mutable FmmM2LOperatorCache operatorCache_;
    FmmPeerExchange exchange_;
    MPI_Comm comm_;
    int rank_;
    std::uint64_t topologyEpoch_;
};

#endif // RICH_MPI

#endif // FMM_LET_PLAN_HPP
