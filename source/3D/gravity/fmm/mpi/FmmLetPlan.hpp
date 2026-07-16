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
    std::uint32_t targetNode = 0;
    int sourceRank = -1;
    std::uint64_t sourceKey = 0;
};

struct FmmLetP2PInteraction
{
    std::uint32_t targetNode = 0;
    int sourceRank = -1;
    std::uint64_t sourceKey = 0;
};

static_assert(sizeof(FmmLetM2LInteraction) == 16,
              "LET M2L interaction must remain compact");
static_assert(sizeof(FmmLetP2PInteraction) == 16,
              "LET P2P interaction must remain compact");

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

    void beginExecute(const FmmTree& localTree,
                      const std::vector<Vector3D>& positions,
                      const std::vector<double>& masses,
                      const std::vector<std::uint64_t>& cellIds,
                      const FmmTaylorExpansion& layout,
                      const std::vector<double>& localMultipoles,
                      const std::vector<double>& localLocals,
                      const std::vector<Vector3D>& acceleration,
                      const std::vector<double>* positiveKernelPotential,
                      std::size_t maxRemoteBytes,
                      FmmSolveStats& stats);

    void progressExecute();

    void finishExecute(const FmmTree& localTree,
                       const std::vector<Vector3D>& positions,
                       const FmmTaylorExpansion& layout,
                       std::vector<double>& localLocals,
                       std::vector<Vector3D>& acceleration,
                       std::vector<double>* positiveKernelPotential,
                       FmmM2LOperatorCache& operatorCache,
                       std::size_t maxRemoteBytes,
                       std::size_t maxOperatorCacheBytes,
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
                 FmmM2LOperatorCache& operatorCache,
                 std::size_t maxRemoteBytes,
                 std::size_t maxOperatorCacheBytes,
                 FmmSolveStats& stats);

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
    struct RemoteLatticeRoot
    {
        std::uint64_t latticeId = 0;
        std::int64_t center[3] = {0, 0, 0};
        std::uint64_t halfUnits = 0;
    };

    struct M2LSource
    {
        int sourceRank = -1;
        std::uint64_t spatialKey = 0;
        FmmNode node;
    };

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
    std::vector<RemoteLatticeRoot> remoteLatticeRoots_;
    std::unordered_map<int,
        std::unordered_map<std::uint64_t, FmmRemoteNodeDescriptor>> remoteDescriptors_;
    std::vector<FmmLetM2LInteraction> m2lInteractions_;
    std::vector<M2LSource> m2lSources_;
    std::vector<std::uint32_t> m2lSourceIndices_;
    std::vector<FmmM2LOperatorCache::PreparedGeometry>
        m2lOperatorGeometries_;
    std::vector<std::uint32_t> m2lOperatorGeometryIndices_;
    std::vector<std::uint64_t> m2lOperatorGeometryUseCounts_;
    std::vector<FmmLetP2PInteraction> p2pInteractions_;
    std::unordered_map<int, std::vector<FmmSubscription>> subscriptionsToSend_;
    std::unordered_map<int, std::vector<FmmSubscription>> subscriptionsReceived_;
    FmmPeerExchange exchange_;
    FmmPeerExchangeRequest pendingExchange_;
    bool executePending_;
    std::size_t pendingMaxRemoteBytes_;
    double pendingExchangePreparationSeconds_;
    std::uint64_t pendingProgressCallCount_;
    std::uint64_t pendingProgressIncompleteCount_;
    std::uint64_t pendingCompletionProgressCall_;
    MPI_Comm comm_;
    int rank_;
    std::uint64_t topologyEpoch_;
};

#endif // RICH_MPI

#endif // FMM_LET_PLAN_HPP
