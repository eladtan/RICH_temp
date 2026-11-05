#ifndef FMM_LET_PLAN_HPP
#define FMM_LET_PLAN_HPP

#ifdef RICH_MPI

#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

#include <mpi.h>

#include "3D/gravity/fmm/FmmDiagnostics.hpp"
#include "3D/gravity/fmm/FmmM2LOperatorCache.hpp"
#include "3D/gravity/fmm/FmmTaylorExpansion.hpp"
#include "3D/gravity/fmm/FmmTree.hpp"
#include "3D/gravity/fmm/mpi/FmmPackets.hpp"
#include "3D/gravity/fmm/mpi/FmmPatchKey.hpp"
#include "3D/gravity/fmm/mpi/FmmPeerExchange.hpp"
#include "3D/gravity/fmm/mpi/FmmProcessTraversal.hpp"

struct FmmLetM2LInteraction
{
    std::uint32_t targetNode = 0;
    std::uint32_t sourceIndex = 0;
    std::uint32_t geometryIndex = 0;
};

struct FmmLetP2PInteraction
{
    std::uint32_t targetNode = 0;
    std::uint32_t sourceIndex = 0;
};

// Remote multipole evaluated directly at each particle of a target leaf. The
// source index refers to the shared multipole source table, as for M2L.
struct FmmLetM2PInteraction
{
    std::uint32_t targetNode = 0;
    std::uint32_t sourceIndex = 0;
};

static_assert(sizeof(FmmLetM2LInteraction) == 12,
              "LET M2L interaction must remain compact");
static_assert(sizeof(FmmLetP2PInteraction) == 8,
              "LET P2P interaction must remain compact");
static_assert(sizeof(FmmLetM2PInteraction) == 8,
              "LET M2P interaction must remain compact");

class FmmLetPlan
{
public:
    FmmLetPlan();

    void build(const FmmTree& localTree,
               const std::vector<Vector3D>& positions,
               const std::vector<FmmPatchRootDescriptor>& rootDescriptors,
               const FmmProcessPairPlan& processPlan,
               double thetaCritical,
               std::uint64_t topologyEpoch,
               const MPI_Comm& comm,
               bool reuseBuildStorage,
               bool enableLeafM2P,
               bool stalenessDiagnostics,
               bool compactParticlePayload,
               bool quantizedParticlePayload,
               bool compactMultipolePayload,
               std::size_t maxLetWaveBytes,
               std::size_t multipoleCoefficientCount,
               FmmSolveStats& stats);

    // Number of payload waves every rank must execute. Collectively agreed in
    // build(), so all ranks call the neighborhood collective the same number of
    // times even when only a few of them need more than one wave.
    std::size_t waveCount() const { return waveCount_; }
    std::size_t m2lSourceCount() const { return m2lSources_.size(); }
    std::size_t p2pSourceCount() const { return p2pSources_.size(); }


    void beginExecute(std::size_t wave,
                      const FmmTree& localTree,
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

    void finishExecute(std::size_t wave,
                       const FmmTree& localTree,
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

    const std::vector<FmmLetM2PInteraction>& m2pInteractions() const
    {
        return m2pInteractions_;
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
        FmmPatchKey sourcePatch;
        std::uint64_t spatialKey = 0;
        std::uint32_t ownerSubscriptionIndex =
            std::numeric_limits<std::uint32_t>::max();
        FmmNode node;
    };

    struct RemoteSource
    {
        FmmPatchKey sourcePatch;
        std::uint64_t spatialKey = 0;
        std::uint32_t ownerSubscriptionIndex =
            std::numeric_limits<std::uint32_t>::max();
    };

    struct ResolvedSubscription
    {
        FmmSubscription subscription;
        std::uint32_t localNodeIndex = 0;
    };

    // Interactions are retained in wave/target order. A warm solve can skip
    // every interaction of an empty target with one particle-count test rather
    // than scanning the complete persistent interaction array.
    struct InteractionTargetRange
    {
        std::uint32_t targetNode = 0;
        std::uint32_t begin = 0;
        std::uint32_t end = 0;
    };

    struct PendingInteraction
    {
        std::size_t targetNode = 0;
        FmmPatchKey sourcePatch;
        std::uint64_t sourceKey = 0;
    };

    struct PendingPair
    {
        std::size_t targetNode = 0;
        FmmPatchKey sourcePatch;
        std::uint64_t sourceKey = 0;
    };

    // Debug: with RICH_FMM_PLAN_DUMP set, write the finished plan of this
    // build to fmm_plan_dump_r<rank>_<n>.bin in the working directory, so two
    // builds can be compared with cmp: the source tables, interactions,
    // prepared M2L geometries and their use counts, wave and target ranges,
    // the subscription wire bytes sent, the subscriptions received, and the
    // remote descriptors kept after pruning.  The multipole build radii are
    // not written (they exist only with letStalenessDiagnostics).
    void dumpPlanForComparison(
        const std::unordered_map<int, std::vector<char>>& subscriptionBuffers) const;

    static FmmRemoteNodeDescriptor descriptorForNode(const FmmNode& node,
                                                     const FmmPatchKey& patch,
                                                     std::uint64_t topologyEpoch);
    static bool admissible(const FmmNode& target,
                           const FmmRemoteNodeDescriptor& source,
                           double thetaCritical);
    // True when the source multipole may be evaluated directly at every
    // particle of a target leaf. This drops the target radius from the test,
    // which a leaf cannot otherwise reduce because it has no children.
    static bool m2pAdmissible(const FmmNode& target,
                              const FmmRemoteNodeDescriptor& source,
                              const std::vector<std::size_t>& particleOrder,
                              const std::vector<Vector3D>& positions,
                              double thetaCritical);

    std::unordered_map<std::uint64_t, std::size_t> localNodeByKey_;
    std::vector<RemoteLatticeRoot> remoteLatticeRoots_;
    std::unordered_map<FmmPatchKey,
        std::unordered_map<std::uint64_t, FmmRemoteNodeDescriptor>,
        FmmPatchKeyHash> remoteDescriptors_;
    std::vector<FmmLetM2LInteraction> m2lInteractions_;
    std::vector<M2LSource> m2lSources_;
    // Each multipole source's descriptor radius when the plan was built (zero
    // for a source that was empty then).  Filled only with
    // letStalenessDiagnostics, which is its only reader.
    std::vector<double> m2lSourceBuildRadius_;
    std::vector<FmmM2LOperatorCache::PreparedGeometry>
        m2lOperatorGeometries_;
    std::vector<std::uint64_t> m2lOperatorGeometryUseCounts_;
    std::vector<FmmLetP2PInteraction> p2pInteractions_;
    std::vector<RemoteSource> p2pSources_;
    std::vector<FmmLetM2PInteraction> m2pInteractions_;
    // Half-open [start, end) index ranges into the interaction arrays, one per
    // wave. Interactions are sorted by source wave so each executes exactly
    // once, in the wave whose payload carries its source.
    std::vector<std::pair<std::size_t, std::size_t>> m2lWaveRanges_;
    std::vector<std::pair<std::size_t, std::size_t>> p2pWaveRanges_;
    std::vector<std::pair<std::size_t, std::size_t>> m2pWaveRanges_;
    std::vector<InteractionTargetRange> m2lTargetRanges_;
    std::vector<InteractionTargetRange> p2pTargetRanges_;
    std::vector<InteractionTargetRange> m2pTargetRanges_;
    std::vector<std::pair<std::size_t, std::size_t>> m2lTargetWaveRanges_;
    std::vector<std::pair<std::size_t, std::size_t>> p2pTargetWaveRanges_;
    std::vector<std::pair<std::size_t, std::size_t>> m2pTargetWaveRanges_;
    std::size_t waveCount_;
    std::vector<std::uint32_t> activeM2LInteractionIndices_;
    std::vector<std::uint32_t> activeP2PInteractionIndices_;
    std::vector<std::uint32_t> activeM2PInteractionIndices_;
    std::vector<unsigned char> activeM2LSourceFlags_;
    std::vector<unsigned char> activeP2PSourceFlags_;
    std::vector<PendingPair> pendingScratch_;
    std::vector<PendingPair> workScratch_;
    std::vector<PendingPair> blockedScratch_;
    std::unordered_map<int, std::vector<FmmSubscription>> subscriptionsToSend_;
    std::unordered_map<int, std::vector<ResolvedSubscription>>
        subscriptionsReceived_;
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
    bool compactParticlePayload_;
    bool quantizedParticlePayload_;
    bool compactMultipolePayload_;
    // letStalenessDiagnostics: the opening criterion of the last build and
    // whether finishExecute counts stale retained interactions.
    double thetaCritical_;
    bool stalenessDiagnostics_;
    // Bracket on a multipole source's current radius: its descriptor radius
    // at build (zero when it was empty then) and its cube radius.
    double optimisticSourceRadius(std::uint32_t sourceIndex) const;
    double boundSourceRadius(std::uint32_t sourceIndex) const;
};

#endif // RICH_MPI

#endif // FMM_LET_PLAN_HPP
