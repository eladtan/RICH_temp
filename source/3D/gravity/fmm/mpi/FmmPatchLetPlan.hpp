#ifndef FMM_PATCH_LET_PLAN_HPP
#define FMM_PATCH_LET_PLAN_HPP

#ifdef RICH_MPI

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <mpi.h>

#include "3D/gravity/fmm/FmmDiagnostics.hpp"
#include "3D/gravity/fmm/FmmM2LOperatorCache.hpp"
#include "3D/gravity/fmm/FmmTaylorExpansion.hpp"
#include "3D/gravity/fmm/mpi/FmmPatchForest.hpp"
#include "3D/gravity/fmm/mpi/FmmPeerExchange.hpp"
#include "3D/gravity/fmm/mpi/FmmProcessTraversal.hpp"

struct FmmPatchLetM2LInteraction
{
    std::uint32_t targetPatchIndex = 0;
    std::uint32_t targetNode = 0;
    std::uint32_t sourceIndex = 0;
    std::uint32_t geometryIndex = 0;
};

struct FmmPatchLetP2PInteraction
{
    std::uint32_t targetPatchIndex = 0;
    std::uint32_t targetNode = 0;
    std::uint32_t sourceIndex = 0;
};

struct FmmPatchLetM2PInteraction
{
    std::uint32_t targetPatchIndex = 0;
    std::uint32_t targetNode = 0;
    std::uint32_t sourceIndex = 0;
};

class FmmPatchLetPlan
{
public:
    FmmPatchLetPlan();

    void build(const FmmPatchForest& forest,
               const std::vector<FmmPatchRootDescriptor>& globalDescriptors,
               const FmmProcessPairPlan& processPlan,
               double thetaCritical,
               std::uint64_t topologyEpoch,
               std::size_t maxLetWaveBytes,
               std::size_t maxTargetPatchesPerWave,
               std::size_t multipoleCoefficientCount,
               std::size_t maxParticlePayloadCount,
               bool enableLeafM2P,
               const MPI_Comm& comm,
               FmmSolveStats& stats,
               bool reuseUnaffectedTargetSubplans = false);

    // Reuse a previously built patch LET when patch identities and node
    // structure are unchanged. Payload counts and coefficients are refreshed
    // by execute(); only the interaction topology and subscriptions persist.
    void reuse(const FmmPatchForest& forest,
               std::uint64_t topologyEpoch,
               FmmSolveStats& stats);

    // True when every retained particle subscription still fits the payload
    // capacity used to construct its wave. A false result requires a LET
    // rebuild before any payload exchange is started.
    bool localPayloadShapeReusable(const FmmPatchForest& forest) const;

    void execute(FmmPatchForest& forest,
                 const FmmTaylorExpansion& layout,
                 FmmM2LOperatorCache& operatorCache,
                 std::size_t maxRemoteBytes,
                 std::size_t maxOperatorCacheBytes,
                 FmmSolveStats& stats);

    std::size_t waveCount() const { return waveCount_; }
    const std::vector<FmmPatchLetM2LInteraction>& m2lInteractions() const
    {
        return m2lInteractions_;
    }
    const std::vector<FmmPatchLetP2PInteraction>& p2pInteractions() const
    {
        return p2pInteractions_;
    }
    const std::vector<FmmPatchLetM2PInteraction>& m2pInteractions() const
    {
        return m2pInteractions_;
    }
    std::size_t bytesOwned() const;

private:
    struct RemoteRootGeometry
    {
        std::uint64_t latticeId = 0;
        std::int64_t center[3] = {0, 0, 0};
        std::uint64_t halfUnits = 0;
    };

    struct PendingPair
    {
        std::size_t targetPatchIndex = 0;
        std::size_t targetNode = 0;
        FmmPatchKey sourcePatch;
        std::uint64_t sourceKey = 0;
    };

    struct PendingInteraction
    {
        std::size_t targetPatchIndex = 0;
        std::size_t targetNode = 0;
        FmmPatchKey sourcePatch;
        std::uint64_t sourceKey = 0;
        int kind = 0;
    };

    typedef std::tuple<FmmPatchKey, std::uint64_t, int> SourceIdentity;
    typedef std::tuple<std::size_t, SourceIdentity> WaveSourceIdentity;

    struct SourceRecord
    {
        FmmRemoteNodeKey key;
        int kind = 0;
        std::size_t wave = 0;
        FmmRemoteNodeDescriptor descriptor;
        FmmNode sourceNode;
    };

    struct CachedSource
    {
        FmmRemoteNodeDescriptor descriptor;
        RemoteRootGeometry root;
    };

    struct CachedTerminal
    {
        std::uint64_t targetSpatialKey = 0;
        SourceIdentity source;
    };

    struct CachedTargetSubplan
    {
        FmmPatchKey targetPatch;
        std::uint64_t targetTopologyHash = 0;
        std::set<FmmPatchKey> sourcePatches;
        std::map<SourceIdentity, CachedSource> sources;
        std::vector<CachedTerminal> m2l;
        std::vector<CachedTerminal> p2p;
        std::vector<CachedTerminal> m2p;
    };

    struct PayloadView
    {
        const char* data = nullptr;
        std::size_t count = 0;
    };

    static FmmRemoteNodeDescriptor descriptorForNode(
        const FmmNode& node,
        const FmmPatchKey& patch,
        std::uint64_t topologyEpoch);
    static bool admissible(const FmmNode& target,
                           const FmmRemoteNodeDescriptor& source,
                           double thetaCritical);
    static bool m2pAdmissible(const FmmNode& target,
                              const FmmRemoteNodeDescriptor& source,
                              const std::vector<std::size_t>& particleOrder,
                              const std::vector<Vector3D>& positions,
                              double thetaCritical);
    static FmmNode sourceNodeFromDescriptor(
        const FmmRemoteNodeDescriptor& descriptor,
        const RemoteRootGeometry& root);

    std::size_t sourceRecordBytes(const SourceIdentity& source) const;
    std::size_t ensureSourceRecord(
        std::size_t wave,
        const SourceIdentity& identity,
        std::map<WaveSourceIdentity, std::uint32_t>& sourceIndexByWave);
    void buildWaveRanges();
    void executeWave(std::size_t wave,
                     FmmPatchForest& forest,
                     const FmmTaylorExpansion& layout,
                     FmmM2LOperatorCache& operatorCache,
                     std::size_t maxRemoteBytes,
                     FmmSolveStats& stats);

    std::vector<std::unordered_map<std::uint64_t, std::size_t>>
        localNodeByPatch_;
    std::map<FmmPatchKey, RemoteRootGeometry> remoteRoots_;
    std::map<FmmPatchKey,
             std::unordered_map<std::uint64_t, FmmRemoteNodeDescriptor>>
        remoteDescriptors_;

    std::vector<SourceRecord> sources_;
    std::vector<FmmPatchLetM2LInteraction> m2lInteractions_;
    std::vector<FmmPatchLetP2PInteraction> p2pInteractions_;
    std::vector<FmmPatchLetM2PInteraction> m2pInteractions_;
    std::vector<FmmM2LOperatorCache::PreparedGeometry> m2lGeometries_;
    std::vector<std::pair<std::size_t, std::size_t>> m2lWaveRanges_;
    std::vector<std::pair<std::size_t, std::size_t>> p2pWaveRanges_;
    std::vector<std::pair<std::size_t, std::size_t>> m2pWaveRanges_;

    std::unordered_map<int, std::vector<FmmSubscription>> subscriptionsReceived_;
    std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t>
        localParticlePayloadCaps_;
    std::map<FmmPatchKey, CachedTargetSubplan> targetSubplans_;
    std::map<FmmPatchKey, std::uint64_t> sourceTopologyHashes_;
    FmmPeerExchange exchange_;
    std::size_t waveCount_;
    std::size_t localWaveCount_;
    std::size_t maxLetWaveBytes_;
    std::size_t multipoleCoefficientCount_;
    std::size_t maxParticlePayloadCount_;
    std::uint64_t topologyEpoch_;
    MPI_Comm comm_;
    int rank_;
    bool initialized_;
};

#endif // RICH_MPI

#endif // FMM_PATCH_LET_PLAN_HPP
