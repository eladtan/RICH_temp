#ifndef DISTRIBUTED_FMM_GRAVITY_CALCULATOR_HPP
#define DISTRIBUTED_FMM_GRAVITY_CALCULATOR_HPP

#ifdef RICH_MPI

#include <cstddef>
#include <cstdint>
#include <vector>

#include <mpi.h>

#include "3D/gravity/fmm/FmmConfig.hpp"
#include "3D/gravity/fmm/FmmDiagnostics.hpp"
#include "3D/gravity/fmm/FmmDualTreeTraversal.hpp"
#include "3D/gravity/fmm/FmmM2LOperatorCache.hpp"
#include "3D/gravity/fmm/FmmRootGeometry.hpp"
#include "3D/gravity/fmm/FmmTree.hpp"
#include "3D/gravity/fmm/mpi/FmmLetPlan.hpp"
#include "3D/gravity/fmm/mpi/FmmPeerExchange.hpp"
#include "3D/gravity/fmm/mpi/FmmProcessTraversal.hpp"
#include "3D/gravity/fmm/mpi/FmmProcessTree.hpp"

struct FmmDistributedOptions
{
    double rootSlackFactor = 1.25;
    std::size_t maxRemoteBytes = static_cast<std::size_t>(2) * 1024 * 1024 * 1024;
    bool rebuildTopologyEverySolve = false;
    bool reuseInteractionPlansAcrossLeafCountChanges = true;

    // Keep persistent-tree controls last for aggregate compatibility.
    bool persistentLocalTreeTopology = true;
    double persistentLeafSplitFactor = 1.5;
    double persistentLeafMergeFactor = 0.5;

    // Bounds non-empty leaves to globalRootHalfSize / 2^level, so that ranks
    // owning a spatially large sparse domain cannot end up with leaves whose
    // own radius exceeds thetaCritical * distance to every remote source.
    // Zero disables the bound; FmmGravityOptions::maxLeafHalfSize overrides it.
    int maxLeafHalfSizeLevel = 0;

    // Lets an inadmissible leaf target accept a remote multipole evaluated at
    // each of its particles. This removes the target radius from the accuracy
    // bound, but pays an uncached translation operator per particle and source,
    // so it is off by default and only worth enabling for distributions with
    // spatially oversized ranks.
    bool enableLeafM2P = false;

    // Upper bound on the LET payload a single rank may request in one exchange.
    // Requests above this are split into several waves, so peak LET memory is
    // set by configuration rather than by the union of a rank's near fields.
    // Zero disables splitting and restores the single-exchange behaviour.
    std::size_t maxLetWaveBytes =
        static_cast<std::size_t>(256) * 1024 * 1024;
};

class DistributedFmmGravityCalculator
{
public:
    explicit DistributedFmmGravityCalculator(
        FmmGravityOptions options = FmmGravityOptions(),
        FmmDistributedOptions distributedOptions = FmmDistributedOptions(),
        const MPI_Comm& comm = MPI_COMM_WORLD);

    ~DistributedFmmGravityCalculator();

    void solve(const std::vector<Vector3D>& positions,
               const std::vector<double>& masses,
               const std::vector<std::uint64_t>& cellIds,
               const Vector3D& domainLower,
               const Vector3D& domainUpper,
               std::vector<Vector3D>& acceleration,
               std::vector<double>* positiveKernelPotential = nullptr);

    const FmmSolveStats& stats() const noexcept { return stats_; }

private:
    struct LocalTopologyChange
    {
        bool rootGeometryChanged = false;
        bool leafTopologyChanged = false;
        bool leafOccupancyChanged = false;
        bool countOnlyLeafChange = false;
        bool persistentTreeRefit = false;
        std::size_t persistentLeafSplits = 0;
        std::size_t persistentSubtreeMerges = 0;
        std::size_t persistentEmptyLeaves = 0;
    };

    void validateInputs(const std::vector<Vector3D>& positions,
                        const std::vector<double>& masses,
                        const std::vector<std::uint64_t>& cellIds,
                        const Vector3D& domainLower,
                        const Vector3D& domainUpper,
                        std::vector<double>* positiveKernelPotential) const;
    LocalTopologyChange prepareLocalTree(const std::vector<Vector3D>& positions,
                                         const Vector3D& domainLower,
                                         const Vector3D& domainUpper);
    void rebuildTopology(const std::vector<Vector3D>& positions,
                         bool rebuildProcessTopology);
    FmmRankRootDescriptor localRootDescriptor() const;
    double effectiveMaxLeafHalfSize(const Vector3D& domainLower,
                                    const Vector3D& domainUpper) const;
    // Collective on comm_.
    void logPatchCountSurvey(const std::vector<Vector3D>& positions,
                             const Vector3D& domainLower,
                             const Vector3D& domainUpper) const;

    FmmGravityOptions options_;
    FmmDistributedOptions distributedOptions_;
    MPI_Comm comm_;
    int rank_;
    int size_;

    FmmSolveStats stats_;
    FmmRootGeometry localRoot_;
    bool rootInitialized_;
    double lastEffectiveMaxLeafHalfSize_;
    std::uint64_t lastLocalTopologyHash_;
    std::vector<std::uint64_t> lastLocalStructuralSignature_;
    std::vector<std::uint64_t> lastLocalOccupancySignature_;
    std::uint64_t topologyEpoch_;
    std::uint64_t topologyRebuildCount_;
    std::uint64_t processTopologyRebuildCount_;
    std::uint64_t letTopologyRebuildCount_;

    FmmTree localTree_;
    FmmM2LOperatorCache operatorCache_;
    FmmLocalInteractionPlan localInteractionPlan_;
    std::vector<double> localMultipoles_;
    std::vector<double> localLocals_;
    std::vector<FmmRankRootDescriptor> rootDescriptors_;
    FmmProcessTree processTree_;
    FmmProcessPairPlan processPlan_;
    FmmLetPlan letPlan_;
    FmmPeerExchange processUpExchange_;
    FmmPeerExchange processM2LExchange_;
    FmmPeerExchange processDownExchange_;
};

#endif // RICH_MPI

#endif // DISTRIBUTED_FMM_GRAVITY_CALCULATOR_HPP
