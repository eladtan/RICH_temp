#ifndef FMM_DISTRIBUTED_OPTIONS_HPP
#define FMM_DISTRIBUTED_OPTIONS_HPP

#include <cstddef>

#include "3D/gravity/fmm/FmmConfig.hpp"

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

    // Patch-forest options. Phases 3-4 rebuild the forest topology each solve;
    // persistent patch reuse is introduced separately.
    bool enablePatchForest = false;
    int minimumPatchLevel = 0;
    int maximumPatchLevel = FMM_MAX_TREE_DEPTH;
    std::size_t targetParticlesPerPatch = 0;
    std::size_t maxLocalPatchCount = 65536;
    std::size_t maxTargetPatchesPerWave = 64;
    bool useLocalPatchLet = true;

    // The patch-root directory is replicated on every rank in Phases 3-4.
    // Crossing this guard requires a distributed directory rather than an
    // unbounded MPI_Allgatherv allocation.
    std::size_t maxReplicatedDescriptorBytes =
        static_cast<std::size_t>(512) * 1024 * 1024;
};

#endif // FMM_DISTRIBUTED_OPTIONS_HPP
