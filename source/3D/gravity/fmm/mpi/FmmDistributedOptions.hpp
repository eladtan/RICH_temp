#ifndef FMM_DISTRIBUTED_OPTIONS_HPP
#define FMM_DISTRIBUTED_OPTIONS_HPP

#include <cstddef>
#include <cstdint>

#include "3D/gravity/fmm/FmmConfig.hpp"

struct FmmDistributedOptions
{
    double rootSlackFactor = 1.25;
    // In patch mode this bounds persistent LET-plan storage plus the peak
    // transient storage of one payload wave.
    std::size_t maxRemoteBytes = static_cast<std::size_t>(2) * 1024 * 1024 * 1024;
    bool rebuildTopologyEverySolve = false;
    // Emit the detailed fmm_solve_trace line without relying on a process
    // environment variable. The legacy RICH_FMM_TRACE switch remains supported
    // for existing runs.
    bool emitSolveTrace = true;
    bool reuseInteractionPlansAcrossLeafCountChanges = true;
    // Retain bounded-wave source assignments when only leaf counts change.
    // The current payload is still checked against the hard remote-memory
    // budget at execution time; structural splits rebuild and resize waves.
    bool reuseBoundedLetWavesAcrossLeafCountChanges = true;

    // Legacy process-tree planning restricts every internal node to one of its
    // two child owners.  On strongly nonuniform decompositions this can funnel
    // most internal nodes and process M2L work onto a few ranks.  When enabled,
    // internal nodes may instead use any active rank and are assigned
    // deterministically to the currently least-loaded owner.  Patch leaves
    // remain on their hydro owners and the existing process coefficient routes
    // carry data between independent internal owners.
    bool globallyBalanceProcessNodeOwners = false;

    // Keep persistent-tree controls last for aggregate compatibility.
    bool persistentLocalTreeTopology = true;
    double persistentLeafSplitFactor = 4.0;
    double persistentLeafMergeFactor = 0.2;

    // Retained particle subscriptions reserve more than the current leaf
    // occupancy.  This keeps count-only moving-mesh changes from rebuilding the
    // complete LET merely because a max-depth or hysteretic leaf gained a few
    // particles.  Capacity is max(stable leaf occupancy, multiplicative slack,
    // additive slack).  The actual packet still carries only current particles.
    double letParticlePayloadSlackFactor = 1.10;
    std::size_t letParticlePayloadSlackCount = 8;

    // Bounds non-empty leaves to globalRootHalfSize / 2^level, so that ranks
    // owning a spatially large sparse domain cannot end up with leaves whose
    // own radius exceeds thetaCritical * distance to every remote source.
    // Zero disables the bound; FmmGravityOptions::maxLeafHalfSize overrides it.
    int maxLeafHalfSizeLevel = 0;

    // Lets an inadmissible leaf target accept a remote multipole evaluated at
    // each of its particles. This removes the target radius from the accuracy
    // bound, but pays an uncached translation operator per particle and source,
    // This is enabled by default to avoid transferring remote near-field
    // particles when a target leaf's extent alone prevents M2L.
    bool enableLeafM2P = true;

    // Encode patch-LET P2P particles as float offsets from their source-leaf
    // center plus a float mass (16 bytes instead of four doubles / 32 bytes).
    // Relative coordinates retain substantially more useful precision than
    // casting absolute domain coordinates to float. This format passed the TDE
    // and Lane-Emden accuracy and performance gates.
    bool compactLetParticlePayload = true;

    // Further compress compatibility-LET near-field particles to three
    // signed 16-bit leaf-relative coordinates plus one signed 16-bit mass
    // fraction. Each source leaf carries one double mass scale. Requires the
    // compact particle mode and leaves all force arithmetic in double. This
    // mode passed the TDE and Lane-Emden validation gates.
    bool quantizedLetParticlePayload = true;

    // Encode compatibility-LET multipole coefficients as binary32 on the
    // wire and use the source rank plus a compact 16-byte record header in
    // place of the repeated 48-byte stamped header.  Coefficients are restored
    // to double before any FMM kernel runs, so this changes only MPI storage
    // and transfer precision. Disabled by default pending accuracy validation.
    bool compactLetMultipolePayload = false;

    // Deliver each remote LET source only once per physical compute node.
    // A node leader receives the deduplicated payload and publishes a sorted
    // directory plus the wire data through an MPI-3 shared-memory window.
    // Local ranks then read only the sources in their own retained plan.  This
    // removes the rank fanout that otherwise sends the same dense TDE source
    // as many as 192 times to one Genoa node.
    bool shareLetPayloadsWithinNode = false;
    // Stripe node-level receive/copy work across several MPI ranks.  Twenty-four
    // matches the CCD count of a dual-socket 192-core Genoa node and avoids a
    // single handler becoming the critical path.
    std::size_t letPayloadHandlersPerNode = 24;

    // Temporarily repartition particles by a high-resolution Morton key for
    // gravity, then return accelerations to their hydro owners.  This gives the
    // one-tree-per-rank FMM compact, balanced spatial domains even when the
    // moving Voronoi decomposition overlaps the dense TDE core on many ranks.
    bool spatiallyRedistributeForGravity = true;

    // Hilbert intervals have substantially better spatial locality than
    // Morton intervals at curve discontinuities.  Use them for the temporary
    // gravity ownership split to reduce pathological rank-root extents and
    // the resulting worst-rank LET fanout.  This remains a pure MPI
    // redistribution with one rank per core.
    bool useHilbertGravityRedistribution = true;

    // Blend particle-count and Hilbert-volume quantiles when choosing gravity
    // owners.  A positive value assigns more ranks to sparse, physically large
    // TDE atmosphere intervals instead of giving every interval the same body
    // count.  The implementation also enforces a minimum sampled occupancy so
    // empty Hilbert gaps cannot consume ranks.  Must be in [0,1).
    double hilbertGravityVolumeWeight = 0.0;

    // Keep ownership ranges stable between warm solves, but periodically
    // resample them so secular motion cannot leave a long-running calculation
    // with the decomposition chosen at process startup.  The refresh also
    // rebuilds the retained local tree from a compact root.  Zero preserves
    // the initial splitters for the lifetime of the process.
    std::uint64_t gravityRedistributionRebalanceInterval = 256;

    // Replicate the small process-tree multipole array with one Allgather.
    // For the one-tree-per-rank compatibility solver this replaces one sparse
    // neighborhood collective per process-tree level and makes every remote
    // process M2L source immediately available. Local particle trees and LET
    // payloads remain rank-private.
    bool replicateProcessMultipoles = true;

    // On the selected (one-based) solve, compare a deterministic stratified
    // random sample of target accelerations with a distributed direct sum over
    // every source particle.  Each MPI rank accumulates its own sources and a
    // small Allreduce combines the reference values; particles are never
    // gathered onto one rank.  Zero samples disables the validation.
    std::size_t directErrorSampleCount = 0;
    std::uint64_t directErrorSampleSolve = 2;

    // Upper bound on the LET payload a single rank may request in one exchange.
    // Requests above this are split into several waves, so peak LET memory is
    // set by configuration rather than by the union of a rank's near fields.
    // Zero disables splitting and restores the single-exchange behaviour.
    std::size_t maxLetWaveBytes =
        static_cast<std::size_t>(256) * 1024 * 1024;

    // The one-tree-per-rank compatibility path is the production default. Patch
    // forest remains available explicitly for patch-specific regression and
    // development coverage. Its fixed level-7 partition is deliberately
    // conservative; adaptive splitting remains opt-in.
    bool enablePatchForest = false;
    int minimumPatchLevel = 7;
    int maximumPatchLevel = 7;
    std::size_t targetParticlesPerPatch = 0;
    std::size_t maxLocalPatchCount = 65536;
    std::size_t maxTargetPatchesPerWave = 65536;
    bool useLocalPatchLet = true;

    // The patch-root directory and compact process tree are replicated on every
    // rank.  Crossing this guard requires a distributed directory rather than
    // an unbounded MPI_Allgatherv allocation.
    std::size_t maxReplicatedDescriptorBytes =
        static_cast<std::size_t>(256) * 1024 * 1024;
};

#endif // FMM_DISTRIBUTED_OPTIONS_HPP
