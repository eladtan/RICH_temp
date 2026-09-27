#ifndef DISTRIBUTED_FMM_GRAVITY_CALCULATOR_HPP
#define DISTRIBUTED_FMM_GRAVITY_CALCULATOR_HPP

#ifdef RICH_MPI

#include <cstddef>
#include <cstdint>
#include <memory>
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

#include "3D/gravity/fmm/mpi/FmmDistributedOptions.hpp"

class FmmPatchDistributedSolver;

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
               std::vector<double>* positiveKernelPotential = nullptr,
               const std::vector<unsigned char>* targetMask = nullptr);

    const FmmSolveStats& stats() const noexcept { return stats_; }

    // True when RICH_FMM_TRACE requested the per-solve trace.  Read once, on
    // rank 0 of this calculator's communicator, at construction.
    bool solveTraceRequested() const noexcept
    {
        return runtime_.balanceTrace;
    }

private:
    // Environment overrides, read once per calculator at construction: rank 0
    // of comm_ parses and validates them and broadcasts the result on comm_,
    // so every rank of this calculator holds identical settings and no
    // process-wide static couples calculators on different communicators.
    struct RuntimeSettings
    {
        bool balanceTrace = false;               // RICH_FMM_TRACE
        bool gravityResplit = true;              // RICH_FMM_GRAVITY_RESPLIT
        std::uint64_t structuralChangeWindow = 0; // RICH_FMM_STRUCTURAL_INTERVAL
        std::uint64_t directErrorSamples = 0;    // RICH_FMM_DIRECT_ERROR_SAMPLES
        std::uint64_t directErrorEvery = 0;      // RICH_FMM_DIRECT_ERROR_EVERY
        bool geometryLog = false;                // RICH_FMM_GEOM_LOG
        bool targetPrune = true;                 // RICH_FMM_TARGET_PRUNE
        bool maskedOwned = false;                // RICH_FMM_TARGET_OWNED
    };
    void readRuntimeSettings();

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
    void solveOwned(const std::vector<Vector3D>& positions,
                    const std::vector<double>& masses,
                    const std::vector<std::uint64_t>& cellIds,
                    const Vector3D& domainLower,
                    const Vector3D& domainUpper,
                    std::vector<Vector3D>& acceleration,
                    std::vector<double>* positiveKernelPotential,
                    const std::vector<unsigned char>* targetMask);
    void solveRedistributed(const std::vector<Vector3D>& positions,
                            const std::vector<double>& masses,
                            const std::vector<std::uint64_t>& cellIds,
                            const Vector3D& domainLower,
                            const Vector3D& domainUpper,
                            std::vector<Vector3D>& acceleration,
                            std::vector<double>* positiveKernelPotential,
                            const std::vector<unsigned char>* targetMask);
    void sampleDirectAccelerationError(
        const std::vector<Vector3D>& positions,
        const std::vector<double>& masses,
        const std::vector<Vector3D>& acceleration);
    LocalTopologyChange prepareLocalTree(const std::vector<Vector3D>& positions,
                                         const Vector3D& domainLower,
                                         const Vector3D& domainUpper);
    void rebuildTopology(const std::vector<Vector3D>& positions,
                         bool rebuildProcessTopology);
    FmmPatchRootDescriptor localRootDescriptor() const;
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
    std::uint64_t solveCount_;
    // Solve index of the last topology rebuild on a solve whose structural
    // window was open (rebuilds forced while it is closed do not restart it);
    // capacity-driven structural leaf changes are deferred until a window has
    // elapsed since then.
    std::uint64_t lastStructuralChangeSolve_;
    RuntimeSettings runtime_;
    // Domain the gravity-owner splitters were sampled for (the Hilbert keys
    // are normalized by it), and the ski-rental state deciding when a
    // re-sampling pays off: straggler seconds accumulated on warm solves since
    // the last sampling beyond the baseline straggler time (measured on the
    // first warm solve after the sampling: what a fresh split cannot remove),
    // and the measured cost of the last process-topology rebuild (what a
    // re-sampling triggers).
    double gravitySplitterDomain_[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double gravityImbalanceDebtSeconds_ = 0.0;
    double gravityBaselineExcessSeconds_ = 0.0;
    bool gravityBaselinePending_ = true;
    double gravityRebuildCostSeconds_ = 0.0;
    std::uint64_t gravityResampleCount_ = 0;
    // Set for the solve right after a re-sampling: ownership changed, so the
    // retained root and leaf structure are no template for the new particle
    // set (retained leaves would absorb the incoming particles up to the
    // persistent split capacity): every local tree is built afresh and the
    // local, process and LET plans are rebuilt.
    bool forceFreshLocalTree_ = false;
    // Particle population of the last solve: 0 redistributed to gravity
    // owners, 1 the callers' own (RICH_FMM_TARGET_OWNED), -1 none yet.  A
    // change rebuilds every plan: retained radii and admissibility decisions
    // describe the previous population.
    int lastOwnershipMode_ = -1;
    bool ownershipModeChanged_ = false;

    FmmTree localTree_;
    FmmM2LOperatorCache operatorCache_;
    FmmLocalInteractionPlan localInteractionPlan_;
    // Per-node target flags of localTree_ for a pruned solve.
    std::vector<unsigned char> nodeTargetMask_;
    std::vector<double> localMultipoles_;
    std::vector<double> localLocals_;
    std::vector<FmmPatchRootDescriptor> rootDescriptors_;
    FmmProcessTree processTree_;
    FmmProcessPairPlan processPlan_;
    std::vector<std::uint64_t> gravityRedistributionSplitters_;
    // Print-only: solve index at which the splitters above were sampled.
    std::uint64_t gravitySplitterSolve_ = 0;
    FmmLetPlan letPlan_;
    FmmPeerExchange processUpExchange_;
    FmmPeerExchange processM2LExchange_;
    FmmPeerExchange processDownExchange_;
    std::unique_ptr<FmmPatchDistributedSolver> patchSolver_;
};

#endif // RICH_MPI

#endif // DISTRIBUTED_FMM_GRAVITY_CALCULATOR_HPP
