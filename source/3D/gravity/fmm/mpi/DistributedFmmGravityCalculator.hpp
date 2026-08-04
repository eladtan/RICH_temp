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
    void solveOwned(const std::vector<Vector3D>& positions,
                    const std::vector<double>& masses,
                    const std::vector<std::uint64_t>& cellIds,
                    const Vector3D& domainLower,
                    const Vector3D& domainUpper,
                    std::vector<Vector3D>& acceleration,
                    std::vector<double>* positiveKernelPotential);
    void solveRedistributed(const std::vector<Vector3D>& positions,
                            const std::vector<double>& masses,
                            const std::vector<std::uint64_t>& cellIds,
                            const Vector3D& domainLower,
                            const Vector3D& domainUpper,
                            std::vector<Vector3D>& acceleration,
                            std::vector<double>* positiveKernelPotential);
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

    FmmTree localTree_;
    FmmM2LOperatorCache operatorCache_;
    FmmLocalInteractionPlan localInteractionPlan_;
    std::vector<double> localMultipoles_;
    std::vector<double> localLocals_;
    std::vector<FmmPatchRootDescriptor> rootDescriptors_;
    FmmProcessTree processTree_;
    FmmProcessPairPlan processPlan_;
    std::vector<std::uint64_t> gravityRedistributionSplitters_;
    FmmLetPlan letPlan_;
    FmmPeerExchange processUpExchange_;
    FmmPeerExchange processM2LExchange_;
    FmmPeerExchange processDownExchange_;
    std::unique_ptr<FmmPatchDistributedSolver> patchSolver_;
};

#endif // RICH_MPI

#endif // DISTRIBUTED_FMM_GRAVITY_CALCULATOR_HPP
