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
    void rebuildTopology(bool rebuildProcessTopology);
    FmmRankRootDescriptor localRootDescriptor() const;

    FmmGravityOptions options_;
    FmmDistributedOptions distributedOptions_;
    MPI_Comm comm_;
    int rank_;
    int size_;

    FmmSolveStats stats_;
    FmmRootGeometry localRoot_;
    bool rootInitialized_;
    std::uint64_t lastLocalTopologyHash_;
    std::vector<std::uint64_t> lastLocalTopologySignature_;
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
