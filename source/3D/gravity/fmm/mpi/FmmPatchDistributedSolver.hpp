#ifndef FMM_PATCH_DISTRIBUTED_SOLVER_HPP
#define FMM_PATCH_DISTRIBUTED_SOLVER_HPP

#ifdef RICH_MPI

#include <cstddef>
#include <cstdint>
#include <vector>

#include <mpi.h>

#include "3D/gravity/fmm/FmmConfig.hpp"
#include "3D/gravity/fmm/FmmDiagnostics.hpp"
#include "3D/gravity/fmm/FmmM2LOperatorCache.hpp"
#include "3D/gravity/fmm/mpi/FmmDistributedOptions.hpp"
#include "3D/gravity/fmm/mpi/FmmPatchForest.hpp"
#include "3D/gravity/fmm/mpi/FmmPatchLetPlan.hpp"
#include "3D/gravity/fmm/mpi/FmmPeerExchange.hpp"
#include "3D/gravity/fmm/mpi/FmmProcessTraversal.hpp"
#include "3D/gravity/fmm/mpi/FmmProcessTree.hpp"

class FmmPatchDistributedSolver
{
public:
    FmmPatchDistributedSolver(const FmmGravityOptions& options,
                              const FmmDistributedOptions& distributedOptions,
                              const MPI_Comm& comm);

    void solve(const std::vector<Vector3D>& positions,
               const std::vector<double>& masses,
               const std::vector<std::uint64_t>& cellIds,
               const Vector3D& domainLower,
               const Vector3D& domainUpper,
               std::vector<Vector3D>& acceleration,
               std::vector<double>* positiveKernelPotential,
               FmmSolveStats& stats);

private:
    FmmGravityOptions options_;
    FmmDistributedOptions distributedOptions_;
    MPI_Comm comm_;
    int rank_;
    int size_;
    std::uint64_t topologyEpoch_;
    std::uint64_t topologyRebuildCount_;

    FmmPatchForest forest_;
    std::vector<FmmPatchRootDescriptor> rootDescriptors_;
    FmmProcessTree processTree_;
    FmmProcessPairPlan processPlan_;
    FmmPatchLetPlan letPlan_;
    FmmM2LOperatorCache operatorCache_;
    FmmPeerExchange processUpExchange_;
    FmmPeerExchange processM2LExchange_;
    FmmPeerExchange processDownExchange_;
};

#endif // RICH_MPI

#endif // FMM_PATCH_DISTRIBUTED_SOLVER_HPP
