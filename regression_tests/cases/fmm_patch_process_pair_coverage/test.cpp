#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <tuple>
#include <vector>

#include <mpi.h>

#include "source/3D/gravity/fmm/mpi/FmmProcessTraversal.hpp"
#include "source/3D/gravity/fmm/mpi/FmmProcessTree.hpp"

namespace
{
std::uint64_t patchId(int firstOctant, int secondOctant)
{
    return (((std::uint64_t(1) << 3u) |
             static_cast<std::uint64_t>(firstOctant)) << 3u) |
           static_cast<std::uint64_t>(secondOctant);
}

std::vector<FmmPatchRootDescriptor> descriptorsForCase(
    int size, int which, std::uint64_t epoch)
{
    std::vector<FmmPatchRootDescriptor> descriptors;
    for(int rank = 0; rank < size; ++rank)
    {
        if(which == 1 && size >= 3 && rank == size - 1)
            continue;
        const int patchesPerRank = which == 0 ? 1 : 2;
        for(int localPatch = 0; localPatch < patchesPerRank; ++localPatch)
        {
            FmmPatchRootDescriptor descriptor;
            descriptor.ownerRank = rank;
            descriptor.patchId = patchesPerRank == 1 ? FMM_COMPAT_PATCH_ID :
                patchId((rank + localPatch) % 8,
                        (3 * rank + 5 * localPatch) % 8);
            descriptor.epoch = epoch;
            descriptor.active = 1;
            descriptor.particleCount =
                static_cast<std::uint64_t>(rank + localPatch + 1);
            descriptor.rootLeaf = 1;
            descriptor.childMask = 0;
            if(which == 0)
            {
                descriptor.center[0] = 3.0 * rank;
                descriptor.center[1] = 0.1 * (rank % 2);
                descriptor.center[2] = 0.0;
                descriptor.halfSize = 0.35;
            }
            else if(which == 1)
            {
                descriptor.center[0] = 0.7 * rank + 0.12 * localPatch;
                descriptor.center[1] = 0.04 * ((rank + localPatch) % 3);
                descriptor.center[2] = -0.03 * ((2 * rank + localPatch) % 4);
                descriptor.halfSize = 0.22;
            }
            else
            {
                // Identical and overlapping leaf geometry on different owners
                // must remain distinct because patch identity includes rank.
                descriptor.center[0] = 0.25 * localPatch;
                descriptor.center[1] = 0.0;
                descriptor.center[2] = 0.0;
                descriptor.halfSize = 0.18;
            }
            descriptor.radius = 0.5 * descriptor.halfSize;
            descriptors.push_back(descriptor);
        }
    }
    std::sort(descriptors.begin(), descriptors.end(),
              [](const FmmPatchRootDescriptor& first,
                 const FmmPatchRootDescriptor& second) {
                  return std::tie(first.ownerRank, first.patchId) <
                         std::tie(second.ownerRank, second.patchId);
              });
    return descriptors;
}

void collectLeafKeys(const FmmProcessTree& tree,
                     std::size_t node,
                     std::vector<FmmPatchKey>& leaves)
{
    const FmmProcessNode& value = tree.nodes()[node];
    if(value.isLeaf())
    {
        leaves.push_back(value.leafKey());
        return;
    }
    collectLeafKeys(tree, value.left, leaves);
    collectLeafKeys(tree, value.right, leaves);
}

bool auditCase(int rank, int size, int which)
{
    const std::uint64_t epoch = static_cast<std::uint64_t>(which + 1);
    const std::vector<FmmPatchRootDescriptor> descriptors =
        descriptorsForCase(size, which, epoch);
    FmmProcessTree tree;
    tree.build(descriptors);
    const FmmProcessPairPlan plan = FmmProcessTraversal::build(
        tree, 0.5, epoch, rank, MPI_COMM_WORLD);

    std::map<FmmPatchKey, std::size_t> indexByPatch;
    for(std::size_t i = 0; i < descriptors.size(); ++i)
    {
        indexByPatch.emplace(
            FmmPatchKey{descriptors[i].ownerRank, descriptors[i].patchId}, i);
    }
    const std::size_t patchCount = descriptors.size();
    const std::size_t matrixSize = patchCount * patchCount;
    std::vector<unsigned long long> local(matrixSize, 0);
    const auto addPair = [&](const FmmPatchKey& target,
                             const FmmPatchKey& source) {
        const auto targetIt = indexByPatch.find(target);
        const auto sourceIt = indexByPatch.find(source);
        if(targetIt == indexByPatch.end() || sourceIt == indexByPatch.end())
            return false;
        ++local[targetIt->second * patchCount + sourceIt->second];
        return true;
    };

    for(const FmmProcessM2LPair& pair : plan.localM2LPairs)
    {
        if(tree.nodes()[pair.targetNode].owner != rank)
            return false;
        std::vector<FmmPatchKey> targets;
        std::vector<FmmPatchKey> sources;
        collectLeafKeys(tree, pair.targetNode, targets);
        collectLeafKeys(tree, pair.sourceNode, sources);
        for(const FmmPatchKey& target : targets)
            for(const FmmPatchKey& source : sources)
                if(!addPair(target, source))
                    return false;
    }
    for(const FmmPatchKey& patch : plan.localSelfPatches)
        if(!addPair(patch, patch))
            return false;
    for(const FmmPatchPair& pair : plan.localCrossPatchPairs)
        if(!addPair(pair.target, pair.source))
            return false;
    for(const FmmPatchPair& pair : plan.remoteLetPairs)
        if(!addPair(pair.target, pair.source))
            return false;

    std::vector<unsigned long long> global(matrixSize, 0);
    MPI_Allreduce(local.data(), global.data(), static_cast<int>(matrixSize),
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return std::all_of(global.begin(), global.end(),
                       [](unsigned long long count) { return count == 1ull; });
}
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int localPass = 1;
    for(int which = 0; which < 3; ++which)
        localPass = localPass && auditCase(rank, size, which);
    int globalPass = 0;
    MPI_Allreduce(&localPass, &globalPass, 1, MPI_INT, MPI_LAND,
                  MPI_COMM_WORLD);

    if(rank == 0)
    {
        std::ofstream output("fmm_patch_process_pair_coverage_metrics.txt");
        output << "ranks " << size << "\n";
        output << "cases 3\n";
        output << "pass " << globalPass << "\n";
        std::cout << "fmm_patch_process_pair_coverage ranks=" << size
                  << " pass=" << globalPass << std::endl;
    }
    MPI_Finalize();
    return globalPass ? 0 : 1;
}
