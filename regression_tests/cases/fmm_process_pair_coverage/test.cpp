#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include <mpi.h>

#include "source/3D/gravity/fmm/mpi/FmmPatchKey.hpp"
#include "source/3D/gravity/fmm/mpi/FmmProcessTraversal.hpp"
#include "source/3D/gravity/fmm/mpi/FmmProcessTree.hpp"

namespace
{
std::vector<FmmPatchRootDescriptor> descriptorsForCase(int size, int which,
                                                       std::uint64_t epoch)
{
    std::vector<FmmPatchRootDescriptor> descriptors(
        static_cast<std::size_t>(size));
    for(int rank = 0; rank < size; ++rank)
    {
        FmmPatchRootDescriptor& descriptor =
            descriptors[static_cast<std::size_t>(rank)];
        descriptor.ownerRank = rank;
        descriptor.patchId = FMM_COMPAT_PATCH_ID;
        descriptor.epoch = epoch;
        const bool empty = which == 1 && size >= 3 && rank == size - 1;
        descriptor.active = empty ? 0 : 1;
        if(empty)
            continue;
        descriptor.particleCount = static_cast<std::uint64_t>(rank + 1);
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
            descriptor.center[0] = 0.2 * rank;
            descriptor.center[1] = 0.03 * ((rank * 5) % 3);
            descriptor.center[2] = -0.02 * ((rank * 7) % 4);
            descriptor.halfSize = 0.45;
        }
        else
        {
            descriptor.center[0] = 0.8 * ((rank * 5) % std::max(1, size));
            descriptor.center[1] = 0.37 * ((rank * 3) % 5);
            descriptor.center[2] = 0.21 * ((rank * 7) % 4);
            descriptor.halfSize = 0.08 + 0.025 * (rank % 4);
        }
    }
    return descriptors;
}

void collectLeafRanks(const FmmProcessTree& tree, std::size_t node,
                      std::vector<int>& leaves)
{
    const FmmProcessNode& value = tree.nodes()[node];
    if(value.isLeaf())
    {
        leaves.push_back(value.leafOwnerRank);
        return;
    }
    collectLeafRanks(tree, value.left, leaves);
    collectLeafRanks(tree, value.right, leaves);
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

    const std::size_t matrixSize = static_cast<std::size_t>(size) *
                                   static_cast<std::size_t>(size);
    std::vector<unsigned long long> local(matrixSize, 0);
    for(const FmmProcessM2LPair& pair : plan.localM2LPairs)
    {
        if(tree.nodes()[pair.targetNode].owner != rank)
            return false;
        std::vector<int> targets;
        std::vector<int> sources;
        collectLeafRanks(tree, pair.targetNode, targets);
        collectLeafRanks(tree, pair.sourceNode, sources);
        for(int target : targets)
            for(int source : sources)
                ++local[static_cast<std::size_t>(target) * size + source];
    }

    const std::size_t localLeaf = tree.leafForRank(rank);
    if(localLeaf != FmmProcessTree::invalidIndex())
    {
        local[static_cast<std::size_t>(rank) * size + rank] +=
            plan.localSelfRankCount;
        for(int source : plan.letSourceRanks)
            ++local[static_cast<std::size_t>(rank) * size + source];
    }
    else if(plan.localSelfRankCount != 0 || !plan.letSourceRanks.empty() ||
            !plan.localM2LPairs.empty())
    {
        return false;
    }

    std::vector<unsigned long long> global(matrixSize, 0);
    MPI_Allreduce(local.data(), global.data(), static_cast<int>(matrixSize),
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    for(int target = 0; target < size; ++target)
    {
        for(int source = 0; source < size; ++source)
        {
            const bool expected = descriptors[static_cast<std::size_t>(target)].active != 0 &&
                                  descriptors[static_cast<std::size_t>(source)].active != 0;
            const unsigned long long count =
                global[static_cast<std::size_t>(target) * size + source];
            if(count != (expected ? 1ull : 0ull))
                return false;
        }
    }
    return true;
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
        std::ofstream output("fmm_process_pair_coverage_metrics.txt");
        output << "ranks " << size << "\n";
        output << "cases 3\n";
        output << "pass " << globalPass << "\n";
        std::cout << "fmm_process_pair_coverage ranks=" << size
                  << " pass=" << globalPass << std::endl;
    }
    MPI_Finalize();
    return globalPass ? 0 : 1;
}
