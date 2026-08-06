#include "3D/gravity/fmm/mpi/FmmDescriptorGather.hpp"

#ifdef RICH_MPI

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <string>
#include <tuple>

#include "3D/gravity/fmm/mpi/FmmGlobalDyadicLattice.hpp"
#include "3D/gravity/fmm/mpi/FmmPatchKey.hpp"
#include "misc/universal_error.hpp"

namespace
{
bool finiteDescriptor(const FmmPatchRootDescriptor& descriptor)
{
    const Vector3D center = descriptor.centerVector();
    const double cubeRadius = std::sqrt(3.0) * descriptor.halfSize;
    const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, cubeRadius);
    return std::isfinite(center.x) && std::isfinite(center.y) &&
        std::isfinite(center.z) && descriptor.halfSize > 0.0 &&
        std::isfinite(descriptor.halfSize) && descriptor.radius >= 0.0 &&
        std::isfinite(descriptor.radius) &&
        descriptor.radius <= cubeRadius + tolerance &&
        std::isfinite(center.x - descriptor.halfSize) &&
        std::isfinite(center.x + descriptor.halfSize) &&
        std::isfinite(center.y - descriptor.halfSize) &&
        std::isfinite(center.y + descriptor.halfSize) &&
        std::isfinite(center.z - descriptor.halfSize) &&
        std::isfinite(center.z + descriptor.halfSize);
}

void validateDescriptor(const FmmPatchRootDescriptor& descriptor,
                        int expectedOwner,
                        std::uint64_t expectedEpoch,
                        const char* context)
{
    if(descriptor.magic != FMM_MPI_PACKET_MAGIC ||
       descriptor.version != FMM_MPI_PACKET_VERSION ||
       descriptor.ownerRank != expectedOwner || descriptor.active != 1 ||
       descriptor.particleCount == 0 ||
       descriptor.epoch != expectedEpoch ||
       !FmmGlobalDyadicLattice::isValidPatchId(descriptor.patchId) ||
       descriptor.latticeId == 0 || descriptor.latticeHalfUnits == 0 ||
       descriptor.latticeHalfUnits > static_cast<std::uint64_t>(
           std::numeric_limits<std::int64_t>::max()) ||
       (descriptor.rootLeaf != 0 && descriptor.rootLeaf != 1) ||
       descriptor.childMask < 0 || descriptor.childMask > 255 ||
       (descriptor.rootLeaf != 0 && descriptor.childMask != 0) ||
       (descriptor.rootLeaf == 0 && descriptor.childMask == 0) ||
       !finiteDescriptor(descriptor))
    {
        UniversalError error(std::string(context) + ": invalid patch descriptor");
        error.addEntry("owner_rank", descriptor.ownerRank);
        error.addEntry("expected_owner", expectedOwner);
        error.addEntry("patch_id", descriptor.patchId);
        error.addEntry("particle_count", descriptor.particleCount);
        error.addEntry("descriptor_epoch", descriptor.epoch);
        error.addEntry("expected_epoch", expectedEpoch);
        throw error;
    }
}

void collectiveRequire(bool localOk,
                       const std::string& localMessage,
                       const char* context,
                       const MPI_Comm& comm)
{
    int local = localOk ? 1 : 0;
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_LAND, comm);
    if(global == 0)
    {
        if(localOk)
            throw UniversalError(std::string(context) +
                                 ": failed on another MPI rank");
        throw UniversalError(localMessage.empty() ? std::string(context) :
                                                   localMessage);
    }
}
}

std::vector<FmmPatchRootDescriptor> FmmDescriptorGather::gather(
    const std::vector<FmmPatchRootDescriptor>& localDescriptors,
    std::uint64_t localParticleCount,
    std::uint64_t topologyEpoch,
    std::size_t maxReplicatedDescriptorBytes,
    const MPI_Comm& comm)
{
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    bool localValid = maxReplicatedDescriptorBytes > 0;
    std::string localError;
    try
    {
        FmmPatchKey previous;
        bool havePrevious = false;
        std::uint64_t descriptorParticles = 0;
        for(const FmmPatchRootDescriptor& descriptor : localDescriptors)
        {
            validateDescriptor(descriptor, rank, topologyEpoch,
                               "FmmDescriptorGather::gather local");
            const FmmPatchKey key{descriptor.ownerRank, descriptor.patchId};
            if(havePrevious && !(previous < key))
                throw UniversalError(
                    "FmmDescriptorGather::gather: local descriptors are not strictly sorted");
            previous = key;
            havePrevious = true;
            if(descriptor.particleCount >
               std::numeric_limits<std::uint64_t>::max() - descriptorParticles)
                throw UniversalError(
                    "FmmDescriptorGather::gather: local particle count overflow");
            descriptorParticles += descriptor.particleCount;
        }
        if(descriptorParticles != localParticleCount)
            throw UniversalError(
                "FmmDescriptorGather::gather: local descriptor particle sum mismatch");
    }
    catch(const UniversalError& error)
    {
        localValid = false;
        localError = error.getErrorMessage();
    }
    collectiveRequire(localValid, localError,
                      "FmmDescriptorGather::gather local validation", comm);

    const unsigned long long localCount =
        static_cast<unsigned long long>(localDescriptors.size());
    std::vector<unsigned long long> counts(static_cast<std::size_t>(size), 0);
    MPI_Allgather(&localCount, 1, MPI_UNSIGNED_LONG_LONG,
                  counts.data(), 1, MPI_UNSIGNED_LONG_LONG, comm);

    std::vector<int> byteCounts(static_cast<std::size_t>(size), 0);
    std::vector<int> displacements(static_cast<std::size_t>(size), 0);
    std::size_t totalCount = 0;
    std::size_t totalBytes = 0;
    bool layoutValid = true;
    std::string layoutError;
    for(int owner = 0; owner < size; ++owner)
    {
        const unsigned long long count64 = counts[static_cast<std::size_t>(owner)];
        if(count64 > static_cast<unsigned long long>(
               std::numeric_limits<std::size_t>::max()))
        {
            layoutValid = false;
            layoutError = "FmmDescriptorGather::gather: descriptor count exceeds size_t";
            break;
        }
        const std::size_t count = static_cast<std::size_t>(count64);
        if(count > std::numeric_limits<std::size_t>::max() /
                       sizeof(FmmPatchRootDescriptor))
        {
            layoutValid = false;
            layoutError = "FmmDescriptorGather::gather: descriptor byte count overflow";
            break;
        }
        const std::size_t bytes = count * sizeof(FmmPatchRootDescriptor);
        if(bytes > static_cast<std::size_t>(INT_MAX) ||
           totalBytes > static_cast<std::size_t>(INT_MAX) - bytes)
        {
            layoutValid = false;
            layoutError =
                "FmmDescriptorGather::gather: MPI_Allgatherv int count/displacement limit exceeded";
            break;
        }
        if(totalCount > std::numeric_limits<std::size_t>::max() - count)
        {
            layoutValid = false;
            layoutError = "FmmDescriptorGather::gather: total descriptor count overflow";
            break;
        }
        displacements[static_cast<std::size_t>(owner)] =
            static_cast<int>(totalBytes);
        byteCounts[static_cast<std::size_t>(owner)] = static_cast<int>(bytes);
        totalBytes += bytes;
        totalCount += count;
    }
    if(layoutValid && totalBytes > maxReplicatedDescriptorBytes)
    {
        layoutValid = false;
        layoutError =
            "FmmDescriptorGather::gather: replicated descriptor budget exceeded; a distributed patch directory is required";
    }
    collectiveRequire(layoutValid, layoutError,
                      "FmmDescriptorGather::gather layout validation", comm);

    std::vector<FmmPatchRootDescriptor> result(totalCount);
    const int localBytes = static_cast<int>(
        localDescriptors.size() * sizeof(FmmPatchRootDescriptor));
    MPI_Allgatherv(localDescriptors.empty() ? nullptr : localDescriptors.data(),
                   localBytes, MPI_BYTE,
                   result.empty() ? nullptr : result.data(),
                   byteCounts.data(), displacements.data(), MPI_BYTE, comm);

    std::sort(result.begin(), result.end(),
              [](const FmmPatchRootDescriptor& first,
                 const FmmPatchRootDescriptor& second) {
                  return std::tie(first.ownerRank, first.patchId) <
                         std::tie(second.ownerRank, second.patchId);
              });

    std::uint64_t descriptorParticleCount = 0;
    std::uint64_t commonLatticeId = 0;
    FmmPatchKey previous;
    bool havePrevious = false;
    for(const FmmPatchRootDescriptor& descriptor : result)
    {
        validateDescriptor(descriptor, descriptor.ownerRank, topologyEpoch,
                           "FmmDescriptorGather::gather global");
        if(descriptor.ownerRank < 0 || descriptor.ownerRank >= size)
            throw UniversalError(
                "FmmDescriptorGather::gather: descriptor owner rank out of range");
        if(commonLatticeId == 0)
            commonLatticeId = descriptor.latticeId;
        else if(descriptor.latticeId != commonLatticeId)
            throw UniversalError(
                "FmmDescriptorGather::gather: inconsistent global lattice IDs");
        const FmmPatchKey key{descriptor.ownerRank, descriptor.patchId};
        if(havePrevious && !(previous < key))
            throw UniversalError(
                "FmmDescriptorGather::gather: duplicate global patch key");
        previous = key;
        havePrevious = true;
        if(descriptor.particleCount >
           std::numeric_limits<std::uint64_t>::max() - descriptorParticleCount)
            throw UniversalError(
                "FmmDescriptorGather::gather: global descriptor particle sum overflow");
        descriptorParticleCount += descriptor.particleCount;
    }

    unsigned long long localParticles =
        static_cast<unsigned long long>(localParticleCount);
    unsigned long long globalParticles = 0;
    MPI_Allreduce(&localParticles, &globalParticles, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
    if(descriptorParticleCount != static_cast<std::uint64_t>(globalParticles))
        throw UniversalError(
            "FmmDescriptorGather::gather: global descriptor particle sum mismatch");

    return result;
}

#endif // RICH_MPI
