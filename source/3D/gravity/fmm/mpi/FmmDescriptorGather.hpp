#ifndef FMM_DESCRIPTOR_GATHER_HPP
#define FMM_DESCRIPTOR_GATHER_HPP

#ifdef RICH_MPI

#include <cstddef>
#include <cstdint>
#include <vector>

#include <mpi.h>

#include "3D/gravity/fmm/mpi/FmmPackets.hpp"

class FmmDescriptorGather
{
public:
    static std::vector<FmmPatchRootDescriptor> gather(
        const std::vector<FmmPatchRootDescriptor>& localDescriptors,
        std::uint64_t localParticleCount,
        std::uint64_t topologyEpoch,
        std::size_t maxReplicatedDescriptorBytes,
        const MPI_Comm& comm);
};

#endif // RICH_MPI

#endif // FMM_DESCRIPTOR_GATHER_HPP
