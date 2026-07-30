#ifndef FMM_PACKETS_HPP
#define FMM_PACKETS_HPP

#ifdef RICH_MPI

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "3D/elementary/Vector3D.hpp"
#include "misc/universal_error.hpp"

static constexpr std::uint32_t FMM_MPI_PACKET_MAGIC = 0x52464d4du; // "RFMM"
static constexpr std::uint16_t FMM_MPI_PACKET_VERSION = 7u;

enum class FmmPacketKind : std::uint16_t
{
    ProcessPairTask = 1,
    ProcessDependency = 2,
    DescriptorRequest = 3,
    DescriptorReply = 4,
    Subscription = 5,
    ProcessCoefficient = 6,
    LetPayload = 7,
    PayloadCapacity = 8
};

struct FmmPacketStamp
{
    std::uint32_t magic = FMM_MPI_PACKET_MAGIC;
    std::uint16_t version = FMM_MPI_PACKET_VERSION;
    std::uint16_t kind = 0;
    std::uint64_t topologyEpoch = 0;
};

inline FmmPacketStamp fmmPacketStamp(FmmPacketKind kind,
                                     std::uint64_t topologyEpoch)
{
    FmmPacketStamp result;
    result.kind = static_cast<std::uint16_t>(kind);
    result.topologyEpoch = topologyEpoch;
    return result;
}

inline void validateFmmPacketStamp(const FmmPacketStamp& stamp,
                                   FmmPacketKind expectedKind,
                                   std::uint64_t expectedEpoch,
                                   const char* context)
{
    if(stamp.magic != FMM_MPI_PACKET_MAGIC ||
       stamp.version != FMM_MPI_PACKET_VERSION ||
       stamp.kind != static_cast<std::uint16_t>(expectedKind) ||
       stamp.topologyEpoch != expectedEpoch)
    {
        UniversalError error(std::string(context) + ": invalid or stale MPI packet");
        error.addEntry("packet_magic", stamp.magic);
        error.addEntry("packet_version", stamp.version);
        error.addEntry("packet_kind", stamp.kind);
        error.addEntry("packet_epoch", stamp.topologyEpoch);
        error.addEntry("expected_epoch", expectedEpoch);
        throw error;
    }
}

struct FmmByteView
{
    const char* data = nullptr;
    std::size_t size = 0;
};

struct FmmPatchRootDescriptor
{
    double center[3] = {0.0, 0.0, 0.0};
    double halfSize = 0.0;
    double radius = 0.0;
    std::uint64_t particleCount = 0;
    std::uint64_t topologyHash = 0;
    std::uint64_t epoch = 0;
    std::uint64_t patchId = 0;
    std::uint64_t latticeId = 0;
    std::int64_t latticeCenter[3] = {0, 0, 0};
    std::uint64_t latticeHalfUnits = 0;
    std::uint32_t magic = FMM_MPI_PACKET_MAGIC;
    std::uint16_t version = FMM_MPI_PACKET_VERSION;
    std::uint16_t reserved16 = 0;
    int ownerRank = -1;
    int active = 0;
    int rootLeaf = 1;
    int childMask = 0;

    Vector3D centerVector() const { return Vector3D(center[0], center[1], center[2]); }
};

using FmmRankRootDescriptor = FmmPatchRootDescriptor;

struct FmmRemoteNodeDescriptor
{
    double center[3] = {0.0, 0.0, 0.0};
    double halfSize = 0.0;
    double radius = 0.0;
    std::uint64_t spatialKey = 0;
    std::uint64_t patchId = 0;
    std::uint64_t particleCount = 0;
    std::uint64_t topologyEpoch = 0;
    int sourceRank = -1;
    int isLeaf = 1;
    int childMask = 0;
    int reserved = 0;

    Vector3D centerVector() const { return Vector3D(center[0], center[1], center[2]); }
    double geometricRadius() const;
};

struct FmmProcessPairTask
{
    FmmPacketStamp stamp;
    std::uint64_t target = 0;
    std::uint64_t source = 0;
};

struct FmmDescriptorRequest
{
    FmmPacketStamp stamp;
    std::uint64_t patchId = 0;
    std::uint64_t spatialKey = 0;
};

struct FmmDescriptorReply
{
    FmmPacketStamp stamp;
    std::uint64_t requestedParentKey = 0;
    FmmRemoteNodeDescriptor child;
    int childCount = 0;
    int childOrdinal = 0;
};

enum class FmmSubscriptionKind : int
{
    Multipole = 1,
    Particles = 2
};

struct FmmSubscription
{
    FmmPacketStamp stamp;
    std::uint64_t patchId = 0;
    std::uint64_t spatialKey = 0;
    int kind = 0;
    int waveIndex = 0;
};

struct FmmProcessDependency
{
    FmmPacketStamp stamp;
    std::uint64_t sourceNode = 0;
    int kind = 0;
    int reserved = 0;
};

struct FmmProcessCoefficientHeader
{
    FmmPacketStamp stamp;
    std::uint64_t nodeIndex = 0;
};

struct FmmPayloadRecordHeader
{
    FmmPacketStamp stamp;
    std::uint64_t patchId = 0;
    std::uint64_t spatialKey = 0;
    std::uint64_t count = 0;
    int kind = 0;
    int waveIndex = 0;
};

struct FmmPayloadCapacity
{
    FmmPacketStamp stamp;
    std::uint64_t patchId = 0;
    std::uint64_t spatialKey = 0;
    std::uint64_t particleCount = 0;
};

struct FmmWireParticle
{
    double position[3] = {0.0, 0.0, 0.0};
    double mass = 0.0;
    std::uint64_t cellId = 0;
    std::uint64_t ownerLocalIndex = 0;
    int ownerRank = -1;
    int reserved = 0;

    Vector3D positionVector() const
    {
        return Vector3D(position[0], position[1], position[2]);
    }
};

// Patch LET P2P traffic never needs body identity: source and target patches
// are owned by different MPI ranks, and the patch interaction plan already
// identifies the source leaf. Keep only the values used by the force kernel.
struct FmmPatchWireParticle
{
    double position[3] = {0.0, 0.0, 0.0};
    double mass = 0.0;

    Vector3D positionVector() const
    {
        return Vector3D(position[0], position[1], position[2]);
    }
};

static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
              "Distributed FMM wire protocol requires IEEE-754 binary64 doubles");
static_assert(sizeof(int) == 4,
              "Distributed FMM wire protocol requires 32-bit int");
static_assert(sizeof(std::uint64_t) == 8,
              "Distributed FMM wire protocol requires 64-bit uint64_t");
static_assert(sizeof(std::size_t) <= sizeof(std::uint64_t),
              "Distributed FMM wire protocol requires <=64-bit size_t");
static_assert(sizeof(FmmPacketStamp) == 16,
              "Distributed FMM packet stamp has unsupported padding");
static_assert(sizeof(FmmPatchRootDescriptor) == 136,
              "Distributed FMM root descriptor has unsupported padding");
static_assert(sizeof(FmmRemoteNodeDescriptor) == 88,
              "Distributed FMM node descriptor has unsupported padding");
static_assert(sizeof(FmmProcessPairTask) == 32,
              "Distributed FMM process task has unsupported padding");
static_assert(sizeof(FmmDescriptorRequest) == 32,
              "Distributed FMM descriptor request has unsupported padding");
static_assert(sizeof(FmmDescriptorReply) == 120,
              "Distributed FMM descriptor reply has unsupported padding");
static_assert(sizeof(FmmSubscription) == 40,
              "Distributed FMM subscription has unsupported padding");
static_assert(sizeof(FmmProcessDependency) == 32,
              "Distributed FMM dependency has unsupported padding");
static_assert(sizeof(FmmProcessCoefficientHeader) == 24,
              "Distributed FMM coefficient header has unsupported padding");
static_assert(sizeof(FmmPayloadRecordHeader) == 48,
              "Distributed FMM payload header has unsupported padding");
static_assert(sizeof(FmmPayloadCapacity) == 40,
              "Distributed FMM payload capacity has unsupported padding");
static_assert(sizeof(FmmWireParticle) == 56,
              "Distributed FMM wire particle has unsupported padding");
static_assert(sizeof(FmmPatchWireParticle) == 32,
              "Distributed patch FMM wire particle has unsupported padding");
static_assert(std::is_trivially_copyable<FmmPacketStamp>::value,
              "FMM packet stamp must be trivially copyable");
static_assert(std::is_trivially_copyable<FmmPatchRootDescriptor>::value,
              "FMM root descriptor must be trivially copyable");
static_assert(std::is_trivially_copyable<FmmRemoteNodeDescriptor>::value,
              "FMM node descriptor must be trivially copyable");
static_assert(std::is_trivially_copyable<FmmProcessPairTask>::value,
              "FMM process task must be trivially copyable");
static_assert(std::is_trivially_copyable<FmmDescriptorRequest>::value,
              "FMM descriptor request must be trivially copyable");
static_assert(std::is_trivially_copyable<FmmDescriptorReply>::value,
              "FMM descriptor reply must be trivially copyable");
static_assert(std::is_trivially_copyable<FmmSubscription>::value,
              "FMM subscription must be trivially copyable");
static_assert(std::is_trivially_copyable<FmmProcessDependency>::value,
              "FMM dependency must be trivially copyable");
static_assert(std::is_trivially_copyable<FmmProcessCoefficientHeader>::value,
              "FMM coefficient header must be trivially copyable");
static_assert(std::is_trivially_copyable<FmmPayloadRecordHeader>::value,
              "FMM payload header must be trivially copyable");
static_assert(std::is_trivially_copyable<FmmPayloadCapacity>::value,
              "FMM payload capacity must be trivially copyable");
static_assert(std::is_trivially_copyable<FmmWireParticle>::value,
              "FMM wire particle must be trivially copyable");
static_assert(std::is_trivially_copyable<FmmPatchWireParticle>::value,
              "FMM patch wire particle must be trivially copyable");

namespace FmmPacketIO
{
template<typename T>
void appendPod(std::vector<char>& buffer, const T& value)
{
    static_assert(std::is_trivially_copyable<T>::value,
                  "FMM packets require trivially copyable values");
    if(sizeof(T) > std::numeric_limits<std::size_t>::max() - buffer.size())
        throw UniversalError("FmmPacketIO::appendPod: size overflow");
    const std::size_t old = buffer.size();
    buffer.resize(old + sizeof(T));
    std::memcpy(buffer.data() + old, &value, sizeof(T));
}

template<typename T>
T readPod(FmmByteView buffer, std::size_t& offset)
{
    static_assert(std::is_trivially_copyable<T>::value,
                  "FMM packets require trivially copyable values");
    if(offset > buffer.size || sizeof(T) > buffer.size - offset)
        throw UniversalError("FmmPacketIO::readPod: truncated packet");
    T value;
    std::memcpy(&value, buffer.data + offset, sizeof(T));
    offset += sizeof(T);
    return value;
}

template<typename T>
T readPod(const std::vector<char>& buffer, std::size_t& offset)
{
    return readPod<T>(FmmByteView{buffer.data(), buffer.size()}, offset);
}

inline void appendDoubles(std::vector<char>& buffer,
                          const double* values,
                          std::size_t count)
{
    if(count > (std::numeric_limits<std::size_t>::max() - buffer.size()) /
               sizeof(double))
        throw UniversalError("FmmPacketIO::appendDoubles: size overflow");
    const std::size_t bytes = count * sizeof(double);
    const std::size_t old = buffer.size();
    buffer.resize(old + bytes);
    if(bytes != 0)
        std::memcpy(buffer.data() + old, values, bytes);
}

inline void readDoubles(FmmByteView buffer,
                        std::size_t& offset,
                        double* values,
                        std::size_t count)
{
    if(count > std::numeric_limits<std::size_t>::max() / sizeof(double))
        throw UniversalError("FmmPacketIO::readDoubles: size overflow");
    const std::size_t bytes = count * sizeof(double);
    if(offset > buffer.size || bytes > buffer.size - offset)
        throw UniversalError("FmmPacketIO::readDoubles: truncated packet");
    if(bytes != 0)
        std::memcpy(values, buffer.data + offset, bytes);
    offset += bytes;
}

inline void readDoubles(const std::vector<char>& buffer,
                        std::size_t& offset,
                        double* values,
                        std::size_t count)
{
    readDoubles(FmmByteView{buffer.data(), buffer.size()}, offset, values, count);
}

inline std::size_t remaining(FmmByteView buffer, std::size_t offset)
{
    if(offset > buffer.size)
        throw UniversalError("FmmPacketIO::remaining: invalid packet offset");
    return buffer.size - offset;
}
}

#endif // RICH_MPI

#endif // FMM_PACKETS_HPP
