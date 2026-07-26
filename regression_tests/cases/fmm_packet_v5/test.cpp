#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "source/3D/gravity/fmm/mpi/FmmPackets.hpp"

namespace
{
template<typename T>
bool roundTrip(const T& value)
{
    std::vector<char> buffer;
    FmmPacketIO::appendPod(buffer, value);
    std::size_t offset = 0;
    const T decoded = FmmPacketIO::readPod<T>(buffer, offset);
    return offset == buffer.size() && std::memcmp(&value, &decoded, sizeof(T)) == 0;
}

bool testPatchRootDescriptor()
{
    FmmPatchRootDescriptor descriptor;
    descriptor.ownerRank = 5;
    descriptor.patchId = 0x1234abcd5678ef00ull;
    descriptor.active = 1;
    descriptor.particleCount = 99;
    descriptor.center[0] = 1.25;
    descriptor.halfSize = 0.5;
    descriptor.radius = 0.866;
    return roundTrip(descriptor);
}

bool testRemoteNodeDescriptor()
{
    FmmRemoteNodeDescriptor descriptor;
    descriptor.sourceRank = 2;
    descriptor.patchId = 0xfeedfacecafebeefull;
    descriptor.spatialKey = 0x13579;
    descriptor.particleCount = 17;
    descriptor.halfSize = 0.25;
    descriptor.radius = 0.433;
    return roundTrip(descriptor);
}

bool testWireMessages()
{
    const std::uint64_t epoch = 42;
    FmmDescriptorRequest request;
    request.stamp = fmmPacketStamp(FmmPacketKind::DescriptorRequest, epoch);
    request.patchId = 0xabcdu;
    request.spatialKey = 0x2468u;
    if(!roundTrip(request))
        return false;

    FmmSubscription subscription;
    subscription.stamp = fmmPacketStamp(FmmPacketKind::Subscription, epoch);
    subscription.patchId = 0x9876543210abcdefull;
    subscription.spatialKey = 0x777;
    subscription.kind = static_cast<int>(FmmSubscriptionKind::Particles);
    subscription.waveIndex = 3;
    if(!roundTrip(subscription))
        return false;

    FmmPayloadRecordHeader header;
    header.stamp = fmmPacketStamp(FmmPacketKind::LetPayload, epoch);
    header.patchId = 0x1111222233334444ull;
    header.spatialKey = 0x55;
    header.count = 128;
    header.kind = static_cast<int>(FmmSubscriptionKind::Multipole);
    header.waveIndex = 7;
    return roundTrip(header);
}
}

int main()
{
    const bool pass = testPatchRootDescriptor() && testRemoteNodeDescriptor() &&
                      testWireMessages();
    std::ofstream output("fmm_packet_v5_metrics.txt");
    output << "protocol_version " << FMM_MPI_PACKET_VERSION << "\n";
    output << "pass " << (pass ? 1 : 0) << "\n";
    std::cout << "fmm_packet_v5 pass=" << pass << std::endl;
    return pass ? 0 : 1;
}
