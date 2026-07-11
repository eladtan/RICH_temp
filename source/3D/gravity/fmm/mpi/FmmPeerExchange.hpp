#ifndef FMM_PEER_EXCHANGE_HPP
#define FMM_PEER_EXCHANGE_HPP

#ifdef RICH_MPI

#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

#include <mpi.h>

#include "3D/gravity/fmm/mpi/FmmPackets.hpp"

struct FmmReceivedMessage
{
    int source = -1;
    std::size_t offset = 0;
    std::size_t size = 0;
};

class FmmPeerExchangeResult
{
public:
    FmmByteView view(const FmmReceivedMessage& message) const;
    std::size_t totalBytes() const { return storage_.size(); }
    const std::vector<FmmReceivedMessage>& messages() const { return messages_; }
    void releaseStorage();

private:
    friend class FmmPeerExchange;
    std::vector<char> storage_;
    std::vector<FmmReceivedMessage> messages_;
};

class FmmPeerExchange
{
public:
    FmmPeerExchange();
    FmmPeerExchange(const MPI_Comm& parent, const std::vector<int>& outgoingPeers);
    FmmPeerExchange(FmmPeerExchange&& other) noexcept;
    FmmPeerExchange& operator=(FmmPeerExchange&& other) noexcept;
    FmmPeerExchange(const FmmPeerExchange&) = delete;
    FmmPeerExchange& operator=(const FmmPeerExchange&) = delete;
    ~FmmPeerExchange();

    void reset(const MPI_Comm& parent, const std::vector<int>& outgoingPeers);
    void clear();

    FmmPeerExchangeResult exchangeBytes(
        const std::unordered_map<int, std::vector<char>>& sendByRank,
        std::uint64_t* bytesSent = nullptr,
        std::uint64_t* bytesReceived = nullptr,
        std::size_t maxReceiveBytes = std::numeric_limits<std::size_t>::max()) const;

    const std::vector<int>& sources() const { return sources_; }
    const std::vector<int>& destinations() const { return destinations_; }
    bool valid() const { return graph_ != MPI_COMM_NULL; }
    std::size_t bytesOwned() const;

private:
    MPI_Comm graph_;
    std::vector<int> sources_;
    std::vector<int> destinations_;
    std::unordered_map<int, int> destinationSlot_;
};

#endif // RICH_MPI

#endif // FMM_PEER_EXCHANGE_HPP
