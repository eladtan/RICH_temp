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

struct FmmPeerExchangeTimings
{
    double flattenSeconds = 0;
    double countExchangeSeconds = 0;
    double receiveSetupSeconds = 0;
    double payloadLaunchSeconds = 0;
};

class FmmPeerExchangeRequest;

class FmmPeerExchangeResult
{
public:
    FmmByteView view(const FmmReceivedMessage& message) const;
    std::size_t totalBytes() const { return storage_.size(); }
    std::size_t bytesOwned() const;
    const std::vector<FmmReceivedMessage>& messages() const { return messages_; }
    void releaseStorage();

private:
    friend class FmmPeerExchange;
    friend class FmmPeerExchangeRequest;
    std::vector<char> storage_;
    std::vector<FmmReceivedMessage> messages_;
};

class FmmPeerExchangeRequest
{
public:
    FmmPeerExchangeRequest();
    FmmPeerExchangeRequest(const FmmPeerExchangeRequest&) = delete;
    FmmPeerExchangeRequest& operator=(const FmmPeerExchangeRequest&) = delete;
    FmmPeerExchangeRequest(FmmPeerExchangeRequest&&) = delete;
    FmmPeerExchangeRequest& operator=(FmmPeerExchangeRequest&&) = delete;
    ~FmmPeerExchangeRequest();

    bool active() const;
    bool progress();
    FmmPeerExchangeResult wait(
        std::uint64_t* bytesSent = nullptr,
        std::uint64_t* bytesReceived = nullptr);
    void clear();
    std::size_t bytesOwned() const;
    double payloadLifetimeSeconds() const { return payloadLifetimeSeconds_; }
    double residualWaitSeconds() const { return residualWaitSeconds_; }
    bool completedByProgress() const { return completedByProgress_; }

private:
    friend class FmmPeerExchange;

    enum class State
    {
        Idle,
        Payload,
        Complete
    };

    void finalizeMessages();

    MPI_Comm graph_;
    State state_;
    MPI_Request payloadRequest_;
    std::vector<int> sourceRanks_;
    std::vector<int> sendCounts_;
    std::vector<int> sendDisplacements_;
    std::vector<int> receiveCounts_;
    std::vector<int> receiveDisplacements_;
    std::vector<char> sendBuffer_;
    FmmPeerExchangeResult result_;
    std::size_t totalSend_;
    std::size_t totalReceive_;
    double payloadLaunchTime_;
    double payloadLifetimeSeconds_;
    double residualWaitSeconds_;
    bool completedByProgress_;
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
    // Collective on parent.  Rebuild the distributed-graph communicator only
    // when at least one rank's normalized outgoing peer set changed.  Returns
    // true when a rebuild was performed.
    bool resetIfChanged(const MPI_Comm& parent,
                        const std::vector<int>& outgoingPeers);
    void clear();

    FmmPeerExchangeResult exchangeBytes(
        const std::unordered_map<int, std::vector<char>>& sendByRank,
        std::uint64_t* bytesSent = nullptr,
        std::uint64_t* bytesReceived = nullptr,
        std::size_t maxReceiveBytes = std::numeric_limits<std::size_t>::max()) const;

    void beginExchangeBytes(
        const std::unordered_map<int, std::vector<char>>& sendByRank,
        FmmPeerExchangeRequest& request,
        std::size_t maxReceiveBytes = std::numeric_limits<std::size_t>::max(),
        std::size_t maxRequestBytes =
            std::numeric_limits<std::size_t>::max(),
        FmmPeerExchangeTimings* timings = nullptr) const;

    const std::vector<int>& sources() const { return sources_; }
    const std::vector<int>& destinations() const { return destinations_; }
    bool valid() const { return graph_ != MPI_COMM_NULL; }
    std::size_t bytesOwned() const;

private:
    void resetValidated(const MPI_Comm& parent,
                        const std::vector<int>& normalizedOutgoingPeers);

    MPI_Comm graph_;
    std::vector<int> sources_;
    std::vector<int> destinations_;
    std::unordered_map<int, int> destinationSlot_;
    bool dense_;
};

#endif // RICH_MPI

#endif // FMM_PEER_EXCHANGE_HPP
