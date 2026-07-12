#include "3D/gravity/fmm/mpi/FmmPeerExchange.hpp"

#ifdef RICH_MPI

#include <algorithm>
#include <limits>
#include <string>
#include <utility>

#include "misc/universal_error.hpp"

namespace
{
void checkMpi(int status, const char* operation)
{
    if(status == MPI_SUCCESS)
        return;
    char text[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(status, text, &length);
    throw UniversalError(std::string(operation) + ": " +
                         std::string(text, static_cast<std::size_t>(length)));
}

[[noreturn]] void abortInvariant(const MPI_Comm& comm, const char* message)
{
    MPI_Abort(comm, 91);
    throw UniversalError(message);
}
}

FmmByteView FmmPeerExchangeResult::view(const FmmReceivedMessage& message) const
{
    if(message.offset > storage_.size() ||
       message.size > storage_.size() - message.offset)
        throw UniversalError("FmmPeerExchangeResult::view: invalid message range");
    return FmmByteView{storage_.data() + message.offset, message.size};
}

void FmmPeerExchangeResult::releaseStorage()
{
    std::vector<char>().swap(storage_);
    std::vector<FmmReceivedMessage>().swap(messages_);
}

FmmPeerExchange::FmmPeerExchange(): graph_(MPI_COMM_NULL) {}

FmmPeerExchange::FmmPeerExchange(const MPI_Comm& parent,
                                 const std::vector<int>& outgoingPeers):
    graph_(MPI_COMM_NULL)
{
    reset(parent, outgoingPeers);
}

FmmPeerExchange::FmmPeerExchange(FmmPeerExchange&& other) noexcept:
    graph_(other.graph_),
    sources_(std::move(other.sources_)),
    destinations_(std::move(other.destinations_)),
    destinationSlot_(std::move(other.destinationSlot_))
{
    other.graph_ = MPI_COMM_NULL;
}

FmmPeerExchange& FmmPeerExchange::operator=(FmmPeerExchange&& other) noexcept
{
    if(this != &other)
    {
        clear();
        graph_ = other.graph_;
        sources_ = std::move(other.sources_);
        destinations_ = std::move(other.destinations_);
        destinationSlot_ = std::move(other.destinationSlot_);
        other.graph_ = MPI_COMM_NULL;
    }
    return *this;
}

FmmPeerExchange::~FmmPeerExchange()
{
    clear();
}

void FmmPeerExchange::clear()
{
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if(initialized != 0)
        MPI_Finalized(&finalized);
    if(graph_ != MPI_COMM_NULL && initialized != 0 && finalized == 0)
        MPI_Comm_free(&graph_);
    graph_ = MPI_COMM_NULL;
    sources_.clear();
    destinations_.clear();
    destinationSlot_.clear();
}

void FmmPeerExchange::reset(const MPI_Comm& parent,
                            const std::vector<int>& outgoingPeers)
{
    clear();
    if(parent == MPI_COMM_NULL)
        throw UniversalError("FmmPeerExchange::reset: parent communicator is null");
    int rank = 0;
    int size = 0;
    checkMpi(MPI_Comm_rank(parent, &rank), "FmmPeerExchange::reset MPI_Comm_rank");
    checkMpi(MPI_Comm_size(parent, &size), "FmmPeerExchange::reset MPI_Comm_size");
    std::vector<int> peers = outgoingPeers;
    std::sort(peers.begin(), peers.end());
    peers.erase(std::unique(peers.begin(), peers.end()), peers.end());
    peers.erase(std::remove(peers.begin(), peers.end(), rank), peers.end());

    int localInvalid = peers.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ? 1 : 0;
    for(int peer : peers)
        localInvalid = localInvalid || peer < 0 || peer >= size;
    int globalInvalid = 0;
    checkMpi(MPI_Allreduce(&localInvalid, &globalInvalid, 1, MPI_INT, MPI_LOR, parent),
             "FmmPeerExchange::reset peer validation");
    if(globalInvalid != 0)
        throw UniversalError("FmmPeerExchange::reset: invalid graph peer on at least one rank");

    const int degree = static_cast<int>(peers.size());
    const int sourceCount = degree == 0 ? 0 : 1;
    checkMpi(MPI_Dist_graph_create(parent, sourceCount,
                                   sourceCount == 0 ? nullptr : &rank,
                                   sourceCount == 0 ? nullptr : &degree,
                                   sourceCount == 0 ? nullptr : peers.data(),
                                   MPI_UNWEIGHTED, MPI_INFO_NULL, 0, &graph_),
             "FmmPeerExchange::reset MPI_Dist_graph_create");

    int indegree = 0;
    int outdegree = 0;
    int weighted = 0;
    checkMpi(MPI_Dist_graph_neighbors_count(graph_, &indegree, &outdegree, &weighted),
             "FmmPeerExchange::reset MPI_Dist_graph_neighbors_count");
    sources_.resize(static_cast<std::size_t>(indegree));
    destinations_.resize(static_cast<std::size_t>(outdegree));
    checkMpi(MPI_Dist_graph_neighbors(graph_, indegree,
                                      indegree == 0 ? nullptr : sources_.data(),
                                      MPI_UNWEIGHTED,
                                      outdegree,
                                      outdegree == 0 ? nullptr : destinations_.data(),
                                      MPI_UNWEIGHTED),
             "FmmPeerExchange::reset MPI_Dist_graph_neighbors");
    for(int i = 0; i < outdegree; ++i)
        destinationSlot_[destinations_[static_cast<std::size_t>(i)]] = i;
}

std::size_t FmmPeerExchange::bytesOwned() const
{
    const std::size_t mapEntryBytes = sizeof(std::pair<const int, int>) +
        2 * sizeof(void*);
    return sources_.capacity() * sizeof(int) +
           destinations_.capacity() * sizeof(int) +
           destinationSlot_.size() * mapEntryBytes;
}

FmmPeerExchangeResult FmmPeerExchange::exchangeBytes(
    const std::unordered_map<int, std::vector<char>>& sendByRank,
    std::uint64_t* bytesSent,
    std::uint64_t* bytesReceived,
    std::size_t maxReceiveBytes) const
{
    if(graph_ == MPI_COMM_NULL)
        throw UniversalError("FmmPeerExchange::exchangeBytes: communicator is not initialized");

    int localInvalid = 0;
    for(const auto& entry : sendByRank)
    {
        if(!entry.second.empty() && destinationSlot_.find(entry.first) ==
           destinationSlot_.end())
            localInvalid = 1;
    }

    std::vector<int> sendCounts(destinations_.size(), 0);
    std::vector<int> sendDisplacements(destinations_.size(), 0);
    std::size_t totalSend = 0;
    for(std::size_t i = 0; i < destinations_.size(); ++i)
    {
        const auto it = sendByRank.find(destinations_[i]);
        const std::size_t count = it == sendByRank.end() ? 0 : it->second.size();
        if(count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
           totalSend > static_cast<std::size_t>(std::numeric_limits<int>::max()) - count)
        {
            localInvalid = 1;
            continue;
        }
        sendCounts[i] = static_cast<int>(count);
        sendDisplacements[i] = static_cast<int>(totalSend);
        totalSend += count;
    }
    if(localInvalid != 0)
        abortInvariant(graph_,
            "FmmPeerExchange::exchangeBytes: invalid or oversized send");

    std::vector<char> sendBuffer(totalSend);
    for(std::size_t i = 0; i < destinations_.size(); ++i)
    {
        const auto it = sendByRank.find(destinations_[i]);
        if(it == sendByRank.end() || it->second.empty())
            continue;
        std::copy(it->second.begin(), it->second.end(),
                  sendBuffer.begin() + sendDisplacements[i]);
    }

    std::vector<int> recvCounts(sources_.size(), 0);
    checkMpi(MPI_Neighbor_alltoall(sendCounts.empty() ? nullptr : sendCounts.data(),
                                   1, MPI_INT,
                                   recvCounts.empty() ? nullptr : recvCounts.data(),
                                   1, MPI_INT, graph_),
             "FmmPeerExchange::exchangeBytes MPI_Neighbor_alltoall");

    std::vector<int> recvDisplacements(sources_.size(), 0);
    std::size_t totalRecv = 0;
    localInvalid = 0;
    for(std::size_t i = 0; i < sources_.size(); ++i)
    {
        if(recvCounts[i] < 0 ||
           totalRecv > static_cast<std::size_t>(std::numeric_limits<int>::max()) -
                       static_cast<std::size_t>(std::max(0, recvCounts[i])))
        {
            localInvalid = 1;
            continue;
        }
        recvDisplacements[i] = static_cast<int>(totalRecv);
        totalRecv += static_cast<std::size_t>(recvCounts[i]);
    }
    if(totalRecv > maxReceiveBytes)
        localInvalid = 1;
    if(localInvalid != 0)
        abortInvariant(graph_,
            "FmmPeerExchange::exchangeBytes: receive size or memory budget exceeded");

    FmmPeerExchangeResult result;
    result.storage_.resize(totalRecv);
    checkMpi(MPI_Neighbor_alltoallv(sendBuffer.empty() ? nullptr : sendBuffer.data(),
                                    sendCounts.empty() ? nullptr : sendCounts.data(),
                                    sendDisplacements.empty() ? nullptr : sendDisplacements.data(),
                                    MPI_BYTE,
                                    result.storage_.empty() ? nullptr : result.storage_.data(),
                                    recvCounts.empty() ? nullptr : recvCounts.data(),
                                    recvDisplacements.empty() ? nullptr : recvDisplacements.data(),
                                    MPI_BYTE, graph_),
             "FmmPeerExchange::exchangeBytes MPI_Neighbor_alltoallv");

    if(bytesSent != nullptr)
        *bytesSent += static_cast<std::uint64_t>(totalSend);
    if(bytesReceived != nullptr)
        *bytesReceived += static_cast<std::uint64_t>(totalRecv);

    result.messages_.reserve(sources_.size());
    for(std::size_t i = 0; i < sources_.size(); ++i)
    {
        if(recvCounts[i] == 0)
            continue;
        result.messages_.push_back(FmmReceivedMessage{
            sources_[i], static_cast<std::size_t>(recvDisplacements[i]),
            static_cast<std::size_t>(recvCounts[i])});
    }
    return result;
}

#endif // RICH_MPI
