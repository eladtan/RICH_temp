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

std::size_t saturatingAdd(std::size_t first, std::size_t second)
{
    return second > std::numeric_limits<std::size_t>::max() - first ?
        std::numeric_limits<std::size_t>::max() : first + second;
}

std::size_t saturatingMultiply(std::size_t first, std::size_t second)
{
    return first != 0 &&
           second > std::numeric_limits<std::size_t>::max() / first ?
        std::numeric_limits<std::size_t>::max() : first * second;
}
}

FmmByteView FmmPeerExchangeResult::view(const FmmReceivedMessage& message) const
{
    if(message.offset > storage_.size() ||
       message.size > storage_.size() - message.offset)
        throw UniversalError("FmmPeerExchangeResult::view: invalid message range");
    return FmmByteView{storage_.data() + message.offset, message.size};
}

std::size_t FmmPeerExchangeResult::bytesOwned() const
{
    return saturatingAdd(
        saturatingMultiply(storage_.capacity(), sizeof(char)),
        saturatingMultiply(messages_.capacity(), sizeof(FmmReceivedMessage)));
}

void FmmPeerExchangeResult::releaseStorage()
{
    std::vector<char>().swap(storage_);
    std::vector<FmmReceivedMessage>().swap(messages_);
}

FmmPeerExchangeRequest::FmmPeerExchangeRequest():
    graph_(MPI_COMM_NULL), state_(State::Idle),
    payloadRequest_(MPI_REQUEST_NULL), totalSend_(0), totalReceive_(0),
    payloadLaunchTime_(0.0), payloadLifetimeSeconds_(0.0),
    residualWaitSeconds_(0.0), completedByProgress_(false) {}

FmmPeerExchangeRequest::~FmmPeerExchangeRequest()
{
    if(!active())
        return;
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    if(initialized != 0)
        MPI_Finalized(&finalized);
    if(initialized != 0 && finalized == 0 && graph_ != MPI_COMM_NULL)
        MPI_Abort(graph_, 93);
}

bool FmmPeerExchangeRequest::active() const
{
    return state_ != State::Idle;
}

void FmmPeerExchangeRequest::finalizeMessages()
{
    if(state_ != State::Payload && state_ != State::Complete)
        throw UniversalError(
            "FmmPeerExchangeRequest::finalizeMessages: invalid request state");
    result_.messages_.clear();
    result_.messages_.reserve(sourceRanks_.size());
    for(std::size_t i = 0; i < sourceRanks_.size(); ++i)
    {
        if(receiveCounts_[i] == 0)
            continue;
        result_.messages_.push_back(FmmReceivedMessage{
            sourceRanks_[i],
            static_cast<std::size_t>(receiveDisplacements_[i]),
            static_cast<std::size_t>(receiveCounts_[i])});
    }
    state_ = State::Complete;
}

bool FmmPeerExchangeRequest::progress()
{
    if(state_ == State::Idle || state_ == State::Complete)
        return true;
    int complete = 0;
    checkMpi(MPI_Test(&payloadRequest_, &complete, MPI_STATUS_IGNORE),
             "FmmPeerExchangeRequest MPI_Test payload");
    if(complete == 0)
        return false;
    payloadLifetimeSeconds_ = MPI_Wtime() - payloadLaunchTime_;
    completedByProgress_ = true;
    finalizeMessages();
    return true;
}

FmmPeerExchangeResult FmmPeerExchangeRequest::wait(
    std::uint64_t* bytesSent,
    std::uint64_t* bytesReceived)
{
    if(state_ == State::Idle)
        throw UniversalError("FmmPeerExchangeRequest::wait: no active exchange");
    if(state_ == State::Payload)
    {
        const double waitStart = MPI_Wtime();
        checkMpi(MPI_Wait(&payloadRequest_, MPI_STATUS_IGNORE),
                 "FmmPeerExchangeRequest MPI_Wait payload");
        residualWaitSeconds_ += MPI_Wtime() - waitStart;
        payloadLifetimeSeconds_ = MPI_Wtime() - payloadLaunchTime_;
        finalizeMessages();
    }
    if(bytesSent != nullptr)
        *bytesSent += static_cast<std::uint64_t>(totalSend_);
    if(bytesReceived != nullptr)
        *bytesReceived += static_cast<std::uint64_t>(totalReceive_);

    FmmPeerExchangeResult result = std::move(result_);
    state_ = State::Idle;
    graph_ = MPI_COMM_NULL;
    payloadRequest_ = MPI_REQUEST_NULL;
    totalSend_ = 0;
    totalReceive_ = 0;
    sendBuffer_.clear();
    result_.releaseStorage();
    return result;
}

void FmmPeerExchangeRequest::clear()
{
    if(active())
        throw UniversalError(
            "FmmPeerExchangeRequest::clear: exchange is still active");
    std::vector<int>().swap(sourceRanks_);
    std::vector<int>().swap(sendCounts_);
    std::vector<int>().swap(sendDisplacements_);
    std::vector<int>().swap(receiveCounts_);
    std::vector<int>().swap(receiveDisplacements_);
    std::vector<char>().swap(sendBuffer_);
    result_.releaseStorage();
    payloadLaunchTime_ = 0.0;
    payloadLifetimeSeconds_ = 0.0;
    residualWaitSeconds_ = 0.0;
    completedByProgress_ = false;
}

std::size_t FmmPeerExchangeRequest::bytesOwned() const
{
    std::size_t result = 0;
    result = saturatingAdd(result,
        saturatingMultiply(sourceRanks_.capacity(), sizeof(int)));
    result = saturatingAdd(result,
        saturatingMultiply(sendCounts_.capacity(), sizeof(int)));
    result = saturatingAdd(result,
        saturatingMultiply(sendDisplacements_.capacity(), sizeof(int)));
    result = saturatingAdd(result,
        saturatingMultiply(receiveCounts_.capacity(), sizeof(int)));
    result = saturatingAdd(result,
        saturatingMultiply(receiveDisplacements_.capacity(), sizeof(int)));
    result = saturatingAdd(result,
        saturatingMultiply(sendBuffer_.capacity(), sizeof(char)));
    return saturatingAdd(result, result_.bytesOwned());
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
    // Keep the distributed-graph constructor signature uniform across ranks.
    // Open MPI validates array arguments before considering a zero degree, so
    // provide a valid dummy destination while retaining n=1 on every rank.
    const int dummyDestination = rank;
    const int* destinations = degree == 0 ? &dummyDestination : peers.data();

    // Some Open MPI topology components can deadlock when
    // MPI_Dist_graph_create is invoked repeatedly on the same parent
    // communicator. Construct through a fresh context each time. The graph
    // communicator is independent after creation, so the temporary duplicate
    // can be released immediately.
    MPI_Comm constructionParent = MPI_COMM_NULL;
    checkMpi(MPI_Comm_dup(parent, &constructionParent),
             "FmmPeerExchange::reset MPI_Comm_dup construction parent");
    const int graphStatus = MPI_Dist_graph_create(
        constructionParent, 1, &rank, &degree, destinations, MPI_UNWEIGHTED,
        MPI_INFO_NULL, 0, &graph_);
    const int freeStatus = MPI_Comm_free(&constructionParent);
    checkMpi(graphStatus, "FmmPeerExchange::reset MPI_Dist_graph_create");
    checkMpi(freeStatus,
             "FmmPeerExchange::reset MPI_Comm_free construction parent");

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

void FmmPeerExchange::beginExchangeBytes(
    const std::unordered_map<int, std::vector<char>>& sendByRank,
    FmmPeerExchangeRequest& request,
    std::size_t maxReceiveBytes,
    std::size_t maxRequestBytes,
    FmmPeerExchangeTimings* timings) const
{
    if(graph_ == MPI_COMM_NULL)
        throw UniversalError(
            "FmmPeerExchange::beginExchangeBytes: communicator is not initialized");
    if(request.active())
        throw UniversalError(
            "FmmPeerExchange::beginExchangeBytes: request is already active");

    int localInvalid = 0;
    for(const auto& entry : sendByRank)
    {
        if(!entry.second.empty() && destinationSlot_.find(entry.first) ==
           destinationSlot_.end())
            localInvalid = 1;
    }

    request.payloadLaunchTime_ = 0.0;
    request.payloadLifetimeSeconds_ = 0.0;
    request.residualWaitSeconds_ = 0.0;
    request.completedByProgress_ = false;

    const double flattenStart = MPI_Wtime();
    request.graph_ = graph_;
    request.sourceRanks_ = sources_;
    request.sendCounts_.assign(destinations_.size(), 0);
    request.sendDisplacements_.assign(destinations_.size(), 0);
    request.totalSend_ = 0;
    for(std::size_t i = 0; i < destinations_.size(); ++i)
    {
        const auto found = sendByRank.find(destinations_[i]);
        const std::size_t count =
            found == sendByRank.end() ? 0 : found->second.size();
        if(count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
           request.totalSend_ >
               static_cast<std::size_t>(std::numeric_limits<int>::max()) - count)
        {
            localInvalid = 1;
            continue;
        }
        request.sendCounts_[i] = static_cast<int>(count);
        request.sendDisplacements_[i] = static_cast<int>(request.totalSend_);
        request.totalSend_ += count;
    }
    if(localInvalid != 0)
        abortInvariant(graph_,
            "FmmPeerExchange::beginExchangeBytes: invalid or oversized send");

    request.sendBuffer_.resize(request.totalSend_);
    for(std::size_t i = 0; i < destinations_.size(); ++i)
    {
        const auto found = sendByRank.find(destinations_[i]);
        if(found == sendByRank.end() || found->second.empty())
            continue;
        std::copy(found->second.begin(), found->second.end(),
            request.sendBuffer_.begin() + request.sendDisplacements_[i]);
    }
    if(timings != nullptr)
        timings->flattenSeconds += MPI_Wtime() - flattenStart;

    // Count exchange is tiny and deliberately blocking. It lets us allocate
    // the exact receive buffer and launch the large payload collective before
    // local traversal begins, maximizing the useful overlap window.
    request.receiveCounts_.assign(sources_.size(), 0);
    const double countExchangeStart = MPI_Wtime();
    checkMpi(MPI_Neighbor_alltoall(
        request.sendCounts_.empty() ? nullptr : request.sendCounts_.data(),
        1, MPI_INT,
        request.receiveCounts_.empty() ? nullptr : request.receiveCounts_.data(),
        1, MPI_INT, graph_),
        "FmmPeerExchange::beginExchangeBytes MPI_Neighbor_alltoall");
    if(timings != nullptr)
        timings->countExchangeSeconds += MPI_Wtime() - countExchangeStart;

    const double receiveSetupStart = MPI_Wtime();
    request.receiveDisplacements_.assign(sources_.size(), 0);
    request.totalReceive_ = 0;
    localInvalid = 0;
    for(std::size_t i = 0; i < sources_.size(); ++i)
    {
        if(request.receiveCounts_[i] < 0 ||
           request.totalReceive_ >
               static_cast<std::size_t>(std::numeric_limits<int>::max()) -
                   static_cast<std::size_t>(
                       std::max(0, request.receiveCounts_[i])))
        {
            localInvalid = 1;
            continue;
        }
        request.receiveDisplacements_[i] =
            static_cast<int>(request.totalReceive_);
        request.totalReceive_ +=
            static_cast<std::size_t>(request.receiveCounts_[i]);
    }
    if(request.totalReceive_ > maxReceiveBytes)
        localInvalid = 1;
    if(localInvalid != 0)
        abortInvariant(graph_,
            "FmmPeerExchange::beginExchangeBytes: receive size or memory budget exceeded");

    request.result_.storage_.clear();
    request.result_.storage_.resize(request.totalReceive_);
    request.result_.messages_.clear();
    request.result_.messages_.reserve(request.sourceRanks_.size());
    if(request.bytesOwned() > maxRequestBytes)
        abortInvariant(graph_,
            "FmmPeerExchange::beginExchangeBytes: request workspace exceeds memory budget");
    if(timings != nullptr)
        timings->receiveSetupSeconds += MPI_Wtime() - receiveSetupStart;

    const double payloadLaunchStart = MPI_Wtime();
    checkMpi(MPI_Ineighbor_alltoallv(
        request.sendBuffer_.empty() ? nullptr : request.sendBuffer_.data(),
        request.sendCounts_.empty() ? nullptr : request.sendCounts_.data(),
        request.sendDisplacements_.empty() ? nullptr :
                                             request.sendDisplacements_.data(),
        MPI_BYTE,
        request.result_.storage_.empty() ? nullptr :
                                           request.result_.storage_.data(),
        request.receiveCounts_.empty() ? nullptr :
                                         request.receiveCounts_.data(),
        request.receiveDisplacements_.empty() ? nullptr :
                                                request.receiveDisplacements_.data(),
        MPI_BYTE, graph_, &request.payloadRequest_),
        "FmmPeerExchange::beginExchangeBytes MPI_Ineighbor_alltoallv");
    const double payloadLaunchEnd = MPI_Wtime();
    if(timings != nullptr)
        timings->payloadLaunchSeconds += payloadLaunchEnd - payloadLaunchStart;
    request.payloadLaunchTime_ = payloadLaunchEnd;
    request.state_ = FmmPeerExchangeRequest::State::Payload;
}

FmmPeerExchangeResult FmmPeerExchange::exchangeBytes(
    const std::unordered_map<int, std::vector<char>>& sendByRank,
    std::uint64_t* bytesSent,
    std::uint64_t* bytesReceived,
    std::size_t maxReceiveBytes) const
{
    FmmPeerExchangeRequest request;
    beginExchangeBytes(sendByRank, request, maxReceiveBytes);
    return request.wait(bytesSent, bytesReceived);
}

#endif // RICH_MPI
