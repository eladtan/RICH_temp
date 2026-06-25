#ifndef BUFFERS_MANAGER_HPP
#define BUFFERS_MANAGER_HPP

#include <cassert>
#include "misc/universal_error.hpp"
#include <limits>
#ifdef RICH_MPI

#include <mpi.h>
#include <vector>
#include <functional>
#include <boost/container/flat_map.hpp>
#include "mpi/mpi_commands.hpp"
#include <mpi_utils/serialize/Serializer.hpp>

template<typename T>
class BuffersManager
{
public:
    BuffersManager(const MPI_Comm &comm, const std::function<void(const T *newValues, size_t newValuesCount, rank_t fromRank)> &receiveCallback, int tag, size_t buffersSize, size_t minSizeToDispatch, size_t minCyclesToDispatch, size_t initialReceiveBuffers, const std::vector<rank_t> &neighbors = std::vector<rank_t>());

    ~BuffersManager();

    void Add(rank_t rank, const T &value);

    void Receive(bool ignore = false);

    void HandleIncomingOutcoming();

    void Destroy(void);

    inline size_t CountOutcoming() const{return this->sendRequests.size();};

    inline size_t GetPendingNumber() const{return this->ranksSendBuffers.size();};

    inline size_t GetActiveSendsNumber() const{return this->activeSendRequests;};

    inline size_t GetSentCounter() const {return this->sendCounter;};

    inline size_t GetRecvCounter() const {return this->recvCounter;};

    inline const std::vector<size_t> &GetAllSendCounters() const {return this->sendCounters;};

    inline const std::vector<size_t> &GetAllRecvCounters() const {return this->recvCounters;};

    inline size_t GetTotalMemoryBytes() const {return this->buffers.size() * this->buffersSize;};

private:
    MPI_Comm comm;
    bool destroyed;
    rank_t rank_world, size_world;
    std::vector<Serializer> buffers;
    std::vector<MPI_Request> sendRequests;
    size_t activeSendRequests;
    std::vector<MPI_Request> receiveRequests;
    std::vector<rank_t> receiveSources;
    boost::container::flat_set<size_t> availableBuffersIndices;
    boost::container::flat_map<size_t, size_t> sendBuffersByRequests;
    boost::container::flat_map<size_t, size_t> recvBuffersByRequests;
    std::vector<size_t> buffersToCycles;
    
    // buffering mechanism
    boost::container::flat_map<rank_t, size_t> ranksSendBuffers;

    size_t sendCounter;
    size_t recvCounter;

    std::vector<size_t> sendCounters;
    std::vector<size_t> recvCounters;

    void Dispatch(rank_t rank);

    bool ShouldSend(rank_t rank);

    void CleanSendRequests();

    size_t buffersSize;
    size_t minSizeToDispatch;
    size_t minCyclesToDispatch;
    int tag;
    std::function<void(const T *newValues, size_t newValuesCount, rank_t fromRank)> receiveCallback;
};

template<typename T>
BuffersManager<T>::BuffersManager(const MPI_Comm &comm, const std::function<void(const T *newValues, size_t newValuesCount, rank_t fromRank)> &receiveCallback, int tag, size_t buffersSize, size_t minSizeToDispatch, size_t minCyclesToDispatch, size_t numReceiveBuffers, const std::vector<rank_t> &neighbors)
    : comm(comm), destroyed(false), receiveCallback(receiveCallback), tag(tag), buffersSize(buffersSize), minSizeToDispatch(minSizeToDispatch), minCyclesToDispatch(minCyclesToDispatch)
{
    this->buffersSize += sizeof(size_t); // for size
    this->minSizeToDispatch += sizeof(size_t); // for size

    MPI_Comm_rank(comm, &this->rank_world);
    MPI_Comm_size(comm, &this->size_world);

    this->activeSendRequests = 0;
    this->sendCounter = 0;
    this->recvCounter = 0;

    this->sendCounters.resize(this->size_world, 0);
    this->recvCounters.resize(this->size_world, 0);

    if(this->buffersSize < this->minSizeToDispatch)
    {
        UniversalError eo("BuffersManager: buffersSize is less than minSizeToDispatch");
        eo.addEntry("Buffers size", this->buffersSize - sizeof(size_t));
        eo.addEntry("Min size to dispatch", this->minSizeToDispatch - sizeof(size_t));
        throw eo;
    }

    if(neighbors.empty() and numReceiveBuffers == 0)
    {
        UniversalError eo("BuffersManager: numReceiveBuffers is 0");
        eo.addEntry("Num receive buffers", numReceiveBuffers);
        throw eo;
    }

    // initialize receive
    if(neighbors.empty())
    {
        this->receiveSources.assign(numReceiveBuffers, MPI_ANY_SOURCE);
    }
    else
    {
        this->receiveSources = neighbors;
    }
    this->receiveRequests.resize(this->receiveSources.size(), MPI_REQUEST_NULL);
    for(size_t i = 0; i < this->receiveRequests.size(); i++)
    {
        Serializer &serializer = this->buffers.emplace_back();
        serializer.resize(this->buffersSize);
        MPI_Request &request = this->receiveRequests[i];
        MPI_Irecv(serializer.getData(), serializer.size(), MPI_BYTE, this->receiveSources[i], this->tag, comm, &request);
        assert(request != MPI_REQUEST_NULL);
        this->recvBuffersByRequests[i] = i;
    }

    this->buffersToCycles = std::vector<size_t>(this->size_world, 0);
}

template<typename T>
BuffersManager<T>::~BuffersManager()
{
    this->Destroy();
}

template<typename T>
void BuffersManager<T>::Destroy(void)
{
    if(this->destroyed)
    {
        return;
    }
    this->destroyed = true;
    if(not this->sendRequests.empty())
    {
        MPI_Waitall(this->sendRequests.size(), this->sendRequests.data(), MPI_STATUSES_IGNORE);
        this->activeSendRequests = 0;
    }

    if(this->activeSendRequests != 0)
    {
        UniversalError eo("BuffersManager: not all send requests were handled");
        eo.addEntry("Active send requests", this->activeSendRequests);
        throw eo;
    }

    size_t sentToMe = 0;
    std::vector<int> counts(this->size_world, 1);
    MPI_Reduce_scatter(this->sendCounters.data(), &sentToMe, counts.data(), MPI_UNSIGNED_LONG_LONG, MPI_SUM, this->comm);
    
    size_t shouldReceiveMore = std::numeric_limits<size_t>::max();
    if(sentToMe < this->recvCounter)
    {
        UniversalError eo("BuffersManager: too many received were handled");
        eo.addEntry("Total Receive requests", this->recvCounter);
        eo.addEntry("Sent to me", sentToMe);
        throw eo;
    }
    else if(sentToMe > this->recvCounter)
    {
        shouldReceiveMore = sentToMe - this->recvCounter;
        std::vector<int> array_of_indices(this->receiveRequests.size());
        std::vector<MPI_Status> array_of_statuses(this->receiveRequests.size());
        while(this->recvCounter < sentToMe)
        {
            int outcount;
            MPI_Waitsome(this->receiveRequests.size(), this->receiveRequests.data(), &outcount, array_of_indices.data(), array_of_statuses.data());
            if(outcount == MPI_UNDEFINED)
            {
                UniversalError eo("BuffersManager: no active receive requests while draining");
                eo.addEntry("Total Receive requests", this->recvCounter);
                eo.addEntry("Sent to me", sentToMe);
                throw eo;
            }
            for(int i = 0; i < outcount; i++)
            {
                size_t requestIndex = array_of_indices[i];
                rank_t fromRank = array_of_statuses[i].MPI_SOURCE;
                size_t bufferIndex = this->recvBuffersByRequests.at(requestIndex);
                Serializer &serializer = this->buffers[bufferIndex];
                MPI_Request &request = this->receiveRequests[requestIndex];
                this->recvCounter++;
                this->recvCounters[fromRank]++;
                if(this->recvCounter < sentToMe)
                {
                    MPI_Irecv(serializer.getData(), serializer.size(), MPI_BYTE, this->receiveSources[requestIndex], this->tag, this->comm, &request);
                }
            }
        }
    }

    // should not reach here anyway
    if(this->recvCounter != sentToMe)
    {
        UniversalError eo("BuffersManager: not all receive requests were handled");
        eo.addEntry("Total Receive requests", this->recvCounter);
        eo.addEntry("Should Have Received More", shouldReceiveMore);
        eo.addEntry("Sent to me", sentToMe);
        throw eo;
    }
    for(MPI_Request &request : this->receiveRequests)
    {
        if(request != MPI_REQUEST_NULL)
        {
            MPI_Cancel(&request);
        }
    }
    if(not this->receiveRequests.empty())
    {
        MPI_Waitall(this->receiveRequests.size(), this->receiveRequests.data(), MPI_STATUSES_IGNORE);
    }
    MPI_Barrier(this->comm);
}

template<typename T>
void BuffersManager<T>::Receive(bool ignore)
{
    static std::vector<int> array_of_indices;
    static std::vector<MPI_Status> array_of_statuses;
    static std::vector<T> data;

    if(array_of_indices.size() != this->receiveRequests.size())
    {
        array_of_indices.resize(this->receiveRequests.size());
    }
    if(array_of_statuses.size() != this->receiveRequests.size())
    {
        array_of_statuses.resize(this->receiveRequests.size());
    }

    int outcount;
    MPI_Testsome(this->receiveRequests.size(), this->receiveRequests.data(), &outcount, array_of_indices.data(), array_of_statuses.data());
    
    for(int i = 0; i < outcount; i++)
    {
        size_t requestIndex = array_of_indices[i];
        MPI_Status &status = array_of_statuses[i];
        rank_t fromRank = status.MPI_SOURCE;
        size_t bufferIndex = this->recvBuffersByRequests.at(requestIndex);
        Serializer &serializer = this->buffers[bufferIndex];

        MPI_Request &request = this->receiveRequests[requestIndex];
        if(not ignore)
        {
            size_t n;
            size_t bytes = serializer.extract(n, 0);
            data.clear();
            serializer.extract(data, bytes, n);
            this->receiveCallback(data.data(), data.size(), fromRank);
        }

        // recycle buffer and request, use for receiving again
        this->recvBuffersByRequests[requestIndex] = bufferIndex;
        this->recvCounter++;
        this->recvCounters[fromRank]++;
        MPI_Irecv(serializer.getData(), serializer.size(), MPI_BYTE, this->receiveSources[requestIndex], this->tag, this->comm, &request);
    }
}

template<typename T>
void BuffersManager<T>::Add(rank_t rank, const T &value)
{
    auto it = this->ranksSendBuffers.find(rank);
    size_t bufferIndex;
    if(it == this->ranksSendBuffers.end())
    {
        // need to allocate a new buffer. First check if one exists
        if(this->availableBuffersIndices.empty())
        {
            // no available buffers, need to create a new one
            bufferIndex = this->buffers.size();
            Serializer &serializer = this->buffers.emplace_back();
            serializer.insert(static_cast<size_t>(0)); // for size
        }
        else
        {
            // use an available buffer (recycle)
            auto it2 = this->availableBuffersIndices.begin();
            bufferIndex = *it2;
            this->availableBuffersIndices.erase(it2);
            Serializer &serializer = this->buffers[bufferIndex];
            serializer.reset();
            serializer.insert(static_cast<size_t>(0)); // for size
        }
    }
    else
    {
        // already has a waiting buffer
        bufferIndex = it->second;
        // std::cout << "Buffer of rank " << rank << " already exists, index " << bufferIndex << std::endl;
    }
    
    // std::cout << "Uses buffer " << bufferIndex << " to rank " << rank << std::endl;
    this->ranksSendBuffers[rank] = bufferIndex;
    
    // // std::cout << "Adds " << value << " to rank " << rank << "'s buffer" << std::endl;
    // assert(std::find(this->buffers[bufferIndex].cbegin(), this->buffers[bufferIndex].cend(), value) == this->buffers[bufferIndex].cend());
    
    // add value to internal buffer
    Serializer &serializer = this->buffers[bufferIndex];
    size_t bytes = 0;
    serializer.extract(bytes, 0);
    bytes += serializer.insert(value);
    serializer.hardSet(0, bytes); // change number of bytes

    assert(bytes <= this->buffersSize);

    // check if the buffer should be sent
    if(this->ShouldSend(rank))
    {
        this->Dispatch(rank);
    }
}

template<typename T>
bool BuffersManager<T>::ShouldSend(rank_t rank)
{
    const auto it = this->ranksSendBuffers.find(rank);
    if(it == this->ranksSendBuffers.end())
    {
        return false;
    }
    size_t bufferIndex = it->second;
    size_t bufferSize;
    this->buffers[bufferIndex].extract(bufferSize, 0);
    if(bufferSize == sizeof(size_t))
    {
        return false;
    }
    if(bufferSize >= this->minSizeToDispatch)
    {
        return true;
    }
    if(this->buffersToCycles[rank] >= this->minCyclesToDispatch)
    {
        return true;
    }
    this->buffersToCycles[rank] += 1;
    return false;
}

template<typename T>
void BuffersManager<T>::Dispatch(rank_t rank)
{
    assert(this->ranksSendBuffers.find(rank) != this->ranksSendBuffers.end());
    size_t bufferIndex = this->ranksSendBuffers.at(rank);

    assert(this->buffers[bufferIndex].size() >= sizeof(size_t));
    assert(this->buffers[bufferIndex].size() <= this->buffersSize);
    // std::cout << "Dispatches buffer " << bufferIndex << " to rank " << rank << std::endl;
    MPI_Request &request = this->sendRequests.emplace_back(MPI_REQUEST_NULL);
    this->sendBuffersByRequests[this->sendRequests.size() - 1] = bufferIndex;
    MPI_Issend(this->buffers[bufferIndex].getData(), this->buffers[bufferIndex].size(), MPI_BYTE, rank, this->tag, this->comm, &request);
    // std::cout << "Sended buffer, request " << &request << " to rank " << rank << " (currently: " << this->sendRequests.size() << " requests)" << std::endl;
    this->activeSendRequests++;
    this->buffersToCycles[rank] = 0; // reset cycles
    this->ranksSendBuffers.erase(rank); // remove rank from send buffers
    this->sendCounters[rank]++;
    this->sendCounter++;
}

template<typename T>
void BuffersManager<T>::CleanSendRequests(void)
{
    static std::vector<int> array_of_indices;
    static std::vector<MPI_Status> array_of_statuses;
    if(this->sendRequests.empty())
    {
        return;
    }
    
    if(array_of_indices.size() != this->sendRequests.size())
    {
        array_of_indices.resize(this->sendRequests.size());
    }
    if(array_of_statuses.size() != this->sendRequests.size())
    {
        array_of_statuses.resize(this->sendRequests.size());
    }

    int outcount;
    MPI_Testsome(this->sendRequests.size(), this->sendRequests.data(), &outcount, array_of_indices.data(), array_of_statuses.data());

    for(int i = outcount - 1; i >= 0; i--)
    {
        int requestNum = array_of_indices[i];
        MPI_Request &request = this->sendRequests.at(requestNum);
        size_t bufferIndex = this->sendBuffersByRequests.at(requestNum);
        this->availableBuffersIndices.insert(bufferIndex);
        this->sendBuffersByRequests.erase(requestNum);

        // delete request
        size_t otherRequestNumber = this->sendRequests.size() - 1;
        if(otherRequestNumber != requestNum)
        {
            // it changes other request's number, so change its buffer in the map
            this->sendBuffersByRequests[requestNum] = this->sendBuffersByRequests.at(otherRequestNumber);
            this->sendBuffersByRequests.erase(otherRequestNumber); // todo: GOOD?
            // std::cout << "Now request " << otherRequestNumber << " is request " << requestNum << ", mapped to buffer " << this->sendBuffersByRequests[requestNum] << std::endl;
        }
        std::swap(request, this->sendRequests.back());
        this->sendRequests.pop_back();
        // std::cout << "Deleted request " << &request << " (currently: " << this->sendRequests.size() << " requests)" << std::endl;

        this->activeSendRequests--;
    }
}

template<typename T>
void BuffersManager<T>::HandleIncomingOutcoming(void)
{
    std::vector<rank_t> ranksToDispatch;
    ranksToDispatch.reserve(this->ranksSendBuffers.size());
    for(const auto &it : this->ranksSendBuffers)
    {
        rank_t rank = it.first;
        // std::cout << "Rank " << this->rank_world << " handles rank " << rank << std::endl;
        if(this->ShouldSend(rank))
        {
            ranksToDispatch.push_back(rank);
        }
    }
    for(rank_t rank : ranksToDispatch)
    {
        this->Dispatch(rank);
    }
    this->Receive();
    this->CleanSendRequests();
}

#endif // RICH_MPI

#endif // BUFFERS_MANAGER_HPP
