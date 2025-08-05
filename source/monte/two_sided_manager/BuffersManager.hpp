#ifndef BUFFERS_MANAGER_HPP
#define BUFFERS_MANAGER_HPP

#ifdef RICH_MPI

#include <mpi.h>
#include <vector>
#include <functional>
#include <boost/container/flat_map.hpp>
#include "mpi/mpi_commands.hpp"

template<typename T>
class BuffersManager
{
public:
    BuffersManager(const MPI_Comm &comm, const std::function<void(const T *newValues, size_t newValuesCount, rank_t fromRank)> &receiveCallback, int tag, size_t buffersSize, size_t minSizeToDispatch, size_t minCyclesToDispatch, size_t initialReceiveBuffers);

    ~BuffersManager();

    void Add(rank_t rank, const T &value);

    void Receive(void);

    void HandleIncomingOutcoming();

    inline size_t CountOutcoming() const{return activeSendRequests;};

    inline size_t GetPendingNumber() const{return this->ranksSendBuffers.size();};

    inline size_t GetSentCounter() const {return this->sendCounter;};

    inline size_t GetRecvCounter() const {return this->recvCounter;};

private:
    MPI_Comm comm;
    rank_t rank_world, size_world;
    std::vector<std::vector<T>> buffers;
    std::vector<MPI_Request> sendRequests;
    size_t activeSendRequests;
    std::vector<MPI_Request> receiveRequests;
    boost::container::flat_set<size_t> availableBuffersIndices;
    boost::container::flat_map<size_t, size_t> sendBuffersByRequests;
    boost::container::flat_map<size_t, size_t> recvBuffersByRequests;
    std::vector<size_t> buffersToCycles;
    
    // buffering mechanism
    boost::container::flat_map<rank_t, size_t> ranksSendBuffers;

    size_t sendCounter;
    size_t recvCounter;

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
BuffersManager<T>::BuffersManager(const MPI_Comm &comm, const std::function<void(const T *newValues, size_t newValuesCount, rank_t fromRank)> &receiveCallback, int tag, size_t buffersSize, size_t minSizeToDispatch, size_t minCyclesToDispatch, size_t numReceiveBuffers)
    : comm(comm), receiveCallback(receiveCallback), tag(tag), buffersSize(buffersSize), minSizeToDispatch(minSizeToDispatch), minCyclesToDispatch(minCyclesToDispatch)
{
    MPI_Comm_rank(comm, &this->rank_world);
    MPI_Comm_size(comm, &this->size_world);

    this->activeSendRequests = 0;
    this->sendCounter = 0;
    this->recvCounter = 0;

    if(this->buffersSize < this->minSizeToDispatch)
    {
        UniversalError eo("BuffersManager: buffersSize is less than minSizeToDispatch");
        eo.addEntry("Buffers size", this->buffersSize);
        eo.addEntry("Min size to dispatch", this->minSizeToDispatch);
        throw eo;
    }

    // initialize receive
    this->receiveRequests.resize(numReceiveBuffers, MPI_REQUEST_NULL);
    for(size_t i = 0; i < numReceiveBuffers; i++)
    {
        this->buffers.push_back(std::vector<T>(this->buffersSize));
        MPI_Request &request = this->receiveRequests[i];
        MPI_Irecv(this->buffers.back().data(), this->buffersSize * sizeof(T), MPI_BYTE, MPI_ANY_SOURCE, this->tag, comm, &request);
        assert(request != MPI_REQUEST_NULL);
        this->recvBuffersByRequests[i] = i;
    }

    this->buffersToCycles = std::vector<size_t>(this->size_world, 0);
}

template<typename T>
BuffersManager<T>::~BuffersManager()
{
    for(MPI_Request &request : this->receiveRequests)
    {
        MPI_Cancel(&request);
    }
}

template<typename T>
void BuffersManager<T>::Receive(void)
{
    static std::vector<int> array_of_indices;
    static std::vector<MPI_Status> array_of_statuses;

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
        size_t requestIndex = static_cast<size_t>(array_of_indices[i]);
        rank_t fromRank = array_of_statuses[i].MPI_SOURCE;
        int count;
        MPI_Get_count(&array_of_statuses[i], MPI_BYTE, &count);
        count /= sizeof(T);
        size_t bufferIndex = this->recvBuffersByRequests.at(requestIndex);

        MPI_Request &request = this->receiveRequests[requestIndex];
        this->receiveCallback(this->buffers[bufferIndex].data(), static_cast<size_t>(count), fromRank);

        // recycle buffer and request, use for receiving again
        this->recvBuffersByRequests[requestIndex] = bufferIndex;
        this->recvCounter++;
        MPI_Irecv(this->buffers[bufferIndex].data(), this->buffersSize * sizeof(T), MPI_BYTE, MPI_ANY_SOURCE, this->tag, MPI_COMM_WORLD, &request);
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
            this->buffers.push_back(std::vector<T>());
            this->buffers.back().reserve(this->buffersSize);
            bufferIndex = this->buffers.size() - 1;
        }
        else
        {
            // use an available buffer (recycle)
            auto it2 = this->availableBuffersIndices.begin();
            bufferIndex = *it2;
            this->availableBuffersIndices.erase(it2);
            this->buffers[bufferIndex].clear();
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

    // std::cout << "Adds " << value << " to rank " << rank << "'s buffer" << std::endl;
    assert(std::find(this->buffers[bufferIndex].cbegin(), this->buffers[bufferIndex].cend(), value) == this->buffers[bufferIndex].cend());

    // add value to internal buffer
    this->buffers[bufferIndex].push_back(value);

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
    if(this->buffers[bufferIndex].empty())
    {
        return false;
    }
    if(this->buffers[bufferIndex].size() >= this->minSizeToDispatch)
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

    // std::cout << "Dispatches buffer " << bufferIndex << " to rank " << rank << std::endl;
    MPI_Request &request = this->sendRequests.emplace_back(MPI_REQUEST_NULL);
    MPI_Isend(this->buffers[bufferIndex].data(), this->buffers[bufferIndex].size() * sizeof(T), MPI_BYTE, rank, this->tag, this->comm, &request);
    this->sendBuffersByRequests[this->sendRequests.size() - 1] = bufferIndex;
    this->activeSendRequests++;
    this->buffersToCycles[rank] = 0; // reset cycles
    this->ranksSendBuffers.erase(rank); // remove rank from send buffers
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
            // std::cout << "Now request " << otherRequestNumber << " is request " << requestNum << ", mapped to buffer " << this->sendBuffersByRequests[requestNum] << std::endl;
        }
        std::swap(request, this->sendRequests.back());
        this->sendRequests.pop_back();


        this->activeSendRequests--;
    }
}

template<typename T>
void BuffersManager<T>::HandleIncomingOutcoming(void)
{
    this->CleanSendRequests();
    for(const auto &it : this->ranksSendBuffers)
    {
        rank_t rank = it.first;
        // std::cout << "Rank " << this->rank_world << " handles rank " << rank << std::endl;
        if(this->ShouldSend(rank))
        {
            this->Dispatch(rank);
        }
    }
    this->Receive();
}

#endif // RICH_MPI

#endif // BUFFERS_MANAGER_HPP