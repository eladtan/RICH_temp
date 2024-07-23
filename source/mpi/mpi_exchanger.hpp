#ifndef MPI_EXCHANGER_HPP
#define MPI_EXCHANGER_HPP

#ifdef RICH_MPI

#include <iostream> // todo remove
#include <mpi.h>
#include <boost/container/flat_map.hpp>
#include <vector>
#include <boost/bimap/bimap.hpp>

using rank_t = int;

class MPI_Exchanger
{
public:
    MPI_Exchanger(const std::vector<rank_t> &sendProcessors, const std::vector<int> &weights = std::vector<int>(), const MPI_Comm &comm = MPI_COMM_WORLD);

    ~MPI_Exchanger();

    template<typename T>
    std::vector<T> exchange(const std::vector<rank_t> &sentProc, const std::vector<std::vector<T>> &sentData) const;

    template<typename T, typename IndexT = size_t>
    std::vector<T> exchange_indices(const std::vector<T> &data, const std::vector<int> &sentProc, const std::vector<std::vector<IndexT>> &sentIndices) const;

    template<typename T>
    std::vector<std::vector<T>> exchange_seperated(const std::vector<rank_t> &sentProc, const std::vector<std::vector<T>> &sentData) const;

    template<typename T, typename IndexT = size_t>
    std::vector<std::vector<T>> exchange_indices_seperated(const std::vector<T> &data, const std::vector<rank_t> &sentProc, const std::vector<std::vector<IndexT>> &sentIndices) const;

private:
    using RanksMapper = boost::bimaps::bimap<rank_t, rank_t>;

    struct SendInformation
    {
        std::vector<double> sendBuffer;
        std::vector<int> sendCounts;
        std::vector<int> sendDisplacements;
    };

    struct ReceiveInformation
    {
        std::vector<int> recvCounts;
        std::vector<int> recvDisplacements;
        size_t totalReceive;
    };

    template<typename T>
    SendInformation prepareToSend(const std::vector<rank_t> &sentProc, const std::vector<std::vector<T>> &sentData) const;

    template<typename T, typename IndexT>
    SendInformation prepareToSendIndices(const std::vector<T> &data, const std::vector<rank_t> &sentProc, const std::vector<std::vector<IndexT>> &sentIndices) const;

    ReceiveInformation prepareToReceive(const SendInformation &sendInfo) const;

    std::vector<double> makeSerializableExchange(const SendInformation &sendInfo, const ReceiveInformation &recvInfo) const;

    template<typename T>
    void serializableVectorToData(std::vector<T> &data, const std::vector<double>::iterator &first, const std::vector<double>::iterator &last) const;
    
    template<typename T>
    std::vector<T> exchange_with_sendInfo(const std::vector<rank_t> &sentProc, const SendInformation &sendInfo) const;
    
    template<typename T>
    std::vector<std::vector<T>> exchange_seperated_with_sendInfo(const std::vector<rank_t> &sentProc, const SendInformation &sendInfo) const;

    MPI_Comm graph;
    RanksMapper ranksToSources;
    RanksMapper ranksToDestinations;
};

template<typename T>
inline MPI_Exchanger::SendInformation MPI_Exchanger::prepareToSend(const std::vector<rank_t> &sentProc, const std::vector<std::vector<T>> &sentData) const
{
    SendInformation sendInfo;
    for(auto dest : this->ranksToDestinations.right)
    {
        rank_t _rank = dest.second;
        size_t rankIndex = std::distance(sentProc.begin(), std::find(sentProc.begin(), sentProc.end(), _rank));
        size_t amountSend = 0;
        if(rankIndex != sentProc.size())
        {
            for(const T &val : sentData[rankIndex])
            {
                std::vector<double> _serialized = val.serialize();
                sendInfo.sendBuffer.insert(sendInfo.sendBuffer.end(), _serialized.begin(), _serialized.end());
                amountSend += _serialized.size();
            }
        }
        size_t displacement = (sendInfo.sendDisplacements.empty())? 0 : sendInfo.sendDisplacements.back() + sendInfo.sendCounts.back();
        sendInfo.sendDisplacements.push_back(static_cast<int>(displacement));
        sendInfo.sendCounts.push_back(static_cast<int>(amountSend));
    }
    return sendInfo;
}

template<typename T, typename IndexT>
inline MPI_Exchanger::SendInformation MPI_Exchanger::prepareToSendIndices(const std::vector<T> &data, const std::vector<rank_t> &sentProc, const std::vector<std::vector<IndexT>> &sentIndices) const
{
    SendInformation sendInfo;
    for(auto dest : this->ranksToDestinations.right)
    {
        // std::cout << "Iterating dest, " << dest.first << ", " << dest.second << std::endl;
        rank_t _rank = dest.second;
        size_t rankIndex = std::distance(sentProc.begin(), std::find(sentProc.begin(), sentProc.end(), _rank));
        size_t amountSend = 0;
        if(rankIndex != sentProc.size())
        {
            for(const size_t &ind : sentIndices[rankIndex])
            {
                std::vector<double> _serialized = data[ind].serialize();
                sendInfo.sendBuffer.insert(sendInfo.sendBuffer.end(), _serialized.begin(), _serialized.end());
                amountSend += _serialized.size();
            }
        }
        size_t displacement = (sendInfo.sendDisplacements.empty())? 0 : sendInfo.sendDisplacements.back() + sendInfo.sendCounts.back();
        sendInfo.sendDisplacements.push_back(static_cast<int>(displacement));
        sendInfo.sendCounts.push_back(static_cast<int>(amountSend));
    }
    return sendInfo;
}

template<typename T>
inline void MPI_Exchanger::serializableVectorToData(std::vector<T> &data, const std::vector<double>::iterator &first, const std::vector<double>::iterator &last) const
{
    auto it = first;
    while(it != last)
    {
        data.emplace_back();
        size_t chunkSize = data.back().getChunkSize();
        data.back().unserialize(std::vector<double>(it, it + chunkSize));
        it += chunkSize;
    }
}

template<typename T>
inline std::vector<T> MPI_Exchanger::exchange_with_sendInfo(const std::vector<rank_t> &sentProc, const SendInformation &sendInfo) const
{
    ReceiveInformation recvInfo = this->prepareToReceive(sendInfo);
    std::vector<double> recvBuffer = this->makeSerializableExchange(sendInfo, recvInfo);
    std::vector<T> result;
    for(rank_t _rank : sentProc)
    {
        size_t srcIndex = this->ranksToSources.right.at(_rank);
        size_t count = recvInfo.recvCounts[srcIndex];
        size_t displacement = recvInfo.recvDisplacements[srcIndex];
        std::vector<double>::iterator first = recvBuffer.begin() + displacement;
        std::vector<double>::iterator last = first + count;
        serializableVectorToData(result, first, last);
    }
    return result;
}

template<typename T>
inline std::vector<std::vector<T>> MPI_Exchanger::exchange_seperated_with_sendInfo(const std::vector<rank_t> &sentProc, const SendInformation &sendInfo) const
{
    ReceiveInformation recvInfo = this->prepareToReceive(sendInfo);
    std::vector<double> recvBuffer = this->makeSerializableExchange(sendInfo, recvInfo);
    std::vector<std::vector<T>> result;
    result.reserve(sentProc.size());
    for(rank_t _rank : sentProc)
    {
        result.emplace_back();
        size_t srcIndex = this->ranksToSources.left.at(_rank);
        size_t count = recvInfo.recvCounts[srcIndex];
        size_t displacement = recvInfo.recvDisplacements[srcIndex];
        std::vector<double>::iterator first = recvBuffer.begin() + displacement;
        std::vector<double>::iterator last = first + count;
        serializableVectorToData(result.back(), first, last);
    }
    return result;
}

template<typename T>
inline std::vector<T> MPI_Exchanger::exchange(const std::vector<rank_t> &sentProc, const std::vector<std::vector<T>> &sentData) const
{
    SendInformation sendInfo = this->prepareToSend<T>(sentProc, sentData);
    return this->exchange_with_sendInfo<T>(sentProc, sendInfo);
}

template<typename T>
inline std::vector<std::vector<T>> MPI_Exchanger::exchange_seperated(const std::vector<rank_t> &sentProc, const std::vector<std::vector<T>> &sentData) const
{
    SendInformation sendInfo = this->prepareToSend<T>(sentProc, sentData);
    return this->exchange_seperated_with_sendInfo<T>(sentProc, sendInfo);
}

template<typename T, typename IndexT>
inline std::vector<T> MPI_Exchanger::exchange_indices(const std::vector<T> &data, const std::vector<rank_t> &sentProc, const std::vector<std::vector<IndexT>> &sentIndices) const
{
    SendInformation sendInfo = this->prepareToSendIndices<T, IndexT>(data, sentProc, sentIndices);
    return this->exchange_with_sendInfo<T>(sentProc, sendInfo);
}

template<typename T, typename IndexT>
inline std::vector<std::vector<T>> MPI_Exchanger::exchange_indices_seperated(const std::vector<T> &data, const std::vector<rank_t> &sentProc, const std::vector<std::vector<IndexT>> &sentIndices) const
{
    SendInformation sendInfo = this->prepareToSendIndices<T, IndexT>(data, sentProc, sentIndices);
    return this->exchange_seperated_with_sendInfo<T>(sentProc, sendInfo);
}

inline MPI_Exchanger::MPI_Exchanger(const std::vector<rank_t> &sendProcessors, const std::vector<int> &weights, const MPI_Comm &comm)
{
    int commRank;
    MPI_Comm_rank(comm, &commRank);
    int degree = static_cast<int>(sendProcessors.size());
    const int *weightsVector = (weights.size() != sendProcessors.size())? MPI_UNWEIGHTED : weights.data();
    MPI_Dist_graph_create(comm, 1, &commRank, &degree, sendProcessors.data(), weightsVector, MPI_INFO_NULL, 0, &this->graph);

    // get a list of sources
    std::vector<rank_t> sources(sendProcessors.size());
    std::vector<rank_t> destinations(sendProcessors.size());
    std::vector<int> sourcesWeights(sources.size()), destinationsWeights(destinations.size());

    MPI_Dist_graph_neighbors(this->graph, sources.size(), sources.data(), sourcesWeights.data(), destinations.size(), destinations.data(), destinationsWeights.data());

    for(size_t i = 0; i < sources.size(); i++)
    {
        // std::cout << "Rank " << commRank << ", source " << i << " is " << sources[i] << std::endl;
        this->ranksToSources.insert({sources[i], static_cast<rank_t>(i)});
    }
    for(size_t i = 0; i < sources.size(); i++)
    {
        // std::cout << "Rank " << commRank << ", destination " << i << " is " << destinations[i] << std::endl;
        this->ranksToDestinations.insert({destinations[i], static_cast<rank_t>(i)});
    }
}

inline MPI_Exchanger::~MPI_Exchanger()
{
    MPI_Comm_free(&this->graph);
}

inline MPI_Exchanger::ReceiveInformation MPI_Exchanger::prepareToReceive(const SendInformation &sendInfo) const
{
    ReceiveInformation recvInfo;
    recvInfo.recvCounts.resize(sendInfo.sendCounts.size());
    MPI_Neighbor_alltoall(sendInfo.sendCounts.data(), 1, MPI_INT, recvInfo.recvCounts.data(), 1, MPI_INT, this->graph);
    
    for(auto src : this->ranksToSources.right)
    {
        // std::cout << "Preparing receive, src = " << src.first << ", " << src.second << std::endl;
        size_t srcIndex = static_cast<size_t>(src.first);
        size_t displacement = recvInfo.totalReceive;
        recvInfo.recvDisplacements.push_back(static_cast<int>(displacement));
        recvInfo.totalReceive += recvInfo.recvCounts[srcIndex];
    }
    
    return recvInfo;
}

inline std::vector<double> MPI_Exchanger::makeSerializableExchange(const SendInformation &sendInfo, const ReceiveInformation &recvInfo) const
{
    std::vector<double> recvBuffer(recvInfo.totalReceive);
    MPI_Neighbor_alltoallv(sendInfo.sendBuffer.data(), sendInfo.sendCounts.data(), sendInfo.sendDisplacements.data(), MPI_DOUBLE, recvBuffer.data(), recvInfo.recvCounts.data(), recvInfo.recvDisplacements.data(), MPI_DOUBLE, this->graph);
    return recvBuffer;    
}

#endif // RICH_MPI

#endif // MPI_EXCHANGER_HPP