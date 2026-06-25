#ifndef MPI_EXCHANGE_COMMANDS_HPP
#define MPI_EXCHANGE_COMMANDS_HPP

#ifdef RICH_MPI

#include "utils/printing/print.hpp" // todo remove
#include <iostream> // todo remove
#include <mpi.h>
#include <vector>
#include <functional>

template<typename T>
inline size_t AgreeChunkSize(const std::vector<T> &data)
{
    size_t chunkSize = data.empty()? 0 : data[0].getChunkSize();
    MPI_Allreduce(MPI_IN_PLACE, &chunkSize, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    return chunkSize;
}

inline MPI_Comm MPI_Create_graph_comm(const std::vector<int> &sendProcs, const std::vector<int> &sendCounts, const MPI_Comm &comm)
{
    int commRank;
    MPI_Comm_rank(comm, &commRank);
    int degree = static_cast<int>(sendProcs.size());
    MPI_Comm graphComm;
    MPI_Dist_graph_create(comm, 1, &commRank, &degree, sendProcs.data(), sendCounts.data(), MPI_INFO_NULL, 0, &graphComm);
    int graphRank;
    MPI_Comm_rank(graphComm, &graphRank);
    std::cout << "graphRank = " << graphRank << std::endl;
    return graphComm;
}

template<typename T, typename RankToDataFunction = std::function<const std::vector<T>&(int)>>
inline std::tuple<std::vector<double>, std::vector<int>, std::vector<int>> CalculateSerializableSendData(const std::vector<int> &sentProc, const RankToDataFunction &rankToData)
{
    size_t size = sentProc.size();
    std::vector<double> sendData;
    std::vector<int> sendCounts;
    std::vector<int> sendDisplacements;
    sendCounts.reserve(size);
    sendDisplacements.reserve(size);
    std::vector<double> serialized;
    for(size_t i = 0; i < size; i++)
    {
        sendDisplacements.push_back((i == 0)? 0 : sendDisplacements.back() + sendCounts.back());
        int _rank = sentProc[i];
        const std::vector<T> &data = rankToData(_rank);
        size_t sendCounter = 0;
        for(const T &element : data)
        {
            serialized = element.serialize();
            sendData.insert(sendData.end(), serialized.cbegin(), serialized.cend());
            sendCounter += serialized.size();
        }
        sendCounts.push_back(static_cast<int>(sendCounter));
    }
    return {sendData, sendCounts, sendDisplacements};
}

template<typename T>
inline std::tuple<std::vector<int>, std::vector<int>, size_t> SyncReceiveData(const std::vector<int> &sendCounts, const MPI_Comm &comm)
{
    size_t size = sendCounts.size();
    std::vector<int> recvCounts(size);
    MPI_Neighbor_alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, comm);
    
    std::vector<int> recvDisplacements;
    size_t totalRecvSize = 0;
    for(size_t i = 0; i < size; i++)
    {
        if(i == 0)
        {
            recvDisplacements.push_back(0);
        }
        else
        {
            recvDisplacements.push_back(recvDisplacements.back() + recvCounts[i - 1]);
        }
        totalRecvSize += recvCounts[i];
    }
    return {recvCounts, recvDisplacements, totalRecvSize};
}

template<typename T, typename InputIterator>
inline void TranslateSerializableVector(std::vector<T> &result, const InputIterator &first, const InputIterator &last)
{
    size_t chunkSize = T().getChunkSize();
    result.reserve(result.size() + std::distance(first, last) / chunkSize);
    std::vector<double> chunk;
    chunk.reserve(chunkSize);
    for(auto it = first; it != last; it += chunkSize)
    {
        chunk.assign(it, it + chunkSize);
        result.emplace_back();
        result.back().unserialize(chunk);
    }
}

template<typename T, typename RankToDataFunction = std::function<const std::vector<T>&(int)>>
inline std::vector<T> MPI_Exchange_data_with_data_function(const std::vector<int> &sentProc, const RankToDataFunction &rankToData, const MPI_Comm &comm = MPI_COMM_WORLD)
{
    std::vector<int> _sendCounts;
    for(int _rank : sentProc)
    {
        const std::vector<T> &data = rankToData(_rank);
        _sendCounts.push_back(static_cast<int>(data.empty()? 0 : data[0].getChunkSize() * data.size()));
    }
    MPI_Comm graphComm = MPI_Create_graph_comm(sentProc, _sendCounts, comm);

    // reorder ranks
    std::vector<int> sources(sentProc.size()), destinations(sentProc.size());
    std::vector<int> srcWeights(sources.size()), destWeights(destinations.size());
    MPI_Dist_graph_neighbors(graphComm, sources.size(), sources.data(), srcWeights.data(), destinations.size(), destinations.data(), destWeights.data());
    
    const auto &[sendData, sendCounts, sendDisplacements] = CalculateSerializableSendData<T>(destinations, rankToData);

    const auto &[recvCounts, recvDisplacements, totalRecvSize] = SyncReceiveData<T>(sendCounts, graphComm);
    
    std::vector<double> recvData(totalRecvSize);
    MPI_Neighbor_alltoallv(sendData.data(), sendCounts.data(), sendDisplacements.data(), MPI_DOUBLE, recvData.data(), recvCounts.data(), recvDisplacements.data(), MPI_DOUBLE, graphComm);
    
    std::vector<T> dataToReturn;
    dataToReturn.reserve(totalRecvSize / T().getChunkSize());
    for(int _rank : sentProc)
    {
        size_t index = std::distance(destinations.cbegin(), std::find(destinations.cbegin(), destinations.cend(), _rank));
        TranslateSerializableVector<T>(dataToReturn, recvData.cbegin() + recvDisplacements[index], recvData.cend() + recvDisplacements[index] + recvCounts[index]);
    }
    MPI_Comm_free(&graphComm);
    return dataToReturn;
}

// template<typename T, typename RankToDataFunction = std::function<const std::vector<T>&(int)>>
// inline std::vector<std::vector<T>> MPI_Exchange_data_seperate_with_data_function(const std::vector<int> &sentProc, const RankToDataFunction &rankToData, const MPI_Comm &comm = MPI_COMM_WORLD)
// {
//     std::vector<int> _sendCounts;
//     for(int _rank : sentProc)
//     {
//         const std::vector<T> &data = rankToData(_rank);
//         _sendCounts.push_back(static_cast<int>(data.empty() ? 0 : data[0].getChunkSize() * data.size()));
//     }
//     MPI_Comm graphComm = MPI_Create_graph_comm(sentProc, _sendCounts, comm);

//     // reorder ranks
//     std::vector<int> sources(sentProc.size()), destinations(sentProc.size());
//     std::vector<int> srcWeights(sources.size()), destWeights(destinations.size());
//     MPI_Dist_graph_neighbors(graphComm, sources.size(), sources.data(), srcWeights.data(), destinations.size(), destinations.data(), destWeights.data());
    
//     int commRank;
//     MPI_Comm_rank(comm, &commRank);
//     std::cout << "rank " << commRank << ", sentproc is " << sentProc << ", sources is " << sources << ", destinations is " << destinations << std::endl;

//     const auto &[sendData, sendCounts, sendDisplacements] = CalculateSerializableSendData<T>(destinations, rankToData);
    
//     std::cout << "rank " << commRank << ", sentproc is " << sentProc << ", sendCounts is " << sendCounts << ", sendDisplacements is " << sendDisplacements << std::endl;

//     const auto &[recvCounts, recvDisplacements, totalRecvSize] = SyncReceiveData<T>(sendCounts, graphComm);

//     std::cout << "rank " << commRank << ", sentproc is " << sentProc << ", recvCounts is " << recvCounts << ", recvDisplacements is " << recvDisplacements << std::endl;

//     std::vector<double> recvData(totalRecvSize);
//     MPI_Neighbor_alltoallv(sendData.data(), sendCounts.data(), sendDisplacements.data(), MPI_DOUBLE, recvData.data(), recvCounts.data(), recvDisplacements.data(), MPI_DOUBLE, graphComm);
    
//     std::vector<std::vector<T>> dataToReturn;
//     dataToReturn.reserve(sentProc.size());
//     for(int _rank : sentProc)
//     {
//         size_t index = std::distance(destinations.cbegin(), std::find(destinations.cbegin(), destinations.cend(), _rank));
//         dataToReturn.emplace_back();
//         TranslateSerializableVector<T>(dataToReturn.back(), recvData.cbegin() + recvDisplacements[index], recvData.cbegin() + recvDisplacements[index] + recvCounts[index]);
//     }
//     return dataToReturn;
// }

template<typename T, typename IndexingType = size_t>
inline std::vector<T> MPI_Exchange_data_indices(const std::vector<T> &data, const std::vector<int> &sentProc, const std::vector<std::vector<IndexingType>> &sentIndices, const std::function<size_t(IndexingType)> &indicesTranslation, const MPI_Comm &comm = MPI_COMM_WORLD)
{
    auto function = [&data, &sentProc, &sentIndices, &indicesTranslation](int _rank)
    {
        size_t index = std::distance(sentProc.cbegin(), std::find(sentProc.cbegin(), sentProc.cend(), _rank));
        if(index == sentProc.size())
        {
            return std::vector<T>();
        }
        const std::vector<IndexingType> &indices = sentIndices[index];
        std::vector<T> result;
        result.reserve(indices.size());
        for(IndexingType ind : indices)
        {
            result.push_back(data[indicesTranslation(ind)]);
        }
        return result;
    };
    return MPI_Exchange_data_with_data_function<T>(sentProc, function, comm);
}

// template<typename T, typename IndexingType = size_t>
// inline std::vector<std::vector<T>> MPI_Exchange_data_seperate_indices(const std::vector<T> &data, const std::vector<int> &sentProc, const std::vector<std::vector<IndexingType>> &sentIndices, const std::function<size_t(IndexingType)> &indicesTranslation, const MPI_Comm &comm = MPI_COMM_WORLD)
// {
//     auto function = [&data, &sentProc, &sentIndices, &indicesTranslation](int _rank)
//     {   
//         size_t index = std::distance(sentProc.cbegin(), std::find(sentProc.cbegin(), sentProc.cend(), _rank));
//         if(index == sentProc.size())
//         {
//             return std::vector<T>();
//         }
//         const std::vector<IndexingType> &indices = sentIndices[index];
//         std::vector<T> result;
//         result.reserve(indices.size());
//         for(IndexingType ind : indices)
//         {
//             result.push_back(data[indicesTranslation(ind)]);
//         }
//         return result;
//     };
//     return MPI_Exchange_data_seperate_with_data_function<T>(sentProc, function, comm);
// }

template<typename T, typename OwnershipFunction = std::function<std::vector<int>(const T&)>>
inline std::pair<std::vector<int>, std::vector<std::vector<size_t>>> PrepareSendValuesOwnership(const std::vector<T> &data, const OwnershipFunction &ownership)
{
    std::vector<int> sentProc;
    std::vector<std::vector<size_t>> sentIndices;
    std::vector<int> owners;
    for(size_t i = 0; i < data.size(); i++)
    {
        owners = ownership(data[i]);
        for(int owner : owners)
        {
            size_t index = std::distance(sentProc.cbegin(), std::find(sentProc.cbegin(), sentProc.cend(), owner));
            if(index == sentProc.size())
            {
                sentProc.push_back(owner);
                sentIndices.push_back({i});
            }
            else
            {
                sentIndices[index].push_back(i);
            }
        }
    }
    return {sentProc, sentIndices};
}

template<typename T, typename OwnershipFunction = std::function<std::vector<int>(const T&)>>
inline std::vector<T> MPI_Exchange_data_ownership(const std::vector<T> &data, const OwnershipFunction &ownership, const MPI_Comm &comm = MPI_COMM_WORLD)
{
    const auto &[sentProc, sentIndices] = PrepareSendValuesOwnership<T, OwnershipFunction>(data, ownership);
    return MPI_Exchange_data_indices<T>(data, sentProc, sentIndices, [](const size_t &ind){return ind;}, comm);
}

// template<typename T, typename OwnershipFunction = std::function<std::vector<int>(const T&)>>
// inline std::vector<std::vector<T>> MPI_Exchange_data_seperate_ownership(const std::vector<T> &data, const OwnershipFunction &ownership, const MPI_Comm &comm = MPI_COMM_WORLD)
// {
//     const auto &[sentProc, sentIndices] = PrepareSendValuesOwnership<T, OwnershipFunction>(data, ownership);
//     return MPI_Exchange_data_seperate_indices<T>(data, sentProc, sentIndices, [](const size_t &ind){return ind;}, comm);
// }

template<typename T, typename IndexingType = size_t>
inline std::vector<T> MPI_Exchange_data(const std::vector<T> &data, const std::vector<int> &sentProc, const std::vector<std::vector<IndexingType>> &sentIndices, const MPI_Comm &comm = MPI_COMM_WORLD)
{
    return MPI_Exchange_data_indices<T, IndexingType>(data, sentProc, sentIndices, [](const IndexingType &ind){return static_cast<size_t>(ind);}, comm);
}

// template<typename T, typename IndexingType = size_t>
// inline std::vector<std::vector<T>> MPI_Exchange_data_seperate(const std::vector<T> &data, const std::vector<int> &sentProc, const std::vector<std::vector<IndexingType>> &sentIndices, const MPI_Comm &comm = MPI_COMM_WORLD)
// {
//     return MPI_Exchange_data_seperate_indices<T, IndexingType>(data, sentProc, sentIndices, [](const IndexingType &ind){return static_cast<size_t>(ind);}, comm);
// }

#endif // RICH_MPI

#endif // MPI_EXCHANGE_COMMANDS_HPP