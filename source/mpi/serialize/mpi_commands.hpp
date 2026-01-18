#ifndef MPI_SERIALIZABLE_COMMANDS_HPP
#define MPI_SERIALIZABLE_COMMANDS_HPP

#ifdef RICH_MPI

#include <functional>
#include <mpi.h>
#include "Serializer.hpp"
#include "mpi/mpi_commands.hpp"
#include "misc/universal_error.hpp"

#define MPI_EXCHANGE_ALLTOALL_TAG 1039

template<typename T, template<typename...> class Container, typename... Ts>
std::vector<std::vector<T>> MPI_Iexchange_all_to_all(const std::vector<Container<T, Ts...>> &data, const MPI_Comm &comm)
{
    rank_t size;
    MPI_Comm_size(comm, &size);
    std::vector<MPI_Request> requests(size);
    std::vector<Serializer> senders(size);
    for(rank_t i = 0; i < size; i++)
    {
        senders[i].insert_all(data[i]);
        MPI_Isend((senders[i].size() > 0)? senders[i].getData() : NULL, senders[i].size(), MPI_BYTE, i, MPI_EXCHANGE_ALLTOALL_TAG, comm, &requests[i]);
    }

    std::vector<Serializer> receivers(size);
    for(rank_t i = 0; i < size; i++)
    {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_EXCHANGE_ALLTOALL_TAG, comm, &status);
        int count;
        MPI_Get_count(&status, MPI_BYTE, &count);
        receivers[status.MPI_SOURCE].resize(count);
        MPI_Recv(receivers[status.MPI_SOURCE].getData(), count, MPI_BYTE, status.MPI_SOURCE, status.MPI_TAG, comm, MPI_STATUS_IGNORE);
    }
    std::vector<std::vector<T>> result(size);
    for(rank_t i = 0; i < size; i++)
    {
        receivers[i].extract_all(result[i]);
    }
    if(not requests.empty())
    {
        MPI_Waitall(static_cast<int>(size), requests.data(), MPI_STATUSES_IGNORE);
    }
    MPI_Barrier(comm);
    return result;
}

template<typename T, template<typename...> class Container, typename... Ts>
std::vector<std::vector<T>> MPI_Iexchange_by_ranks(const std::vector<Container<T, Ts...>> &data, const std::vector<rank_t> &correspondents, const MPI_Comm &comm)
{
    rank_t rank;
    MPI_Comm_rank(comm, &rank);

    size_t sendSize = correspondents.size();

    std::vector<MPI_Request> requests(sendSize);
    std::vector<Serializer> senders(sendSize);
    for(size_t i = 0; i < sendSize; i++)
    {
        senders[i].insert_all(data[i]);
        MPI_Isend((senders[i].size() > 0)? senders[i].getData() : NULL, senders[i].size(), MPI_BYTE, correspondents[i], MPI_EXCHANGE_ALLTOALL_TAG, comm, &requests[i]);
    }

    std::vector<std::vector<T>> result(sendSize);
    for(size_t i = 0; i < sendSize; i++)
    {
        Serializer receiver;
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_EXCHANGE_ALLTOALL_TAG, comm, &status);
        size_t index = std::distance(correspondents.begin(), std::find(correspondents.begin(), correspondents.end(), status.MPI_SOURCE));
        if(index >= sendSize)
        {
            UniversalError eo("MPI_Iexchange_by_ranks: received from an unexpected rank");
            eo.addEntry("My rank", rank);
            eo.addEntry("Received From", status.MPI_SOURCE);
            eo.addEntry("Correspondents", correspondents);
            throw eo;
        }
        int count;
        MPI_Get_count(&status, MPI_BYTE, &count);
        receiver.resize(count);
        MPI_Recv(receiver.getData(), count, MPI_BYTE, status.MPI_SOURCE, status.MPI_TAG, comm, MPI_STATUS_IGNORE);
        receiver.extract(result[index], 0);
    }
    if(not requests.empty())
    {
        MPI_Waitall(static_cast<int>(sendSize), requests.data(), MPI_STATUSES_IGNORE);
    }
    MPI_Barrier(comm);
    return result;
}

template<typename T, template<typename...> class Container, typename... Ts>
std::vector<std::vector<T>> MPI_Exchange_all_to_all(const std::vector<Container<T, Ts...>> &data, const MPI_Comm &comm)
{
    rank_t size;
    Serializer send;
    MPI_Comm_size(comm, &size);
    assert(data.size() == size);
    
    std::vector<int> sendDisplacements(size, 0), recvDisplacements(size, 0);
    std::vector<int> sendCounts(size, 0), recvCounts(size, 0);

    for(rank_t _rank = 0; _rank < size; _rank++)
    {
        sendCounts[_rank] = static_cast<int>(send.insert_all(data[_rank]));
        if(_rank > 0)
        {
            sendDisplacements[_rank] = sendDisplacements[_rank - 1] + sendCounts[_rank - 1];
        }
    }

    int totalSize = 0;
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, comm);

    for(rank_t _rank = 0; _rank < size; _rank++)
    {
        totalSize += recvCounts[_rank];
        if(_rank > 0)
        if(_rank > 0)
        {
            recvDisplacements[_rank] = recvDisplacements[_rank - 1] + recvCounts[_rank - 1];
        }
    }

    Serializer recv;
    recv.resize(totalSize);

    MPI_Alltoallv(send.getData(), sendCounts.data(), sendDisplacements.data(), MPI_BYTE, recv.getData(), recvCounts.data(), recvDisplacements.data(), MPI_BYTE, comm);

    std::vector<std::vector<T>> result(size);
    for(rank_t _rank = 0; _rank < size; _rank++)
    {
        size_t bytesRead = recv.extract(result[_rank], recvDisplacements[_rank], recvCounts[_rank]);
        assert(bytesRead == static_cast<size_t>(recvCounts[_rank]));
    }

    return result;
}

template<typename T, typename FunctionType = std::function<rank_t(const T&)>>
std::vector<std::vector<T>> MPI_Exchange_by_ownership_by_ranks(const std::vector<T> &data, const FunctionType &ownership, const MPI_Comm &comm)
{
	rank_t rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	std::vector<std::vector<T>> dataByRanks(size);
	for(const T &value : data)
	{
		rank_t _rank = ownership(value);
		if(_rank < 0 or _rank >= size)
		{
			UniversalError eo("MPI_Exchange_by_ownership_by_ranks: ownership function returned invalid rank");
			eo.addEntry("Rank", _rank);
			eo.addEntry("Size", size);
			throw eo;
		}
		dataByRanks[_rank].push_back(value);
	}

	return MPI_Exchange_all_to_all(dataByRanks, comm);
}

template<typename T, typename FunctionType = std::function<rank_t(const T&)>>
std::vector<T> MPI_Exchange_by_ownership(const std::vector<T> &data, const FunctionType &ownership, const MPI_Comm &comm)
{
	std::vector<std::vector<T>> exchangedData = MPI_Exchange_by_ownership_by_ranks(data, ownership, comm);
	std::vector<T> result;
	for(const std::vector<T> &values : exchangedData)
	{
		result.insert(result.end(), values.cbegin(), values.cend());
	}
	return result;
}

template<typename T, template<typename...> class Container, typename... Ts>
std::vector<std::vector<T>> MPI_All_cast_by_ranks(const Container<T, Ts...> &data, const MPI_Comm &comm = MPI_COMM_WORLD)
{
    rank_t size;
    MPI_Comm_size(comm, &size);

    // first know how much data is being sent from each one
    Serializer send;
    int count = static_cast<int>(send.insert_all(data));
    std::vector<int> recvCounts(size, 0);
    MPI_Allgather(&count, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, comm);

    std::vector<int> recvDisplacements(size, 0);
    size_t totalToReceive = 0;
    for(rank_t _rank = 0; _rank < size; _rank++)
    {
        totalToReceive += static_cast<size_t>(recvCounts[_rank]);
        if(_rank > 0)
        {
            recvDisplacements[_rank] = recvDisplacements[_rank - 1] + recvCounts[_rank - 1];
        }
    }

    std::vector<int> sendDisplacements(size, 0);
    std::vector<int> sendCounts(size, count);
    Serializer recv;
    recv.resize(totalToReceive);
    MPI_Alltoallv(send.getData(), sendCounts.data(), sendDisplacements.data(), MPI_BYTE,
                    recv.getData(), recvCounts.data(), recvDisplacements.data(), MPI_BYTE, comm);

    std::vector<std::vector<T>> resultByRanks(size);
    for(rank_t _rank = 0; _rank < size; _rank++)
    {
        size_t readCount = recv.extract(resultByRanks[_rank], recvDisplacements[_rank], recvCounts[_rank]);
        assert(readCount == recvCounts[_rank]);
    }
    return resultByRanks;
}

template<typename T, template<typename...> class Container, typename... Ts>
std::vector<T> MPI_All_cast(const Container<T, Ts...> &data, const MPI_Comm &comm)
{
	std::vector<std::vector<T>> resultByRanks = MPI_All_cast_by_ranks(data, comm);
	std::vector<T> result;
	for(const std::vector<T> &values : resultByRanks)
	{
		result.insert(result.end(), values.cbegin(), values.cend());
	}
	return result;
}

template<typename T>
T MPI_Bcast_serializable(const T &data, rank_t owner, const MPI_Comm &comm = MPI_COMM_WORLD)
{
    Serializer send;
    size_t sizeSent = send.insert(data);
    MPI_Bcast(&sizeSent, 1, MPI_UNSIGNED_LONG_LONG, owner, comm);
    
    Serializer recv;
    recv.resize(sizeSent);
    MPI_Bcast(recv.getData(), sizeSent, MPI_BYTE, owner, comm);

    T value;
    size_t sizeRead = recv.extract(value, 0);
    assert(sizeRead == sizeSent);
    return value;
}

template<typename T, template<typename...> class Container, typename... Ts>
std::vector<T> MPI_Gatherv_serializable(const Container<T, Ts...> &data, rank_t root, const MPI_Comm &comm)
{
    rank_t rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if(size == 1)
    {
        return data;
    }
    Serializer send;
    int bytes = static_cast<int>(send.insert_all(data));

    if(rank == root)
    {
        std::vector<int> toRecvBytes(size);
        MPI_Gather(&bytes, 1, MPI_INT, toRecvBytes.data(), 1, MPI_INT, root, comm);
        std::vector<int> toRecvDisplacements(size, 0);
        size_t totalSize = 0;
        for(int _rank = 0; _rank < size; _rank++)
        {
            totalSize += toRecvBytes[_rank];
            if(_rank > 0)
            {
                toRecvDisplacements[_rank] = toRecvDisplacements[_rank - 1] + toRecvBytes[_rank - 1];
            }
        }
        Serializer recv;
        recv.resize(totalSize);
        MPI_Gatherv(send.getData(), bytes, MPI_BYTE, recv.getData(), toRecvBytes.data(), toRecvDisplacements.data(), MPI_BYTE, root, comm);
        std::vector<T> toReturn;
        recv.extract_all(toReturn);
        return toReturn;
    }
    // else
    MPI_Gather(&bytes, 1, MPI_INT, NULL, 0, MPI_INT, root, comm);
    MPI_Gatherv(send.getData(), bytes, MPI_BYTE, NULL, NULL, NULL, MPI_BYTE, root, comm);
    return std::vector<T>();
}

template<typename T, template<typename...> class Container, typename... Ts>
std::vector<T> MPI_Spread(const Container<T, Ts...> &data, rank_t root, const MPI_Comm &comm)
{
	rank_t rank, size;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &size);

	if(size == 1)
	{
		return data;
	}

	Serializer send;
	Serializer recv;
	int mySize;
	if(rank == root)
	{
		size_t totalSize = data.size();
		size_t idealSize = totalSize / size;
		std::vector<int> counts(size, 0);
		std::vector<int> offsets(size, 0);
		size_t current = 0;
		for(rank_t _rank = 0; _rank < size; _rank++)
		{
			size_t _begin = _rank * idealSize;
			size_t _end = (_rank == size - 1)? totalSize : ((_rank + 1) * idealSize);
			size_t length = _end - _begin;
			offsets[_rank] = current;
			counts[_rank] = send.insert_elements(data, _begin, length);
            current += counts[_rank];
		}
		MPI_Scatter(counts.data(), 1, MPI_INT, &mySize, 1, MPI_INT, root, comm);
		recv.resize(mySize);
		MPI_Scatterv(send.getData(), counts.data(), offsets.data(), MPI_BYTE, recv.getData(), mySize, MPI_BYTE, root, comm);
	}
	else
	{
		MPI_Scatter(NULL, 1, MPI_INT, &mySize, 1, MPI_INT, root, comm);
		recv.resize(mySize);
		MPI_Scatterv(NULL, NULL, NULL, MPI_BYTE, recv.getData(), mySize, MPI_BYTE, root, comm);
	}

	std::vector<T> toReturn;
	recv.extract_all(toReturn);
	return toReturn;
}

template<typename T, typename Index_T = size_t>
std::vector<std::vector<T>> MPI_Ask_data(const std::vector<rank_t> &correspondents, const std::vector<T> &myData, const std::vector<std::vector<Index_T>> &myRequestedIndices = std::vector<std::vector<Index_T>>())
{
    rank_t rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<rank_t> allRanks(size);
    std::iota(allRanks.begin(), allRanks.end(), 0);

    std::vector<std::vector<Index_T>> indicesToSend(size);
    for(size_t i = 0; i < correspondents.size(); i++)
    {
        const rank_t &_rank = correspondents[i];
        indicesToSend[_rank] = myRequestedIndices[i];
    }

    std::vector<std::vector<Index_T>> requestedIndices = MPI_Exchange_all_to_all(indicesToSend, MPI_COMM_WORLD);
    // std::vector<Index_T> allOutcomingIndices;
    // std::vector<int> outcomingIndicesLengths(size, 0), incomingIndicesLengths(size);
    // for(size_t i = 0; i < correspondents.size(); i++)
    // {
    //     const rank_t &_rank = correspondents[i];
    //     outcomingIndicesLengths[_rank] = static_cast<int>(myRequestedIndices[i].size()) * sizeof(Index_T);
    //     allOutcomingIndices.insert(allOutcomingIndices.end(), myRequestedIndices[i].cbegin(), myRequestedIndices[i].cend());
    // }

    // std::vector<int> outcomingIndicesDisplacements(size, 0);
    // for(int _rank = 1; _rank < size; _rank++)
    // {
    //     outcomingIndicesDisplacements[_rank] = outcomingIndicesDisplacements[_rank - 1] + outcomingIndicesLengths[_rank - 1];
    // }
    // MPI_Alltoall(outcomingIndicesLengths.data(), 1, MPI_INT, incomingIndicesLengths.data(), 1, MPI_INT, MPI_COMM_WORLD);

    // std::vector<int> incomingIndicesDisplacements(size, 0);
    // size_t totalIndicesLengths = incomingIndicesLengths[0];
    // for(rank_t _rank = 1; _rank < size; _rank++)
    // {
    //     incomingIndicesDisplacements[_rank] = incomingIndicesDisplacements[_rank - 1] + incomingIndicesLengths[_rank - 1];
    //     totalIndicesLengths += incomingIndicesLengths[_rank];
    // }

    // std::vector<Index_T> allIncomingIndices(totalIndicesLengths);
    // MPI_Alltoallv(allOutcomingIndices.data(), outcomingIndicesLengths.data(), outcomingIndicesDisplacements.data(), MPI_BYTE,
    //               allIncomingIndices.data(), incomingIndicesLengths.data(), incomingIndicesDisplacements.data(), MPI_BYTE, MPI_COMM_WORLD);

    // std::vector<std::vector<Index_T>> requestedIndices;
    // for(rank_t _rank = 0; _rank < size; _rank++)
    // {
    //     auto begin = allIncomingIndices.cbegin() + incomingIndicesDisplacements[_rank];
    //     size_t length = incomingIndicesLengths[_rank] / sizeof(Index_T);
    //     auto end = begin + length;
    //     requestedIndices.emplace_back(std::vector<Index_T>(begin, end));
    // }

    
    // size_t selfIndex = std::distance(correspondents.begin(), std::find(correspondents.begin(), correspondents.end(), rank));
    // assert(std::equal(myRequestedIndices[selfIndex].cbegin(), myRequestedIndices[selfIndex].cend(), requestedIndices[rank].cbegin()));
    
    std::vector<std::vector<T>> resultOfAllRanks = MPI_exchange_data_indexed(allRanks, myData, requestedIndices);
    assert(resultOfAllRanks.size() == size);
    std::vector<std::vector<T>> resultByRanks;

    // return only requested ranks
    for(rank_t _rank : correspondents)
    {
        resultByRanks.emplace_back(resultOfAllRanks[_rank]);
    }

    return resultByRanks;
}

#endif // RICH_MPI

#endif // MPI_SERIALIZABLE_COMMANDS_HPP