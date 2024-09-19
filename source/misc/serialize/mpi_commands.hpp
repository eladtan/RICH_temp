#ifndef MPI_SERIALIZABLE_COMMANDS_HPP
#define MPI_SERIALIZABLE_COMMANDS_HPP

#ifdef RICH_MPI

#include <functional>
#include <mpi.h>
#include "Serializer.hpp"
#include "misc/universal_error.hpp"

using rank_t = int;

template<typename T, template<typename...> class Container, typename... Ts>
std::vector<std::vector<T>> MPI_Exchange_all_to_all(const std::vector<Container<T, Ts...>> &data, const MPI_Comm &comm)
{
    rank_t size;
    Serializer send;
    MPI_Comm_size(comm, &size);

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
        assert(bytesRead != static_cast<size_t>(recvCounts[_rank]));
    }

    return result;
}

template<typename T>
std::vector<std::vector<T>> MPI_Exchange_by_ownership_by_ranks(const std::vector<T> &data, const std::function<rank_t(const T&)> &ownership, const MPI_Comm &comm)
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

template<typename T>
std::vector<T> MPI_Exchange_by_ownership(const std::vector<T> &data, const std::function<rank_t(const T&)> &ownership, const MPI_Comm &comm)
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

#endif // RICH_MPI

#endif // MPI_SERIALIZABLE_COMMANDS_HPP