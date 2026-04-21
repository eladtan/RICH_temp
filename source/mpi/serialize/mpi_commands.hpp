#ifndef MPI_SERIALIZABLE_COMMANDS_HPP
#define MPI_SERIALIZABLE_COMMANDS_HPP

#ifdef RICH_MPI

#include <functional>
#include <mpi.h>
#include "Serializer.hpp"
#include "mpi/mpi_commands.hpp"
#include "mpi/MPI_complex_dtype.hpp"
#include "misc/universal_error.hpp"

#define MPI_EXCHANGE_ALLTOALL_TAG 1039

template<typename T, template<typename...> class Container, typename... Ts>
std::vector<std::vector<T>> MPI_Iexchange_all_to_all(std::vector<Container<T, Ts...>> &data, const MPI_Comm &comm)
{
    rank_t size, myRank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &myRank);

    if constexpr (MPI_has_complex_dtype<T>::value)
    {
        // Native-dtype path: send T elements directly, count = number of elements
        MPI_Datatype dtype = MPI_has_complex_dtype<T>::getDatatype();

        std::vector<int> sendCounts(size), recvCounts(size);
        for(rank_t i = 0; i < size; i++)
            sendCounts[i] = static_cast<int>(data[i].size());
        MPI_Alltoall(sendCounts.data(), 1, MPI_INT,
                     recvCounts.data(), 1, MPI_INT, comm);

        std::vector<MPI_Request> requests(size);
        for(rank_t i = 0; i < size; i++)
        {
            MPI_Isend(data[i].data(), sendCounts[i], dtype, i,
                      MPI_EXCHANGE_ALLTOALL_TAG, comm, &requests[i]);
        }

        std::vector<std::vector<T>> result(size);
        for(rank_t i = 0; i < size; i++)
        {
            result[i].resize(static_cast<size_t>(recvCounts[i]));
            MPI_Recv(result[i].data(), recvCounts[i], dtype, i,
                     MPI_EXCHANGE_ALLTOALL_TAG, comm, MPI_STATUS_IGNORE);
        }

        MPI_Waitall(static_cast<int>(size), requests.data(), MPI_STATUSES_IGNORE);
        for(rank_t i = 0; i < size; i++)
            data[i] = Container<T, Ts...>();

        MPI_Barrier(comm);
        return result;
    }
    else
    {
        std::vector<MPI_Request> requests(size);
        std::vector<Serializer> senders(size);
        for(rank_t i = 0; i < size; i++)
        {
            senders[i].insert_all(data[i]);
            data[i] = Container<T, Ts...>();
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

        if(not requests.empty())
        {
            MPI_Waitall(static_cast<int>(size), requests.data(), MPI_STATUSES_IGNORE);
        }
        std::vector<std::vector<T>> result(size);
        for(rank_t i = 0; i < size; i++)
        {
            receivers[i].extract_all(result[i]);
            receivers[i].reset();
        }

        MPI_Barrier(comm);
        return result;
    }
}

template<typename T>
std::vector<std::vector<T>> MPI_Iexchange_all_to_all_serializers(std::vector<Serializer> &senders, const MPI_Comm &comm)
{
    rank_t size, myRank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &myRank);
    assert(static_cast<rank_t>(senders.size()) == size);

    if constexpr (MPI_has_complex_dtype<T>::value)
    {
        // Native-dtype path: deserialize into vectors, then send T elements directly
        MPI_Datatype dtype = MPI_has_complex_dtype<T>::getDatatype();

        std::vector<std::vector<T>> sendVecs(size);
        for(rank_t i = 0; i < size; i++)
        {
            senders[i].extract_all(sendVecs[i]);
            senders[i].reset();
        }

        std::vector<int> sendCounts(size), recvCounts(size);
        for(rank_t i = 0; i < size; i++)
            sendCounts[i] = static_cast<int>(sendVecs[i].size());
        MPI_Alltoall(sendCounts.data(), 1, MPI_INT,
                     recvCounts.data(), 1, MPI_INT, comm);

        std::vector<MPI_Request> requests(size);
        for(rank_t i = 0; i < size; i++)
        {
            MPI_Isend(sendVecs[i].data(), sendCounts[i], dtype, i,
                      MPI_EXCHANGE_ALLTOALL_TAG, comm, &requests[i]);
        }

        std::vector<std::vector<T>> result(size);
        for(rank_t i = 0; i < size; i++)
        {
            result[i].resize(static_cast<size_t>(recvCounts[i]));
            MPI_Recv(result[i].data(), recvCounts[i], dtype, i,
                     MPI_EXCHANGE_ALLTOALL_TAG, comm, MPI_STATUS_IGNORE);
        }

        MPI_Waitall(static_cast<int>(size), requests.data(), MPI_STATUSES_IGNORE);

        MPI_Barrier(comm);
        return result;
    }
    else
    {
        std::vector<MPI_Request> requests(size);
        for(rank_t i = 0; i < size; i++)
        {
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

        if(not requests.empty())
        {
            MPI_Waitall(static_cast<int>(size), requests.data(), MPI_STATUSES_IGNORE);
        }
        std::vector<std::vector<T>> result(size);
        for(rank_t i = 0; i < size; i++)
        {
            receivers[i].extract_all(result[i]);
            receivers[i].reset();
        }

        MPI_Barrier(comm);
        return result;
    }
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
    Serializer receiver;
    for(size_t i = 0; i < sendSize; i++)
    {
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
        receiver.reset();
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
    rank_t rank;
    MPI_Comm_rank(comm, &rank);

    Serializer buf;
    size_t sizeSent = 0;
    if(rank == owner)
    {
        sizeSent = buf.insert(data);
    }
    MPI_Bcast(&sizeSent, 1, MPI_UNSIGNED_LONG_LONG, owner, comm);

    if(rank != owner)
    {
        buf.resize(sizeSent);
    }
    MPI_Bcast(buf.getData(), sizeSent, MPI_BYTE, owner, comm);

    T value;
    size_t sizeRead = buf.extract(value, 0);
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

template<typename T>
void MPI_Distribute(std::vector<T> &data, const MPI_Comm &comm)
{
    static_assert(sizeof(size_t) == sizeof(unsigned long long),
                  "MPI_UNSIGNED_LONG_LONG size mismatch with size_t");

    rank_t rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    size_t localCount = data.size();
    std::vector<size_t> counts(size);
    MPI_Allgather(&localCount, 1, MPI_UNSIGNED_LONG_LONG,
                  counts.data(), 1, MPI_UNSIGNED_LONG_LONG, comm);

    size_t total = 0;
    for (rank_t r = 0; r < size; r++)
        total += counts[r];

    if (total == 0) return;

    size_t base = total / size;
    size_t remainder = total % size;

    std::vector<size_t> curPfx(size + 1, 0), tgtPfx(size + 1, 0);
    for (rank_t r = 0; r < size; r++)
    {
        curPfx[r + 1] = curPfx[r] + counts[r];
        tgtPfx[r + 1] = tgtPfx[r] + base + (static_cast<size_t>(r) < remainder ? 1 : 0);
    }

    size_t myStart = curPfx[rank];
    size_t myEnd   = curPfx[rank + 1];

    // Target ranges partition the global array, so local slices are non-overlapping.
    // The keep range (r == rank) is never moved-from by a prior iteration.
    std::vector<std::vector<T>> sendData(size);
    size_t keepBegin = localCount, keepEnd = 0;

    for (rank_t r = 0; r < size; r++)
    {
        size_t oStart = std::max(myStart, tgtPfx[r]);
        size_t oEnd   = std::min(myEnd,   tgtPfx[r + 1]);
        if (oStart >= oEnd) continue;

        size_t lStart = oStart - myStart;
        size_t lEnd   = oEnd   - myStart;
        if (r == rank)
        {
            keepBegin = lStart;
            keepEnd   = lEnd;
        }
        else
        {
            sendData[r].assign(
                std::make_move_iterator(data.begin() + lStart),
                std::make_move_iterator(data.begin() + lEnd));
        }
    }

    std::vector<T> kept;
    if (keepBegin < keepEnd)
        kept.assign(
            std::make_move_iterator(data.begin() + keepBegin),
            std::make_move_iterator(data.begin() + keepEnd));

    data.clear();

    auto received = MPI_Iexchange_all_to_all(sendData, comm);

    size_t targetCount = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    data = std::move(kept);
    data.reserve(targetCount);
    for (rank_t r = 0; r < size; r++)
        data.insert(data.end(),
            std::make_move_iterator(received[r].begin()),
            std::make_move_iterator(received[r].end()));
}

#endif // RICH_MPI

#endif // MPI_SERIALIZABLE_COMMANDS_HPP