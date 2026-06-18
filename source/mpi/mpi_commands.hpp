#ifndef MPI_COMMANDS_HPP
#define MPI_COMMANDS_HPP 1

using rank_t = int;

#ifdef RICH_MPI

#include <vector>
#include <chrono>
#include <mpi.h>
#include <functional>
#include "misc/utils.hpp"
#include "3D/tessellation/Tessellation3D.hpp"
#include "mpi/serialize/Serializer.hpp"
#define MPI_TIMED_BARRIER_TAG 110503
#define MPI_EXCHANGE_TAG 5

using std::vector;

namespace rich_mpi_detail {

constexpr size_t SOFT_RELEASE_BYTES = 32ull  * 1024ull * 1024ull;
constexpr size_t HARD_RELEASE_BYTES = 128ull * 1024ull * 1024ull;
constexpr size_t WASTE_FACTOR = 4;

inline bool should_release_serializer(const Serializer& s)
{
	const size_t cap = s.capacity();
	const size_t live = s.size();
	if(cap >= HARD_RELEASE_BYTES)
		return true;
	if(cap >= SOFT_RELEASE_BYTES && cap > WASTE_FACTOR * std::max<size_t>(live, 1))
		return true;
	return false;
}

inline void release_changed_neighbor_slots(
	std::vector<Serializer>& senders,
	std::vector<Serializer>& receivers,
	const std::vector<rank_t>& prev,
	const std::vector<rank_t>& curr)
{
	for(size_t i = curr.size(); i < prev.size(); ++i)
	{
		if(i < senders.size())   senders[i].release();
		if(i < receivers.size()) receivers[i].release();
	}
	const size_t common = std::min(prev.size(), curr.size());
	for(size_t i = 0; i < common; ++i)
	{
		if(prev[i] != curr[i])
		{
			if(i < senders.size())   senders[i].release();
			if(i < receivers.size()) receivers[i].release();
		}
	}
}

inline void release_oversized_serializers(std::vector<Serializer>& bufs)
{
	for(auto& s : bufs)
		if(should_release_serializer(s))
			s.release();
}

} // namespace rich_mpi_detail

template<typename T, typename Index_T = size_t>
std::vector<std::vector<T>> MPI_exchange_data_indexed(const std::vector<rank_t> &correspondents, const std::vector<T> &data, const std::vector<std::vector<Index_T>> &indices = std::vector<std::vector<Index_T>>(), const size_t &extent = 1)
{
	static std::vector<Serializer> senders;
	static std::vector<Serializer> receivers;
	static std::vector<rank_t> previous_correspondents;

	rich_mpi_detail::release_changed_neighbor_slots(senders, receivers, previous_correspondents, correspondents);

	std::vector<MPI_Request> req(correspondents.size());
	senders.resize(correspondents.size());
	for(size_t i = 0; i < correspondents.size(); ++i)
	{
		senders[i].reset();
		senders[i].insert_all_indexed(data, indices[i], extent);
		MPI_Isend((senders[i].size() > 0)? senders[i].getData() : NULL, senders[i].size(), MPI_CHAR, correspondents[i], MPI_EXCHANGE_TAG, MPI_COMM_WORLD, &req[i]);
	}

	receivers.resize(correspondents.size());
	for(size_t i = 0; i < correspondents.size(); ++i)
		receivers[i].reset();
	for(size_t i = 0; i < correspondents.size(); ++i)
	{
		MPI_Status status;
		MPI_Probe(MPI_ANY_SOURCE, MPI_EXCHANGE_TAG, MPI_COMM_WORLD, &status);
		int count;
		MPI_Get_count(&status, MPI_CHAR, &count);
		size_t location = std::distance(correspondents.begin(), std::find(correspondents.begin(), correspondents.end(), status.MPI_SOURCE));
		if(location >= correspondents.size())
		{
			UniversalError eo("Bad location in mpi exchange");
			eo.addEntry("Type", typeid(T).name());
			eo.addEntry("Location (Index)", location);
			eo.addEntry("Correspondents.size()", correspondents.size());
			throw eo;
		}
		receivers[location].resize(count);
		MPI_Recv(receivers[location].getData(), count, MPI_CHAR, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	std::vector<std::vector<T>> result(correspondents.size());
	for(size_t i = 0; i < correspondents.size(); ++i)
	{
		receivers[i].extract_all(result[i]);
	}
	if(not req.empty())
	{
		MPI_Waitall(static_cast<int>(correspondents.size()), &req[0], MPI_STATUSES_IGNORE);
	}
	rich_mpi_detail::release_oversized_serializers(senders);
	rich_mpi_detail::release_oversized_serializers(receivers);
	previous_correspondents.assign(correspondents.begin(), correspondents.end());
	MPI_Barrier(MPI_COMM_WORLD);
	return result;
}

template<typename T>
std::vector<std::vector<T>> MPI_exchange_data(const std::vector<rank_t>& correspondents, const std::vector<std::vector<T>>& data)
{
	static std::vector<Serializer> senders;
	static std::vector<Serializer> receivers;
	static std::vector<rank_t> previous_correspondents;

	rich_mpi_detail::release_changed_neighbor_slots(senders, receivers, previous_correspondents, correspondents);

	std::vector<MPI_Request> req(correspondents.size());
	senders.resize(correspondents.size());
	for(size_t i = 0; i < correspondents.size(); ++i)
	{
		senders[i].reset();
		senders[i].insert_all(data[i]);
		MPI_Isend((senders[i].size() > 0)? senders[i].getData() : NULL, senders[i].size(), MPI_CHAR, correspondents[i], MPI_EXCHANGE_TAG, MPI_COMM_WORLD, &req[i]);
	}

	receivers.resize(correspondents.size());
	for(size_t i = 0; i < correspondents.size(); ++i)
		receivers[i].reset();
	for(size_t i = 0; i < correspondents.size(); ++i)
	{
		MPI_Status status;
		MPI_Probe(MPI_ANY_SOURCE, MPI_EXCHANGE_TAG, MPI_COMM_WORLD, &status);
		int count;
		MPI_Get_count(&status, MPI_CHAR, &count);
		size_t location = std::distance(correspondents.begin(), std::find(correspondents.begin(), correspondents.end(), status.MPI_SOURCE));
		if(location >= correspondents.size())
		{
			UniversalError eo("Bad location in mpi exchange");
			eo.addEntry("Type", typeid(T).name());
			eo.addEntry("Location (Index)", location);
			eo.addEntry("Correspondents.size()", correspondents.size());
			throw eo;
		}
		receivers[location].resize(count);
		MPI_Recv(receivers[location].getData(), count, MPI_CHAR, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	std::vector<std::vector<T>> result(correspondents.size());
	for(size_t i = 0; i < correspondents.size(); ++i)
	{
		receivers[i].extract_all(result[i]);
	}
	if(not req.empty())
	{
		MPI_Waitall(static_cast<int>(correspondents.size()), &req[0], MPI_STATUSES_IGNORE);
	}
	rich_mpi_detail::release_oversized_serializers(senders);
	rich_mpi_detail::release_oversized_serializers(receivers);
	previous_correspondents.assign(correspondents.begin(), correspondents.end());
	MPI_Barrier(MPI_COMM_WORLD);
	return result;
}

template<typename T>
std::pair<rank_t, T> MPI_Minmax_loc(const T &data, const MPI_Comm &comm, bool max)
{
	rank_t rank;
	MPI_Comm_rank(comm, &rank);

	MPI_Datatype dtype;
	if constexpr(std::is_same_v<T, double>)
	{
		dtype = MPI_DOUBLE_INT;
	}
	else if constexpr(std::is_same_v<T, int>)
	{
		dtype = MPI_2INT;
	}
	else
	{
		UniversalError eo("Unsupported type for MPI_Minmax_loc");
		eo.addEntry("Type", typeid(T).name());
		throw eo;
	}
	struct
	{
		T value;
		rank_t rank;
	} myVal = {data, rank}, outVal;
	MPI_Allreduce(&myVal, &outVal, 1, dtype, max ? MPI_MAXLOC : MPI_MINLOC, comm);
	return std::make_pair(outVal.rank, outVal.value);
}

template<typename T>
std::pair<rank_t, T> MPI_Max_loc(const T &data, const MPI_Comm &comm = MPI_COMM_WORLD)
{
	return MPI_Minmax_loc(data, comm, true);
}

template<typename T>
std::pair<rank_t, T> MPI_Min_loc(const T &data, const MPI_Comm &comm = MPI_COMM_WORLD)
{
	return MPI_Minmax_loc(data, comm, false);
}


template<typename T>
std::pair<rank_t, T> MPI_Max_loc(const std::vector<T> &data, const MPI_Comm &comm = MPI_COMM_WORLD)
{
	return MPI_Minmax_loc(data.empty()? std::numeric_limits<T>::lowest() : *std::max_element(data.begin(), data.end()), comm, true);
}

template<typename T>
std::pair<rank_t, T> MPI_Min_loc(const std::vector<T> &data, const MPI_Comm &comm = MPI_COMM_WORLD)
{
	return MPI_Minmax_loc(data.empty()? std::numeric_limits<T>::max() : *std::min_element(data.begin(), data.end()), comm, false);
}

#include "mpi_commands_2d.hpp"
#include "mpi_commands_3d.hpp"

void MPI_Timed_barrier(const MPI_Comm &comm, double seconds, std::string const &place);

#endif //RICH_MPI

#endif // MPI_COMMANDS_HPP

