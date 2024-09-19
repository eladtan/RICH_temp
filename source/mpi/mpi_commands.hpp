#ifndef MPI_COMMANDS_HPP
#define MPI_COMMANDS_HPP 1

#ifdef RICH_MPI

#include <vector>
#include <chrono>
#include <mpi.h>
#include <functional>
#include "misc/utils.hpp"
#include "3D/tesselation/Tessellation3D.hpp"
#include "misc/serialize/Serializer.hpp"
#include "stdint.h"

#define MPI_TIMED_BARRIER_TAG 110503
#define MPI_EXCHANGE_TAG 5

using rank_t = int;

using std::vector;

template<typename T, typename Index_T = size_t>
std::vector<std::vector<T>> MPI_exchange_data_indexed(const std::vector<rank_t>& correspondents, const std::vector<T>& data, const std::vector<std::vector<Index_T>> &indices = std::vector<std::vector<Index_T>>())
{
	std::vector<MPI_Request> req(correspondents.size());
	std::vector<Serializer> senders(correspondents.size());
	for(size_t i = 0; i < correspondents.size(); ++i)
	{
		senders[i].insert_all_indexed(data, indices[i]);
		MPI_Isend((senders[i].size() > 0)? senders[i].getData() : NULL, senders[i].size(), MPI_BYTE, correspondents[i], MPI_EXCHANGE_TAG, MPI_COMM_WORLD, &req[i]);
	}

	std::vector<Serializer> receivers(correspondents.size());
	for(size_t i = 0; i < correspondents.size(); ++i)
	{
		MPI_Status status;
		MPI_Probe(MPI_ANY_SOURCE, MPI_EXCHANGE_TAG, MPI_COMM_WORLD, &status);
		int count;
		MPI_Get_count(&status, MPI_BYTE, &count);
		size_t location = std::distance(correspondents.begin(), std::find(correspondents.begin(), correspondents.end(), status.MPI_SOURCE));
		if(location >= correspondents.size())
		{
			UniversalError eo("Bad location in mpi exchange");
			eo.addEntry("Location (Index)", location);
			eo.addEntry("Correspondents.size()", correspondents.size());
			throw eo;
		}
		receivers[location].resize(count);
		MPI_Recv(receivers[location].getData(), count, MPI_BYTE, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
	MPI_Barrier(MPI_COMM_WORLD);
	return result;
}

template<typename T>
std::vector<std::vector<T>> MPI_exchange_data(const std::vector<rank_t>& correspondents, const std::vector<std::vector<T>>& data)
{
	std::vector<MPI_Request> req(correspondents.size());
	std::vector<Serializer> senders(correspondents.size());
	for(size_t i = 0; i < correspondents.size(); ++i)
	{
		senders[i].insert_all(data[i]);
		MPI_Isend((senders[i].size() > 0)? senders[i].getData() : NULL, senders[i].size(), MPI_BYTE, correspondents[i], MPI_EXCHANGE_TAG, MPI_COMM_WORLD, &req[i]);
	}

	std::vector<Serializer> receivers(correspondents.size());
	for(size_t i = 0; i < correspondents.size(); ++i)
	{
		MPI_Status status;
		MPI_Probe(MPI_ANY_SOURCE, MPI_EXCHANGE_TAG, MPI_COMM_WORLD, &status);
		int count;
		MPI_Get_count(&status, MPI_BYTE, &count);
		size_t location = std::distance(correspondents.begin(), std::find(correspondents.begin(), correspondents.end(), status.MPI_SOURCE));
		if(location >= correspondents.size())
		{
			UniversalError eo("Bad location in mpi exchange");
			eo.addEntry("Location (Index)", location);
			eo.addEntry("Correspondents.size()", correspondents.size());
			throw eo;
		}
		receivers[location].resize(count);
		MPI_Recv(receivers[location].getData(), count, MPI_BYTE, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
	MPI_Barrier(MPI_COMM_WORLD);
	return result;
}

#include "mpi_commands_2d.hpp"
#include "mpi_commands_3d.hpp"

void MPI_Timed_barrier(const MPI_Comm &comm, double seconds, std::string const &place);

#endif //RICH_MPI

#endif // MPI_COMMANDS_HPP

