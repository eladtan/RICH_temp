#ifndef MPI_COMMANDS_2D_HPP
#define MPI_COMMANDS_2D_HPP

#ifdef RICH_MPI

#include <mpi.h>
#include <vector>
#include "newtonian/two_dimensional/computational_cell_2d.hpp"
#include "newtonian/two_dimensional/extensive.hpp"
#include "tessellation/tessellation.hpp"
#include "misc/serialize/Serializer.hpp"

#define MPI_EXCHANGE_2D_TAG 6

/*!
\brief Sends and revs data
\param tess The tessellation
\param cells The data to send/recv
\param ghost_or_sent True for ghost cells false for sent cells.
*/
template<class T>
void MPI_exchange_data(const Tessellation &tess, vector<T>& cells, bool ghost_or_sent, const T *example_cell = nullptr)
{
	if(example_cell == nullptr and cells.empty())
	{
		throw UniversalError("Empty cell vector in MPI_exchange_data");
	}
	if(example_cell == nullptr)
	{
		example_cell = &cells[0];
	}

	const std::vector<rank_t> &correspondents = (ghost_or_sent)? tess.GetDuplicatedProcs() : tess.GetSentProcs();
	const std::vector<std::vector<int>> &duplicated_points = (ghost_or_sent)? tess.GetDuplicatedPoints() : tess.GetSentPoints();
	std::vector<MPI_Request> req(correspondents.size());
	std::vector<vector<double>> tempsend(correspondents.size());
	std::vector<Serializer> senders(correspondents.size());
	double temp = 0;
	for(size_t i = 0; i < correspondents.size(); ++i)
	{
		bool isempty = duplicated_points[i].empty();
		if(!isempty)
		{
			senders[i].insert_all_indexed(cells, duplicated_points[i]);
		}
		MPI_Isend((senders[i].size() > 0)? senders[i].getData() : NULL, senders[i].size(), MPI_CHAR, correspondents[i], MPI_EXCHANGE_2D_TAG, MPI_COMM_WORLD, &req[i]);
	}
	const std::vector<std::vector<int>> &ghost_indices = tess.GetGhostIndeces();
	if(ghost_or_sent)
	{
		cells.resize(tess.GetTotalPointNumber(), *example_cell);
	}
	else
	{
		cells = VectorValues(cells, tess.GetSelfPoint());
	}

	std::vector<Serializer> receivers;
	for(size_t i = 0; i < correspondents.size(); ++i)
	{
		MPI_Status status;
		MPI_Probe(MPI_ANY_SOURCE, MPI_EXCHANGE_2D_TAG, MPI_COMM_WORLD, &status);
		int count;
		MPI_Get_count(&status, MPI_CHAR, &count);
		size_t location = std::distance(correspondents.begin(), std::find(correspondents.begin(), correspondents.end(), status.MPI_SOURCE));
		if(location >= correspondents.size())
		{
			UniversalError eo("Bad location in mpi exchange");
			eo.addEntry("Location (Index)", location);
			eo.addEntry("Correspondents.size()", correspondents.size());
			throw eo;
		}
		receivers[location].resize(count);
		MPI_Recv(receivers[location].getData(), count, MPI_CHAR, status.MPI_SOURCE, status.MPI_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	for(size_t i = 0; i < correspondents.size(); ++i)
	{
		std::vector<T> data;
		receivers[i].extract_all(data);
		size_t count = data.size();
		if (ghost_or_sent)
		{
			for(size_t j = 0; j < count; ++j)
			{
				cells.at(ghost_indices.at(i).at(j)) = data[j];
			}
		}
		else
		{
			for(size_t j = 0; j < count; ++j)
			{
				cells.push_back(data[j]);
			}
		}
	}
	if(not req.empty())
	{
		MPI_Waitall(static_cast<int>(correspondents.size()), &req[0], MPI_STATUSES_IGNORE);
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

#endif // RICH_MPI
#endif // MPI_COMMANDS_2D_HPP