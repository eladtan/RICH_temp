#ifndef MPI_COMMANDS_2D_HPP
#define MPI_COMMANDS_2D_HPP

#ifdef RICH_MPI

#include <mpi.h>
#include <vector>
#include "newtonian/two_dimensional/extensive.hpp"
#include "tessellation/tessellation.hpp"
#include <mpi_utils/serialize/Serializer.hpp>

/*!
\brief Sends and revs data
\param tess The tessellation
\param cells The data to send/recv
\param ghost_or_sent True for ghost cells false for sent cells.
*/
template<class T>
void MPI_exchange_data(const Tessellation& tess, vector<T>& cells, bool ghost_or_sent, const T *example_cell = nullptr)
{
	T default_cell_storage{};
	if(example_cell == nullptr)
	{
		if(!cells.empty())
			example_cell = &cells[0];
		else
			example_cell = &default_cell_storage;
	}

	const std::vector<rank_t> &correspondents = (ghost_or_sent)? tess.GetDuplicatedProcs() : tess.GetSentProcs();
	const std::vector<std::vector<int>> &indices = (ghost_or_sent)? tess.GetDuplicatedPoints() : tess.GetSentPoints();
	std::vector<MPI_Request> req(correspondents.size());
	
	std::vector<std::vector<T>> exchange = MPI_exchange_data_indexed(correspondents, cells, indices);
	const std::vector<std::vector<int>> &ghost_indices = tess.GetGhostIndeces();
	if(ghost_or_sent)
	{
		cells.resize(tess.GetTotalPointNumber(), *example_cell);
	}
	else
	{
		cells = VectorValues(cells, tess.GetSelfPoint());
	}

	for(size_t i = 0; i < correspondents.size(); ++i)
	{
		const std::vector<T> &data = exchange[i];
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
}

#endif // RICH_MPI
#endif // MPI_COMMANDS_2D_HPP