#ifndef MPI_COMMANDS_3D_HPP
#define MPI_COMMANDS_3D_HPP

#ifdef RICH_MPI

#include <mpi.h>
#include <vector>
#include "3D/tessellation/Tessellation3D.hpp"
#include <mpi_utils/serialize/Serializer.hpp>
#include "misc/utils.hpp"

namespace rich_mpi_3d_detail {
template<class T>
inline void compact_to_indices_exact_order(std::vector<T>& data,
                                           const std::vector<size_t>& keep)
{
    std::vector<T> tmp;
    tmp.reserve(keep.size());
    for(size_t idx : keep)
        tmp.push_back(std::move(data[idx]));
    data.swap(tmp);
}
} // namespace rich_mpi_3d_detail
#include "misc/memory_profile.hpp"

/*!
\brief Sends and revs data
\param tess The tessellation
\param cells The data to send/recv
\param ghost_or_sent True for ghost cells false for sent cells.
*/
template<class T>
inline void MPI_exchange_data(const Tessellation3D& tess, std::vector<T>& cells, bool ghost_or_sent, const size_t extent = 1, const T *example_cell = nullptr)
{
	MEMORY_PROFILE_SCOPE("MPI exchange");
	T default_cell_storage{};
	if(example_cell == nullptr)
	{
		if(!cells.empty())
			example_cell = &cells[0];
		else
			example_cell = &default_cell_storage;
	}

	const std::vector<rank_t> &correspondents = (ghost_or_sent)? tess.GetDuplicatedProcs() : tess.GetSentProcs();
	const std::vector<std::vector<size_t>> &indices = (ghost_or_sent)? tess.GetDuplicatedPoints() : tess.GetSentPoints();
	std::vector<MPI_Request> req(correspondents.size());
	
	std::vector<std::vector<T>> exchange = MPI_exchange_data_indexed(correspondents, cells, indices, extent);
	const std::vector<std::vector<size_t>> &ghost_indices = tess.GetGhostIndeces();
	if(ghost_or_sent)
	{
		cells.resize(tess.GetTotalPointNumber() * extent, *example_cell);
	}
	else
	{
		rich_mpi_3d_detail::compact_to_indices_exact_order(cells, tess.GetSelfIndex());
	}
	for(size_t i = 0; i < correspondents.size(); ++i)
	{
		const std::vector<T> &data = exchange[i];
		if(data.size() % extent != 0) 
		{
			throw UniversalError("Extent size does not match");
		}
		size_t count = data.size();
		if (ghost_or_sent)
		{
			count = data.size() / extent;
			for(size_t j = 0; j < count; ++j)
			{
				for(size_t k = 0; k < extent; ++k)
					cells.at(ghost_indices.at(i).at(j) * extent + k) = data[j * extent + k];
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
	if(!ghost_or_sent)
		conditional_shrink(cells);
}

template<class T>
inline std::vector<std::vector<T>> MPI_exchange_data(const Tessellation3D& tess, std::vector<std::vector<T>>& data)
{
    const std::vector<rank_t> &correspondents = tess.GetDuplicatedProcs();
    return MPI_exchange_data(correspondents, data);
}

#endif // RICH_MPI

#endif // MPI_COMMANDS_3D_HPP
