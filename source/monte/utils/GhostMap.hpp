#ifndef GHOST_MAP_HPP
#define GHOST_MAP_HPP

#ifdef RICH_MPI

#include "mpi/mpi_commands.hpp"
#include <boost/container/flat_map.hpp>

template<typename Grid>
boost::container::flat_map<size_t, std::pair<rank_t, size_t>> GetGhostMap(const Grid &grid)
{
    boost::container::flat_map<size_t, std::pair<rank_t, size_t>> ranks_ghost_map;
    std::vector<std::vector<size_t>> incoming = MPI_exchange_data(grid.GetDuplicatedProcs(), grid.GetDuplicatedPoints());
    const std::vector<std::vector<size_t>> &ghosts = grid.GetGhostIndeces();
    for(size_t i = 0; i < incoming.size(); i++)
    {
        int _rank = grid.GetDuplicatedProcs()[i];
        for(size_t j = 0; j < incoming[i].size(); j++)
        {
            assert(incoming[i].size() == ghosts[i].size());
            if(ranks_ghost_map.find(ghosts[i][j]) != ranks_ghost_map.end())
            {
                UniversalError eo("Duplicate in ranks ghost map");
                eo.addEntry("Index", ghosts[i][j]);
                eo.addEntry("Rank", _rank);
                eo.addEntry("Incoming", incoming[i][j]);
                eo.addEntry("Current Value", ranks_ghost_map[ghosts[i][j]]);
                throw eo;
            }
            ranks_ghost_map.insert({ghosts[i][j], {_rank, incoming[i][j]}});
        }
    }
    return ranks_ghost_map;
}

#endif // RICH_MPI

#endif // GHOST_MAP_HPP