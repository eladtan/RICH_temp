#ifdef RICH_MPI

#include "ExchangeGhosts.hpp"

boost::container::flat_map<size_t, std::pair<rank_t, size_t>> ExchangeGhosts(const Tessellation3D &tess)
{
    boost::container::flat_map<size_t, std::pair<rank_t, size_t>> ghosts_map;
    std::vector<std::vector<size_t>> incoming = MPI_exchange_data(tess.GetDuplicatedProcs(), tess.GetDuplicatedPoints());
    const std::vector<std::vector<size_t>> &ghosts = tess.GetGhostIndeces();
    for(size_t i = 0; i < incoming.size(); i++)
    {
        int _rank = tess.GetDuplicatedProcs()[i];
        for(size_t j = 0; j < incoming[i].size(); j++)
        {
            assert(incoming[i].size() == ghosts[i].size());
            ghosts_map.insert({ghosts[i][j], {_rank, incoming[i][j]}});
        }
    }
    return ghosts_map;
}

#endif // RICH_MPI