#ifdef RICH_MPI

#include "printGhostPoints.hpp"

std::vector<size_t> GetGhostPoints(const Voronoi3D &voronoi, int ofRank)
{
    const std::vector<int> &procs = voronoi.GetDuplicatedProcs();
    size_t rankIndex = std::distance(procs.cbegin(), std::find(procs.cbegin(), procs.cend(), ofRank));
    if(rankIndex == procs.size())
    {
        return std::vector<size_t>();
    }
    return voronoi.GetDuplicatedPoints()[rankIndex];
}

#endif // RICH_MPI