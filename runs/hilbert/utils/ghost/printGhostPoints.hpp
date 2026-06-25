#ifndef PRINT_GHOST_POINTS_HPP
#define PRINT_GHOST_POINTS_HPP

#ifdef RICH_MPI
    
#include "3D/tessellation/Voronoi3D.hpp"
#include <algorithm>
#include <vector>
#include <mpi.h>

std::vector<size_t> GetGhostPoints(const Voronoi3D &voronoi, int ofRank);

inline std::vector<double> GetWhetherGhostPoint(const Voronoi3D &voronoi, int ofRank)
{
    std::vector<double> result(voronoi.GetPointNo(), 0);
    for(const size_t &index : GetGhostPoints(voronoi, ofRank))
    {
        result[index] = 1;
    }
    return result;
}

#endif // RICH_MPI

#endif // PRINT_GHOST_POINTS_HPP