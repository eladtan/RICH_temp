#ifndef RICH_REGRESSION_TESTS_DENSMORE2012_MESH_HPP
#define RICH_REGRESSION_TESTS_DENSMORE2012_MESH_HPP

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "source/3D/elementary/Vector3D.hpp"

namespace densmore2012_mesh
{
    // Densmore, Thompson, and Urbatsch (2012), first heterogeneous problem:
    //   100 cells of width 0.02 cm in 0 < x < 2 cm;
    //   200 cells of width 0.005 cm in 2 < x < 3 cm, except that the
    //   leftmost thick cell is replaced by ten geometrically growing cells.
    constexpr double domainLength = 3.0;
    constexpr double interfacePosition = 2.0;
    constexpr double thinCellWidth = 0.02;
    constexpr double thickCellWidth = 0.005;
    constexpr double refinementRatio = 1.47394;

    constexpr std::size_t thinCellCount = 100;
    constexpr std::size_t refinedCellCount = 10;
    constexpr std::size_t regularThickCellCount = 199;
    constexpr std::size_t cellCount =
        thinCellCount + refinedCellCount + regularThickCellCount;

    inline std::vector<double> BuildCellEdges()
    {
        std::vector<double> edges;
        edges.reserve(cellCount + 1);
        edges.push_back(0.0);

        for(std::size_t i = 1; i <= thinCellCount; ++i)
            edges.push_back(static_cast<double>(i) * thinCellWidth);

        double const firstRefinedWidth = thickCellWidth *
            (refinementRatio - 1.0) /
            (std::pow(refinementRatio,
                      static_cast<double>(refinedCellCount)) - 1.0);

        double x = interfacePosition;
        for(std::size_t i = 0; i < refinedCellCount; ++i)
        {
            x += firstRefinedWidth *
                std::pow(refinementRatio, static_cast<double>(i));
            edges.push_back(x);
        }

        // Remove accumulated roundoff at the end of the subdivided 0.005-cm
        // cell before appending the remaining regular thick cells.
        edges.back() = interfacePosition + thickCellWidth;
        for(std::size_t i = 1; i <= regularThickCellCount; ++i)
        {
            edges.push_back(interfacePosition + thickCellWidth +
                            static_cast<double>(i) * thickCellWidth);
        }
        edges.back() = domainLength;

        if(edges.size() != cellCount + 1)
            throw std::runtime_error("Densmore mesh has the wrong edge count");
        return edges;
    }

    inline std::vector<Vector3D> BuildVoronoiSites()
    {
        std::vector<double> const edges = BuildCellEdges();
        std::vector<Vector3D> points;
        points.reserve(cellCount);

        // A Voronoi face is the perpendicular bisector of two generators.
        // Simply placing generators at the nominal cell midpoints would move
        // the x=2 opacity interface by roughly 0.005 cm because the first
        // refined cell is much smaller than the last thin cell.  Instead,
        // reflect each new generator through the prescribed face.  This puts
        // every x-face exactly on the paper's cell edge, including x=2.
        double const firstRefinedWidth = edges[thinCellCount + 1] -
                                         edges[thinCellCount];
        double siteX = 0.5 * firstRefinedWidth;
        for(std::size_t i = 0; i < cellCount; ++i)
        {
            if(i > 0)
                siteX = 2.0 * edges[i] - points.back().x;

            if(!(siteX > edges[i] && siteX < edges[i + 1]))
            {
                throw std::runtime_error(
                    "Densmore Voronoi generator lies outside its cell");
            }
            points.emplace_back(siteX, 0.0, 0.0);
        }

        return points;
    }
}

#endif // RICH_REGRESSION_TESTS_DENSMORE2012_MESH_HPP
