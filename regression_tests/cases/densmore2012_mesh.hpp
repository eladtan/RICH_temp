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

        // Start from the paper-zone midpoints.  The former implementation
        // recursively reflected every generator through every requested edge.
        // The abrupt 0.02 cm -> 5e-5 cm transition then forced generators in
        // the entire domain to alternate between opposite cell faces; the
        // first generator ended up only 2.5e-5 cm from x=0.  Besides making
        // boundary sampling fragile, that corrupts DDMC centre-to-face lengths.
        for(std::size_t i = 0; i < cellCount; ++i)
        {
            double const siteX = 0.5 * (edges[i] + edges[i + 1]);
            points.emplace_back(siteX, 0.0, 0.0);
        }

        // An unweighted Voronoi mesh cannot represent an arbitrary structured
        // edge list while keeping every generator at its zone midpoint.  Use
        // the exact-face recurrence only locally, where it matters physically:
        // keep x=2 and every edge of the ten-cell refined layer through x=2.005
        // exact, then return to midpoint generators in the regular thick mesh.
        points[thinCellCount - 1].x =
            2.0 * interfacePosition - points[thinCellCount].x;

        std::size_t const firstRefined = thinCellCount;
        std::size_t const lastRefined =
            thinCellCount + refinedCellCount - 1;
        for(std::size_t i = firstRefined + 1; i <= lastRefined; ++i)
            points[i].x = 2.0 * edges[i] - points[i - 1].x;

        // Enforce the end face of the subdivided 0.005-cm cell.  Stopping the
        // recurrence here confines the unavoidable Voronoi transition error to
        // one face in the regular thick mesh instead of alternating forever.
        std::size_t const firstRegularThick = lastRefined + 1;
        points[firstRegularThick].x =
            2.0 * edges[firstRegularThick] - points[lastRefined].x;

        for(std::size_t i = 0; i < points.size(); ++i)
        {
            if(!(points[i].x > 0.0 && points[i].x < domainLength) ||
               (i > 0 && !(points[i].x > points[i - 1].x)))
            {
                throw std::runtime_error(
                    "Densmore Voronoi generators are not strictly ordered");
            }
        }

        auto facePosition = [&points](std::size_t rightCell) {
            return 0.5 * (points[rightCell - 1].x + points[rightCell].x);
        };
        if(std::abs(facePosition(thinCellCount) - interfacePosition) > 1e-13 ||
           std::abs(facePosition(firstRegularThick) -
                    (interfacePosition + thickCellWidth)) > 1e-13)
        {
            throw std::runtime_error(
                "Densmore refined-layer boundary is misplaced");
        }

        return points;
    }
}

#endif // RICH_REGRESSION_TESTS_DENSMORE2012_MESH_HPP
