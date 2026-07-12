#ifndef RICH_REGRESSION_TESTS_DENSMORE2012_INTERFACE_MESH_HPP
#define RICH_REGRESSION_TESTS_DENSMORE2012_INTERFACE_MESH_HPP

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "source/3D/elementary/Vector3D.hpp"

namespace densmore2012_interface_mesh
{
    // Diagnostic mesh for an exposed IMC-DDMC material interface.
    //
    // The physical target is 100 zones of width 0.02 cm on the thin side and
    // 200 zones of width 0.005 cm on the thick side, with the opacity jump at
    // x=2 cm and no geometrically refined buffer.  An unweighted Voronoi mesh
    // cannot keep every generator at the corresponding structured-zone center
    // while also placing the 0.02-to-0.005 transition face exactly at x=2.
    // Keep the complete thick region exact and move only the final thin-side
    // generator.  Consequently, only the last two thin-side Voronoi widths are
    // slightly distorted; the material face and every thick-side width are
    // exact.  Both the pure-IMC and DDMC tests use this identical tessellation.
    constexpr double domainLength = 3.0;
    constexpr double interfacePosition = 2.0;
    constexpr double thinCellWidth = 0.02;
    constexpr double thickCellWidth = 0.005;

    constexpr std::size_t thinCellCount = 100;
    constexpr std::size_t thickCellCount = 200;
    constexpr std::size_t cellCount = thinCellCount + thickCellCount;

    inline std::vector<Vector3D> BuildVoronoiSites()
    {
        std::vector<Vector3D> points;
        points.reserve(cellCount);

        // Keep the first 99 thin-side generators at their regular centers.
        for(std::size_t i = 0; i + 1 < thinCellCount; ++i)
        {
            double const x = (static_cast<double>(i) + 0.5) * thinCellWidth;
            points.emplace_back(x, 0.0, 0.0);
        }

        // The first thick generator is at 2.0025 cm.  Reflect it through x=2
        // to place the material interface exactly at x=2.
        double const firstThickCenter =
            interfacePosition + 0.5 * thickCellWidth;
        points.emplace_back(2.0 * interfacePosition - firstThickCenter,
                            0.0, 0.0);

        // The full thick region is a regular 0.005-cm Voronoi mesh.
        for(std::size_t i = 0; i < thickCellCount; ++i)
        {
            double const x = interfacePosition +
                (static_cast<double>(i) + 0.5) * thickCellWidth;
            points.emplace_back(x, 0.0, 0.0);
        }

        if(points.size() != cellCount)
            throw std::runtime_error(
                "Densmore interface mesh has the wrong point count");

        for(std::size_t i = 0; i < points.size(); ++i)
        {
            if(!(points[i].x > 0.0 && points[i].x < domainLength) ||
               (i > 0 && !(points[i].x > points[i - 1].x)))
            {
                throw std::runtime_error(
                    "Densmore interface mesh generators are not ordered");
            }
        }

        auto facePosition = [&points](std::size_t rightCell) {
            return 0.5 *
                (points[rightCell - 1].x + points[rightCell].x);
        };

        if(std::abs(facePosition(thinCellCount) - interfacePosition) > 1e-14)
            throw std::runtime_error(
                "Densmore interface mesh does not place the jump at x=2");

        // The first two thick-side faces must be exactly 2.0 and 2.005 cm.
        if(std::abs(facePosition(thinCellCount + 1) -
                    (interfacePosition + thickCellWidth)) > 1e-14)
        {
            throw std::runtime_error(
                "Densmore interface mesh has the wrong first thick-cell width");
        }

        return points;
    }
}

#endif // RICH_REGRESSION_TESTS_DENSMORE2012_INTERFACE_MESH_HPP
