
#ifndef MESH_HELPER_HPP
#define MESH_HELPER_HPP

#include "problem_config_3d.hpp"
#include "3D/elementary/Vector3D.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/common/equation_of_state.hpp"
#include "3D/tesselation/voronoi/Voronoi3D.hpp"
#include "3D/GeometryCommon/RoundGrid3D.hpp"
#include "misc/mesh_generator3D.hpp"
#include <vector>
#include <iostream>
#include <stdexcept>

namespace rich3d {

using ::ComputationalCell3D;
using ::EquationOfState;
using ::Vector3D;
using ::Voronoi3D;
using std::pair;
using std::vector;

// Type alias for mesh setup result
using MeshSetupResult = pair<vector<Vector3D>, vector<ComputationalCell3D>>;

inline bool is_point_in_domain(const Vector3D& p, const Vector3D& lower, const Vector3D& upper, double margin = 0.0) {
    return p.x >= lower.x - margin && p.x <= upper.x + margin && p.y >= lower.y - margin && p.y <= upper.y + margin &&
           p.z >= lower.z - margin && p.z <= upper.z + margin;
}

inline ComputationalCell3D create_default_new_cell_state(const EquationOfState& eos, double time) {
    ComputationalCell3D cell;
    cell.density = 1e-20;
    cell.pressure = eos.dp2e(cell.density, 1e-10, cell.tracers);
    cell.velocity = Vector3D(0., 0., 0.);
    cell.internal_energy = eos.dp2e(cell.density, cell.pressure, cell.tracers);
    cell.temperature = 1e3;
    return cell;
}

inline ComputationalCell3D get_expansion_reference_cell(const Problem3DConfig& config, double time) {
    if (config.domain.dynamic.new_cell_state) {
        return config.domain.dynamic.new_cell_state(*config.physics.eos, time);
    }
    return create_default_new_cell_state(*config.physics.eos, time);
}

inline vector<Vector3D> create_mesh_points(const DomainConfig& domain, const MeshConfig& mesh) {
    vector<Vector3D> points;

    switch (mesh.type) {
        case MeshConfig::Type::CARTESIAN: {
            std::cout << "  Generating " << (mesh.nx * mesh.ny * mesh.nz) << " Cartesian mesh points...\n";
            points = ::CartesianMesh(mesh.nx, mesh.ny, mesh.nz, domain.lower_bound, domain.upper_bound);
            std::cout << "    Done!\n";
            break;
        }

        case MeshConfig::Type::RANDOM: {
            std::cout << "  Generating " << mesh.num_points << " random points...\n";
            points = ::RandRectangular(mesh.num_points, domain.lower_bound, domain.upper_bound);

            // Apply RoundGrid3D
            if (mesh.round_iterations > 0) {
                std::cout << "  Applying RoundGrid3D with " << mesh.round_iterations << " iterations...\n";
                points = ::RoundGrid3D(points, domain.lower_bound, domain.upper_bound, mesh.round_iterations);
            }
            std::cout << "    Done!\n";
            break;
        }

        case MeshConfig::Type::CUSTOM: {
            // User provides points
            if (mesh.custom_points.empty()) {
                throw std::runtime_error("CUSTOM mesh type requires custom_points vector");
            }
            points = mesh.custom_points;
            std::cout << "  Using " << points.size() << " custom mesh points\n";
            break;
        }

        case MeshConfig::Type::SPHERICAL:
        case MeshConfig::Type::CYLINDRICAL:
            throw std::runtime_error("SPHERICAL and CYLINDRICAL mesh types not yet implemented");

        default:
            throw std::runtime_error("Unknown mesh type");
    }

    return points;
}

inline MeshSetupResult setup_initial_mesh(const Problem3DConfig& config, Voronoi3D& tess) {
    // Create initial mesh points
    vector<Vector3D> points = create_mesh_points(config.domain, config.mesh);
    std::cout << "  Created " << points.size() << " initial mesh points\n";

    // Check all points are within domain
    bool all_valid = true;
    for (const auto& p : points) {
        if (!is_point_in_domain(p, config.domain.lower_bound, config.domain.upper_bound)) {
            std::cout << "    ERROR: Point " << p << " is outside domain!\n";
            all_valid = false;
            break;
        }
    }
    if (all_valid) {
        std::cout << "    All points are within domain bounds\n";
    }

    // Build tessellation
    std::cout << "  Building Voronoi tessellation...\n";
    tess.Build(points);
    std::cout << "    Tessellation built successfully!\n";

    // Apply initial conditions
    vector<ComputationalCell3D> cells;
    cells.reserve(points.size());
    for (const auto& point : points) {
        cells.push_back(config.initial_condition(point, *config.physics.eos));
    }
    std::cout << "  Applied initial conditions\n";

    return {points, cells};
}

} // namespace rich3d

#endif // MESH_HELPER_HPP
