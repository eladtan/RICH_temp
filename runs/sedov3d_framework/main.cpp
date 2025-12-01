#include "source/framework/problem_config_3d.hpp"
#include "source/framework/simulation_builder_3d.hpp"
#include "source/misc/universal_error.hpp"

using namespace rich3d;

ComputationalCell3D sedov_initial_condition(const Vector3D& position, const EquationOfState& eos) {
    ComputationalCell3D cell;

    // Sedov blast wave parameters
    const double ambient_density = 1.0;
    const double ambient_energy = 0.1;
    const double blast_energy = 1e5;
    const double blast_radius = 0.2;
    const Vector3D blast_center(0, 0, 0);

    // Distance from blast center
    double r = abs(position - blast_center);

    // Set common properties
    cell.density = ambient_density;
    cell.velocity = Vector3D(0, 0, 0);

    if (r < blast_radius) {
        cell.internal_energy = blast_energy;
        cell.pressure = eos.de2p(cell.density, cell.internal_energy);
    } else {
        // Ambient medium
        cell.internal_energy = ambient_energy;
        cell.pressure = eos.de2p(cell.density, cell.internal_energy);
    }

    cell.temperature = 0.0;
    cell.Erad = 0.0;
    for (size_t i = 0; i < cell.Eg.size(); ++i) {
        cell.Eg[i] = 0.0;
    }

    return cell;
}

// Reference cell for expanded vacuum regions
ComputationalCell3D vacuum_reference_cell(const EquationOfState& eos, double time) {
    ComputationalCell3D cell;

    // Very low density vacuum
    cell.density = 1e-10;
    cell.velocity = Vector3D(0, 0, 0);

    // Low energy ambient state
    cell.internal_energy = 1e-10;
    cell.pressure = eos.de2p(cell.density, cell.internal_energy);

    // Zero radiation
    cell.temperature = 0.0;
    cell.Erad = 0.0;
    for (size_t i = 0; i < cell.Eg.size(); ++i) {
        cell.Eg[i] = 0.0;
    }

    return cell;
}

int main() {
    Problem3DConfig config;
    config.name = "sedov3d_framework";

    // Domain: Start with small box, will expand automatically
    double box_size = 1;
    config.domain.lower_bound = Vector3D(-box_size, -box_size, -box_size);
    config.domain.upper_bound = Vector3D(box_size, box_size, box_size);

    // Enable automatic domain expansion as blast wave propagates
    //config.domain.dynamic.enabled = true;
    //config.domain.dynamic.update_frequency = 1;   // Check every 10 cycles
    //config.domain.dynamic.min_velocity = 0.5;      // Track material moving faster than 0.5
    //config.domain.dynamic.volume_fraction = 1e-5;  // New point density in expanded region

    // config.mesh.type = MeshConfig::Type::RANDOM;  // Default
    config.mesh.num_points = 1e4; // 10,000 particles (matching sedov_3d)
    // config.mesh.round_iterations = 10;  // Default

    // Physics (matching sedov_3d exactly)
    config.physics.eos = std::make_shared<IdealGas>(5.0 / 3.0);

    // Boundary conditions (using defaults: FREE_FLOW on all sides)
    /*config.boundary.x_lower = BoundaryConfig::Type::FREE_FLOW;  // Default
     config.boundary.x_upper = BoundaryConfig::Type::FREE_FLOW;  // Default
     config.boundary.y_lower = BoundaryConfig::Type::FREE_FLOW;  // Default
     config.boundary.y_upper = BoundaryConfig::Type::FREE_FLOW;  // Default
     config.boundary.z_lower = BoundaryConfig::Type::FREE_FLOW;  // Default
     config.boundary.z_upper = BoundaryConfig::Type::FREE_FLOW;  // Default*/

    config.boundary.x_lower = BoundaryConfig::Type::RIGID_WALL;
    config.boundary.x_upper = BoundaryConfig::Type::RIGID_WALL;
    config.boundary.y_lower = BoundaryConfig::Type::RIGID_WALL;
    config.boundary.y_upper = BoundaryConfig::Type::RIGID_WALL;
    config.boundary.z_lower = BoundaryConfig::Type::RIGID_WALL;
    config.boundary.z_upper = BoundaryConfig::Type::RIGID_WALL;

    // Initial condition: Sedov blast wave (user-defined function)
    config.initial_condition = sedov_initial_condition;

    // Reference cell for domain expansion (new vacuum cells)
    //config.domain.dynamic.new_cell_state = vacuum_reference_cell;

    // Output configuration (matching sedov_3d behavior)
    config.output.output_directory = "./";          // Current directory, not ./output
    config.output.output_prefix = "sedov";          // Match sedov_3d naming
    config.output.mode = OutputConfig::Mode::CYCLE; // Cycle-based output!
    config.output.cycle_interval = 10;              // Every 100 cycles (matching sedov_3d)
    config.output.final_time = 0.018;
    config.output.max_cycles = 1000000;

    try {
        Simulation3DBuilder::build_and_run(config);
    } catch (const UniversalError& e) {
        std::cerr << "\n=== UNIVERSALERROR ===\n";
        std::cerr << "Error message: " << e.getErrorMessage() << "\n";
        std::cerr << "======================\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "\n=== STD::EXCEPTION ===\n";
        std::cerr << "Exception caught: " << e.what() << "\n";
        std::cerr << "======================\n";
        return 1;
    } catch (...) {
        std::cerr << "\n=== UNKNOWN ERROR ===\n";
        std::cerr << "Unknown exception caught!\n";
        std::cerr << "=====================\n";
        return 1;
    }

    return 0;
}
