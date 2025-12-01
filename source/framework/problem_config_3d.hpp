
#ifndef PROBLEM_CONFIG_3D_HPP
#define PROBLEM_CONFIG_3D_HPP

#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <cstddef>

#include <iostream>
#include <cmath>

#include "3D/elementary/Vector3D.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/common/equation_of_state.hpp"
#include "newtonian/common/ideal_gas.hpp"

class Ghost3D;
class PointMotion3D;
class SourceTerm3D;
class Acceleration3D;
class CellsToRefine3D;
class CellsToRemove3D;
class DiagnosticAppendix3D;
class DiffusionCoefficientCalculator;
class Tessellation3D;
class HDSim3D;

namespace rich3d {

// Import RICH types from global namespace
using ::Acceleration3D;
using ::CellsToRefine3D;
using ::CellsToRemove3D;
using ::ComputationalCell3D;
using ::DiagnosticAppendix3D;
using ::DiffusionCoefficientCalculator;
using ::EquationOfState;
using ::Ghost3D;
using ::HDSim3D;
using ::IdealGas;
using ::PointMotion3D;
using ::SourceTerm3D;
using ::Tessellation3D;
using ::Vector3D;

using SimulationHookFunc = std::function<void(HDSim3D& sim, double time)>;

using InitialConditionFunc = std::function<ComputationalCell3D(const Vector3D& position, const EquationOfState& eos)>;

using SourceTermFunc = std::function<std::vector<double>(const std::vector<ComputationalCell3D>& cells,
                                                         const Tessellation3D& tess, double time)>;

using NewCellStateFunc = std::function<ComputationalCell3D(const EquationOfState& eos, double time)>;

struct DynamicExpansionConfig {
    bool enabled = false;            ///< Enable automatic domain expansion
    int update_frequency = 7;        ///< Check for expansion every N cycles
    double min_velocity = 0.5;       ///< Velocity threshold for "active" material
    double volume_fraction = 1e-5;   ///< Density of new points in expanded region
    NewCellStateFunc new_cell_state; ///< Initial state for cells in expanded region
};

struct DomainConfig {
    Vector3D lower_bound;    ///< Lower corner (xmin, ymin, zmin)
    Vector3D upper_bound;    ///< Upper corner (xmax, ymax, zmax)
    bool periodic_x = false; ///< Periodic in x-direction
    bool periodic_y = false; ///< Periodic in y-direction
    bool periodic_z = false; ///< Periodic in z-direction

    DynamicExpansionConfig dynamic;
};

struct MeshConfig {
    // For CARTESIAN type
    size_t nx = 10; ///< Initial grid points in x
    size_t ny = 10; ///< Initial grid points in y
    size_t nz = 10; ///< Initial grid points in z

    enum class Type {
        CARTESIAN,   ///< Regular Cartesian grid
        SPHERICAL,   ///< Spherical coordinates (r, theta, phi)
        CYLINDRICAL, ///< Cylindrical coordinates (r, phi, z)
        RANDOM,      ///< Random points (will auto-apply RoundGrid3D)
        CUSTOM       ///< User-provided point distribution (pre-processed)
    };

    Type type = Type::RANDOM;

    // For RANDOM type
    size_t num_points = 100000;   ///< Number of random points
    size_t round_iterations = 10; ///< RoundGrid3D iterations (0 = skip)

    // For CUSTOM type: user provides initial mesh points
    std::vector<Vector3D> custom_points;
};

struct PhysicsConfig {
    // Equation of state
    std::shared_ptr<EquationOfState> eos;
};

struct NumericalConfig {
    // Time stepping
    double cfl = 0.3;
    double max_time_step = 1e100;

    // Reconstruction order
    enum class Reconstruction {
        PCM,         ///< Piecewise constant (1st order)
        LINEAR_GAUSS ///< Linear with Gauss gradients (2nd order)
    };
    Reconstruction reconstruction = Reconstruction::LINEAR_GAUSS;

    // Riemann solver
    enum class RiemannSolver {
        HLLC, ///< HLLC approximate solver
        EXACT ///< Exact Riemann solver
    };
    RiemannSolver riemann_solver = RiemannSolver::HLLC;

    // Grid motion
    enum class GridMotion {
        EULERIAN,   ///< Fixed grid
        LAGRANGIAN, ///< Moves with fluid
        ROUND_CELLS ///< Regularizes cell shapes
    };
    GridMotion grid_motion = GridMotion::ROUND_CELLS;

    void print() const {
        std::cout << "  CFL number: " << cfl << "\n";

        std::cout << "  Reconstruction: ";
        switch (reconstruction) {
            case Reconstruction::LINEAR_GAUSS:
                std::cout << "LINEAR_GAUSS (2nd order)\n";
                break;
            case Reconstruction::PCM:
                std::cout << "PCM (1st order)\n";
                break;
        }

        std::cout << "  Riemann solver: ";
        switch (riemann_solver) {
            case RiemannSolver::HLLC:
                std::cout << "HLLC\n";
                break;
            case RiemannSolver::EXACT:
                std::cout << "EXACT\n";
                break;
        }

        std::cout << "  Grid motion: ";
        switch (grid_motion) {
            case GridMotion::LAGRANGIAN:
                std::cout << "LAGRANGIAN\n";
                break;
            case GridMotion::EULERIAN:
                std::cout << "EULERIAN\n";
                break;
            case GridMotion::ROUND_CELLS:
                std::cout << "ROUND_CELLS (Lagrangian with regularization)\n";
                break;
        }
    }
};

struct BoundaryConfig {
    enum class Type {
        RIGID_WALL, ///< Reflective boundary (RigidWallGenerator3D)
        FREE_FLOW,  ///< Outflow boundary (FreeFlowGenerator3D)
        PERIODIC,   ///< Periodic (handled by tessellation, not Ghost3D)
        CUSTOM      ///< User-provided Ghost3D
    };

    Type x_lower = Type::FREE_FLOW;
    Type x_upper = Type::FREE_FLOW;
    Type y_lower = Type::FREE_FLOW;
    Type y_upper = Type::FREE_FLOW;
    Type z_lower = Type::FREE_FLOW;
    Type z_upper = Type::FREE_FLOW;

    // If a face is marked CUSTOM but no custom_* is provided, defaults to FREE_FLOW
    std::shared_ptr<Ghost3D> custom_x_lower;
    std::shared_ptr<Ghost3D> custom_x_upper;
    std::shared_ptr<Ghost3D> custom_y_lower;
    std::shared_ptr<Ghost3D> custom_y_upper;
    std::shared_ptr<Ghost3D> custom_z_lower;
    std::shared_ptr<Ghost3D> custom_z_upper;

    void print() const {
        auto type_name = [](Type t) -> std::string {
            switch (t) {
                case Type::RIGID_WALL:
                    return "RIGID_WALL";
                case Type::FREE_FLOW:
                    return "FREE_FLOW";
                case Type::PERIODIC:
                    return "PERIODIC";
                case Type::CUSTOM:
                    return "CUSTOM";
                default:
                    return "UNKNOWN";
            }
        };

        std::cout << "  Boundary conditions:\n";
        std::cout << "    x-: " << type_name(x_lower) << "\n";
        std::cout << "    x+: " << type_name(x_upper) << "\n";
        std::cout << "    y-: " << type_name(y_lower) << "\n";
        std::cout << "    y+: " << type_name(y_upper) << "\n";
        std::cout << "    z-: " << type_name(z_lower) << "\n";
        std::cout << "    z+: " << type_name(z_upper) << "\n";
    }
};

struct RadiationConfig {
    bool enabled = false;

    enum class Type {
        GRAY_DIFFUSION, ///< Flux-limited diffusion (1 group)
        MULTIGROUP      ///< Multigroup diffusion
    };
    Type type = Type::GRAY_DIFFUSION;

    // Opacity model (must implement DiffusionCoefficientCalculator interface)
    std::shared_ptr<DiffusionCoefficientCalculator> opacity;

    // Boundary conditions for radiation
    enum class BoundaryType {
        OPEN,   ///< Open boundary (radiation escapes)
        CLOSED, ///< Closed/reflecting boundary
        SIDE    ///< Side boundary (zero gradient)
    };
    BoundaryType boundary_type = BoundaryType::OPEN;

    // Multigroup parameters (for MULTIGROUP type)
    size_t n_groups = 1;      ///< Number of energy groups
    double energy_min = 1e-2; ///< Minimum energy (keV)
    double energy_max = 1e4;  ///< Maximum energy (keV)

    // Physics flags
    bool compton_cooling = true; ///< Include Compton cooling/heating
    bool doppler_shift = false;  ///< Include Doppler shifting
    bool mixed_frame = false;    ///< Use mixed frame
    bool flux_limiter_on = true; ///< Enable flux limiter

    // Numerical parameters
    double flux_limiter = 1.0 / 3.0; ///< Flux limiter parameter
    double convergence_tolerance = 1e-6;
    double temperature_floor = 1e3; ///< Minimum temperature (K)
    size_t max_iterations = 2000;   ///< Max solver iterations

    // Radiation time stepping
    bool use_radiation_timestep = true; ///< Use radiation-limited time step
    double min_timestep = 2.01e-4;      ///< Minimum allowed time step
    double max_timestep = 0.01;         ///< Maximum allowed time step
    double timestep_smoothing = 0.5;    ///< Prevent rapid drops (new_dt >= smoothing * old_dt)

    void print() const {
        if (!enabled)
            return;

        std::cout << "  Radiation: ";
        switch (type) {
            case Type::GRAY_DIFFUSION:
                std::cout << "GRAY_DIFFUSION";
                break;
            case Type::MULTIGROUP:
                std::cout << "MULTIGROUP (" << n_groups << " groups)";
                break;
        }
        std::cout << " enabled\n";

        if (compton_cooling)
            std::cout << "    - Compton cooling/heating\n";
        if (doppler_shift)
            std::cout << "    - Doppler shifting\n";
        if (flux_limiter_on)
            std::cout << "    - Flux limiter (lambda=" << flux_limiter << ")\n";
        if (use_radiation_timestep) {
            std::cout << "    - Radiation timestep control: min_dt=" << min_timestep << ", max_dt=" << max_timestep
                      << "\n";
        }
    }
};

struct SourceTermConfig {
    // These include: GravityAcceleration3D, ConstantAcceleration3D, TDEGravity, etc.
    std::vector<std::shared_ptr<Acceleration3D>> accelerations;
    bool conservative_mass_flux = false; ///< Include mass flux term

    // These include: DiffusionForce, custom cooling, etc.
    std::vector<std::shared_ptr<SourceTerm3D>> source_terms;

    void print(bool include_radiation) const {
        size_t total_sources = accelerations.size() + source_terms.size() + (include_radiation ? 1 : 0);

        if (total_sources == 0) {
            std::cout << "  Source terms: None (ZeroForce3D)\n";
        } else {
            std::cout << "  Source terms: " << total_sources << " source(s) configured\n";
            std::cout << "    - " << accelerations.size() << " acceleration(s)\n";
            std::cout << "    - " << source_terms.size() << " custom source term(s)\n";
            if (include_radiation) {
                std::cout << "    - Radiation diffusion force\n";
            }
        }
    }
};

struct AMRConfig {
    bool enabled = false;

    int frequency = 10; ///< Run AMR every N cycles

    // User-provided AMR criteria
    std::shared_ptr<CellsToRefine3D> custom_refine;
    std::shared_ptr<CellsToRemove3D> custom_remove;

    void print() const {
        if (!enabled)
            return;

        std::cout << "  AMR: Enabled\n";
        std::cout << "    - Frequency: every " << frequency << " cycles\n";
    }
};

struct OutputConfig {
    std::string output_directory = "./";

    std::string output_prefix = "snapshot";

    // Output mode: time-based or cycle-based
    enum class Mode {
        TIME, ///< Output at fixed time intervals
        CYCLE ///< Output at fixed cycle intervals
    };
    Mode mode = Mode::TIME;

    double snapshot_interval = 0.1; ///< Time between snapshots (Mode::TIME)
    size_t cycle_interval = 100;    ///< Cycles between snapshots (Mode::CYCLE)

    double final_time = 1.0;     ///< Simulation end time
    size_t max_cycles = 1000000; ///< Max timesteps

    // Custom diagnostics
    std::vector<std::shared_ptr<DiagnosticAppendix3D>> diagnostics;

    void print() const {
        std::cout << "\nStarting time evolution...\n";
        std::cout << "  Target time: " << final_time << "\n";

        if (mode == Mode::CYCLE) {
            std::cout << "  Output mode: Every " << cycle_interval << " cycles\n";
        } else {
            std::cout << "  Output mode: Every " << snapshot_interval << " time units\n";
        }
    }
};

struct SimulationHooks {
    SimulationHookFunc pre_step;    ///< Called before each time step
    SimulationHookFunc post_step;   ///< Called after each time step
    SimulationHookFunc post_amr;    ///< Called after AMR refinement
    SimulationHookFunc pre_output;  ///< Called before writing snapshot
    SimulationHookFunc post_output; ///< Called after writing snapshot

    void print() const {
        int hook_count = 0;
        if (pre_step)
            hook_count++;
        if (post_step)
            hook_count++;
        if (pre_output)
            hook_count++;
        if (post_output)
            hook_count++;
        if (post_amr)
            hook_count++;

        if (hook_count > 0) {
            std::cout << "  Lifecycle hooks: " << hook_count << " hook(s) configured\n";
            if (pre_step)
                std::cout << "    - pre_step\n";
            if (post_step)
                std::cout << "    - post_step\n";
            if (pre_output)
                std::cout << "    - pre_output\n";
            if (post_output)
                std::cout << "    - post_output\n";
            if (post_amr)
                std::cout << "    - post_amr\n";
        }
    }
};

struct Problem3DConfig {
    std::string name = "simulation";

    DomainConfig domain;
    MeshConfig mesh;
    PhysicsConfig physics;
    NumericalConfig numerical;
    BoundaryConfig boundary;
    SourceTermConfig sources;
    OutputConfig output;
    InitialConditionFunc initial_condition; ///< Initial condition function

    // Optional advanced features
    RadiationConfig radiation; ///< Radiation transport (set .enabled = true to use)
    AMRConfig amr;             ///< Adaptive mesh refinement (set .enabled = true to use)
    SimulationHooks hooks;     ///< Lifecycle hooks for custom behavior

    // Tracer and sticker names
    std::vector<std::string> tracer_names;
    std::vector<std::string> sticker_names;

    // Restart support
    bool restart = false;     ///< Load from checkpoint
    std::string restart_file; ///< Checkpoint file path
    int restart_cycle = 0;    ///< Starting cycle number

    // MPI configuration (for future)
    // bool use_mpi = false;
    // int mpi_ranks = 1;
};

} // namespace rich3d

#endif
