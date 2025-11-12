
#include "simulation_builder_3d.hpp"
#include "boundary_helper.hpp"
#include "mesh_helper.hpp"
#include "amr_helper.hpp"
#include "source_term_helper.hpp"
#include "numerical_scheme_helper.hpp"

#include "3D/elementary/Vector3D.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"

#include "3D/tesselation/voronoi/Voronoi3D.hpp"
#include "3D/GeometryCommon/UpdateBox.hpp"

#include "newtonian/three_dimensional/hdsim_3d.hpp"

#include "3D/output/write3D.hpp"
#include "misc/universal_error.hpp"

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <string>

namespace rich3d {

using std::pair;
using std::string;
using std::vector;

using TracerStickerNames = pair<vector<string>, vector<string>>;

static void run_simulation_loop(const Problem3DConfig& config, ::HDSim3D& sim, ::CourantFriedrichsLewy& timestep,
                                SourceTermComponents& source_components, AMRComponents& amr_components) {
    //function to join paths properly
    auto join_path = [](const std::string& dir, const std::string& file) -> std::string {
        if (dir.empty())
            return file;
        if (dir.back() == '/')
            return dir + file;
        return dir + "/" + file;
    };

    // Write initial condition
    std::string initial_filename = join_path(config.output.output_directory, config.output.output_prefix + "_0.h5");
    ::WriteSnapshot3D(sim, initial_filename);
    std::cout << "  >> Wrote initial snapshot: " << initial_filename << "\n\n";

    size_t cycle = 0;
    size_t snapshot_counter = 1;
    const size_t report_interval = 10;
    size_t next_snapshot_cycle = config.output.cycle_interval;
    double next_snapshot_time = config.output.snapshot_interval;

    while (sim.getTime() < config.output.final_time && cycle < config.output.max_cycles) {
        // Pre-step hook
        if (config.hooks.pre_step) {
            config.hooks.pre_step(sim, sim.getTime());
        }

        // Radiation time stepping
        if (config.radiation.enabled && config.radiation.use_radiation_timestep && source_components.diffusion_solver) {
            double old_dt = sim.getTimeStep();
            double new_dt = sim.RadiationTimeStep(old_dt, *source_components.diffusion_solver);

            // Apply bounds and smoothing
            new_dt = std::min(config.radiation.max_timestep, std::max(config.radiation.min_timestep, new_dt));

            // Prevent rapid time step drops
            if (new_dt < config.radiation.timestep_smoothing * old_dt) {
                new_dt = config.radiation.timestep_smoothing * old_dt;
            }

            timestep.SetTimeStep(new_dt);
        }

        sim.timeAdvance2();

        // Post-step hook
        if (config.hooks.post_step) {
            config.hooks.post_step(sim, sim.getTime());
        }

        // Out-of-domain cell removal
        if (amr_components.out_of_domain_amr && cycle % 10 == 0 && cycle > 0) {
            (*amr_components.out_of_domain_amr)(sim);
        }

        // AMR
        if (config.amr.enabled && amr_components.amr_ptr && cycle % config.amr.frequency == 0 && cycle > 0) {
            (*amr_components.amr_ptr)(sim);

            // Post-AMR hook
            if (config.hooks.post_amr) {
                config.hooks.post_amr(sim, sim.getTime());
            }
        }

        // Domain expansion
        if (config.domain.dynamic.enabled && cycle % config.domain.dynamic.update_frequency == 0 && cycle > 0) {
            // Get reference cell for new vacuum regions
            ::ComputationalCell3D ref_cell = get_expansion_reference_cell(config, sim.getTime());

            // Call UpdateBox to expand domain if needed
            ::UpdateBox(sim.getTesselation(), sim, config.domain.dynamic.min_velocity,
                        config.domain.dynamic.volume_fraction, ref_cell);

            // Report new domain size
            auto box = sim.getTesselation().GetBoxCoordinates();
            std::cout << "  Domain expansion check (cycle " << cycle << ")\n";
            std::cout << "    Domain: " << box.first << " to " << box.second << "\n";
        }

        // Progress reporting
        if (cycle % report_interval == 0) {
            std::cout << "Cycle " << cycle << "  t = " << sim.getTime() << "  dt = " << sim.getTimeStep() << "\n";
        }

        // Periodic snapshot output
        bool should_output = false;
        if (config.output.mode == OutputConfig::Mode::CYCLE) {
            should_output = (cycle >= next_snapshot_cycle && cycle > 0);
        } else { // OutputConfig::Mode::TIME
            should_output = (sim.getTime() >= next_snapshot_time);
        }

        if (should_output) {
            std::string snapshot_filename =
                join_path(config.output.output_directory,
                          config.output.output_prefix + "_" + std::to_string(snapshot_counter) + ".h5");

            // Pre-output hook
            if (config.hooks.pre_output) {
                config.hooks.pre_output(sim, sim.getTime());
            }

            ::WriteSnapshot3D(sim, snapshot_filename);
            std::cout << "  >> Wrote snapshot: " << snapshot_filename << " (cycle " << cycle << ")\n";

            // Post-output hook
            if (config.hooks.post_output) {
                config.hooks.post_output(sim, sim.getTime());
            }

            snapshot_counter++;

            // Update next snapshot trigger
            if (config.output.mode == OutputConfig::Mode::CYCLE) {
                next_snapshot_cycle += config.output.cycle_interval;
            } else {
                next_snapshot_time += config.output.snapshot_interval;
            }
        }

        cycle++;
    }

    // Final output
    std::string output_filename = join_path(config.output.output_directory, config.output.output_prefix + "_final.h5");
    ::WriteSnapshot3D(sim, output_filename);

    std::cout << "\nSimulation complete!\n";
    std::cout << "  Final time: " << sim.getTime() << "\n";
    std::cout << "  Total cycles: " << cycle << "\n";
    std::cout << "  Output written to " << output_filename << "\n";
}

void Simulation3DBuilder::build_and_run(const Problem3DConfig& config) {
    std::cout << "================================================================\n";
    std::cout << "RICH 3D Simulation: " << config.name << "\n";
    std::cout << "================================================================\n\n";

    // Validate required configuration
    if (!config.physics.eos) {
        throw std::runtime_error("PhysicsConfig: equation of state (eos) is required");
    }
    if (!config.initial_condition) {
        throw std::runtime_error("Problem3DConfig: initial_condition function is required");
    }

    // Create output directory
    string mkdir_cmd = "mkdir -p " + config.output.output_directory;
    std::system(mkdir_cmd.c_str());

    std::cout << "Building simulation components...\n";

    // Build tessellation
    ::Voronoi3D tess(config.domain.lower_bound, config.domain.upper_bound);

    // Setup mesh and initial conditions
    auto [points, cells] = setup_initial_mesh(config, tess);

    // Setup boundary conditions
    auto [ghost_ptr, ghost_owner] =
        create_boundary_conditions(config.boundary, config.domain.lower_bound, config.domain.upper_bound);

    // Setup source terms (accelerations + custom + radiation)
    SourceTermComponents source_components = setup_source_terms(config);

    // Setup numerical scheme
    NumericalSchemeComponents numerical(config.numerical, *config.physics.eos, *ghost_ptr,
                                        *source_components.force_ptr);

    // Setup AMR
    AMRComponents amr_components = setup_amr(config, numerical.reconstruction_ptr);

    std::cout << "  Created all simulation components\n\n";

    // Print all configuration details
    config.boundary.print();
    config.radiation.print();
    config.sources.print(config.radiation.enabled);
    config.numerical.print();
    config.amr.print();
    config.hooks.print();
    config.output.print();

    // Create HDSim3D simulator
    TracerStickerNames tracer_sticker_names{
        config.tracer_names.empty() ? ::ComputationalCell3D::tracerNames : config.tracer_names,
        config.sticker_names.empty() ? ::ComputationalCell3D::stickerNames : config.sticker_names};
    ::HDSim3D sim(tess, cells, *config.physics.eos, *numerical.point_motion_ptr, numerical.timestep,
                  *numerical.flux_calc, numerical.cell_updater, numerical.extensive_updater,
                  *source_components.force_ptr, tracer_sticker_names);
    std::cout << "  Created HDSim3D simulator\n";

    // Run the main simulation loop
    run_simulation_loop(config, sim, numerical.timestep, source_components, amr_components);
}

} // namespace rich3d
