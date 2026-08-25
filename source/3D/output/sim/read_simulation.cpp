#include "newtonian/three_dimensional/simulation/Simulation.hpp"
#include "3D/output/cellData.hpp"
#include <filesystem>
#include <thread>
#include <chrono>
#include <cstdint>
#include <unordered_map>
#include "newtonian/three_dimensional/simulation/steps/io/HydroStepIOHandler.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/RadiationStepIOHandler.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/RadiationMCStepIOHandler.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/PhysicsStepIOHandlerFactory.hpp"
#include "misc/universal_error.hpp"

#ifdef RICH_MPI
    #include <mpi.h>
    #include "mpi/mpi_commands_3d.hpp"
    #include "3D/tessellation/io/load_balancing/HilbertLoadBalancerIOHandler.hpp"
    #include "3D/tessellation/io/load_balancing/LoadBalancerIOHandlerFactory.hpp"
    #include "3D/tessellation/Voronoi3D.hpp"
    #include "3D/tessellation/io/points_manager/HilbertPointsManagerIOHandler.hpp"
    #include "3D/tessellation/io/points_manager/PointsManagerIOHandlerFactory.hpp"
    #include "3D/tessellation/io/hilbert/RectangularConvertorIOHandler.hpp"
    #include "3D/tessellation/io/hilbert/ConvertorIOHandlerFactory.hpp"
#endif

namespace fs = std::filesystem;

namespace
{
    void openReader(HDF5Reader &reader, const std::string &filename)
    {
        for(int attempt = 1; attempt <= 50; ++attempt)
        {
            try
            {
                reader.Load(filename);
                return;
            }
            catch(const H5::FileIException &)
            {
                if(attempt == 50)
                {
                    throw UniversalError("Failed to open HDF5 file after 50 attempts: " + filename);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }

    void readGeneralInfo(const HDF5Reader &reader, Simulation &sim)
    {
        BoundingBox<Vector3D> box;
        reader.ReadElement("/Box", box);
        sim.getTessellation().SetBox(box.getLL(), box.getUR());

        double time = 0;
        reader.ReadElement("/Time", time);
        sim.SetTime(time);

        size_t cycle = 0;
        reader.ReadElement("/Cycle", cycle);
        sim.SetCycle(cycle);

        if(reader.Exists("/TimeStep"))
        {
            double dt = 0;
            reader.ReadElement("/TimeStep", dt);
            sim.SetTimeStep(dt);
        }

        if(reader.Exists("/WallclockTime"))
        {
            double wct = 0;
            reader.ReadElement("/WallclockTime", wct);
            sim.SetWallclockTime(wct);
        }
    }

    void readIndividualTimeSteps(const HDF5Reader &reader, Simulation &sim)
    {
        std::string const group = "/individual_time_steps";
        if(!reader.Exists(group + "/version"))
            return;

        std::uint64_t version = 0;
        reader.ReadElement(group + "/version", version);
		if(version != 1 && version != 2 && version != 3 && version != 4 &&
		   version != 5 && version != 6 && version != 7)
			throw UniversalError("Unsupported individual timestep restart version");

        IndividualTimeStepOptions options;
        reader.ReadElement(group + "/time_quantum", options.time_quantum);
        reader.ReadElement(group + "/initial_bin", options.initial_bin);
        reader.ReadElement(group + "/maximum_bin", options.maximum_bin);
        reader.ReadElement(group + "/maximum_neighbor_bin_difference",
                           options.maximum_neighbor_bin_difference);
        std::uint8_t mesh_policy = 0;
        reader.ReadElement(group + "/mesh_build_policy", mesh_policy);
        if(mesh_policy > static_cast<std::uint8_t>(IndividualMeshBuildPolicy::AutoPartial))
            throw UniversalError("Invalid individual mesh policy in restart");
        options.mesh_build_policy = static_cast<IndividualMeshBuildPolicy>(mesh_policy);
        reader.ReadElement(group + "/partial_build_fraction",
                           options.partial_build_fraction);
        std::uint8_t verify_partial = 0;
		reader.ReadElement(group + "/verify_partial_build", verify_partial);
		options.verify_partial_build = verify_partial != 0;
		if(version >= 2)
		{
			std::uint8_t force_synchronized = 0;
			reader.ReadElement(group + "/force_synchronized", force_synchronized);
			options.force_synchronized = force_synchronized != 0;
		}

        std::vector<std::uint64_t> ids, begin, end, primitive;
        std::vector<std::uint8_t> bins, gravity_phase;
        std::vector<Vector3D> point_velocity, acceleration;
        reader.ReadElement(group + "/cell_ids", ids);
        reader.ReadElement(group + "/begin_ticks", begin);
        reader.ReadElement(group + "/end_ticks", end);
        reader.ReadElement(group + "/primitive_ticks", primitive);
        reader.ReadElement(group + "/bins", bins);
        reader.ReadElement(group + "/point_velocity", point_velocity);
        reader.ReadElement(group + "/cached_acceleration", acceleration);
        reader.ReadElement(group + "/gravity_half_kick_pending", gravity_phase);
        std::size_t const count = ids.size();
        if(begin.size() != count || end.size() != count || primitive.size() != count ||
           bins.size() != count || point_velocity.size() != count ||
           acceleration.size() != count || gravity_phase.size() != count)
            throw UniversalError("Individual timestep restart arrays have inconsistent lengths");

        std::unordered_map<std::size_t, std::size_t> stored_index;
        stored_index.reserve(count);
        for(std::size_t i = 0; i < count; ++i)
            if(!stored_index.emplace(static_cast<std::size_t>(ids[i]), i).second)
                throw UniversalError("Duplicate cell ID in individual timestep restart");

        std::vector<ComputationalCell3D> const& cells = sim.getCells();
        std::vector<CellTimeState> states(cells.size());
        for(std::size_t i = 0; i < cells.size(); ++i)
        {
            auto const found = stored_index.find(cells[i].ID);
            if(found == stored_index.end())
                throw UniversalError("Cell ID is missing from individual timestep restart");
            std::size_t const source = found->second;
            states[i].cell_id = cells[i].ID;
            states[i].begin_tick = begin[source];
            states[i].end_tick = end[source];
            states[i].last_primitive_tick = primitive[source];
            states[i].time_bin = bins[source];
            states[i].point_velocity = point_velocity[source];
            states[i].cached_acceleration = acceleration[source];
            states[i].gravity_half_kick_pending = gravity_phase[source] != 0;
        }

        double time_origin = 0;
        double time_quantum = 0;
        std::uint64_t current_tick = 0;
        reader.ReadElement(group + "/time_origin", time_origin);
        reader.ReadElement(group + "/time_quantum", time_quantum);
        reader.ReadElement(group + "/current_tick", current_tick);
        RadiationRepairAccounting repair_accounting;
        if(version >= 3) {
            std::string const repair_group = group + "/radiation_repair";
            reader.ReadElement(repair_group + "/repaired_cells",
                               repair_accounting.repaired_cells);
            reader.ReadElement(repair_group + "/repaired_groups",
                               repair_accounting.repaired_groups);
            reader.ReadElement(repair_group + "/cumulative_injected_energy",
                               repair_accounting.cumulative_injected_energy);
            reader.ReadElement(repair_group + "/maximum_relative_deficit",
                               repair_accounting.maximum_relative_deficit);
            reader.ReadElement(repair_group + "/representative_cell_id",
                               repair_accounting.representative_cell_id);
            reader.ReadElement(repair_group + "/representative_group",
                               repair_accounting.representative_group);
            reader.ReadElement(repair_group + "/representative_rank",
                               repair_accounting.representative_rank);
            reader.ReadElement(repair_group + "/representative_original_extent",
                               repair_accounting.representative_original_extent);
            reader.ReadElement(repair_group + "/representative_floor_extent",
                               repair_accounting.representative_floor_extent);
            reader.ReadElement(repair_group + "/representative_injected_extent",
                               repair_accounting.representative_injected_extent);
            reader.ReadElement(repair_group + "/maximum_global_radiation_energy",
                               repair_accounting.maximum_global_radiation_energy);
            reader.ReadElement(repair_group + "/next_warning_fraction",
                               repair_accounting.next_warning_fraction);
            if(version >= 4) {
                reader.ReadElement(
                    repair_group + "/residual_correction_limited_groups",
                    repair_accounting.residual_correction_limited_groups);
                reader.ReadElement(
                    repair_group + "/residual_correction_signed_energy_bias",
                    repair_accounting.
                        residual_correction_signed_energy_bias);
                reader.ReadElement(
                    repair_group + "/residual_correction_absolute_energy_bias",
                    repair_accounting.
                        residual_correction_absolute_energy_bias);
                reader.ReadElement(
                    repair_group +
                        "/residual_correction_signed_bias_by_group",
                    repair_accounting.
                        residual_correction_signed_bias_by_group);
                reader.ReadElement(
                    repair_group +
                        "/residual_correction_absolute_bias_by_group",
                    repair_accounting.
                        residual_correction_absolute_bias_by_group);
                reader.ReadElement(
                    repair_group + "/residual_correction_minimum_scale",
                    repair_accounting.residual_correction_minimum_scale);
                reader.ReadElement(
                    repair_group + "/positivity_rescue_events",
                    repair_accounting.positivity_rescue_events);
                reader.ReadElement(
                    repair_group + "/positivity_rescue_blocks",
                    repair_accounting.positivity_rescue_blocks);
                reader.ReadElement(
                    repair_group +
                        "/positivity_rescue_additional_iterations",
                    repair_accounting.
                        positivity_rescue_additional_iterations);
                reader.ReadElement(
                    repair_group + "/residual_positive_floor_events",
                    repair_accounting.residual_positive_floor_events);
                reader.ReadElement(
                    repair_group + "/residual_positive_floor_cells",
                    repair_accounting.residual_positive_floor_cells);
                reader.ReadElement(
                    repair_group + "/residual_positive_floor_groups",
                    repair_accounting.residual_positive_floor_groups);
                reader.ReadElement(
                    repair_group +
                        "/residual_positive_floor_cumulative_injected_energy",
                    repair_accounting.
                        residual_positive_floor_cumulative_injected_energy);
                reader.ReadElement(
                    repair_group +
                        "/residual_positive_floor_maximum_cell_injection_ratio",
                    repair_accounting.
                        residual_positive_floor_maximum_cell_injection_ratio);
                reader.ReadElement(
                    repair_group +
                        "/residual_positive_floor_maximum_global_injection_ratio",
                    repair_accounting.
                        residual_positive_floor_maximum_global_injection_ratio);
                reader.ReadElement(
                    repair_group +
                        "/residual_positive_floor_maximum_post_true_residual_error",
                    repair_accounting.
                        residual_positive_floor_maximum_post_true_residual_error);
                reader.ReadElement(
                    repair_group +
                        "/residual_positive_floor_initial_global_radiation_energy",
                    repair_accounting.
                        residual_positive_floor_initial_global_radiation_energy);
            }
        }
        IndividualRadiationDefectAccounting defect_accounting;
		if(version >= 5) {
			std::string const defect_group = group + "/radiation_defect";
			if(version == 5) {
				double cumulative_signed_extent = 0;
				double cumulative_absolute_extent = 0;
				reader.ReadElement(defect_group + "/cumulative_signed_extent",
					cumulative_signed_extent);
				reader.ReadElement(defect_group + "/cumulative_absolute_extent",
					cumulative_absolute_extent);
				defect_accounting.cumulative_signed_extent =
					static_cast<long double>(cumulative_signed_extent);
				defect_accounting.cumulative_absolute_extent =
					static_cast<long double>(cumulative_absolute_extent);
			}
			else {
				double cumulative_signed_extent_hi = 0;
				double cumulative_signed_extent_lo = 0;
				double cumulative_absolute_extent_hi = 0;
				double cumulative_absolute_extent_lo = 0;
				reader.ReadElement(
					defect_group + "/cumulative_signed_extent_hi",
					cumulative_signed_extent_hi);
				reader.ReadElement(
					defect_group + "/cumulative_signed_extent_lo",
					cumulative_signed_extent_lo);
				reader.ReadElement(
					defect_group + "/cumulative_absolute_extent_hi",
					cumulative_absolute_extent_hi);
				reader.ReadElement(
					defect_group + "/cumulative_absolute_extent_lo",
					cumulative_absolute_extent_lo);
				defect_accounting.cumulative_signed_extent =
					static_cast<long double>(cumulative_signed_extent_hi) +
					static_cast<long double>(cumulative_signed_extent_lo);
				defect_accounting.cumulative_absolute_extent =
					static_cast<long double>(cumulative_absolute_extent_hi) +
					static_cast<long double>(cumulative_absolute_extent_lo);
			}
            reader.ReadElement(defect_group + "/initial_positive_global_extent",
                defect_accounting.initial_positive_global_extent);
            reader.ReadElement(defect_group + "/last_normalization_scale",
                defect_accounting.last_normalization_scale);
            reader.ReadElement(
                defect_group + "/maximum_event_absolute_fraction",
                defect_accounting.maximum_event_absolute_fraction);
            reader.ReadElement(defect_group + "/maximum_local_fraction",
                defect_accounting.maximum_local_fraction);
            reader.ReadElement(defect_group + "/accepted_dirichlet_candidates",
                defect_accounting.accepted_dirichlet_candidates);
            reader.ReadElement(defect_group + "/defect_rejections",
                defect_accounting.defect_rejections);
            reader.ReadElement(defect_group + "/defect_retry_substeps",
                defect_accounting.defect_retry_substeps);
            reader.ReadElement(defect_group + "/config_version",
                defect_accounting.config_version);
            reader.ReadElement(defect_group + "/local_withdrawal_limit",
                defect_accounting.local_withdrawal_limit);
            reader.ReadElement(defect_group + "/event_absolute_target",
                defect_accounting.event_absolute_target);
            reader.ReadElement(defect_group + "/cumulative_signed_limit",
                defect_accounting.cumulative_signed_limit);
            reader.ReadElement(defect_group + "/cumulative_absolute_limit",
                defect_accounting.cumulative_absolute_limit);
            reader.ReadElement(defect_group + "/cooldown_accepted_candidates",
                defect_accounting.cooldown_accepted_candidates);
            reader.ReadElement(defect_group + "/cooldown_required_candidates",
                defect_accounting.cooldown_required_candidates);
            reader.ReadElement(defect_group + "/cooldown_fraction_ceiling",
                defect_accounting.cooldown_fraction_ceiling);
            std::uint8_t history_complete = 0;
            reader.ReadElement(defect_group + "/history_complete",
                               history_complete);
            defect_accounting.history_complete = history_complete != 0;
        }
        else {
            defect_accounting = IndividualRadiationDefectAccounting();
            defect_accounting.history_complete = false;
        }
        bool force_all_active_latched = false;
        if(version >= 7)
        {
            std::uint8_t stored_latch = 0;
            reader.ReadElement(group + "/force_all_active_latched",
                               stored_latch);
            if(stored_latch > 1)
                throw UniversalError(
                    "Invalid forced-active latch in individual timestep restart");
            force_all_active_latched = stored_latch != 0;
        }
        sim.EnableIndividualTimeSteps(options);
        sim.GetIndividualTimeStepScheduler()->restore(
            cells, time_origin, time_quantum, current_tick, std::move(states),
            repair_accounting, defect_accounting, force_all_active_latched);

        double const restored_time = time_origin +
            time_quantum * static_cast<double>(current_tick);
        if(restored_time != sim.GetTime())
            throw UniversalError("Individual timestep restart clock differs from snapshot time");
        sim.SetTimeStep(sim.GetIndividualTimeStepScheduler()->nextEventTimeStep());
    }

    #ifdef RICH_MPI
    std::string readLoadBalancers(const HDF5Reader &reader, Simulation &sim)
    {
        std::string currentLBName;

        if(!reader.Exists("/load_balance"))
        {
            return currentLBName;
        }

        if(reader.Exists("/load_balance/current"))
        {
            reader.ReadElement("/load_balance/current", currentLBName);
        }

        auto lbEntries = reader.ReadGroupNames("/load_balance");
        for(const auto &name : lbEntries)
        {
            std::string group = "/load_balance/" + name;
            if(!reader.Exists(group + "/type"))
            {
                continue;
            }

            auto lb = LoadBalancerIO::readLoadBalancer(reader, group);
            sim.storeLoadBalance(name, lb);
        }

        return currentLBName;
    }
    #endif

    void readTessellation(const HDF5Reader &reader, const std::string &prefix,
                          Simulation &sim
#ifdef RICH_MPI
                          , const std::shared_ptr<LoadBalancer<Vector3D>>& restart_load
#endif
                          )
    {
        Tessellation3D &tess = sim.getTessellation();

#ifdef RICH_MPI
        // readGeneralInfo resets the Voronoi points manager when it restores
        // the box.  Seed that fresh manager before a no-exchange build;
        // otherwise its environment agent is constructed with a null load
        // balancer.
        if(restart_load)
            tess.PresetLoadBalancer(restart_load);
#endif

        if(reader.Exists(prefix + "/volumes"))
        {
            std::vector<double> vols;
            reader.ReadElement(prefix + "/volumes", vols);
            // tess.GetAllVolumes() = std::move(vols);
        }
        if(reader.Exists(prefix + "/CM"))
        {
            std::vector<Vector3D> cm;
            reader.ReadElement(prefix + "/CM", cm);
            // tess.GetAllCM() = std::move(cm);
        }

#ifdef RICH_MPI
        if(reader.Exists(prefix + "/points_manager/type"))
        {
            Voronoi3D *voronoi = dynamic_cast<Voronoi3D *>(&tess);
            if(voronoi)
            {
                auto coords = tess.GetBoxCoordinates();
                auto pm = PointsManagerIO::readPointsManager(reader, prefix + "/points_manager", coords.first, coords.second);
                voronoi->SetPointsManager(pm);
                if(restart_load)
                    tess.PresetLoadBalancer(restart_load);
            }
        }
#endif

        int rank = 0;
        #ifdef RICH_MPI
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        #endif // RICH_MPI

        if(reader.Exists(prefix + "/mesh_points"))
        {
            std::vector<Vector3D> points;
            reader.ReadElement(prefix + "/mesh_points", points);
            #ifdef RICH_MPI
                std::cout << "Rank " << rank << " has " << points.size() << " points read." << std::endl;
                tess.BuildParallel(points, true, true);
            #else
                tess.Build(points);
            #endif
        }

        std::cout << "After first build, rank " << rank << " has " <<  tess.GetPointNo() << " points" << std::endl;
    }

    void readPhysicsGroups(const HDF5Reader &reader, const std::string &prefix, Simulation &sim)
    {
        for(auto &step : sim.getPhysicsSteps())
        {
            PhysicsStepIO::readStep(reader, prefix, *step);
        }
    }

    void readPrivateInfo(const HDF5Reader &reader, const std::string &prefix, Simulation &sim)
    {
        if(!reader.Exists(prefix + "/cells"))
        {
            return;
        }
        std::vector<ComputationalCell3D> cells;
        reader.ReadElement(prefix + "/cells", cells);
        sim.getCells() = std::move(cells);
        if(reader.Exists(prefix + "/extensives"))
        {
            std::vector<HDF5Utils::Conserved3DRecord> records;
            reader.ReadElement(prefix + "/extensives", records);
            if(records.size() != sim.getCells().size())
                throw UniversalError("Restart conserved extent count differs from cells");
            std::vector<Conserved3D> extensives(records.size());
            for(std::size_t i = 0; i < records.size(); ++i)
                extensives[i] = HDF5Utils::UnpackConserved3D(records[i]);
            sim.getExtensives() = std::move(extensives);
        }
    }
}

void ReadSimulation(const std::string &filename,
                    Simulation &sim
                    #ifdef RICH_MPI
                        , bool parallel
                        , int fake_rank
                    #endif
                    )
{
    HDF5Reader globalReader;
    openReader(globalReader, filename);
    readGeneralInfo(globalReader, sim);

    std::shared_ptr<HDF5Reader> dataReader;

    #ifdef RICH_MPI
    std::string currentLBName;
    std::shared_ptr<LoadBalancer<Vector3D>> restartLoad;
    if(parallel)
    {
        currentLBName = readLoadBalancers(globalReader, sim);

        for(const auto& entry : sim.GetLoads())
            if(entry.first == currentLBName)
            {
                restartLoad = entry.second;
                break;
            }

        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int rank_to_read = (fake_rank >= 0) ? fake_rank : rank;

        std::string dir = fs::path(filename).replace_extension("").string();
        std::string rankFile = dir + "/" + std::to_string(rank_to_read) + ".h5";
        if(!fs::exists(rankFile))
        {
            throw UniversalError("ReadSimulation: rank file not found: " + rankFile);
        }

        dataReader = std::make_shared<HDF5Reader>();
        openReader(*dataReader, rankFile);
    }
    else
    #endif
    {
        dataReader = std::make_shared<HDF5Reader>();
        openReader(*dataReader, filename);
    }

#ifdef RICH_MPI
    readTessellation(*dataReader, "/tess", sim, restartLoad);
#else
    readTessellation(*dataReader, "/tess", sim);
#endif

#ifdef RICH_MPI
    if(parallel && !currentLBName.empty())
    {
        auto loads = sim.GetLoads();
        for(const auto &[name, lb] : loads)
        {
            if(name == currentLBName)
            {
                sim.getTessellation().PresetLoadBalancer(lb);
                break;
            }
        }
        sim.PresetLoadBalance(currentLBName);
    }
    #endif

    readPhysicsGroups(*dataReader, "", sim);
    readPrivateInfo(*dataReader, "", sim);
#ifdef RICH_MPI
    bool const rank_local_individual = parallel &&
        dataReader->Exists("/individual_time_steps/version");
    if(parallel && !rank_local_individual)
        MPI_exchange_data(sim.getTessellation(), sim.getCells(), true);
    // Parallel snapshots store canonical scheduler state beside each rank's
    // owned cells.  Fall back to the top-level group for serial and legacy
    // files written before rank-local scheduler persistence was added.
    if(rank_local_individual)
        readIndividualTimeSteps(*dataReader, sim);
    else
#endif
        readIndividualTimeSteps(globalReader, sim);
    sim.recomputeMaxID();
}
