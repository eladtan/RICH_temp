#include "newtonian/three_dimensional/simulation/Simulation.hpp"
#include "3D/output/cellData.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/HydroStepIOHandler.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/RadiationStepIOHandler.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/RadiationMCStepIOHandler.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/PhysicsStepIOHandlerFactory.hpp"
#include <filesystem>
#include <thread>
#include <chrono>
#include <cstdint>
#include "misc/universal_error.hpp"
#include "misc/memory_debug.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"

#ifdef RICH_MPI
    #include <mpi.h>
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
#ifdef RICH_MPI
    std::string SerializedLoadBalanceName(const std::string &name)
    {
        return name.empty() ? "__rich_restart_current__" : name;
    }
#endif

    HDF5Writer openWriter(const std::string &filename)
    {
        for(int attempt = 1; attempt <= 50; ++attempt)
        {
            try
            {
                return HDF5Writer(filename);
            }
            catch(const H5::FileIException &)
            {
                if(attempt == 50)
                {
                    throw UniversalError("Failed to create HDF5 file after 50 attempts: " + filename);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        throw UniversalError("Unreachable: openWriter");
    }

    void writeGeneralInfo(HDF5Writer &writer, const Simulation &sim)
    {
        const Tessellation3D &tess = sim.getTessellation();
        auto coords = tess.GetBoxCoordinates();
        BoundingBox<Vector3D> box(coords.first, coords.second);
        writer.WriteElement("/Box", box);
        writer.WriteElement("/Time", sim.GetTime());
        writer.WriteElement("/Cycle", sim.GetCycle());
        writer.WriteElement("/TimeStep", sim.GetTimeStep());
        writer.WriteElement("/WallclockTime", sim.GetWallclockTime());

        #ifdef RICH_MPI
        {
            std::vector<std::vector<std::string>> lb_table;
            for(const auto &step : sim.getPhysicsSteps())
            {
                lb_table.push_back({step->getName(), step->getRequiredLB()});
            }
            writer.WriteElement("/load_balance/names", lb_table);
            writer.WriteElement("/load_balance/current",
                                SerializedLoadBalanceName(sim.getCurrentLB()));
        }
        #endif
    }

    void writeIndividualTimeSteps(HDF5Writer &writer, const Simulation &sim)
    {
        if(sim.GetTimeIntegrationMode() != TimeIntegrationMode::Individual)
            return;
        IndividualTimeStepScheduler const* scheduler =
            sim.GetIndividualTimeStepScheduler();
        if(scheduler == nullptr || !scheduler->initialized())
            return;

        std::string const group = "/individual_time_steps";
		writer.WriteElement(group + "/version", std::uint64_t(7));
        writer.WriteElement(group + "/time_origin", scheduler->timeOrigin());
        writer.WriteElement(group + "/time_quantum", scheduler->timeQuantum());
        writer.WriteElement(group + "/current_tick", scheduler->currentTick());
        IndividualTimeStepOptions const& options = scheduler->options();
        writer.WriteElement(group + "/initial_bin", options.initial_bin);
        writer.WriteElement(group + "/maximum_bin", options.maximum_bin);
        writer.WriteElement(group + "/maximum_neighbor_bin_difference",
                            options.maximum_neighbor_bin_difference);
        writer.WriteElement(group + "/mesh_build_policy",
                            static_cast<std::uint8_t>(options.mesh_build_policy));
        writer.WriteElement(group + "/partial_build_fraction",
                            options.partial_build_fraction);
		writer.WriteElement(group + "/verify_partial_build",
		                    static_cast<std::uint8_t>(options.verify_partial_build));
		writer.WriteElement(group + "/force_synchronized",
		                    static_cast<std::uint8_t>(options.force_synchronized));

        RadiationRepairAccounting const& repair =
            scheduler->radiationRepairAccounting();
        std::string const repair_group = group + "/radiation_repair";
        writer.WriteElement(repair_group + "/repaired_cells",
                            repair.repaired_cells);
        writer.WriteElement(repair_group + "/repaired_groups",
                            repair.repaired_groups);
        writer.WriteElement(repair_group + "/cumulative_injected_energy",
                            repair.cumulative_injected_energy);
        writer.WriteElement(repair_group + "/maximum_relative_deficit",
                            repair.maximum_relative_deficit);
        writer.WriteElement(repair_group + "/representative_cell_id",
                            repair.representative_cell_id);
        writer.WriteElement(repair_group + "/representative_group",
                            repair.representative_group);
        writer.WriteElement(repair_group + "/representative_rank",
                            repair.representative_rank);
        writer.WriteElement(repair_group + "/representative_original_extent",
                            repair.representative_original_extent);
        writer.WriteElement(repair_group + "/representative_floor_extent",
                            repair.representative_floor_extent);
        writer.WriteElement(repair_group + "/representative_injected_extent",
                            repair.representative_injected_extent);
        writer.WriteElement(repair_group + "/maximum_global_radiation_energy",
                            repair.maximum_global_radiation_energy);
        writer.WriteElement(repair_group + "/next_warning_fraction",
                            repair.next_warning_fraction);
        writer.WriteElement(
            repair_group + "/residual_correction_limited_groups",
            repair.residual_correction_limited_groups);
        writer.WriteElement(
            repair_group + "/residual_correction_signed_energy_bias",
            repair.residual_correction_signed_energy_bias);
        writer.WriteElement(
            repair_group + "/residual_correction_absolute_energy_bias",
            repair.residual_correction_absolute_energy_bias);
        writer.WriteElement(
            repair_group + "/residual_correction_signed_bias_by_group",
            repair.residual_correction_signed_bias_by_group);
        writer.WriteElement(
            repair_group + "/residual_correction_absolute_bias_by_group",
            repair.residual_correction_absolute_bias_by_group);
        writer.WriteElement(
            repair_group + "/residual_correction_minimum_scale",
            repair.residual_correction_minimum_scale);
        writer.WriteElement(repair_group + "/positivity_rescue_events",
                            repair.positivity_rescue_events);
        writer.WriteElement(repair_group + "/positivity_rescue_blocks",
                            repair.positivity_rescue_blocks);
        writer.WriteElement(
            repair_group + "/positivity_rescue_additional_iterations",
            repair.positivity_rescue_additional_iterations);
        writer.WriteElement(
            repair_group + "/residual_positive_floor_events",
            repair.residual_positive_floor_events);
        writer.WriteElement(
            repair_group + "/residual_positive_floor_cells",
            repair.residual_positive_floor_cells);
        writer.WriteElement(
            repair_group + "/residual_positive_floor_groups",
            repair.residual_positive_floor_groups);
        writer.WriteElement(
            repair_group +
                "/residual_positive_floor_cumulative_injected_energy",
            repair.residual_positive_floor_cumulative_injected_energy);
        writer.WriteElement(
            repair_group +
                "/residual_positive_floor_maximum_cell_injection_ratio",
            repair.residual_positive_floor_maximum_cell_injection_ratio);
        writer.WriteElement(
            repair_group +
                "/residual_positive_floor_maximum_global_injection_ratio",
            repair.residual_positive_floor_maximum_global_injection_ratio);
        writer.WriteElement(
            repair_group +
                "/residual_positive_floor_maximum_post_true_residual_error",
            repair.residual_positive_floor_maximum_post_true_residual_error);
        writer.WriteElement(
            repair_group +
                "/residual_positive_floor_initial_global_radiation_energy",
            repair.residual_positive_floor_initial_global_radiation_energy);

        IndividualRadiationDefectAccounting const& defect =
            scheduler->radiationDefectAccounting();
        std::string const defect_group = group + "/radiation_defect";
        double const cumulative_signed_extent_hi =
            static_cast<double>(defect.cumulative_signed_extent);
        double const cumulative_signed_extent_lo = static_cast<double>(
            defect.cumulative_signed_extent -
            static_cast<long double>(cumulative_signed_extent_hi));
        double const cumulative_absolute_extent_hi =
            static_cast<double>(defect.cumulative_absolute_extent);
        double const cumulative_absolute_extent_lo = static_cast<double>(
            defect.cumulative_absolute_extent -
            static_cast<long double>(cumulative_absolute_extent_hi));
        writer.WriteElement(defect_group + "/cumulative_signed_extent_hi",
            cumulative_signed_extent_hi);
        writer.WriteElement(defect_group + "/cumulative_signed_extent_lo",
            cumulative_signed_extent_lo);
        writer.WriteElement(defect_group + "/cumulative_absolute_extent_hi",
            cumulative_absolute_extent_hi);
        writer.WriteElement(defect_group + "/cumulative_absolute_extent_lo",
            cumulative_absolute_extent_lo);
        writer.WriteElement(defect_group + "/initial_positive_global_extent",
            defect.initial_positive_global_extent);
        writer.WriteElement(defect_group + "/last_normalization_scale",
            defect.last_normalization_scale);
        writer.WriteElement(defect_group + "/maximum_event_absolute_fraction",
            defect.maximum_event_absolute_fraction);
        writer.WriteElement(defect_group + "/maximum_local_fraction",
            defect.maximum_local_fraction);
        writer.WriteElement(defect_group + "/accepted_dirichlet_candidates",
            defect.accepted_dirichlet_candidates);
        writer.WriteElement(defect_group + "/defect_rejections",
            defect.defect_rejections);
        writer.WriteElement(defect_group + "/defect_retry_substeps",
            defect.defect_retry_substeps);
        writer.WriteElement(defect_group + "/config_version",
            defect.config_version);
        writer.WriteElement(defect_group + "/local_withdrawal_limit",
            defect.local_withdrawal_limit);
        writer.WriteElement(defect_group + "/event_absolute_target",
            defect.event_absolute_target);
        writer.WriteElement(defect_group + "/cumulative_signed_limit",
            defect.cumulative_signed_limit);
        writer.WriteElement(defect_group + "/cumulative_absolute_limit",
            defect.cumulative_absolute_limit);
        writer.WriteElement(defect_group + "/cooldown_accepted_candidates",
            defect.cooldown_accepted_candidates);
        writer.WriteElement(defect_group + "/cooldown_required_candidates",
            defect.cooldown_required_candidates);
        writer.WriteElement(defect_group + "/cooldown_fraction_ceiling",
            defect.cooldown_fraction_ceiling);
        writer.WriteElement(defect_group + "/history_complete",
            static_cast<std::uint8_t>(defect.history_complete));

        std::vector<CellTimeState> const& states = scheduler->states();
        std::vector<std::uint64_t> ids(states.size()), begin(states.size()),
            end(states.size()), primitive(states.size());
        std::vector<std::uint8_t> bins(states.size()), gravity_phase(states.size());
        std::vector<Vector3D> point_velocity(states.size()), acceleration(states.size());
        for(std::size_t i = 0; i < states.size(); ++i)
        {
            ids[i] = static_cast<std::uint64_t>(states[i].cell_id);
            begin[i] = states[i].begin_tick;
            end[i] = states[i].end_tick;
            primitive[i] = states[i].last_primitive_tick;
            bins[i] = states[i].time_bin;
            gravity_phase[i] = states[i].gravity_half_kick_pending ? 1 : 0;
            point_velocity[i] = states[i].point_velocity;
            acceleration[i] = states[i].cached_acceleration;
        }
        writer.WriteElement(group + "/cell_ids", ids);
        writer.WriteElement(group + "/begin_ticks", begin);
        writer.WriteElement(group + "/end_ticks", end);
        writer.WriteElement(group + "/primitive_ticks", primitive);
        writer.WriteElement(group + "/bins", bins);
        writer.WriteElement(group + "/point_velocity", point_velocity);
        writer.WriteElement(group + "/cached_acceleration", acceleration);
        writer.WriteElement(group + "/gravity_half_kick_pending", gravity_phase);
        writer.WriteElement(group + "/force_all_active_latched",
            static_cast<std::uint8_t>(scheduler->forceAllActiveLatched()));
    }

    #ifdef RICH_MPI
    void writeLoadBalancers(HDF5Writer &writer, const Simulation &sim)
    {
        auto loads = sim.GetLoads();
        for(const auto &[name, lb] : loads)
        {
            LoadBalancerIO::writeLoadBalancer(writer,
                "/load_balance/" + SerializedLoadBalanceName(name), *lb);
        }
    }
    #endif

    void writeTessellation(HDF5Writer &writer, const std::string &prefix, const Simulation &sim)
    {
        const Tessellation3D &tess = sim.getTessellation();
        size_t N = tess.GetPointNo();

        if(sim.GetTimeIntegrationMode() == TimeIntegrationMode::Individual)
        {
            N = sim.getCells().size();
            std::vector<Vector3D> points;
            for(const std::shared_ptr<PhysicsStep>& step : sim.getPhysicsSteps())
            {
                std::vector<Vector3D> candidate;
                if(step->getIndividualGeneratorPoints(candidate) &&
                   candidate.size() >= N)
                {
                    points.swap(candidate);
                    break;
                }
            }
            if(points.size() < N)
                points = tess.getAllPoints();
            if(points.size() < N)
                throw UniversalError("Individual snapshot is missing canonical generator positions");
            writer.WriteSlice(prefix + "/mesh_points", points, N);
            return;
        }

        writer.WriteSlice(prefix + "/mesh_points", tess.getMeshPoints(), N);
        writer.WriteSlice(prefix + "/volumes", tess.GetAllVolumes(), N);
        writer.WriteSlice(prefix + "/CM", tess.GetAllCM(), N);

#ifdef RICH_MPI
        const Voronoi3D *voronoi = dynamic_cast<const Voronoi3D*>(&tess);
        if(voronoi)
        {
            auto pm = voronoi->GetPointsManager();
            if(pm)
            {
                PointsManagerIO::writePointsManager(writer, prefix + "/points_manager", *pm);
            }
        }
#endif
    }

    void writePhysicsGroups(HDF5Writer &writer, const std::string &prefix, const Simulation &sim)
    {
        for(const auto &step : sim.getPhysicsSteps())
        {
            PhysicsStepIO::writeStep(writer, prefix, *step);
        }
    }

    void writePrivateInfo(HDF5Writer &writer, const std::string &prefix, const Simulation &sim)
    {
        const std::vector<ComputationalCell3D> &cells = sim.getCells();
        size_t const count = sim.GetTimeIntegrationMode() == TimeIntegrationMode::Individual
            ? cells.size()
            : sim.getTessellation().GetPointNo();
        writer.WriteSlice(prefix + "/cells", cells, count);
        std::vector<HDF5Utils::Conserved3DRecord> records(count);
        std::vector<Conserved3D> const& extensives = sim.getExtensives();
        if(extensives.size() < count)
            throw UniversalError("Snapshot has fewer conserved extents than cells");
        for(std::size_t i = 0; i < count; ++i)
            records[i] = HDF5Utils::PackConserved3D(extensives[i]);
        writer.WriteSlice(prefix + "/extensives", records, count);
    }
} // anonymous namespace

void WriteSimulation(const Simulation &sim, const std::string &filename
                     #ifdef RICH_MPI
                         , bool parallel
                     #endif
                     )
{
    MEMORY_DEBUG_PRINT("WriteSimulation: before");
    if(sim.IndividualEventInProgress())
        throw UniversalError("Individual snapshots may only be written after a completed event");
    #ifdef RICH_MPI
        int rank = 0, ws = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ws);
    #endif

    #ifdef RICH_MPI
    if(parallel)
    {
        fs::path path = fs::absolute(filename).parent_path();
        fs::path ranks_dir = path / fs::path(filename).filename().replace_extension();

        if(rank == 0)
        {
            if(fs::exists(ranks_dir))
            {
                fs::remove_all(ranks_dir);
            }
            fs::create_directory(ranks_dir);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        std::string myFile = (ranks_dir / std::to_string(rank)).string() + ".h5";
        {
            HDF5Writer rankWriter = openWriter(myFile);
            writePrivateInfo(rankWriter, "", sim);
            writeTessellation(rankWriter, "/tess", sim);
            writePhysicsGroups(rankWriter, "", sim);
            writeIndividualTimeSteps(rankWriter, sim);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if(rank == 0)
        {
            HDF5Writer globalWriter = openWriter(filename);
            writeGeneralInfo(globalWriter, sim);
            writeIndividualTimeSteps(globalWriter, sim);
            writeLoadBalancers(globalWriter, sim);
            for(int r = 0; r < ws; ++r)
            {
                std::string relRankFile = ranks_dir.filename().string() + "/" + std::to_string(r) + ".h5";
                globalWriter.AddExternalLink(relRankFile, "/", "/rank" + std::to_string(r));
            }
        }
    }
    else
    #endif
    {
        HDF5Writer writer = openWriter(filename);
        writeGeneralInfo(writer, sim);
        writeIndividualTimeSteps(writer, sim);
        writePrivateInfo(writer, "", sim);
        writeTessellation(writer, "/tess", sim);
        writePhysicsGroups(writer, "", sim);
    }
    MEMORY_DEBUG_PRINT("WriteSimulation: after");
}
