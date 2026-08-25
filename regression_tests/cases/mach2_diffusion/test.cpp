#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/newtonian/three_dimensional/eulerian_3d.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/default_extensive_updater.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/HydroStep.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/ConservativeForce3D.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/Radiation/Diffusion.hpp"
#include <libgen.h>
#include <string.h>
#include "source/Radiation/DiffusionForce.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationStep.hpp"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>

namespace
{
    size_t EnvironmentSize(char const* name, size_t fallback)
    {
        char const* value = std::getenv(name);
        if(value == nullptr || value[0] == '\0')
            return fallback;
        char* end = nullptr;
        unsigned long long const parsed = std::strtoull(value, &end, 10);
        if(end == value || *end != '\0')
            throw std::invalid_argument(std::string("Invalid integer in ") + name);
        return static_cast<size_t>(parsed);
    }

    class IsPointLeftRightBox3D : public ConditionActionFlux1::Condition3D
    {
    public:
        pair<bool, bool> operator()(size_t face_index, const Tessellation3D& tess,
            const vector<ComputationalCell3D>& cells) const override
        {
            if (!tess.BoundaryFace(face_index))
                return std::pair<bool, bool>(false, false);

            auto const& box = tess.GetBoxCoordinates();
            Vector3D const& first_point = tess.GetMeshPoint(tess.GetFaceNeighbors(face_index).first);
            Vector3D const& second_point = tess.GetMeshPoint(tess.GetFaceNeighbors(face_index).second);

            const bool is_left = first_point.x < box.first.x || first_point.x > box.second.x;
            const bool is_right = second_point.x < box.first.x || second_point.x > box.second.x;

            return std::make_pair(is_left || is_right, is_right);
        }
    };

    class GhostChooser: public SeveralGhostGenerator3D::GhostCriteria3D
    {
    public:
        size_t GhostChoose(Tessellation3D const& tess, size_t index) const
        {
            auto const& box = tess.GetBoxCoordinates();
            Vector3D const& p = tess.GetMeshPoint(index);
            if (p.x < box.first.x)
                return 0;
            if (p.x > box.second.x)
                return 1;
            else
                return 2;
        }
    };
}

int main(void)
{
    size_t const Np = EnvironmentSize("RICH_TEST_POINT_COUNT", 1024);
    double const box_size = 1e3;
    double const dy = 3 * box_size / (2 * Np);
    Vector3D ll(-box_size, -dy, -dy), ur(2 * box_size, dy, dy);
    int rank = 0;
    int nprocs = 1;
#ifdef RICH_MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#endif

    std::vector<Vector3D> points;
    if (rank == 0)
        points = CartesianMesh(Np, 1, 1, ll, ur);
#ifdef RICH_MPI
    points = MPI_Spread(points, 0, MPI_COMM_WORLD);
#endif

    Voronoi3D tess(ll, ur);
    try {
#ifdef RICH_MPI
        tess.BuildParallel(points);
#else
        tess.Build(points);
#endif
    } catch (UniversalError const& eo) {
        reportError(eo);
        throw;
    }

    IdealGas eos(5./3., CG::boltzmann_constant / (1.67e-24 * (5.0 / 3.0 - 1)), 1, 0);

    size_t const Nlocal = tess.GetPointNo();
#ifdef RICH_MPI
    unsigned long long owned = static_cast<unsigned long long>(Nlocal);
    unsigned long long minimum_owned = owned;
    unsigned long long maximum_owned = owned;
    unsigned long long empty_ranks = owned == 0 ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &minimum_owned, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maximum_owned, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &empty_ranks, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    if(rank == 0)
        std::clog << "GREY_MPI_OWNED_CELL_RANGE minimum=" << minimum_owned
                  << " maximum=" << maximum_owned
                  << " empty_ranks=" << empty_ranks
                  << " ranks=" << nprocs << '\n';
#endif
    std::vector<ComputationalCell3D> cells(Nlocal);
    ComputationalCell3D left_cell, right_cell;
    left_cell.velocity = Vector3D(2.3547e5, 0, 0);
    left_cell.density = 5.45887e-13;
    left_cell.temperature = 100;
    left_cell.internal_energy = eos.dT2e(left_cell.density, left_cell.temperature, left_cell.tracers, ComputationalCell3D::tracerNames);
    left_cell.pressure = eos.de2p(left_cell.density, left_cell.internal_energy, left_cell.tracers, ComputationalCell3D::tracerNames);
    left_cell.Erad = CG::radiation_constant * std::pow(left_cell.temperature, 4) / left_cell.density;

    right_cell.velocity = Vector3D(1.03e5, 0, 0);
    right_cell.density = 1.2479e-12;
    right_cell.temperature = 207.757;
    right_cell.Erad = CG::radiation_constant * std::pow(right_cell.temperature, 4) / right_cell.density;
    right_cell.internal_energy = eos.dT2e(right_cell.density, right_cell.temperature, right_cell.tracers, ComputationalCell3D::tracerNames);
    right_cell.pressure = eos.de2p(right_cell.density, right_cell.internal_energy, right_cell.tracers, ComputationalCell3D::tracerNames);

    for (size_t i = 0; i < Nlocal; ++i)
    {
        if (tess.GetMeshPoint(i).x < 0)
            cells[i] = left_cell;
        else
            cells[i] = right_cell;
    }

    Hllc3D rs;

    RigidWallGenerator3D rigid_ghost;
    ConstantPrimitiveGenerator3D left_ghost(left_cell), right_ghost(right_cell);
    std::vector<Ghost3D*> ghost_list = {&left_ghost, &right_ghost, &rigid_ghost};
    GhostChooser ghost_chooser;
    SeveralGhostGenerator3D ghost(ghost_list, ghost_chooser);

    LinearGauss3D interp(eos, ghost);

    std::vector<pair<const ConditionActionFlux1::Condition3D*,
        const ConditionActionFlux1::Action3D*>> sequence;
    ConditionActionFlux1::Condition3D* isbulk = new IsBulkFace3D();
    ConditionActionFlux1::Condition3D* is_side = new IsPointLeftRightBox3D();
    ConditionActionFlux1::Condition3D* isboundary = new IsBoundaryFace3D();
    ConditionActionFlux1::Action3D* normal_flux = new RegularFlux3D(rs);
    ConditionActionFlux1::Action3D* rigid_flux = new RigidWallFlux3D(rs);
    sequence.push_back(std::pair<const ConditionActionFlux1::Condition3D*,
        const ConditionActionFlux1::Action3D*>(is_side, normal_flux));
    sequence.push_back(std::pair<const ConditionActionFlux1::Condition3D*,
        const ConditionActionFlux1::Action3D*>(isboundary, rigid_flux));
    sequence.push_back(std::pair<const ConditionActionFlux1::Condition3D*,
        const ConditionActionFlux1::Action3D*>(isbulk, normal_flux));
    ConditionActionFlux1 flux(sequence, interp);

    std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D*, const ConditionExtensiveUpdater3D::Action3D*>> eu_sequence;
    ConditionExtensiveUpdater3D eu(eu_sequence);

    PowerLawOpacity opacity(CG::speed_of_light / (3 * 0.848902), 0, 0, 3.93e-5, 0, 0);
    DiffusionXInflowBoundary diffusion_boundary(left_cell, right_cell, opacity);
    Diffusion diffusion(opacity, diffusion_boundary, eos);

    DefaultCellUpdater cu(false, 0, true, 0, &diffusion);

    DiffusionForce force(diffusion, eos);

    double const hydro_cfl = 0.3;
    double const force_cfl = 1;
    auto tsf = std::make_shared<CourantFriedrichsLewy>(hydro_cfl, force_cfl, force);

    Eulerian3D pm;

    Simulation simulation(tess, cells, eos);
    simulation.SetTimeStepFunction(tsf);
    HDSim3D sim(tess, simulation.getCells(), simulation.getExtensives(), eos, simulation.getTracker(), pm, *tsf, flux, cu, eu, force,
        std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));

    auto radStep = std::make_shared<RadiationStep>(tess, simulation.getCells(), simulation.getExtensives(),
        simulation.getTracker(),
#ifdef RICH_MPI
        nullptr,
#endif
        diffusion, false);
    auto hydroStep = std::make_shared<HydroStep>(sim, HydroStep::TIMEADVANCE_2);
    simulation.addPhysics(hydroStep);
    simulation.addPhysics(radStep);
    simulation.SetTimeStep(1e-15);

    char const* individual_mode = std::getenv("RICH_INDIVIDUAL_MODE");
    if(individual_mode != nullptr && individual_mode[0] != '\0')
    {
        IndividualTimeStepOptions options;
	        std::string const mode(individual_mode);
	        bool const synchronized = mode == "full";
	        bool const auto_partial = mode == "partial";
	        if(!synchronized && !auto_partial && mode != "full-variable")
	            throw std::invalid_argument(
	                "RICH_INDIVIDUAL_MODE must be 'full', 'full-variable', or 'partial'");
        options.initial_bin = synchronized ? 0 : 4;
	        options.maximum_bin = synchronized ? 0 : 60;
        options.time_quantum = std::ldexp(1e-15,
            -static_cast<int>(options.initial_bin));
	        options.force_synchronized = synchronized;
	        options.mesh_build_policy = auto_partial ?
	            IndividualMeshBuildPolicy::AutoPartial :
	            IndividualMeshBuildPolicy::FullReference;
        options.verify_partial_build = std::getenv(
            "RICH_VERIFY_PARTIAL_BUILD") != nullptr;
        simulation.EnableIndividualTimeSteps(options);

        if(std::getenv("RICH_TEST_SPARSE_INITIAL_BIN") != nullptr)
        {
            if(synchronized || options.initial_bin < 2)
                throw std::invalid_argument(
                    "RICH_TEST_SPARSE_INITIAL_BIN requires variable individual timesteps");
            IndividualTimeStepScheduler* scheduler =
                simulation.GetIndividualTimeStepScheduler();
            scheduler->initialize(simulation.getCells(), simulation.GetTime(),
                                  simulation.GetTimeStep());
            std::vector<CellTimeState>& states = scheduler->states();
            std::size_t sparse_index = states.size();
            bool const activate_reference_maximum = std::getenv(
                "RICH_TEST_SPARSE_MAX_ER_CELL") != nullptr;
            if(activate_reference_maximum)
            {
#ifdef RICH_MPI
                struct
                {
                    double value;
                    int rank;
                } local_maximum = {
                    -std::numeric_limits<double>::infinity(), rank};
#else
                double local_maximum =
                    -std::numeric_limits<double>::infinity();
#endif
                for(std::size_t index = 0; index < states.size(); ++index)
                {
                    double const value = simulation.getCells()[index].Erad *
                        simulation.getCells()[index].density;
#ifdef RICH_MPI
                    if(value > local_maximum.value)
                    {
                        local_maximum.value = value;
                        sparse_index = index;
                    }
#else
                    if(value > local_maximum)
                    {
                        local_maximum = value;
                        sparse_index = index;
                    }
#endif
                }
#ifdef RICH_MPI
                auto global_maximum = local_maximum;
                MPI_Allreduce(&local_maximum, &global_maximum, 1,
                    MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
                if(rank != global_maximum.rank)
                    sparse_index = states.size();
                if(rank == 0)
                    std::clog << "GREY_ACTIVE_REFERENCE_MAX_TEST value="
                              << global_maximum.value << " owner="
                              << global_maximum.rank << '\n';
#else
                std::clog << "GREY_ACTIVE_REFERENCE_MAX_TEST value="
                          << local_maximum << " owner=0\n";
#endif
            }
            else
            {
#ifdef RICH_MPI
                if(rank == 0 && !states.empty())
                    sparse_index = states.size() - 1;
#else
                if(!states.empty())
                    sparse_index = states.size() / 2;
#endif
            }
            if(sparse_index < states.size())
            {
                CellTimeState& sparse_state = states[sparse_index];
                sparse_state.time_bin = static_cast<std::uint8_t>(
                    options.initial_bin - 2);
                sparse_state.end_tick = std::uint64_t(1) << sparse_state.time_bin;
            }
        }
    }

    size_t const maximum_cycles = EnvironmentSize(
        "RICH_TEST_MAX_CYCLES", std::numeric_limits<size_t>::max());

    while (simulation.GetTime() < 0.01 &&
           simulation.GetCycle() < maximum_cycles)
    {
        try
        {
            auto step_start = std::chrono::steady_clock::now();
            double old_time = simulation.GetTime();

            simulation.step();

            double current_dt = simulation.GetTime() - old_time;
            auto step_end = std::chrono::steady_clock::now();
            double wall_sec = std::chrono::duration<double>(step_end - step_start).count();

            if (rank == 0)
            {
                std::cout << std::endl;
                std::cout << "Cycle " << simulation.GetCycle()
                          << " dt " << std::scientific << std::setprecision(6) << current_dt
                          << " time " << simulation.GetTime()
                          << " wall_time " << std::fixed << std::setprecision(3) << wall_sec << "s"
                          << std::endl;
            }
        }
        catch (UniversalError const& eo)
        {
            reportError(eo);
            throw;
        }
    }

    // Compute output path in the same directory as this source file
    char file_buf[4096];
    const char *artifact_dir = getenv("THUNDER_ARTIFACT_DIR");
    std::string case_dir;
    if(artifact_dir && artifact_dir[0] != '\0')
    {
        case_dir = artifact_dir;
    }
    else
    {
        getcwd(file_buf, sizeof(file_buf));
        case_dir = file_buf;
    }
    std::string profile_path = case_dir + "/mach2_profile.txt";

    // Global geometry output must not interpret canonical owned-cell arrays
    // through a partial tessellation's local ordering.
#ifndef RICH_MPI
    if(tess.GetPointNo() != simulation.getCells().size())
    {
        std::vector<Vector3D> output_points = tess.getAllPoints();
        output_points.resize(simulation.getCells().size());
        tess.Build(output_points);
    }
#endif

    // Gather profile data from all MPI ranks and write to file
    {
        size_t const Nfinal = tess.GetPointNo();
        std::vector<double> local_x(Nfinal), local_rho(Nfinal), local_T(Nfinal), local_Trad(Nfinal);
        auto const& final_cells = simulation.getCells();
        for (size_t i = 0; i < Nfinal; ++i)
        {
            local_x[i] = tess.GetMeshPoint(i).x;
            local_rho[i] = final_cells[i].density;
            local_T[i] = final_cells[i].temperature;
            local_Trad[i] = std::pow(final_cells[i].Erad * final_cells[i].density / CG::radiation_constant, 0.25);
        }

#ifdef RICH_MPI
        int local_n = static_cast<int>(Nfinal);
        std::vector<int> recv_counts(nprocs, 0);
        MPI_Gather(&local_n, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> displs(nprocs, 0);
        int total_n = 0;
        if (rank == 0)
        {
            for (int i = 0; i < nprocs; ++i)
            {
                displs[i] = total_n;
                total_n += recv_counts[i];
            }
        }

        std::vector<double> all_x, all_rho, all_T, all_Trad;
        if (rank == 0)
        {
            all_x.resize(total_n);
            all_rho.resize(total_n);
            all_T.resize(total_n);
            all_Trad.resize(total_n);
        }
        MPI_Gatherv(local_x.data(), local_n, MPI_DOUBLE, all_x.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_rho.data(), local_n, MPI_DOUBLE, all_rho.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_T.data(), local_n, MPI_DOUBLE, all_T.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_Trad.data(), local_n, MPI_DOUBLE, all_Trad.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            std::vector<size_t> idx(total_n);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return all_x[a] < all_x[b]; });

            std::ofstream out(profile_path);
            out << std::scientific << std::setprecision(12);
            for (int i = 0; i < total_n; ++i)
            {
                out << all_x[idx[i]] << " " << all_rho[idx[i]] << " " << all_T[idx[i]] << " " << all_Trad[idx[i]] << "\n";
            }
            out.close();
        }
#else
        std::vector<size_t> idx(Nfinal);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return local_x[a] < local_x[b]; });

        std::ofstream out(profile_path);
        out << std::scientific << std::setprecision(12);
        for (size_t i = 0; i < Nfinal; ++i)
        {
            out << local_x[idx[i]] << " " << local_rho[idx[i]] << " " << local_T[idx[i]] << " " << local_Trad[idx[i]] << "\n";
        }
        out.close();
#endif
    }

#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return 0;
}
