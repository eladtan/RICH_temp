#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/HydroStep.hpp"
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/misc/simple_io.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/newtonian/three_dimensional/Lagrangian3D.hpp"
#include "source/newtonian/three_dimensional/RoundCells3D.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/Ghost3D.hpp"
#include "source/newtonian/three_dimensional/ConservativeForce3D.hpp"
#include "source/newtonian/three_dimensional/GravityAcc3D.hpp"
#include "source/3D/output/read3D.hpp"
#include "source/3D/output/write3D.hpp"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <unistd.h>

#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace
{

size_t EnvironmentSize(char const* name, size_t fallback)
{
    char const* value = std::getenv(name);
    if(value == nullptr || value[0] == '\0')
        return fallback;
    return static_cast<size_t>(std::stoull(value));
}

double EnvironmentDouble(char const* name, double fallback)
{
    char const* value = std::getenv(name);
    if(value == nullptr || value[0] == '\0')
        return fallback;
    return std::stod(value);
}

struct LaneEmdenProfile
{
    vector<double> xsi, theta;
    double n, alpha, rho_c, K;

    LaneEmdenProfile(double M, double R, double G)
    {
        xsi = read_vector("../../../data/xsi32.txt");
        theta = read_vector("../../../data/theta32.txt");
        xsi[0] = 0;
        n = 1.5;
        double endfactor = 2.714;
        alpha = R / xsi.back();
        rho_c = M / (4 * M_PI * alpha * alpha * alpha * endfactor);
        K = G * alpha * alpha * 4 * M_PI / ((n + 1) * std::pow(rho_c, 1.0 / n - 1));
    }

    double densityAt(double r, double R) const
    {
        if (r < R)
        {
            double t = LinearInterpolation(xsi, theta, r / alpha);
            return std::max(rho_c * std::pow(t, n), 1e-5);
        }
        else
        {
            double t = theta.back();
            return rho_c * std::pow(t, n);
        }
    }
};

std::vector<ComputationalCell3D> GetCells(Tessellation3D const &tess, double R,
                                          IdealGas const &eos,
                                          LaneEmdenProfile const &prof)
{
    size_t N = tess.GetPointNo();
    std::vector<ComputationalCell3D> res(N);

    for (size_t i = 0; i < N; ++i)
    {
        Vector3D const &point = tess.GetMeshPoint(i);
        double r = abs(point);
        res[i].density = prof.densityAt(r, R);
        double const P = prof.K * std::pow(res[i].density, 1 + 1.0 / prof.n);
        res[i].pressure = P;
        res[i].internal_energy = eos.dp2e(res[i].density, P, res[i].tracers,
                                          ComputationalCell3D::tracerNames);
    }

    return res;
}
}

int main(void)
{
    int rank = 0;
    int ws = 1;

#ifdef RICH_MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ws);
#endif

    if (rank == 0)
    {
        char cwd_buf[4096];
        if (getcwd(cwd_buf, sizeof(cwd_buf)))
            std::cerr << "CWD: " << cwd_buf << std::endl;
    }

    double const R = 7e10;
    double const M = 2e33;
    double const G = 6.674e-8;

    const double width = 2 * R;
    size_t np = EnvironmentSize("RICH_TEST_POINT_COUNT", static_cast<size_t>(2e6));

    Vector3D ll(-width, -width, -width), ur(width, width, width);
    Voronoi3D tess(ll, ur);

    std::vector<ComputationalCell3D> cells;
    vector<Vector3D> points;

    if (rank == 0)
    {
        size_t np_main = np * 4 / 7;
        size_t np_mid = np * 2 / 7;
        size_t np_far = np - np_main - np_mid;

        points = RandSphereR(np_main, ll, ur, 0, R * 1.1);

        vector<Vector3D> ptemp2 = RandSphereR(np_mid, ll, ur, 0.8 * R, R * 1.05);
        vector<Vector3D> ptemp3 = RandSphereR2(np_far, ll, ur, R, 1.4 * width);

        points.insert(points.end(), ptemp2.begin(), ptemp2.end());
        points.insert(points.end(), ptemp3.begin(), ptemp3.end());

        std::cout << "Total points: " << points.size()
                  << " (main=" << np_main << ", mid=" << np_mid
                  << ", far=" << np_far << ")" << std::endl;
    }

#ifdef RICH_MPI
    points = MPI_Spread(points, 0, MPI_COMM_WORLD);
#endif

    points = RoundGrid3D(points, ll, ur, 10);

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

    if (rank == 0)
        std::cerr << "Finished build" << std::endl;

    IdealGas eos(5.0 / 3.0);

    LaneEmdenProfile prof(M, R, G);
    cells = GetCells(tess, R, eos, prof);
    if (rank == 0)
        std::cerr << "Finished cells" << std::endl;

    Hllc3D rs;
    RigidWallGenerator3D ghost;
    LinearGauss3D interp(eos, ghost);

    Lagrangian3D bpm;
    RoundCells3D pm(bpm, eos);
    PointMotion3D const& point_motion =
        std::getenv("RICH_TEST_DISABLE_ROUNDING") != nullptr ?
        static_cast<PointMotion3D const&>(bpm) :
        static_cast<PointMotion3D const&>(pm);

    DefaultCellUpdater cu;

    RigidWallFlux3D rigidflux(rs);
    RegularFlux3D *regular_flux = new RegularFlux3D(rs);
    IsBoundaryFace3D *boundary_face = new IsBoundaryFace3D();
    IsBulkFace3D *bulk_face = new IsBulkFace3D();
    vector<pair<const ConditionActionFlux1::Condition3D *, const ConditionActionFlux1::Action3D *>> flux_vector;
    flux_vector.push_back(pair<const ConditionActionFlux1::Condition3D *,
                          const ConditionActionFlux1::Action3D *>(boundary_face, &rigidflux));
    flux_vector.push_back(pair<const ConditionActionFlux1::Condition3D *,
                          const ConditionActionFlux1::Action3D *>(bulk_face, regular_flux));
    ConditionActionFlux1 fc(flux_vector, interp);

    vector<pair<const ConditionExtensiveUpdater3D::Condition3D *, const ConditionExtensiveUpdater3D::Action3D *>> eu_sequence;
    ConditionExtensiveUpdater3D eu(eu_sequence);

    double const gravity_strength =
        std::getenv("RICH_TEST_DISABLE_GRAVITY") != nullptr ? 0.0 : G;
    GravityAcceleration3D acc(0.7, true, gravity_strength);
    ConservativeForce3D force(acc);

    auto tsf = std::make_shared<CourantFriedrichsLewy>(0.25, 1, force, std::vector<std::string>(), false);

    Simulation simulation(tess, cells, eos);
    simulation.SetTimeStepFunction(tsf);
    HDSim3D sim(tess, simulation.getCells(), simulation.getExtensives(), eos, simulation.getTracker(), point_motion, *tsf, fc, cu, eu, force,
                std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));

    auto hydroStep = std::make_shared<HydroStep>(sim, HydroStep::TIMEADVANCE_2);
    simulation.addPhysics(hydroStep);
    double const initial_dt = EnvironmentDouble("RICH_TEST_INITIAL_DT", 1.0);
    simulation.SetTimeStep(initial_dt);

    char const* restart_input = std::getenv("RICH_TEST_RESTART_INPUT");
    char const* individual_mode = std::getenv("RICH_INDIVIDUAL_MODE");
    if(restart_input != nullptr && restart_input[0] != '\0')
        ReadSimulation(restart_input, simulation);
    else if(individual_mode != nullptr && individual_mode[0] != '\0')
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
        options.time_quantum = std::ldexp(initial_dt,
            -static_cast<int>(options.initial_bin));
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
            if(states.empty())
                throw std::logic_error(
                    "RICH_TEST_SPARSE_INITIAL_BIN requires at least one cell");
            CellTimeState& sparse_state = states[states.size() / 2];
            sparse_state.time_bin = static_cast<std::uint8_t>(
                options.initial_bin - 2);
            sparse_state.end_tick = std::uint64_t(1) << sparse_state.time_bin;
        }
    }
#ifdef RICH_MPI
    simulation.PresetLoadBalance("hydro");
#endif

    double old_t = simulation.GetTime();
    double metric = 0;

#ifdef RICH_MPI
    double step_tstart = MPI_Wtime();
#endif

    size_t const max_cycles = EnvironmentSize("RICH_TEST_MAX_CYCLES",
        std::numeric_limits<size_t>::max());
    while (simulation.GetTime() < 5000.0 &&
           simulation.GetCycle() < max_cycles)
    {
        try
        {
            simulation.step();

            double dt = simulation.GetTime() - old_t;
            old_t = simulation.GetTime();

#ifdef RICH_MPI
            double step_tend = MPI_Wtime();
            double wall_elapsed = step_tend - step_tstart;
            step_tstart = step_tend;
#else
            double wall_elapsed = 0;
#endif

            size_t N_now = tess.GetPointNo();
            auto const& cur_cells = simulation.getCells();
            double local_sum = 0, local_volume = 0;
            double local_count = 0;
            if(N_now == cur_cells.size())
                for (size_t i = 0; i < N_now; ++i)
                {
                    if (cur_cells[i].density > 1e-2)
                    {
                        double r = abs(tess.GetCellCM(i));
                        double rho0 = prof.densityAt(r, R);
                        local_sum += tess.GetVolume(i) * std::abs(cur_cells[i].density - rho0);
                        local_volume += tess.GetVolume(i);
                        ++local_count;
                    }
                }
            
            double global_count = local_count;
            double global_sum = local_sum;
            double global_volume = local_volume;
#ifdef RICH_MPI
            MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&local_count, &global_count, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&local_volume, &global_volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
            if(global_volume > 0)
                metric = global_sum / global_volume;

            if (rank == 0)
            {
                std::cout<<endl;
                std::cout << "Cycle " << simulation.GetCycle()
                          << " Time " << simulation.GetTime()
                          << " dt " << dt
                          << " WallTime " << wall_elapsed
                          << " Metric " << metric << std::endl;
                std::cerr << "Cycle " << simulation.GetCycle()
                          << " Time " << simulation.GetTime()
                          << " dt " << dt
                          << " Metric " << metric << std::endl;
            }
        }
        catch (UniversalError const &eo)
        {
            reportError(eo);
            throw;
        }
    }

    char const* restart_output = std::getenv("RICH_TEST_RESTART_OUTPUT");
    if(restart_output != nullptr && restart_output[0] != '\0')
        WriteSimulation(simulation, restart_output);

    // Global diagnostics require canonical point ordering, not a partial view.
#ifndef RICH_MPI
    if(tess.GetPointNo() != simulation.getCells().size())
    {
        std::vector<Vector3D> output_points = tess.getAllPoints();
        output_points.resize(simulation.getCells().size());
        tess.Build(output_points);
    }

    {
        double final_sum = 0;
        double final_volume = 0;
        auto const& final_cells = simulation.getCells();
        for(size_t i = 0; i < tess.GetPointNo(); ++i)
            if(final_cells[i].density > 1e-2)
            {
                double const volume = tess.GetVolume(i);
                double const rho0 = prof.densityAt(abs(tess.GetCellCM(i)), R);
                final_sum += volume * std::abs(final_cells[i].density - rho0);
                final_volume += volume;
            }
        metric = final_volume > 0 ? final_sum / final_volume : 0;
    }
#endif

    char output_cwd[4096];
    if(getcwd(output_cwd, sizeof(output_cwd)) == nullptr)
        throw std::runtime_error("Failed to resolve current working directory");
    char const* artifact_dir = std::getenv("THUNDER_ARTIFACT_DIR");
    std::string const output_dir = artifact_dir != nullptr && artifact_dir[0] != '\0'
        ? std::string(artifact_dir) : std::string(output_cwd);
    std::string const profile_path = output_dir + "/lane_profile.txt";

#ifndef RICH_MPI
    if(rank == 0)
    {
        std::ofstream state_out(output_dir + "/lane_state.txt");
        state_out << std::hexfloat;
        auto const& final_cells = simulation.getCells();
        auto const& final_extensives = simulation.getExtensives();
        for(size_t i = 0; i < final_cells.size(); ++i)
        {
            ComputationalCell3D const& cell = final_cells[i];
            Conserved3D const& extensive = final_extensives[i];
            Vector3D const point = tess.GetMeshPoint(i);
            state_out << cell.ID << ' ' << point.x << ' ' << point.y << ' '
                      << point.z << ' ' << cell.density << ' ' << cell.pressure
                      << ' ' << cell.internal_energy << ' ' << cell.velocity.x
                      << ' ' << cell.velocity.y << ' ' << cell.velocity.z << ' '
                      << extensive.mass << ' ' << extensive.momentum.x << ' '
                      << extensive.momentum.y << ' ' << extensive.momentum.z
                      << ' ' << extensive.energy << ' '
                      << extensive.internal_energy << '\n';
        }

        IndividualTimeStepScheduler const* scheduler =
            simulation.GetIndividualTimeStepScheduler();
        if(scheduler != nullptr && scheduler->initialized())
        {
            std::ofstream scheduler_out(output_dir + "/lane_scheduler.txt");
            scheduler_out << std::hexfloat << scheduler->timeOrigin() << ' '
                          << scheduler->timeQuantum() << ' '
                          << scheduler->currentTick() << '\n';
            for(CellTimeState const& state : scheduler->states())
                scheduler_out << state.cell_id << ' ' << state.begin_tick << ' '
                              << state.end_tick << ' ' << state.last_primitive_tick
                              << ' ' << static_cast<unsigned int>(state.time_bin) << ' '
                              << state.point_velocity.x << ' '
                              << state.point_velocity.y << ' '
                              << state.point_velocity.z << ' '
                              << state.cached_acceleration.x << ' '
                              << state.cached_acceleration.y << ' '
                              << state.cached_acceleration.z << ' '
                              << static_cast<unsigned int>(
                                     state.gravity_half_kick_pending) << '\n';
        }
    }
#endif

    // Radial profile output for plotting
    {
        size_t const nbins = 300;
        std::vector<double> r_sum(nbins, 0.0);
        std::vector<double> density_sum(nbins, 0.0);
        std::vector<double> density_analytic_sum(nbins, 0.0);
        std::vector<double> volume_sum(nbins, 0.0);

        double rmax_local = 0.0;
        size_t N_final = tess.GetPointNo();
        for (size_t i = 0; i < N_final; ++i)
        {
            double const r = abs(tess.GetCellCM(i));
            rmax_local = std::max(rmax_local, r);
        }
        double rmax = rmax_local;
#ifdef RICH_MPI
        MPI_Allreduce(&rmax_local, &rmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
        if (rmax <= 0.0)
            rmax = 1.0;

        auto const& out_cells = simulation.getCells();
        for (size_t i = 0; i < N_final; ++i)
        {
            Vector3D const cm = tess.GetCellCM(i);
            double const r = abs(cm);
            size_t bin = static_cast<size_t>((r / rmax) * static_cast<double>(nbins));
            if (bin >= nbins)
                bin = nbins - 1;
            double const cell_volume = tess.GetVolume(i);
            r_sum[bin] += r * cell_volume;
            density_sum[bin] += out_cells[i].density * cell_volume;
            density_analytic_sum[bin] += prof.densityAt(r, R) * cell_volume;
            volume_sum[bin] += cell_volume;
        }

#ifdef RICH_MPI
        std::vector<double> r_sum_g(nbins, 0.0);
        std::vector<double> density_sum_g(nbins, 0.0);
        std::vector<double> density_analytic_sum_g(nbins, 0.0);
        std::vector<double> volume_sum_g(nbins, 0.0);
        MPI_Reduce(r_sum.data(), r_sum_g.data(), static_cast<int>(nbins), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(density_sum.data(), density_sum_g.data(), static_cast<int>(nbins), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(density_analytic_sum.data(), density_analytic_sum_g.data(), static_cast<int>(nbins), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(volume_sum.data(), volume_sum_g.data(), static_cast<int>(nbins), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            std::ofstream prof_out(profile_path);
            for (size_t b = 0; b < nbins; ++b)
            {
                if (volume_sum_g[b] <= 0.0)
                    continue;
                double const inv = 1.0 / volume_sum_g[b];
                prof_out << r_sum_g[b] * inv << " "
                         << density_sum_g[b] * inv << " "
                         << density_analytic_sum_g[b] * inv << "\n";
            }
            prof_out.close();
            std::cerr << "Wrote " << profile_path << std::endl;
        }
#else
        std::ofstream prof_out(profile_path);
        for (size_t b = 0; b < nbins; ++b)
        {
            if (volume_sum[b] <= 0.0)
                continue;
            double const inv = 1.0 / volume_sum[b];
            prof_out << r_sum[b] * inv << " "
                     << density_sum[b] * inv << " "
                     << density_analytic_sum[b] * inv << "\n";
        }
        prof_out.close();
#endif
    }

    bool pass = std::abs(metric) < 4e-2;

    if (rank == 0)
    {
        std::string const metrics_path = output_dir + "/lane_gravity_metrics.txt";

        std::ofstream out(metrics_path);
        if (!out.is_open())
        {
            std::cerr << "ERROR: could not open " << metrics_path << " for writing" << std::endl;
        }
        else
        {
            out << "final_metric " << metric << "\n";
            out << "pass " << (pass ? 1 : 0) << "\n";
            out.close();
            std::cerr << "Wrote metrics to " << metrics_path << std::endl;
        }

        std::cout << "Final metric: " << metric
                  << " pass: " << (pass ? "yes" : "no") << std::endl;
    }

#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return 0;
}
