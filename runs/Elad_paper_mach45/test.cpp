#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include "3D/radiation/MonteCarloPhysics3D.hpp"
#include "mpi/mpi_commands.hpp"
#include "misc/mesh_generator3D.hpp"
#include "3D/tessellation/Voronoi3D.hpp"
#ifdef RICH_MPI
#include <MeshDecomposer3D/load_balancing/OneDimensionalLoadBalancer.hpp>
#endif
#include "CMMC/src/units/units.hpp"
#include "newtonian/common/ideal_gas.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "newtonian/three_dimensional/simulation/Simulation.hpp"
#include "newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "newtonian/three_dimensional/ManualTimeStep.hpp"

#include "newtonian/three_dimensional/hdsim_3d.hpp"
#include "newtonian/three_dimensional/eulerian_3d.hpp"
#include "newtonian/three_dimensional/Hllc3D.hpp"
#include "newtonian/three_dimensional/LinearGauss3D.hpp"
#include "newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "newtonian/three_dimensional/default_cell_updater.hpp"
#include "newtonian/three_dimensional/simulation/steps/HydroStep.hpp"

#include "3D/output/write3D.hpp"
#include "3D/output/read3D.hpp"

#include "3D/radiation/RadiationIMC.hpp"
#include "3D/radiation/PowerLawOpacity.hpp"
#include "monte/population/CombPopulationControl.hpp"
#include "monte/boundary/TwoSidesTemperature.hpp"
#include "newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"
#include "utils/arguments/ArgumentParser.hpp"
#include "runs/mc_results_dir.hpp"

#ifdef RICH_MPI
    #include <mpi.h>
    #include "newtonian/three_dimensional/CostCalculator3D.hpp"
#endif // RICH_MPI

/*
 * Mach 45 Radiative Shock benchmark from:
 *   Steinberg & Heizler (2021), arXiv:2108.13453, Section 5.2.
 *   Original problem: Lowrie & Edwards (2008).
 *
 * Hydro (HydroStep) + Monte Carlo (RadiationMCStep) via operator-split.
 *
 * Gas:         ideal, gamma = 5/3, Cv = 1.45e15 erg/(g keV) ~ 1.25e8 erg/(g K)
 * Absorption:  sigma_a = 0.0142 rho^2 (T/keV)^{-3.5} cm^{-1}
 * Scattering:  sigma_s = 0.4006 rho cm^{-1}
 * Upstream:    rho = 1 g/cc,    v = 0,              T = 0.1 keV
 * Downstream:  rho = 6.43 g/cc, v = -4.82e8 cm/s,   T = 8.36 keV
 * V_shock:     5.71e8 cm/s
 * Domain:      x in [1950, 2450] cm, 4000 cells (default), shock at x = 2300
 * Runtime:     8e-7 s
 *
 * Usage: mpirun -np N ./test [options] [Np] [prefix] [new/cell] [max/cell]
 */

namespace fs = std::filesystem;

namespace
{
    class IsPointLeftRightBox3D : public ConditionActionFlux1::Condition3D
    {
    public:
        pair<bool, bool> operator()(size_t face_index, const Tessellation3D &tess,
                                    const vector<ComputationalCell3D> &) const override
        {
            if(!tess.BoundaryFace(face_index))
                return {false, false};
            auto const &box = tess.GetBoxCoordinates();
            Vector3D const &p1 = tess.GetMeshPoint(tess.GetFaceNeighbors(face_index).first);
            Vector3D const &p2 = tess.GetMeshPoint(tess.GetFaceNeighbors(face_index).second);
            bool left = p1.x < box.first.x || p1.x > box.second.x;
            bool right = p2.x < box.first.x || p2.x > box.second.x;
            return {left || right, right};
        }
    };

    class GhostChooser : public SeveralGhostGenerator3D::GhostCriteria3D
    {
    public:
        size_t GhostChoose(Tessellation3D const &tess, size_t index) const
        {
            auto const &box = tess.GetBoxCoordinates();
            Vector3D const &p = tess.GetMeshPoint(index);
            if(p.x < box.first.x) return 0;  // left x-boundary
            if(p.x > box.second.x) return 1; // right x-boundary
            return 2;                          // y/z boundaries
        }
    };

#ifdef RICH_MPI
    class MCStepCostCalculator : public CostCalculator3D
    {
    public:
        MCStepCostCalculator(const std::shared_ptr<MonteCarloManager3D> &manager) : manager(manager)
        {}

        std::vector<double> CalculateCost(const Tessellation3D &tess, const vector<ComputationalCell3D> &) const override
        {
            size_t N = tess.GetPointNo();
            const std::vector<size_t> &counters = manager->GetCellsStepsCounters();
            std::vector<double> weights(N, 0.01);
            for(size_t j = 0; j < std::min(N, counters.size()); j++)
                weights[j] = std::max(0.01, static_cast<double>(counters[j]));
            return weights;
        }

    private:
        const std::shared_ptr<MonteCarloManager3D> manager;
    };
#endif

    struct ProfilePoint
        #ifdef RICH_MPI
            : public Serializable
        #endif // RICH_MPI
    {
        double x, density, temperature, Erad, velocity;

        ProfilePoint() : x(0), density(0), temperature(0), Erad(0), velocity(0) {}
        ProfilePoint(double x_, double rho_, double T_, double Er_, double v_)
            : x(x_), density(rho_), temperature(T_), Erad(Er_), velocity(v_) {}

        #ifdef RICH_MPI
            size_t dump(Serializer *ser) const override
            {
                size_t off = 0;
                off += ser->insert(x);
                off += ser->insert(density);
                off += ser->insert(temperature);
                off += ser->insert(Erad);
                off += ser->insert(velocity);
                return off;
            }

            size_t load(const Serializer *ser, std::size_t offset)
            {
                size_t rd = 0;
                rd += ser->extract(x, offset);
                rd += ser->extract(density, offset + rd);
                rd += ser->extract(temperature, offset + rd);
                rd += ser->extract(Erad, offset + rd);
                rd += ser->extract(velocity, offset + rd);
                return rd;
            }
        #endif // RICH_MPI
        
        bool operator<(const ProfilePoint &o) const { return x < o.x; }
    };

    void WriteProfile(const Voronoi3D &tess, const std::vector<ComputationalCell3D> &cells,
                        const std::shared_ptr<MonteCarloRadiationPhysics3D> physics,
                      const std::string &filename, double time_us, size_t Np, size_t cycle,
                      const std::string &latestLink = "")
    {
        int rank = 0;
        #ifdef RICH_MPI
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        #endif // RICH_MPI

        const std::vector<double> &Erad_time_avg = physics->getEradTimeAvg();
        size_t N = tess.GetPointNo();
        std::vector<ProfilePoint> data;
        data.reserve(N);
        for(size_t i = 0; i < N; i++)
        {
            const Vector3D &p = tess.GetMeshPoint(i);
            data.emplace_back(p.x, cells[i].density, cells[i].temperature,
                                Erad_time_avg[i], cells[i].velocity.x);
        }

        #ifdef RICH_MPI
            data = MPI_Gatherv_serializable(data, 0, MPI_COMM_WORLD);
        #endif // RICH_MPI
        if(rank == 0)
        {
            std::sort(data.begin(), data.end());
            std::ofstream out(filename);
            out << "# Mach45 MC+Hydro  t_us=" << time_us << "  cycle=" << cycle << "  Np=" << Np << "\n";
            out << "# x(cm), rho(g/cc), T_gas(keV), T_rad(keV), v_x(cm/s)\n";
            for(const auto &pt : data)
            {
                double T_gas_keV = pt.temperature / units::kev_kelvin;
                double T_rad_keV = std::pow(pt.Erad / units::arad, 0.25) / units::kev_kelvin;
                out << pt.x << ", " << pt.density << ", " << T_gas_keV << ", "
                    << T_rad_keV << ", " << pt.velocity << "\n";
            }
            out.close();
            std::cout << "Wrote " << filename << " (" << data.size() << " cells)" << std::endl;
            if(!latestLink.empty())
            {
                std::filesystem::remove(latestLink);
                std::filesystem::create_symlink(std::filesystem::path(filename).filename(), latestLink);
            }
        }
    }

    void WriteVTK(const Voronoi3D &tess, const std::vector<ComputationalCell3D> &cells,
                    const std::shared_ptr<MonteCarloRadiationPhysics3D> physics,
                  const std::string &filename)
    {
        size_t N = tess.GetPointNo();
        std::vector<double> t_keV(N), dens(N), vel_x(N), erad(N), pressure(N);
        for(size_t i = 0; i < N; i++)
        {
            t_keV[i] = cells[i].temperature / units::kev_kelvin;
            dens[i] = cells[i].density;
            vel_x[i] = cells[i].velocity.x;
            erad[i] = cells[i].Erad;
            pressure[i] = cells[i].pressure;
        }
        WriteVoronoiVTKOnly(tess, filename,
                            {t_keV, dens, vel_x, erad, physics->getEradTimeAvg(), pressure},
                            {"T_gas_keV", "density", "velocity_x", "Erad", "Erad_time_avg", "pressure"});
    }

    struct AnalyticProfilePoint
    {
        double x, rho, T_gas_K, Erad, vx;
    };

    std::vector<AnalyticProfilePoint> LoadAnalyticProfile(const std::string &filepath)
    {
        std::vector<AnalyticProfilePoint> pts;
        std::ifstream in(filepath);
        if(!in.is_open())
            throw std::runtime_error("Cannot open profile file: " + filepath);
        std::string line;
        while(std::getline(in, line))
        {
            if(line.empty() || line[0] == '#')
                continue;
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream iss(line);
            AnalyticProfilePoint p;
            if(iss >> p.x >> p.rho >> p.T_gas_K >> p.Erad >> p.vx)
            {
                pts.push_back(p);
            }
        }
        return pts;
    }
}

int main(int argc, char *argv[])
{
    vtune_stop();
    DISABLE_TIMERS();

    #ifdef RICH_MPI
        MPI_Init(&argc, &argv);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    #endif // RICH_MPI

    int rank = 0, ws = 1;
    #ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ws);
    #endif // RICH_MPI

  try
  {
    ArgumentParser arguments("Mach 45 radiative shock benchmark");
    arguments.addPositional<size_t>("Np", 4000, "number of cells along x");
    arguments.addPositional<std::string>("prefix", "mach45_mc", "output prefix");
    arguments.addPositional<size_t>("new_photons_per_cell", 25, "new photons per cell per step");
    arguments.addPositional<size_t>("max_photons_per_cell", 100, "population-control photon cap per cell");
    arguments.addFlag("resume", "resume from the checkpoint if it exists");
    arguments.addOption<std::string>("profile", "", "analytic profile file for initialization");
    arguments.addOption<std::string>("manager", "new-rdma-auto", "Monte Carlo communication manager")
        .choices({"new-rdma-auto", "new-rdma-ibv", "p2p"})
        .flagAlias("new-rdma", "new-rdma-auto")
        .flagAlias("rdma", "new-rdma-auto")
        .flagAlias("new-ibv", "new-rdma-ibv")
        .flagAlias("new_ibv", "new-rdma-ibv")
        .flagAlias("ibv", "new-rdma-ibv")
        .flagAlias("p2p", "p2p");

    try
    {
        if(!arguments.parse(argc, argv))
        {
            if(rank == 0)
                std::cout << arguments.help() << std::endl;
            #ifdef RICH_MPI
                MPI_Finalize();
            #endif
            return 0;
        }
    }
    catch(const std::exception &e)
    {
        if(rank == 0)
        {
            std::cerr << e.what() << std::endl;
            std::cerr << arguments.help() << std::endl;
        }
        #ifdef RICH_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
        #else
            throw;
        #endif
    }

    size_t Np = arguments.get<size_t>("Np");
    std::string prefix = arguments.get<std::string>("prefix");
    if(prefix.find('/') == std::string::npos)
    {
        prefix = McResultsDirectory("Mach45") + "/" + prefix;
    }
    size_t newPhotonsPerCell = arguments.get<size_t>("new_photons_per_cell");
    size_t maxPhotonsPerCell = arguments.get<size_t>("max_photons_per_cell");
    bool doResume = arguments.get<bool>("resume");
    std::string profileFile = arguments.get<std::string>("profile");
    std::string managerName = arguments.get<std::string>("manager");

    #ifdef RICH_MPI
        RadiationMCStep::ManagerType managerType =
            managerName == "p2p" ? RadiationMCStep::ManagerType::P2P :
            managerName == "new-rdma-ibv" ? RadiationMCStep::ManagerType::NEW_IBV_RDMA :
            RadiationMCStep::ManagerType::NEW_RDMA;
    #endif

    // --- Physical parameters (arXiv:2108.13453, Section 5.2) ---
    constexpr double gamma_gas = 5.0 / 3.0;
    const double Cv = 1.45e15 / units::kev_kelvin;  // erg/(g K)

    constexpr double T_up_keV = 0.1, T_dn_keV = 8.36;
    double T_up = T_up_keV * units::kev_kelvin;
    double T_dn = T_dn_keV * units::kev_kelvin;

    constexpr double rho_up = 1.0;     // g/cc
    constexpr double rho_dn = 6.43;    // g/cc

    // Lab frame: upstream v=0, downstream v=-4.82e8, V_shock=5.71e8
    // Shock rest frame: both sides flow to the right
    constexpr double V_shock = 5.71e8;      // cm/s
    constexpr double v_up = V_shock;        // 5.71e8 cm/s (upstream, flowing right)
    constexpr double v_dn = V_shock - 4.82e8;  // 0.89e8 cm/s (downstream, flowing right)

    constexpr double t_final = 3e-6;   // 3 us
    constexpr double xmin = 1950.0, xmax = 2450.0;
    constexpr double shock_x = 2300.0;

    const double dx = (xmax - xmin) / Np;
    double cs_up = std::sqrt(gamma_gas * (gamma_gas - 1) * Cv * T_up);
    double cs_dn = std::sqrt(gamma_gas * (gamma_gas - 1) * Cv * T_dn);
    double max_speed = std::max(v_up + cs_up, v_dn + cs_dn);
    double max_dt = 0.3 * dx / max_speed;

    constexpr size_t boundaryPhotonsPerCell = 50;
    constexpr bool withHydro = true;
    constexpr bool diffusionPressureGradient = false;
    const bool MMC = false;
    constexpr size_t dumpInterval = 50;
    constexpr size_t vtkInterval = 200;

    const std::string simFile = prefix + "_checkpoint.h5";
    EnsureParentDirectory(prefix, rank);
#ifdef RICH_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    if(doResume)
    {
        int found = 0;
        if(rank == 0)
        {
            if(fs::exists(simFile))
            {
                found = 1;
                std::cout << "Found checkpoint: " << simFile << std::endl;
            }
            else
                std::cout << "No checkpoint found at " << simFile << ", starting fresh" << std::endl;
        }
        #ifdef RICH_MPI
            MPI_Bcast(&found, 1, MPI_INT, 0, MPI_COMM_WORLD);
        #endif // RICH_MPI
        if(!found)
            doResume = false;
    }

    double domainLength = xmax - xmin;
    double dy = domainLength / 2.0;
    Vector3D ll(xmin, -dy, -dy), ur(xmax, dy, dy);

    // --- Equation of State: e = Cv * T ---
    IdealGas eos(gamma_gas, Cv, 1, 0);

    // --- Ghost cell templates ---
    ComputationalCell3D left_cell, right_cell;

    left_cell.density = rho_up;
    left_cell.temperature = T_up;
    left_cell.velocity = Vector3D(v_up, 0, 0);
    left_cell.internal_energy = eos.dT2e(left_cell.density, left_cell.temperature,
        left_cell.tracers, ComputationalCell3D::tracerNames);
    left_cell.pressure = eos.de2p(left_cell.density, left_cell.internal_energy,
        left_cell.tracers, ComputationalCell3D::tracerNames);
    left_cell.Erad = units::arad * std::pow(T_up, 4) / left_cell.density;

    right_cell.density = rho_dn;
    right_cell.temperature = T_dn;
    right_cell.velocity = Vector3D(v_dn, 0, 0);
    right_cell.internal_energy = eos.dT2e(right_cell.density, right_cell.temperature,
        right_cell.tracers, ComputationalCell3D::tracerNames);
    right_cell.pressure = eos.de2p(right_cell.density, right_cell.internal_energy,
        right_cell.tracers, ComputationalCell3D::tracerNames);
    right_cell.Erad = units::arad * std::pow(T_dn, 4) / right_cell.density;

    // --- Generate mesh & initial conditions (skipped on resume) ---
    Voronoi3D tess(ll, ur);
#ifdef RICH_MPI
    tess.PresetLoadBalancer(std::make_shared<OneDimensionalLoadBalancer<Vector3D>>(ll, ur, Axis::X));
#endif
    std::vector<ComputationalCell3D> initialCells;
    size_t startCycle = 0;
    double simTime = 0;
    size_t dumpCount = 0;

    if(!doResume)
    {
        std::vector<Vector3D> points;
        if(rank == 0)
            points = CartesianMesh(Np, 1, 1, ll, ur);
        #ifdef RICH_MPI
            points = MPI_Spread(points, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
        #endif // RICH_MPI

        #ifdef RICH_MPI
            tess.BuildParallel(points);
        #else 
            tess.Build(points);
        #endif // RICH_MPI

        size_t Nlocal = tess.GetPointNo();
        initialCells.resize(Nlocal);

        if(!profileFile.empty())
        {
            std::vector<AnalyticProfilePoint> profile;
            if(rank == 0)
            {
                profile = LoadAnalyticProfile(profileFile);
                std::cout << "Loaded analytic profile from " << profileFile
                          << " (" << profile.size() << " points)" << std::endl;
            }
            #ifdef RICH_MPI
            {
                size_t psize = profile.size();
                MPI_Bcast(&psize, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
                profile.resize(psize);
                MPI_Bcast(profile.data(), psize * sizeof(AnalyticProfilePoint), MPI_BYTE, 0, MPI_COMM_WORLD);
            }
            #endif

            std::vector<double> px(profile.size());
            for(size_t j = 0; j < profile.size(); j++)
                px[j] = profile[j].x;

            for(size_t i = 0; i < Nlocal; ++i)
            {
                double xi = tess.GetMeshPoint(i).x;
                auto it = std::lower_bound(px.begin(), px.end(), xi);
                size_t idx = (it == px.end()) ? px.size() - 1 :
                             (it == px.begin()) ? 0 :
                             (xi - *(it - 1) < *it - xi) ? std::distance(px.begin(), it) - 1 :
                             std::distance(px.begin(), it);

                const auto &pp = profile[idx];
                initialCells[i].density = pp.rho;
                initialCells[i].temperature = pp.T_gas_K;
                initialCells[i].velocity = Vector3D(pp.vx, 0, 0);
                initialCells[i].internal_energy = eos.dT2e(pp.rho, pp.T_gas_K,
                    initialCells[i].tracers, ComputationalCell3D::tracerNames);
                initialCells[i].pressure = eos.de2p(pp.rho, initialCells[i].internal_energy,
                    initialCells[i].tracers, ComputationalCell3D::tracerNames);
                initialCells[i].Erad = pp.Erad / pp.rho;
            }

            // Override ghost cells with the actual leftmost/rightmost cell values
            double local_min_x = std::numeric_limits<double>::max();
            double local_max_x = std::numeric_limits<double>::lowest();
            size_t min_idx = 0, max_idx = 0;
            for(size_t i = 0; i < Nlocal; i++)
            {
                double xi = tess.GetMeshPoint(i).x;
                if(xi < local_min_x) { local_min_x = xi; min_idx = i; }
                if(xi > local_max_x) { local_max_x = xi; max_idx = i; }
            }

            #ifdef RICH_MPI
            auto [left_rank, left_x] = MPI_Min_loc(local_min_x);
            auto [right_rank, right_x] = MPI_Max_loc(local_max_x);

            if(rank == left_rank)
                left_cell = initialCells[min_idx];
            left_cell = MPI_Bcast_serializable(left_cell, left_rank);

            if(rank == right_rank)
                right_cell = initialCells[max_idx];
            right_cell = MPI_Bcast_serializable(right_cell, right_rank);

            #else // RICH_MPI
                left_cell = initialCells[min_idx];
                right_cell = initialCells[max_idx];
            #endif // RICH_MPI

            T_up = left_cell.temperature;
            T_dn = right_cell.temperature;
            cs_up = std::sqrt(gamma_gas * (gamma_gas - 1) * Cv * T_up);
            cs_dn = std::sqrt(gamma_gas * (gamma_gas - 1) * Cv * T_dn);
            max_speed = std::max(v_up + cs_up, v_dn + cs_dn);
            max_dt = 0.3 * dx / max_speed;

            if(rank == 0)
                std::cout << "Ghost cells from profile edges:"
                          << " left(rho=" << left_cell.density
                          << ", T=" << left_cell.temperature / units::kev_kelvin << " keV"
                          << ", v=" << left_cell.velocity.x << ")"
                          << " right(rho=" << right_cell.density
                          << ", T=" << right_cell.temperature / units::kev_kelvin << " keV"
                          << ", v=" << right_cell.velocity.x << ")" << std::endl;
        }
        else
        {
            if(rank == 0)
                std::cout << "Using step-function IC" << std::endl;
            for(size_t i = 0; i < Nlocal; ++i)
                initialCells[i] = (tess.GetMeshPoint(i).x < shock_x) ? left_cell : right_cell;
        }
    }

    // ===== Simulation =====
    Simulation sim(tess, initialCells, eos);
    ZeroForce3D force;
    auto tsc = std::make_shared<CourantFriedrichsLewy>(0.3, 1, force);
    sim.SetTimeStepFunction(tsc);

    std::vector<ComputationalCell3D> &cells = sim.getCells();
    std::vector<Conserved3D> &extensives = sim.getExtensives();

    // ===== Hydro setup =====
    Hllc3D rs;

    RigidWallGenerator3D rigid_ghost;
    ConstantPrimitiveGenerator3D left_ghost(left_cell), right_ghost(right_cell);
    std::vector<Ghost3D *> ghost_list = {&left_ghost, &right_ghost, &rigid_ghost};
    GhostChooser ghost_chooser;
    SeveralGhostGenerator3D ghost(ghost_list, ghost_chooser);

    LinearGauss3D interp(eos, ghost);

    IsBulkFace3D isbulk;
    IsPointLeftRightBox3D is_side;
    IsBoundaryFace3D isboundary;
    RegularFlux3D normal_flux(rs);
    RigidWallFlux3D rigid_flux(rs);

    std::vector<pair<const ConditionActionFlux1::Condition3D *,
                     const ConditionActionFlux1::Action3D*>> flux_seq;
    flux_seq.push_back({&is_side, &normal_flux});
    flux_seq.push_back({&isboundary, &rigid_flux});
    flux_seq.push_back({&isbulk, &normal_flux});
    ConditionActionFlux1 flux(flux_seq, interp);

    DefaultCellUpdater cu(false, 0, true);

    std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D *,
                          const ConditionExtensiveUpdater3D::Action3D *>> eu_sequence;
    ConditionExtensiveUpdater3D eu(eu_sequence);

    Eulerian3D pm;

    HDSim3D hdsim(tess, cells, extensives, eos, sim.getTracker(), pm, *tsc, flux, cu, eu, force,
                  std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));

    auto hydroStep = std::make_shared<HydroStep>(hdsim, HydroStep::TIMEADVANCE_2);
    sim.addPhysics(hydroStep);

    // ===== MC radiation setup =====
    // sigma_a = 0.0142 * rho^2 * (T/keV)^{-3.5}
    //         = 0.0142 * kev_kelvin^{3.5} * rho^2 * T_K^{-3.5}
    // sigma_s = 0.4006 * rho
    auto eosPtr = std::make_shared<IdealGas>(eos);
    double sigmaA0 = 0.0142 * std::pow(units::kev_kelvin, 3.5);
    double sigmaS0 = 0.4006;
    auto opacityPtr = std::make_shared<MCPowerLawOpacity>(sigmaA0, sigmaS0, 2, -3.5, 1, 0);

    std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> boundaryCond =
        std::make_shared<STORM::TwoSidesTemperature<Vector3D, Tessellation3D>>(
            tess, T_up, T_dn, boundaryPhotonsPerCell);

    STORM::RadiationIMCParameters<ENERGY_GROUPS_NUM> radiationIMCParameters = {
        .newPhotonsPerCell = newPhotonsPerCell,
        .withHydro = withHydro,
        .diffusionPressureGradient = diffusionPressureGradient,
        .MMC = MMC,
        .withMultigroupOpacity = false,
        .withRandomWalk = false,
        .energyBoundaries = {0.0, 1.0e30},
        .energyBoundariesProvided = true
    };
    std::shared_ptr<MonteCarloRadiationPhysics3D> physics = std::make_shared<::RadiationIMC>(
        tess, boundaryCond, cells, extensives, eosPtr, opacityPtr, radiationIMCParameters);

    std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> popControl =
        std::make_shared<STORM::CombPopulationControl<Vector3D, Tessellation3D>>(tess, maxPhotonsPerCell, 10);

    size_t initialParticlesPerCell = 50;
    std::vector<Particle3D> initialParticles;
    auto mcStep = std::make_shared<RadiationMCStep>(
        tess, cells, extensives, physics, popControl, boundaryCond, initialParticles, initialParticlesPerCell, withHydro
        #ifdef RICH_MPI
            , managerType
        #endif
    );
    #ifdef RICH_MPI
        mcStep->setCost(std::make_shared<MCStepCostCalculator>(mcStep->getManager()));
    #endif
    sim.addPhysics(mcStep);

    double current_dt;
    // ===== Resume from checkpoint =====
    if(doResume)
    {
        ReadSimulation(simFile, sim);
        startCycle = sim.GetCycle();
        simTime = sim.GetTime();
        current_dt = sim.GetTimeStep();
        dumpCount = startCycle / dumpInterval;
        if(rank == 0)
            std::cout << "Resumed from " << simFile
                      << ": cycle=" << startCycle
                      << ", t=" << simTime * 1e6 << " us"
                      << ", dumpCount=" << dumpCount << std::endl;
        
        std::cout << "Outside of read, rank has " << tess.GetPointNo() << " points" << std::endl;
    }
    else
    {
        current_dt = 0.001 * max_dt;
    }

    extensives.resize(cells.size());
    for(size_t i = 0; i < cells.size(); i++)
        PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);



    // --- Print setup info ---
    if(rank == 0)
    {
        std::cout << "Mach 45 Radiative Shock (Hydro + MC)"
                  << "\n  Np=" << Np
                  << ", new/cell=" << newPhotonsPerCell
                  << ", max/cell=" << maxPhotonsPerCell
                  << "\n  T_upstream=" << T_up_keV << " keV"
                  << ", T_downstream=" << T_dn_keV << " keV"
                  << "\n  rho_upstream=" << rho_up
                  << ", rho_downstream=" << rho_dn
                  << "\n  v_up(shock frame)=" << v_up << " cm/s"
                  << ", v_dn(shock frame)=" << v_dn << " cm/s"
                  << "\n  domain=[" << xmin << ", " << xmax << "] cm"
                  << ", shock_x=" << shock_x
                  << "\n  dt=" << current_dt << " s"
                  << ", t_final=" << t_final * 1e6 << " us"
                  << ", prefix=" << prefix
                  << ", manager=" << managerName
                  << (doResume ? ", RESUMED" : "")
                  << std::endl;
    }

    if(rank == 0)
    {
        std::ofstream parameter_file(fs::path(prefix).parent_path() / "mach45_parameters.txt");
        parameter_file << std::setprecision(17) << std::boolalpha;
        parameter_file << "# Mach45 runtime parameters" << std::endl;
        parameter_file << "output_directory=" << fs::path(prefix).parent_path().string() << std::endl;
        parameter_file << "output_prefix=" << prefix << std::endl;
        parameter_file << "mpi_ranks=" << ws << std::endl;
        parameter_file << "mesh_cells_x=" << Np << std::endl;
        parameter_file << "mesh_cells_y=1" << std::endl;
        parameter_file << "mesh_cells_z=1" << std::endl;
        parameter_file << "domain_xmin_cm=" << xmin << std::endl;
        parameter_file << "domain_xmax_cm=" << xmax << std::endl;
        parameter_file << "domain_ymin_cm=" << -dy << std::endl;
        parameter_file << "domain_ymax_cm=" << dy << std::endl;
        parameter_file << "domain_zmin_cm=" << -dy << std::endl;
        parameter_file << "domain_zmax_cm=" << dy << std::endl;
        parameter_file << "domain_length_x_cm=" << domainLength << std::endl;
        parameter_file << "mesh_dx_cm=" << dx << std::endl;
        parameter_file << "shock_x_cm=" << shock_x << std::endl;
        parameter_file << "gamma_gas=" << gamma_gas << std::endl;
        parameter_file << "Cv_erg_per_g_K=" << Cv << std::endl;
        parameter_file << "sigma_a_coefficient=" << 0.0142 << std::endl;
        parameter_file << "sigma_a_rho_exponent=" << 2 << std::endl;
        parameter_file << "sigma_a_temperature_keV_exponent=" << -3.5 << std::endl;
        parameter_file << "sigma_s_coefficient=" << 0.4006 << std::endl;
        parameter_file << "sigma_s_rho_exponent=" << 1 << std::endl;
        parameter_file << "rho_upstream_g_per_cc=" << rho_up << std::endl;
        parameter_file << "rho_downstream_g_per_cc=" << rho_dn << std::endl;
        parameter_file << "T_upstream_keV=" << T_up_keV << std::endl;
        parameter_file << "T_downstream_keV=" << T_dn_keV << std::endl;
        parameter_file << "v_upstream_shock_frame_cm_s=" << v_up << std::endl;
        parameter_file << "v_downstream_shock_frame_cm_s=" << v_dn << std::endl;
        parameter_file << "v_downstream_lab_cm_s=" << -4.82e8 << std::endl;
        parameter_file << "shock_speed_cm_s=" << V_shock << std::endl;
        parameter_file << "t_final_s=" << t_final << std::endl;
        parameter_file << "cfl_factor=" << 0.3 << std::endl;
        parameter_file << "max_speed_cm_s=" << max_speed << std::endl;
        parameter_file << "max_dt_s=" << max_dt << std::endl;
        parameter_file << "initial_dt_s=" << current_dt << std::endl;
        parameter_file << "initial_dt_factor=" << 0.001 << std::endl;
        parameter_file << "timestep_ramp_start_cycle=" << 750 << std::endl;
        parameter_file << "timestep_ramp_factor=" << 1.01 << std::endl;
        parameter_file << "dump_interval_cycles=" << dumpInterval << std::endl;
        parameter_file << "vtk_interval_cycles=" << vtkInterval << std::endl;
        parameter_file << "new_photons_per_cell=" << newPhotonsPerCell << std::endl;
        parameter_file << "max_photons_per_cell=" << maxPhotonsPerCell << std::endl;
        parameter_file << "initial_particles_per_cell=" << initialParticlesPerCell << std::endl;
        parameter_file << "boundary_photons_per_cell=" << boundaryPhotonsPerCell << std::endl;
        parameter_file << "population_control_comb_parameter=" << 10 << std::endl;
        parameter_file << "with_hydro=" << withHydro << std::endl;
        parameter_file << "diffusion_pressure_gradient=" << diffusionPressureGradient << std::endl;
        parameter_file << "MMC=" << MMC << std::endl;
        parameter_file << "multigroup_opacity=" << false << std::endl;
        parameter_file << "random_walk=" << false << std::endl;
        parameter_file << "energy_boundary_min=" << 0.0 << std::endl;
        parameter_file << "energy_boundary_max=" << 1.0e30 << std::endl;
        parameter_file << "manager=" << managerName << std::endl;
        parameter_file << "profile_file=" << profileFile << std::endl;
        parameter_file << "resume=" << doResume << std::endl;
        parameter_file << "start_cycle=" << startCycle << std::endl;
        parameter_file << "start_time_s=" << simTime << std::endl;
        parameter_file.close();
        std::cout << "Wrote " << fs::path(prefix).parent_path() / "mach45_parameters.txt" << std::endl;
    }

    if(!doResume)
    {
        WriteProfile(tess, cells, physics, prefix + "_init.txt", 0, Np, 0, prefix + "_latest.txt");
        WriteVTK(tess, cells, physics, prefix + "_init.vtu");
    }

    // ===== Main time-stepping loop =====
    size_t cycle = startCycle;
    size_t stepsSinceLastDump = (doResume) ? (startCycle % dumpInterval) : 0;

    auto startWall = std::chrono::high_resolution_clock::now();

    sim.SetTimeStep(current_dt);

    while(simTime < t_final)
    {
        auto stepStart = std::chrono::high_resolution_clock::now();

        sim.step();
        cycle++;
        stepsSinceLastDump++;

        simTime = sim.GetTime();
        if(cycle >= 750)
        {
            current_dt = min(1.01 * current_dt, max_dt);
        }
        // sim.SetTimeStep(current_dt);

        auto stepEnd = std::chrono::high_resolution_clock::now();
        double stepSec = std::chrono::duration<double>(stepEnd - stepStart).count();
        double elapsedWall = std::chrono::duration<double>(stepEnd - startWall).count();

        double fraction = simTime / t_final;
        double eta = (fraction > 0) ? elapsedWall * (1.0 - fraction) / fraction : 0;

        if(rank == 0 && (cycle % 50 == 0 || cycle <= 5))
        {
            int pct = static_cast<int>(fraction * 100);
            int etaMin = static_cast<int>(eta) / 60;
            int etaSec = static_cast<int>(eta) % 60;
            std::cout << "Cycle " << cycle
                      << "  t=" << simTime * 1e6 << " us"
                      << " (" << pct << "%)"
                      << "  dt=" << sim.GetTimeStep()
                      << "  step=" << stepSec << "s"
                      << "  ETA=" << etaMin << "m" << etaSec << "s"
                      << std::endl;
        }

        if(stepsSinceLastDump >= dumpInterval)
        {
            stepsSinceLastDump = 0;
            dumpCount++;

            char buf[512];
            std::snprintf(buf, sizeof(buf), "%s_%05zu.txt", prefix.c_str(), dumpCount);
            WriteProfile(tess, cells, physics, buf, simTime * 1e6, Np, cycle, prefix + "_latest.txt");
        }

        if(cycle % vtkInterval == 0)
        {
            char buf[512];
            std::snprintf(buf, sizeof(buf), "%s_%05zu.vtu", prefix.c_str(), cycle / vtkInterval);
            WriteVTK(tess, cells, physics, buf);

            WriteSimulation(sim, simFile);
            if(rank == 0)
                std::cout << "Checkpoint written: " << simFile << std::endl;
        }
    }

    auto endWall = std::chrono::high_resolution_clock::now();
    double wallSec = std::chrono::duration<double>(endWall - startWall).count();
    if(rank == 0)
        std::cout << "Completed " << cycle << " cycles in " << wallSec << "s" << std::endl;

    // --- Final output ---
    WriteProfile(tess, cells, physics, prefix + "_final.txt", simTime * 1e6, Np, cycle, prefix + "_latest.txt");
    WriteVTK(tess, cells, physics, prefix + "_final.vtu");
    WriteSimulation(sim, simFile);

  }
  catch(const UniversalError &e)
  {
      std::cout << std::setprecision(16);
      std::cerr << std::setprecision(16);  
      std::cerr << "=== UniversalError on rank " << rank << " ===" << std::endl;
      reportError(e);
      #ifdef RICH_MPI
          MPI_Abort(MPI_COMM_WORLD, 1);
      #else // RICH_MPI
        return 1;
      #endif // RICH_MPI
  }
  catch(const std::exception &e)
  {
      std::cerr << "=== std::exception on rank " << rank << ": " << e.what() << " ===" << std::endl;
      #ifdef RICH_MPI
          MPI_Abort(MPI_COMM_WORLD, 1);
      #else // RICH_MPI
        return 1;
      #endif // RICH_MPI
  }

    #ifdef RICH_MPI
        MPI_Finalize();
    #endif // RICH_MPI
    return 0;
}
