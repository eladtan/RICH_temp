#include <mpi.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include "3D/radiation/MonteCarloPhysics3D.hpp"
#include "mpi/mpi_commands.hpp"
#include "misc/mesh_generator3D.hpp"
#include "3D/tessellation/Voronoi3D.hpp"
#include "CMMC/src/units/units.hpp"
#include "newtonian/common/ideal_gas.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "newtonian/three_dimensional/simulation/Simulation.hpp"
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
#include "newtonian/three_dimensional/CostCalculator3D.hpp"
#include "utils/arguments/ArgumentParser.hpp"
#include "runs/mc_results_dir.hpp"

/*
 * Mach 2 Radiative Shock benchmark from:
 *   Steinberg & Heizler (2021), arXiv:2108.13453, Section 5.1.
 *   Original problem: Lowrie & Edwards (2008).
 *
 * This implementation uses Hydro (HydroStep) + Monte Carlo (RadiationMCStep)
 * via the Simulation class, in an operator-split scheme.
 *
 * Gas:         ideal, gamma = 5/3, Cv = 1.91e8 erg/(g K)
 * Absorption:  sigma_a = 0.362 rho (T/keV)^{-3.5} cm^{-1}, sigma_s = 0
 * Upstream:    rho = 1 g/cc,    v = 3.4616e7 cm/s, T = 0.122 keV
 * Downstream:  rho = 2.29 g/cc, v = 1.5116e7 cm/s, T = 0.253 keV
 * Domain:      x in [-0.21, 0.25] cm, 1024 cells (default)
 * Runtime:     5 ns
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

    struct AnalyticProfilePoint
    {
        double x, rho, T_gas_K, T_rad_K, vx;
    };

    std::vector<AnalyticProfilePoint> LoadAnalyticProfile(const std::string &filepath)
    {
        std::vector<AnalyticProfilePoint> points;
        std::ifstream in(filepath);
        if(!in.is_open())
        {
            throw std::runtime_error("Cannot open analytic profile: " + filepath);
        }

        std::string line;
        while(std::getline(in, line))
        {
            if(line.empty() || line[0] == '#')
            {
                continue;
            }
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream iss(line);
            AnalyticProfilePoint point;
            if(iss >> point.x >> point.rho >> point.T_gas_K >> point.T_rad_K >> point.vx)
            {
                points.push_back(point);
            }
        }
        if(points.empty())
        {
            throw std::runtime_error("Analytic profile is empty: " + filepath);
        }
        std::sort(points.begin(), points.end(),
                  [](const AnalyticProfilePoint &a, const AnalyticProfilePoint &b)
                  { return a.x < b.x; });
        return points;
    }

    void WriteProfile(const Voronoi3D &tess, const std::vector<ComputationalCell3D> &cells,
                        const std::shared_ptr<MonteCarloRadiationPhysics3D> physics,
                      const std::string &filename, double time_ns, size_t Np)
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
            // At t=0 the time-average estimator is intentionally empty/zero;
            // use the initialized radiation energy for the initial profile.
            // Erad_time_avg is an energy density, while cells[i].Erad is
            // radiation energy per unit mass. Store an energy density in both
            // cases so the output conversion below is identical.
            double erad = (i < Erad_time_avg.size() && Erad_time_avg[i] > 0.0)
                              ? Erad_time_avg[i] : cells[i].Erad * cells[i].density;
            data.emplace_back(p.x, cells[i].density, cells[i].temperature,
                              erad, cells[i].velocity.x);
        }

        #ifdef RICH_MPI
            data = MPI_Gatherv_serializable(data, 0, MPI_COMM_WORLD);
        #endif // RICH_MPI
        if(rank == 0)
        {
            std::sort(data.begin(), data.end());
            std::ofstream out(filename);
            out << "# Mach2 MC+Hydro  t_ns=" << time_ns << "  Np=" << Np << "\n";
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
}

int main(int argc, char *argv[])
{
    vtune_stop();
    DISABLE_TIMERS();

    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);

    int rank = 0, ws = 1;
    #ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ws);
    #endif // RICH_MPI

  try
  {
    ArgumentParser arguments("Mach 2 radiative shock benchmark");
    arguments.addPositional<size_t>("Np", 1024, "number of cells along x");
    arguments.addPositional<std::string>("prefix", "mach2_mc", "output prefix");
    arguments.addPositional<size_t>("new_photons_per_cell", 25, "new photons per cell per step");
    arguments.addPositional<size_t>("max_photons_per_cell", 100, "population-control photon cap per cell");
    arguments.addFlag("resume", "resume from the checkpoint if it exists");
    arguments.addOption<std::string>("profile", "mach2_analytic_ic2.dat", "analytic profile file for initialization");
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
            MPI_Finalize();
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
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t Np = arguments.get<size_t>("Np");
    std::string prefix = arguments.get<std::string>("prefix");
    if(prefix.find('/') == std::string::npos)
    {
        prefix = McResultsDirectory("Mach2") + "/" + prefix;
    }
    EnsureParentDirectory(prefix, rank);
#ifdef RICH_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
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

    // --- Physical parameters (arXiv:2108.13453, Section 5.1) ---
    constexpr double gamma_gas = 5.0 / 3.0;
    constexpr double Cv = 1.91e8;  // erg/(g K)

    constexpr double T_up_keV = 0.122, T_dn_keV = 0.253;
    const double T_up = T_up_keV * units::kev_kelvin;
    const double T_dn = T_dn_keV * units::kev_kelvin;

    constexpr double rho_up = 1.0;     // g/cc
    constexpr double rho_dn = 2.29;    // g/cc
    constexpr double v_up = 3.4616e+07;    // cm/s, stationary shock frame
    constexpr double v_dn = 1.5116e+07;    // cm/s, stationary shock frame

    constexpr double t_final = 5e-9;   // 5 ns
    constexpr double xmin = -0.21, xmax = 0.25;

    // CFL-based constant time step in the stationary shock frame.
    const double dx = (xmax - xmin) / Np;
    const double cs_up = std::sqrt(gamma_gas * (gamma_gas - 1) * Cv * T_up);
    const double cs_dn = std::sqrt(gamma_gas * (gamma_gas - 1) * Cv * T_dn);
    const double max_speed = std::max(std::abs(v_up) + cs_up, std::abs(v_dn) + cs_dn);
    const double dt = 0.3 * dx / max_speed;

    constexpr size_t boundaryPhotonsPerCell = 50;
    constexpr bool withHydro = true;
    constexpr bool diffusionPressureGradient = false;
    constexpr size_t dumpInterval = 10;
    constexpr size_t vtkInterval = 25;

    const std::string simFile = prefix + "_checkpoint.h5";

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

    // 1 cell in y and z.
    double domainLength = xmax - xmin;
    double dy = domainLength / 2.0;
    Vector3D ll(xmin, -dy, -dy), ur(xmax, dy, dy);

    // --- Equation of State: e = Cv * T ---
    IdealGas eos(gamma_gas, Cv, 1, 0);

    std::vector<AnalyticProfilePoint> analyticProfile;
    if(!doResume)
    {
        analyticProfile = LoadAnalyticProfile(profileFile);
        if(rank == 0)
            std::cout << "Loaded analytic profile from " << profileFile
                      << " (" << analyticProfile.size() << " points)" << std::endl;
    }

    // --- Ghost cell templates (needed for hydro BCs and fresh init) ---
    ComputationalCell3D left_cell, right_cell;

    auto makeCell = [&eos](double rho, double T_gas, double T_rad, double vx)
    {
        ComputationalCell3D cell;
        cell.density = rho;
        cell.temperature = T_gas;
        cell.velocity = Vector3D(vx, 0, 0);
        cell.internal_energy = eos.dT2e(cell.density, cell.temperature,
            cell.tracers, ComputationalCell3D::tracerNames);
        cell.pressure = eos.de2p(cell.density, cell.internal_energy,
            cell.tracers, ComputationalCell3D::tracerNames);
        cell.Erad = units::arad * std::pow(T_rad, 4) / cell.density;
        return cell;
    };

    // x increases from the upstream (left) state to the downstream (right) state,
    // matching the stationary shock-frame benchmark in Fig. 9(a).
    left_cell = makeCell(rho_up, T_up, T_up, v_up);
    right_cell = makeCell(rho_dn, T_dn, T_dn, v_dn);
    if(!analyticProfile.empty())
    {
        const auto &upstream = analyticProfile.front();
        const auto &downstream = analyticProfile.back();
        left_cell = makeCell(upstream.rho, upstream.T_gas_K, upstream.T_rad_K, upstream.vx);
        right_cell = makeCell(downstream.rho, downstream.T_gas_K, downstream.T_rad_K, downstream.vx);
    }

    // --- Generate mesh & initial conditions (skipped on resume) ---
    Voronoi3D tess(ll, ur);
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

        std::vector<double> profileX(analyticProfile.size());
        for(size_t j = 0; j < analyticProfile.size(); ++j)
            profileX[j] = analyticProfile[j].x;

        for(size_t i = 0; i < Nlocal; ++i)
        {
            const double x = tess.GetMeshPoint(i).x;
            auto it = std::lower_bound(profileX.begin(), profileX.end(), x);
            size_t index = (it == profileX.end()) ? profileX.size() - 1 :
                           (it == profileX.begin()) ? 0 :
                           (x - *(it - 1) < *it - x) ?
                               std::distance(profileX.begin(), it) - 1 :
                               std::distance(profileX.begin(), it);
            const auto &point = analyticProfile[index];
            initialCells[i] = makeCell(point.rho, point.T_gas_K, point.T_rad_K, point.vx);
        }
        if(rank == 0)
            std::cout << "Initialized cells from the Lowrie-Edwards analytic profile" << std::endl;
    }

    // ===== Simulation =====
    Simulation sim(tess, initialCells, eos);
    auto tsc = std::make_shared<ManualTimeStep>();
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
                     const ConditionActionFlux1::Action3D *>> flux_seq;
    flux_seq.push_back({&is_side, &normal_flux});
    flux_seq.push_back({&isboundary, &rigid_flux});
    flux_seq.push_back({&isbulk, &normal_flux});
    ConditionActionFlux1 flux(flux_seq, interp);

    DefaultCellUpdater cu(false, 0, true);

    std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D *,
                          const ConditionExtensiveUpdater3D::Action3D *>> eu_sequence;
    ConditionExtensiveUpdater3D eu(eu_sequence);

    ZeroForce3D force;
    Eulerian3D pm;

    // HDSim3D is initialized from Simulation's cells, extensives, and tracker
    HDSim3D hdsim(tess, cells, extensives, eos, sim.getTracker(), pm, *tsc, flux, cu, eu, force,
                  std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));

    auto hydroStep = std::make_shared<HydroStep>(hdsim, HydroStep::TIMEADVANCE_2);
    sim.addPhysics(hydroStep);

    // ===== MC radiation setup =====
    // sigma_a = 0.362 * rho * (T/keV)^{-3.5}
    //         = 0.362 * kev_kelvin^{3.5} * rho^1 * T_K^{-3.5}
    auto eosPtr = std::make_shared<IdealGas>(eos);
    double sigmaA0 = 0.362 * std::pow(units::kev_kelvin, 3.5);
    auto opacityPtr = std::make_shared<MCPowerLawOpacity>(sigmaA0, 0, 1, -3.5, 0, 0);

    std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> boundaryCond =
        std::make_shared<TwoSidesTemperature<Vector3D, Tessellation3D>>(
            tess, T_up, T_dn, boundaryPhotonsPerCell);

    STORM::RadiationIMCParameters<ENERGY_GROUPS_NUM> radiationIMCParameters = {
        .newPhotonsPerCell = newPhotonsPerCell,
        .withHydro = withHydro,
        .diffusionPressureGradient = diffusionPressureGradient,
        .MMC = false,
        .withMultigroupOpacity = false,
        .withRandomWalk = false,
        .energyBoundaries = {0.0, 1.0e30},
        .energyBoundariesProvided = true
    };
    std::shared_ptr<MonteCarloRadiationPhysics3D> physics = std::make_shared<::RadiationIMC>(
        tess, boundaryCond, cells, extensives, eosPtr, opacityPtr, radiationIMCParameters);

    std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> popControl =
        std::make_shared<CombPopulationControl<Vector3D, Tessellation3D>>(tess, maxPhotonsPerCell, 10);

    std::vector<Particle3D> initialParticles;
    constexpr size_t initialParticlesPerCell = 50;
    auto mcStep = std::make_shared<RadiationMCStep>(
        tess, cells, extensives, physics, popControl, boundaryCond, initialParticles, initialParticlesPerCell, withHydro
        #ifdef RICH_MPI
            , managerType
        #endif
    );
    #ifdef RICH_MPI
        mcStep->setCost(std::make_shared<MCStepCostCalculator>(mcStep->getManager()));
        sim.setForceRebalanceSteps(4);
        sim.addMigrationBuffer(mcStep->getManager()->GetCellsStepsCounters());
    #endif
    sim.addPhysics(mcStep);

    sim.SetTimeStep(dt);

    // ===== Resume from checkpoint =====
    if(doResume)
    {
        ReadSimulation(simFile, sim);
        startCycle = sim.GetCycle();
        simTime = sim.GetTime();
        dumpCount = startCycle / dumpInterval;

        if(rank == 0)
            std::cout << "Resumed from " << simFile
                      << ": cycle=" << startCycle
                      << ", t=" << simTime * 1e9 << " ns"
                      << ", dumpCount=" << dumpCount << std::endl;
        
        std::cout << "Outside of read, rank has " << tess.GetPointNo() << " points" << std::endl;
    }

    extensives.resize(cells.size());
    for(size_t i = 0; i < cells.size(); i++)
        PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);

    // --- Print setup info ---
    if(rank == 0)
    {
        std::cout << "Mach 2 Radiative Shock (Hydro + MC)"
                  << "\n  Np=" << Np
                  << ", new/cell=" << newPhotonsPerCell
                  << ", max/cell=" << maxPhotonsPerCell
                  << "\n  T_upstream=" << T_up_keV << " keV"
                  << ", T_downstream=" << T_dn_keV << " keV"
                  << "\n  rho_upstream=" << rho_up
                  << ", rho_downstream=" << rho_dn
                  << "\n  v_upstream=" << v_up
                  << ", v_downstream=" << v_dn << " cm/s"
                  << "\n  domain=[" << xmin << ", " << xmax << "] cm"
                  << ", dt=" << dt << " s"
                  << ", t_final=" << t_final * 1e9 << " ns"
                  << ", prefix=" << prefix
                  << ", manager=" << managerName
                  << (doResume ? ", RESUMED" : "")
                  << std::endl;
    }

    if(!doResume)
    {
        WriteProfile(tess, cells, physics, prefix + "_init.txt", 0, Np);
        WriteVTK(tess, cells, physics, prefix + "_init.vtu");
    }

    // ===== Main time-stepping loop =====
    size_t cycle = startCycle;
    size_t stepsSinceLastDump = (doResume) ? (startCycle % dumpInterval) : 0;

    auto startWall = std::chrono::high_resolution_clock::now();

    while(simTime < t_final)
    {
        auto stepStart = std::chrono::high_resolution_clock::now();

        sim.step();
        cycle++;
        stepsSinceLastDump++;

        simTime = sim.GetTime();
        sim.SetTimeStep(dt);

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
                      << "  t=" << simTime * 1e9 << " ns"
                      << " (" << pct << "%)"
                      << "  dt=" << dt
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
            WriteProfile(tess, cells, physics, buf, simTime * 1e9, Np);
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
    WriteProfile(tess, cells, physics, prefix + "_final.txt", simTime * 1e9, Np);
    WriteVTK(tess, cells, physics, prefix + "_final.vtu");
    WriteSimulation(sim, simFile);

  }
  catch(const UniversalError &e)
  {
      std::cerr << "=== UniversalError on rank " << rank << " ===" << std::endl;
      reportError(e);
      MPI_Abort(MPI_COMM_WORLD, 1);
  }
  catch(const std::exception &e)
  {
      std::cerr << "=== std::exception on rank " << rank << ": " << e.what() << " ===" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
  }

    MPI_Finalize();
    return 0;
}
