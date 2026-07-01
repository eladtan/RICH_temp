#include <mpi.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include "mpi/mpi_commands.hpp"
#include "3D/tessellation/Voronoi3D.hpp"
#include "Radiation/CMMC/src/units/units.hpp"
#include "newtonian/common/ideal_gas.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "newtonian/three_dimensional/simulation/Simulation.hpp"
#include "newtonian/three_dimensional/ManualTimeStep.hpp"
#include "3D/output/write3D.hpp"

// Monte Carlo includes
#include "3D/radiation/RadiationIMC.hpp"
#include "3D/radiation/PowerLawOpacity.hpp"
#include "monte/population/CombPopulationControl.hpp"
#include "monte/boundary/SideTemperature.hpp"
#include "newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"
#include "utils/arguments/ArgumentParser.hpp"

// Diffusion includes
#include "Radiation/Diffusion.hpp"
#include "newtonian/three_dimensional/simulation/steps/RadiationStep.hpp"
#include "newtonian/three_dimensional/CostCalculator3D.hpp"

/*
 * 1D Marshak Wave test problem from:
 *   Steinberg & Heizler (2021), "A New Discrete Implicit Monte Carlo Scheme
 *   for Simulating Radiative Transfer Problems", arXiv:2108.13453, Section 4.1.
 *
 * Domain:      x in [0, 4]
 * Left BC:     Blackbody source, T = 1 (normalized)
 * Opacity:     sigma_a = 10 * T^{-3},  sigma_s = 0
 * Heat cap:    Cv = 7.14 * a_rad
 * Init temp:   T_0 = 0.01
 * Run until:   c*t = 500
 * Time step:   c*dt = 0.01  (constant)
 * Photons:     50 new / cell / step,  200 max / cell  (MC only)
 *
 * Modes:
 *   mc              - Implicit Monte Carlo  (default)
 *   diffusion       - Grey flux-limited diffusion (no flux limiter)
 *   diffusion-fl    - Grey flux-limited diffusion (with flux limiter)
 */

int main(int argc, char *argv[])
{
    vtune_stop();
    DISABLE_TIMERS();

    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);

    rank_t rank, ws;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ws);

  try
  {
    ArgumentParser arguments("Marshak wave benchmark");
    arguments.addPositional<size_t>("Nx", "number of cells along x").required();
    arguments.addPositional<std::string>("prefix", "marshak", "output prefix");
    arguments.addPositional<size_t>("Ny", 1, "number of cells along y");
    arguments.addPositional<size_t>("Nz", 1, "number of cells along z");
    arguments.addPositional<std::string>("mode", "mc", "solver mode")
        .choices({"mc", "diffusion", "diffusion-fl"});
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

    size_t Nx = arguments.get<size_t>("Nx");
    std::string prefix = arguments.get<std::string>("prefix");
    size_t Ny = arguments.get<size_t>("Ny");
    size_t Nz = arguments.get<size_t>("Nz");
    std::string mode = arguments.get<std::string>("mode");
    std::string managerName = arguments.get<std::string>("manager");

    #ifdef RICH_MPI
        RadiationMCStep::ManagerType managerType =
            managerName == "p2p" ? RadiationMCStep::ManagerType::P2P :
            managerName == "new-rdma-ibv" ? RadiationMCStep::ManagerType::NEW_IBV_RDMA :
            RadiationMCStep::ManagerType::NEW_RDMA;
    #endif

    // --- Physical parameters (exactly as in the paper, Section 4.1) ---
    constexpr double domainLength = 4.0;
    constexpr double T_init       = 0.01;
    constexpr double T_boundary   = 1.0;
    constexpr double c_dt         = 0.01;    // c * dt
    constexpr double c_totalTime  = 500.0;   // c * t_final
    const     double dt           = c_dt / units::clight;
    const     size_t iterations   = static_cast<size_t>(c_totalTime / c_dt);

    // MC parameters (only used in mc mode)
    const size_t newPhotonsPerCell = std::max<size_t>(1, 50 / (Ny * Nz));
    const size_t particlesPerCell  = std::max<size_t>(4, 200 / (Ny * Nz));

    Vector3D ll(0, 0, 0), ur(domainLength, 1, 1);

    // --- Generate grid points on rank 0, spread to all ranks ---
    std::vector<Vector3D> points;
    if(rank == 0)
    {
        double dx = (ur.x - ll.x) / Nx;
        double dy = (ur.y - ll.y) / Ny;
        double dz = (ur.z - ll.z) / Nz;
        for(size_t i = 0; i < Nx; i++)
            for(size_t j = 0; j < Ny; j++)
                for(size_t k = 0; k < Nz; k++)
                    points.emplace_back(ll.x + (i + 0.5) * dx,
                                        ll.y + (j + 0.5) * dy,
                                        ll.z + (k + 0.5) * dz);
        std::cout << "Generated " << points.size() << " points ("
                  << Nx << " x " << Ny << " x " << Nz << ")" << std::endl;
    }

    points = MPI_Spread(points, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    Voronoi3D tess(ll, ur);
    tess.BuildParallel(points);

    // --- Equation of State: Cv = 7.14 * a_rad ---
    IdealGas eos(1.5, 7.14 * units::arad, 1, 0);

    // --- Initial conditions ---
    ComputationalCell3D initialCell;
    initialCell.density  = 1;
    initialCell.temperature = T_init;
    initialCell.velocity = Vector3D(0, 0, 0);
    initialCell.internal_energy = eos.dT2e(initialCell.density, initialCell.temperature,
                                           initialCell.tracers, ComputationalCell3D::tracerNames);
    initialCell.Erad = units::arad * std::pow(T_init, 4) / initialCell.density;

    size_t N = tess.GetPointNo();
    std::vector<ComputationalCell3D> initialCells(N, initialCell);

    // --- Simulation ---
    Simulation sim(tess, initialCells, eos);
    std::shared_ptr<TimeStepFunction3D> tsc = std::make_shared<ManualTimeStep>();
    sim.SetTimeStepFunction(tsc);

    std::vector<ComputationalCell3D> &cells = sim.getCells();
    std::vector<Conserved3D> &extensives = sim.getExtensives();
    extensives.resize(cells.size());
    for(size_t i = 0; i < cells.size(); i++)
        PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);

    // --- Set up physics depending on mode ---
    auto eosPtr = std::make_shared<IdealGas>(eos);

    if(mode == "mc")
    {
        auto opacityPtr = std::make_shared<MCPowerLawOpacity>(10, 0, 0, -3, 0, 0);
        constexpr bool withHydro = false;
        constexpr size_t boundaryPhotonsPerCell = 100;

        std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> boundaryCond =
            std::make_shared<SideTemperature<Vector3D, Tessellation3D>>(
                tess, cells, T_boundary, boundaryPhotonsPerCell, withHydro);

        STORM::RadiationIMCParameters<ENERGY_GROUPS_NUM> radiationIMCParameters = {
            .newPhotonsPerCell = newPhotonsPerCell,
            .withHydro = withHydro
        };
        std::shared_ptr<MonteCarloRadiationPhysics3D> physics = std::make_shared<::RadiationIMC>(
            tess, boundaryCond, cells, extensives, eosPtr, opacityPtr, radiationIMCParameters);
        std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> popControl =
            std::make_shared<CombPopulationControl<Vector3D, Tessellation3D>>(tess, particlesPerCell);

        std::vector<Particle3D> initialParticles;
        constexpr size_t initialParticlesPerCell = 0;
        auto mcStep = std::make_shared<RadiationMCStep>(
            tess, cells, extensives, physics, popControl, boundaryCond, initialParticles, initialParticlesPerCell, withHydro
            #ifdef RICH_MPI
                , managerType
            #endif
        );
        sim.addPhysics(mcStep);

        if(rank == 0)
        {
            std::cout << "Mode: Monte Carlo (IMC)"
                      << ", new photons/cell=" << newPhotonsPerCell
                      << ", max photons/cell=" << particlesPerCell
                      << ", manager=" << managerName << std::endl;
        }
    }
    else
    {
        // Diffusion modes: "diffusion" or "diffusion-fl"
        bool useFluxLimiter = (mode == "diffusion-fl");

        // D(T) = c/(3*sigma_a) = c*T^3/30
        // PowerLawOpacity: D = D0 * rho^alpha * T^beta
        double D0 = units::clight / 30.0;
        // Planck opacity: sigma_planck = 10 * T^{-3}
        double planck0 = 10.0;

        static PowerLawOpacity diffOpacity(D0, 0, 3, planck0, 0, -3);
        static DiffusionSideBoundary diffBoundary(T_boundary);
        static Diffusion diffusion(diffOpacity, diffBoundary, eos, std::vector<std::string>(),
            useFluxLimiter, /*hydro_on=*/false, /*compton_on=*/false);

        auto radStep = std::make_shared<RadiationStep>(
            tess, cells, extensives, sim.getTracker(),
            #ifdef RICH_MPI
                std::make_shared<CostCalculator3D>(),
            #endif
            diffusion, /*no_hydro=*/false);
        sim.addPhysics(radStep);

        if(rank == 0)
        {
            std::cout << "Mode: Diffusion"
                      << (useFluxLimiter ? " (with flux limiter)" : " (no flux limiter)")
                      << std::endl;
        }
    }

    sim.SetTimeStep(dt);

    if(rank == 0)
    {
        std::cout << "Marshak wave: Nx=" << Nx << ", Ny=" << Ny << ", Nz=" << Nz
                  << ", c*dt=" << c_dt << ", c*t_final=" << c_totalTime
                  << ", iterations=" << iterations << std::endl;
    }

    // --- Helper to write temperature profile ---
    struct CellData : public Serializable
    {
        double x;
        double temperature;

        CellData() : x(0), temperature(0) {}
        CellData(double x_, double T_) : x(x_), temperature(T_) {}

        size_t dump(Serializer *serializer) const override
        {
            size_t off = 0;
            off += serializer->insert(this->x);
            off += serializer->insert(this->temperature);
            return off;
        }

        size_t load(const Serializer *serializer, std::size_t offset)
        {
            size_t rd = 0;
            rd += serializer->extract(this->x, offset);
            rd += serializer->extract(this->temperature, offset + rd);
            return rd;
        }

        bool operator<(const CellData &o) const { return x < o.x; }
    };

    auto writeResults = [&](const std::string &filename, double ct_value)
    {
        size_t nPoints = tess.GetPointNo();
        std::vector<CellData> cellData(nPoints);
        for(size_t i = 0; i < nPoints; i++)
        {
            cellData[i].x = tess.GetMeshPoint(i).x;
            cellData[i].temperature = cells[i].temperature;
        }
        cellData = MPI_Gatherv_serializable(cellData, 0, MPI_COMM_WORLD);
        if(rank == 0)
        {
            std::sort(cellData.begin(), cellData.end());
            std::ofstream out(filename);
            out << "# ct=" << ct_value << " Nx=" << Nx << " mode=" << mode << "\n";
            out << "# x, T\n";
            double prevX = -1;
            double sumT = 0;
            int count = 0;
            for(size_t i = 0; i < cellData.size(); i++)
            {
                if(count > 0 && std::abs(cellData[i].x - prevX) > 1e-10)
                {
                    out << prevX << ", " << sumT / count << "\n";
                    sumT = 0;
                    count = 0;
                }
                prevX = cellData[i].x;
                sumT += cellData[i].temperature;
                count++;
            }
            if(count > 0)
                out << prevX << ", " << sumT / count << "\n";
            out.close();
            std::cout << "Wrote " << filename << std::endl;
        }
    };

    // Snapshot times (in c*t units) at which to dump temperature profiles
    const std::vector<double> snapshotCT = {1, 5, 10, 50, 100, 200, 300, 400, 500};
    size_t nextSnapshot = 0;

    // --- Main time-stepping loop ---
    double simTime = 0;
    auto startWall = std::chrono::high_resolution_clock::now();

    for(size_t i = 0; i < iterations; i++)
    {
        auto stepStart = std::chrono::high_resolution_clock::now();

        sim.step();

        simTime += dt;
        sim.SetTimeStep(dt);

        auto stepEnd = std::chrono::high_resolution_clock::now();
        double stepSec = std::chrono::duration<double>(stepEnd - stepStart).count();
        double elapsedWall = std::chrono::duration<double>(stepEnd - startWall).count();

        double fraction = static_cast<double>(i + 1) / iterations;
        double eta = (fraction > 0) ? elapsedWall * (1.0 - fraction) / fraction : 0;

        if(rank == 0 && (i % 100 == 0 || i + 1 == iterations))
        {
            int pct = static_cast<int>(fraction * 100);
            int etaMin = static_cast<int>(eta) / 60;
            int etaSec = static_cast<int>(eta) % 60;
            std::cout << "Cycle " << i + 1 << "/" << iterations
                      << " (" << pct << "%)"
                      << "  ct=" << simTime * units::clight
                      << "  step=" << stepSec << "s"
                      << "  ETA=" << etaMin << "m" << etaSec << "s"
                      << std::endl;
        }

        // Write snapshot if we've crossed a snapshot time
        double ct = simTime * units::clight;
        while(nextSnapshot < snapshotCT.size() && ct >= snapshotCT[nextSnapshot])
        {
            std::string snapName = prefix + "_ct" + std::to_string(static_cast<int>(snapshotCT[nextSnapshot])) + ".txt";
            writeResults(snapName, ct);
            nextSnapshot++;
        }
    }

    auto endWall = std::chrono::high_resolution_clock::now();
    double wallSec = std::chrono::duration<double>(endWall - startWall).count();
    if(rank == 0)
        std::cout << "Total wall time: " << wallSec << "s" << std::endl;

    // --- Write final results ---
    writeResults(prefix + "_final.txt", simTime * units::clight);

    size_t Nfinal = tess.GetPointNo();
    std::vector<double> temperatures(Nfinal);
    for(size_t i = 0; i < Nfinal; i++)
        temperatures[i] = cells[i].temperature;
    WriteVoronoiParallel(tess, prefix + "_final.vtu", {temperatures}, {"Temperature"});

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
}
