#include <mpi.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include "mpi/mpi_commands.hpp"
#include "3D/tessellation/Voronoi3D.hpp"
#include "CMMC/src/units/units.hpp"
#include "newtonian/common/ideal_gas.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "newtonian/three_dimensional/simulation/Simulation.hpp"
#include "newtonian/three_dimensional/ManualTimeStep.hpp"
#include "misc/mesh_generator3D.hpp"

#include "3D/radiation/RadiationIMC.hpp"
#include "3D/radiation/PowerLawOpacity.hpp"
#include "monte/population/CombPopulationControl.hpp"
#include "monte/boundary/SideTemperature.hpp"
#include "newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"
#include "newtonian/three_dimensional/CostCalculator3D.hpp"
#include "runs/mc_results_dir.hpp"

/*
 * Gray Marshak wave test for Random Walk validation.
 *
 *   Domain:      x in [0, 2000]
 *   Left BC:     Blackbody source, T = 1 keV
 *   Opacity:     sigma = (T / 1keV)^{-3}  (pure absorption, no scattering)
 *   EOS:         ideal gas, gamma = 1.4, Cv = 1e15 / kev_kelvin
 *   Init:        T = 1 eV, rho = 1
 *   Time step:   starts at 1e-9, grows 5% per cycle, capped at 1e-6
 *   No hydro, gray (no multigroup).
 *
 * Usage: mpirun -np N ./rich <Nx> [prefix] [photons/cell] [max_photons] [rw(0/1)]
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

    if(rank == 0 && argc < 2)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <Nx> [prefix] [photons/cell] [max_photons] [rw(0/1)]"
                  << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

  try
  {
    size_t Nx = std::stoul(argv[1]);
    std::string prefix = (argc >= 3) ? argv[2] : (McResultsDirectory("GrayMarshak") + "/gray_marshak");
    if(prefix.find('/') == std::string::npos)
    {
        prefix = McResultsDirectory("GrayMarshak") + "/" + prefix;
    }
    EnsureParentDirectory(prefix, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    size_t newPhotonsPerCell = (argc >= 4) ? std::stoul(argv[3]) : 50;
    size_t maxPhotonsPerCell = (argc >= 5) ? std::stoul(argv[4]) : 50;
    bool useRandomWalk = (argc >= 6) ? (std::stoi(argv[5]) != 0) : true;

    // Energy groups (gray mode — groups unused but must be initialized)
    size_t const G = ENERGY_GROUPS_NUM;
    double const Emin = units::kev * 1e-4;
    double const Emax = units::kev * 1e2;
    std::vector<double> energy_groups_boundary(G + 1);
    energy_groups_boundary[0] = Emin;
    for(size_t g = 0; g < G; g++)
        energy_groups_boundary[g + 1] = std::pow(Emax / Emin, 1.0 / G) * energy_groups_boundary[g];
    for(size_t g = 0; g <= G; g++)
        ComputationalCell3D::energyBoundaries[g] = energy_groups_boundary[g];

    // Physical parameters
    double const cv = 1e9 / units::kev_kelvin;
    IdealGas eos(1.4, cv, 1, 0);

    // sigma = 30 * (T/kev)^{-1} and constant scattering = 0.01
    double const sigmaA0 = 30.0 * units::kev_kelvin;

    constexpr double domainLength = 100.0;
    double const T_init = 50.0 * units::ev_kelvin;
    double const T_boundary = units::kev_kelvin;

    double const tf = 1e-7;
    double const dt_init = 1e-12;

    double const width = domainLength;
    Vector3D ll(0, -0.5 * width / Nx, -0.5 * width / Nx);
    Vector3D ur(width, 0.5 * width / Nx, 0.5 * width / Nx);

    // Generate mesh
    std::vector<Vector3D> points;
    if(rank == 0)
        points = CartesianMesh(Nx, 1, 1, ll, ur);

    points = MPI_Spread(points, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    Voronoi3D tess(ll, ur);
    tess.BuildParallel(points);

    // Initial conditions
    ComputationalCell3D init_cell;
    init_cell.density = 1.0;
    init_cell.temperature = T_init;
    init_cell.velocity = Vector3D(0, 0, 0);
    init_cell.internal_energy = eos.dT2e(init_cell.density, init_cell.temperature,
                                         init_cell.tracers, ComputationalCell3D::tracerNames);
    init_cell.pressure = eos.de2p(init_cell.density, init_cell.internal_energy,
                                  init_cell.tracers, ComputationalCell3D::tracerNames);
    init_cell.Erad = units::arad * std::pow(T_init, 4) / init_cell.density;

    size_t N = tess.GetPointNo();

    std::vector<ComputationalCell3D> initialCells(N, init_cell);

    // Simulation
    Simulation sim(tess, initialCells, eos);
    auto tsc = std::make_shared<ManualTimeStep>();
    sim.SetTimeStepFunction(tsc);

    std::vector<ComputationalCell3D> &cells = sim.getCells();
    std::vector<Conserved3D> &extensives = sim.getExtensives();
    extensives.resize(cells.size());
    for(size_t i = 0; i < cells.size(); i++)
        PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);

    // MC radiation setup — gray
    auto eosPtr = std::make_shared<IdealGas>(eos);
    auto opacityPtr = std::make_shared<MCPowerLawOpacity>(
        sigmaA0, /*sigmaS0=*/0.01,
        /*sigmaA_rho=*/0.0, /*sigmaA_T=*/-1.0,
        /*sigmaS_rho=*/0.0, /*sigmaS_T=*/0.0);

    constexpr bool withHydro = false;
    constexpr size_t boundaryPhotonsPerCell = 100;

    auto boundaryCond = std::make_shared<SideTemperature<Vector3D, Tessellation3D>>(
        tess, cells, T_boundary, boundaryPhotonsPerCell, /*multigroup=*/false);

    STORM::RadiationIMCParameters<ENERGY_GROUPS_NUM> radiationIMCParameters = {
        .newPhotonsPerCell = newPhotonsPerCell,
        .withHydro = withHydro,
        .diffusionPressureGradient = false,
        .MMC = false,
        .withMultigroupOpacity = false,
        .withRandomWalk = useRandomWalk
    };
    auto physics = std::make_shared<::RadiationIMC>(
        tess, boundaryCond, cells, extensives, eosPtr, opacityPtr, radiationIMCParameters);

    auto popControl = std::make_shared<CombPopulationControl<Vector3D, Tessellation3D>>(
        tess, maxPhotonsPerCell, 5);

    std::vector<Particle3D> initialParticles;
    size_t initialParticlesPerCell = 0;
    auto mcStep = std::make_shared<RadiationMCStep>(
        tess, cells, extensives, physics, popControl, boundaryCond,
        initialParticles, initialParticlesPerCell, withHydro
        #ifdef RICH_MPI
            , RadiationMCStep::ManagerType::AUTO_RDMA
        #endif
    );
    sim.addPhysics(mcStep);

    if(rank == 0)
    {
        std::cout << "Gray Marshak wave (RW validation)"
                  << "\n  Nx=" << Nx
                  << ", new/cell=" << newPhotonsPerCell
                  << ", max/cell=" << maxPhotonsPerCell
                  << ", RW=" << useRandomWalk
                  << "\n  T_init=" << T_init / units::ev_kelvin << " eV"
                  << ", T_boundary=" << T_boundary / units::kev_kelvin << " keV"
                  << "\n  sigmaA0=" << sigmaA0 << " (sigma=30*(T/keV)^-1)"
                  << ", sigmaS0=0.01"
                  << "\n  dt start=" << dt_init << ", dt=max(3e-10, sqrt(t)/1e5)"
                  << ", t_final=" << tf
                  << "\n  prefix=" << prefix
                  << std::endl;
    }

    // Helper to write temperature profile
    struct CellData
        #ifdef RICH_MPI
            : public Serializable
        #endif
    {
        double x, temperature, Erad, Erad_time_avg;

        CellData() : x(0), temperature(0), Erad(0), Erad_time_avg(0) {}
        CellData(double x_, double T_, double Er_, double ErAvg_)
            : x(x_), temperature(T_), Erad(Er_), Erad_time_avg(ErAvg_) {}

        #ifdef RICH_MPI
        size_t dump(Serializer *serializer) const override
        {
            size_t off = 0;
            off += serializer->insert(this->x);
            off += serializer->insert(this->temperature);
            off += serializer->insert(this->Erad);
            off += serializer->insert(this->Erad_time_avg);
            return off;
        }
        size_t load(const Serializer *serializer, std::size_t offset)
        {
            size_t rd = 0;
            rd += serializer->extract(this->x, offset);
            rd += serializer->extract(this->temperature, offset + rd);
            rd += serializer->extract(this->Erad, offset + rd);
            rd += serializer->extract(this->Erad_time_avg, offset + rd);
            return rd;
        }
        #endif

        bool operator<(const CellData &o) const { return x < o.x; }
    };

    auto writeResults = [&](const std::string &filename, double time)
    {
        const std::vector<double> &EradAvg = physics->getEradTimeAvg();
        size_t nPoints = tess.GetPointNo();
        std::vector<CellData> cellData(nPoints);
        for(size_t i = 0; i < nPoints; i++)
        {
            cellData[i].x = tess.GetMeshPoint(i).x;
            cellData[i].temperature = cells[i].temperature;
            cellData[i].Erad = cells[i].Erad * cells[i].density;
            cellData[i].Erad_time_avg = (i < EradAvg.size()) ? EradAvg[i] : 0;
        }
        #ifdef RICH_MPI
        cellData = MPI_Gatherv_serializable(cellData, 0, MPI_COMM_WORLD);
        #endif
        if(rank == 0)
        {
            std::sort(cellData.begin(), cellData.end());
            std::ofstream out(filename);
            out << "# Gray Marshak wave  t=" << time << "  Nx=" << Nx << "\n";
            out << "# x, T(K), Erad(erg/cm^3), Erad_time_avg(erg/cm^3)\n";
            double prevX = -1;
            double sumT = 0, sumErad = 0, sumEradAvg = 0;
            int count = 0;
            for(size_t i = 0; i < cellData.size(); i++)
            {
                if(count > 0 && std::abs(cellData[i].x - prevX) > 1e-10)
                {
                    out << prevX << ", " << sumT / count << ", "
                        << sumErad / count << ", " << sumEradAvg / count << "\n";
                    sumT = sumErad = sumEradAvg = 0;
                    count = 0;
                }
                prevX = cellData[i].x;
                sumT += cellData[i].temperature;
                sumErad += cellData[i].Erad;
                sumEradAvg += cellData[i].Erad_time_avg;
                count++;
            }
            if(count > 0)
                out << prevX << ", " << sumT / count << ", "
                    << sumErad / count << ", " << sumEradAvg / count << "\n";
            out.close();
            std::cout << "Wrote " << filename << std::endl;
        }
    };

    // Main time-stepping loop with adaptive dt
    double simTime = 0;
    double dt = dt_init;
    size_t cycle = 0;
    auto startWall = std::chrono::high_resolution_clock::now();

    while(simTime < tf)
    {
        if(simTime + dt > tf)
            dt = tf - simTime;

        sim.SetTimeStep(dt);

        auto stepStart = std::chrono::high_resolution_clock::now();
        sim.step();
        auto stepEnd = std::chrono::high_resolution_clock::now();

        simTime += dt;
        cycle++;
        double stepSec = std::chrono::duration<double>(stepEnd - stepStart).count();
        double elapsedWall = std::chrono::duration<double>(stepEnd - startWall).count();
        double eta = (simTime > 0) ? elapsedWall * (tf - simTime) / simTime : 0;

        if(rank == 0 && (cycle % 20 == 0 || simTime >= tf))
        {
            int etaMin = static_cast<int>(eta) / 60;
            int etaSec = static_cast<int>(eta) % 60;
            std::cout << "Cycle " << cycle
                      << "  t=" << simTime
                      << "  dt=" << dt
                      << "  step=" << stepSec << "s"
                      << "  ETA=" << etaMin << "m" << etaSec << "s"
                      << std::endl;
        }

        if(cycle % 50 == 0)
        {
            char buf[512];
            std::snprintf(buf, sizeof(buf), "%s_%05zu.txt", prefix.c_str(), cycle);
            writeResults(buf, simTime);
        }

        dt = std::max(1e-12, std::sqrt(simTime) / 1e6);
    }

    auto endWall = std::chrono::high_resolution_clock::now();
    double wallSec = std::chrono::duration<double>(endWall - startWall).count();
    if(rank == 0)
        std::cout << "Total wall time: " << wallSec << "s" << std::endl;

    writeResults(prefix + "_final.txt", simTime);

  }
  catch(const UniversalError &e)
  {
      reportError(e, std::cerr);
      std::cerr << std::flush;
      MPI_Abort(MPI_COMM_WORLD, 1);
  }
  catch(const std::exception &e)
  {
      std::cerr << "=== std::exception on rank " << rank << ": " << e.what()
                << " ===" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
  }

    MPI_Finalize();
    return 0;
}
