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
#include "Radiation/CMMC/src/units/units.hpp"
#include "Radiation/CMMC/src/planck_integral/planck_integral.hpp"
#include "newtonian/common/ideal_gas.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "newtonian/three_dimensional/simulation/Simulation.hpp"
#include "newtonian/three_dimensional/ManualTimeStep.hpp"
#include "misc/mesh_generator3D.hpp"

#include "3D/radiation/RadiationIMC.hpp"
#include "3D/radiation/RadiationOpacity.hpp"
#include "3D/radiation/PowerLawOpacity.hpp"
#include "monte/population/Comb.hpp"
#include "monte/boundary/SideTemperature.hpp"
#include "newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"
#include "newtonian/three_dimensional/CostCalculator3D.hpp"
#include "utils/arguments/ArgumentParser.hpp"

/*
 * Desmore 2012 step-opacity test — Monte Carlo (multigroup IMC) version.
 *
 * Same problem as runs/desmore2012_step (multigroup diffusion):
 *   Domain:      x in [0, 3]
 *   Left BC:     Blackbody source, T = 1 keV
 *   Opacity:     sigma_a(E) = sigma_0 / (sqrt(kT) * E^3)
 *                Left  material (x < 2): sigma_0 = 10   * kev^3.5
 *                Right material (x >= 2): sigma_0 = 1000 * kev^3.5
 *   Scattering:  none
 *   EOS:         ideal gas, gamma = 1.4, Cv = 1e15/kev_kelvin
 *   Init:        T = 1 eV, rho = 1
 *   Runtime:     1e-9 s
 *   No hydro.
 *
 * Usage: mpirun -np N ./test [options] <Nx> [output_prefix] [new_photons_per_cell] [max_photons_per_cell]
 */

namespace
{
    class DesmoreMCOpacity : public OpacityCalculator
    {
    public:
        DesmoreMCOpacity(double sigma0_left, double sigma0_right,
                         const std::vector<double> &groupCenters,
                         const std::vector<double> &groupBoundaries,
                         int rank = 0)
            : sigma0_left(sigma0_left), sigma0_right(sigma0_right),
              groupCenters(groupCenters), groupBoundaries(groupBoundaries)
        {}

        double CalcPlanckOpacity(const ComputationalCell3D &cell) const override
        {
            double sigma0 = getSigma0(cell);
            double kT = units::k_boltz * cell.temperature;
            double sqrtKT = std::sqrt(kT);
            size_t G = groupCenters.size();

            double weightedSum = 0;
            double totalWeight = 0;
            for(size_t g = 0; g < G; g++)
            {
                double a = groupBoundaries[g] / kT;
                double b = groupBoundaries[g + 1] / kT;
                double Bg = planck_integral::planck_integral(a, b);
                double sigma_g = sigma0 / (sqrtKT * groupCenters[g] * groupCenters[g] * groupCenters[g]);
                weightedSum += sigma_g * Bg;
                totalWeight += Bg;
            }
            return weightedSum / totalWeight;
        }

        double CalcScatteringOpacity(const ComputationalCell3D &) const override
        {
            return 0.0;
        }

        double CalcAbsorptionOpacity(const ComputationalCell3D &cell, double energy) const override
        {
            double sigma0 = getSigma0(cell);
            double kT = units::k_boltz * cell.temperature;
            energy = std::clamp(energy, groupBoundaries.front(), groupBoundaries.back());
            auto it = std::upper_bound(groupBoundaries.begin(), groupBoundaries.end(), energy);
            size_t idx = static_cast<size_t>(std::distance(groupBoundaries.begin(), it));
            size_t g = (idx == 0) ? 0 : std::min(idx - 1, groupCenters.size() - 1);
            double Eg = groupCenters[g];
            return sigma0 / (std::sqrt(kT) * Eg * Eg * Eg);
        }

    private:
        double getSigma0(const ComputationalCell3D &cell) const
        {
            return cell.tracers[0] > 0.5 ? sigma0_left : sigma0_right;
        }

        double sigma0_left;
        double sigma0_right;
        std::vector<double> groupCenters;
        std::vector<double> groupBoundaries;
    };
}

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
    ArgumentParser arguments("Desmore 2012 step-opacity MC benchmark");
    arguments.addPositional<size_t>("Nx", "number of cells along x").required();
    arguments.addPositional<std::string>("prefix", "desmore_step_mc", "output prefix");
    arguments.addPositional<size_t>("new_photons_per_cell", 50, "new photons per cell per step");
    arguments.addPositional<size_t>("max_photons_per_cell", 200, "population-control photon cap per cell");
    arguments.addPositional<bool>("with_random_walk", true, "enable random walk acceleration")
        .optionAlias("random-walk")
        .flagAlias("rw", true)
        .flagAlias("no-rw", false);
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
    size_t newPhotonsPerCell = arguments.get<size_t>("new_photons_per_cell");
    size_t maxPhotonsPerCell = arguments.get<size_t>("max_photons_per_cell");
    bool useRandomWalk = arguments.get<bool>("with_random_walk");
    std::string managerName = arguments.get<std::string>("manager");

    #ifdef RICH_MPI
        RadiationMCStep::ManagerType managerType =
            managerName == "p2p" ? RadiationMCStep::ManagerType::P2P :
            managerName == "new-rdma-ibv" ? RadiationMCStep::ManagerType::NEW_IBV_RDMA :
            RadiationMCStep::ManagerType::NEW_RDMA;
    #endif

    // --- Energy groups (same as diffusion test) ---
    size_t const G = ENERGY_GROUPS_NUM;
    std::vector<double> energy_groups_center(G);
    std::vector<double> energy_groups_boundary(G + 1);

    double const Emin = units::kev * 1e-4;
    double const Emax = units::kev * 1e2;

    energy_groups_boundary[0] = Emin;
    for(size_t g = 0; g < G; g++)
    {
        energy_groups_boundary[g + 1] = std::pow(Emax / Emin, 1.0 / G) * energy_groups_boundary[g];
        energy_groups_center[g] = 0.5 * (energy_groups_boundary[g + 1] + energy_groups_boundary[g]);
    }

    for(size_t g = 0; g <= G; g++)
        ComputationalCell3D::energyBoundaries[g] = energy_groups_boundary[g];

    // --- Tracers for two-material setup ---
    ComputationalCell3D::stickerNames.push_back("Left");
    ComputationalCell3D::stickerNames.push_back("Right");

    // --- Physical parameters ---
    double const cv = 1e15 / units::kev_kelvin;
    IdealGas eos(1.4, cv, 1, 0);

    double const sigma_0_left  = 10.0   * std::pow(units::kev, 3.5);
    double const sigma_0_right = 1000.0 * std::pow(units::kev, 3.5);

    constexpr double domainLength = 3.0;
    double const T_init = units::ev_kelvin;       // 1 eV in Kelvin
    double const T_boundary = units::kev_kelvin;  // 1 keV in Kelvin

    double const tf = 1e-9;
    double const dt = 5e-12;
    size_t const iterations = static_cast<size_t>(tf / dt);

    double const width = domainLength;
    Vector3D ll(0, -0.5 * width / Nx, -0.5 * width / Nx);
    Vector3D ur(width, 0.5 * width / Nx, 0.5 * width / Nx);

    // --- Generate mesh ---
    std::vector<Vector3D> points;
    if(rank == 0)
        points = CartesianMesh(Nx, 1, 1, ll, ur);

    points = MPI_Spread(points, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    Voronoi3D tess(ll, ur);
    tess.BuildParallel(points);

    // --- Initial conditions ---
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
    for(size_t i = 0; i < N; i++)
    {
        if(tess.GetCellCM(i).x < 2.0)
            initialCells[i].tracers[0] = 1.0;
        else
            initialCells[i].tracers[1] = 1.0;
    }

    // --- Simulation ---
    Simulation sim(tess, initialCells, eos);
    std::shared_ptr<TimeStepFunction3D> tsc = std::make_shared<ManualTimeStep>();
    sim.SetTimeStepFunction(tsc);

    std::vector<ComputationalCell3D> &cells = sim.getCells();
    std::vector<Conserved3D> &extensives = sim.getExtensives();
    extensives.resize(cells.size());
    for(size_t i = 0; i < cells.size(); i++)
        PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);

    // --- MC radiation setup ---
    auto eosPtr = std::make_shared<IdealGas>(eos);
    auto opacityPtr = std::make_shared<DesmoreMCOpacity>(
        sigma_0_left, sigma_0_right, energy_groups_center, energy_groups_boundary, rank);

    constexpr bool withHydro = false;
    constexpr size_t boundaryPhotonsPerCell = 100;

    std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> boundaryCond =
        std::make_shared<SideTemperature<Vector3D, Tessellation3D>>(
            tess, cells, T_boundary, boundaryPhotonsPerCell, /*multigroup=*/true);

    RadiationIMCParameters radiationIMCParameters = {
        .newPhotonsPerCell = newPhotonsPerCell,
        .withHydro = withHydro,
        .diffusionPressureGradient = false,
        .MMC = false,
        .withMultigroupOpacity = true,
        .withRandomWalk = useRandomWalk
    };
    std::shared_ptr<MonteCarloRadiationPhysics3D> physics = std::make_shared<RadiationIMC>(
        tess, boundaryCond, cells, extensives, eosPtr, opacityPtr, radiationIMCParameters);

    std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> popControl =
        std::make_shared<CombPopulationControl<Vector3D, Tessellation3D>>(tess, maxPhotonsPerCell, 5);

    std::vector<Particle3D> initialParticles;
    size_t initialParticlesPerCell = 0;
    auto mcStep = std::make_shared<RadiationMCStep>(
        tess, cells, extensives, physics, popControl, boundaryCond,
        initialParticles, initialParticlesPerCell, withHydro
        #ifdef RICH_MPI
            , managerType
        #endif
    );
    sim.addPhysics(mcStep);

    sim.SetTimeStep(dt);

    if(rank == 0)
    {
        std::cout << "Desmore 2012 step-opacity (MC multigroup IMC)"
                  << "\n  Nx=" << Nx
                  << ", G=" << G
                  << ", new/cell=" << newPhotonsPerCell
                  << ", max/cell=" << maxPhotonsPerCell
                  << "\n  T_init=" << T_init / units::ev_kelvin << " eV"
                  << ", T_boundary=" << T_boundary / units::kev_kelvin << " keV"
                  << "\n  sigma_0_left=" << sigma_0_left
                  << ", sigma_0_right=" << sigma_0_right
                  << "\n  dt=" << dt << " s"
                  << ", t_final=" << tf << " s"
                  << ", iterations=" << iterations
                  << "\n  prefix=" << prefix
                  << ", random_walk=" << useRandomWalk
                  << ", manager=" << managerName
                  << std::endl;
    }

    // --- Helper to write temperature profile ---
    struct CellData
        #ifdef RICH_MPI
            : public Serializable
        #endif
    {
        double x;
        double temperature;
        double Erad;
        double Erad_time_avg;

        CellData() : x(0), temperature(0), Erad(0), Erad_time_avg(0) {}
        CellData(double x_, double T_, double Er_, double ErAvg_) : x(x_), temperature(T_), Erad(Er_), Erad_time_avg(ErAvg_) {}

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
            out << "# Desmore2012 step MC multigroup  t=" << time << "  Nx=" << Nx << "\n";
            out << "# x, T(K), Erad(erg/cm^3), Erad_time_avg(erg/cm^3)\n";
            double prevX = -1;
            double sumT = 0, sumErad = 0, sumEradAvg = 0;
            int count = 0;
            for(size_t i = 0; i < cellData.size(); i++)
            {
                if(count > 0 && std::abs(cellData[i].x - prevX) > 1e-10)
                {
                    out << prevX << ", " << sumT / count << ", " << sumErad / count << ", " << sumEradAvg / count << "\n";
                    sumT = 0;
                    sumErad = 0;
                    sumEradAvg = 0;
                    count = 0;
                }
                prevX = cellData[i].x;
                sumT += cellData[i].temperature;
                sumErad += cellData[i].Erad;
                sumEradAvg += cellData[i].Erad_time_avg;
                count++;
            }
            if(count > 0)
                out << prevX << ", " << sumT / count << ", " << sumErad / count << ", " << sumEradAvg / count << "\n";
            out.close();
            std::cout << "Wrote " << filename << std::endl;
        }
    };

    // Snapshot times (fractions of final time)
    const std::vector<double> snapshotFractions = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
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

        if(rank == 0 && (i % 10 == 0 || i + 1 == iterations))
        {
            int pct = static_cast<int>(fraction * 100);
            int etaMin = static_cast<int>(eta) / 60;
            int etaSec = static_cast<int>(eta) % 60;
            std::cout << "Cycle " << i + 1 << "/" << iterations
                      << " (" << pct << "%)"
                      << "  t=" << simTime
                      << "  step=" << stepSec << "s"
                      << "  ETA=" << etaMin << "m" << etaSec << "s"
                      << std::endl;
        }

        if((i + 1) % 20 == 0)
        {
            char buf[512];
            std::snprintf(buf, sizeof(buf), "%s_%05zu.txt", prefix.c_str(), i + 1);
            writeResults(buf, simTime);
        }

        while(nextSnapshot < snapshotFractions.size() && simTime >= snapshotFractions[nextSnapshot] * tf)
        {
            int snapIdx = static_cast<int>(snapshotFractions[nextSnapshot] * 10);
            std::string snapName = prefix + "_snap" + std::to_string(snapIdx) + ".txt";
            writeResults(snapName, simTime);
            nextSnapshot++;
        }
    }

    auto endWall = std::chrono::high_resolution_clock::now();
    double wallSec = std::chrono::duration<double>(endWall - startWall).count();
    if(rank == 0)
        std::cout << "Total wall time: " << wallSec << "s" << std::endl;

    writeResults(prefix + "_final.txt", simTime);

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
