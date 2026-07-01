#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/Radiation/CMMC/src/units/units.hpp"
#include "source/Radiation/CMMC/src/planck_integral/planck_integral.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"
#include "source/newtonian/three_dimensional/conserved_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/ManualTimeStep.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/3D/radiation/RadiationIMC.hpp"
#include "source/3D/radiation/RadiationOpacity.hpp"
#include "source/3D/radiation/PowerLawOpacity.hpp"
#include "source/monte/population/CombPopulationControl.hpp"
#include "source/monte/boundary/SideTemperature.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"

namespace fs = std::filesystem;

/*
 * Densmore 2012 heterogeneous step-opacity — serial Monte Carlo regression test.
 *
 * Same physics as the MPI variant but runs serially with random walk enabled.
 * Outputs desmore2012_mc_serial_profile.txt for comparison.
 */

namespace
{
    class DesmoreMCOpacity : public OpacityCalculator
    {
    public:
        DesmoreMCOpacity(double sigma0_left, double sigma0_right,
                         const std::vector<double> &groupCenters,
                         const std::vector<double> &groupBoundaries)
            : sigma0_left_(sigma0_left), sigma0_right_(sigma0_right),
              groupCenters_(groupCenters), groupBoundaries_(groupBoundaries)
        {}

        double CalcPlanckOpacity(const ComputationalCell3D &cell) const override
        {
            double sigma0 = getSigma0(cell);
            double kT = units::k_boltz * cell.temperature;
            double sqrtKT = std::sqrt(kT);
            size_t G = groupCenters_.size();

            double weightedSum = 0;
            double totalWeight = 0;
            for(size_t g = 0; g < G; g++)
            {
                double a = groupBoundaries_[g] / kT;
                double b = groupBoundaries_[g + 1] / kT;
                double Bg = ::planck_integral::planck_integral(a, b);
                double sigma_g = sigma0 / (sqrtKT * groupCenters_[g] * groupCenters_[g] * groupCenters_[g]);
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
            energy = std::clamp(energy, groupBoundaries_.front(), groupBoundaries_.back());
            auto it = std::upper_bound(groupBoundaries_.begin(), groupBoundaries_.end(), energy);
            size_t idx = static_cast<size_t>(std::distance(groupBoundaries_.begin(), it));
            size_t g = (idx == 0) ? 0 : std::min(idx - 1, groupCenters_.size() - 1);
            double Eg = groupCenters_[g];
            return sigma0 / (std::sqrt(kT) * Eg * Eg * Eg);
        }

    private:
        double getSigma0(const ComputationalCell3D &cell) const
        {
            return cell.tracers[0] > 0.5 ? sigma0_left_ : sigma0_right_;
        }

        double sigma0_left_;
        double sigma0_right_;
        std::vector<double> groupCenters_;
        std::vector<double> groupBoundaries_;
    };
}

int main(int /*argc*/, char * /*argv*/[])
{
  try
  {
    constexpr size_t Nx = 256;
    constexpr size_t newPhotonsPerCell = 50;
    constexpr size_t maxPhotonsPerCell = 200;
    constexpr bool   useRandomWalk = true;

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

    ComputationalCell3D::stickerNames.push_back("Left");
    ComputationalCell3D::stickerNames.push_back("Right");

    double const cv = 1e15 / units::kev_kelvin;
    IdealGas eos(1.4, cv, 1, 0);

    double const sigma_0_left  = 10.0   * std::pow(units::kev, 3.5);
    double const sigma_0_right = 1000.0 * std::pow(units::kev, 3.5);

    constexpr double domainLength = 3.0;
    double const T_init = units::ev_kelvin;
    double const T_boundary = units::kev_kelvin;

    double const tf = 1e-9;
    double const dt = 5e-12;
    size_t const iterations = static_cast<size_t>(tf / dt);

    double const width = domainLength;
    Vector3D ll(0, -0.5 * width / Nx, -0.5 * width / Nx);
    Vector3D ur(width, 0.5 * width / Nx, 0.5 * width / Nx);

    std::vector<Vector3D> points = CartesianMesh(Nx, 1, 1, ll, ur);

    Voronoi3D tess(ll, ur);
    tess.Build(points);

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

    Simulation sim(tess, initialCells, eos);
    std::shared_ptr<TimeStepFunction3D> tsc = std::make_shared<ManualTimeStep>();
    sim.SetTimeStepFunction(tsc);

    std::vector<ComputationalCell3D> &cells = sim.getCells();
    std::vector<Conserved3D> &extensives = sim.getExtensives();
    extensives.resize(cells.size());
    for(size_t i = 0; i < cells.size(); i++)
        PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);

    auto eosPtr = std::make_shared<IdealGas>(eos);
    auto opacityPtr = std::make_shared<DesmoreMCOpacity>(
        sigma_0_left, sigma_0_right, energy_groups_center, energy_groups_boundary);

    constexpr bool withHydro = false;
    constexpr size_t boundaryPhotonsPerCell = 100;

    std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> boundaryCond =
        std::make_shared<SideTemperature<Vector3D, Tessellation3D>>(
            tess, T_boundary, boundaryPhotonsPerCell, energy_groups_boundary);

    STORM::RadiationIMCParameters<ENERGY_GROUPS_NUM> radiationIMCParameters = {
        .newPhotonsPerCell = newPhotonsPerCell,
        .withHydro = withHydro,
        .diffusionPressureGradient = false,
        .MMC = false,
        .withMultigroupOpacity = true,
        .withRandomWalk = useRandomWalk
    };
    std::shared_ptr<MonteCarloRadiationPhysics3D> physics = std::make_shared<::RadiationIMC>(
        tess, boundaryCond, cells, extensives, eosPtr, opacityPtr, radiationIMCParameters);

    std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> popControl =
        std::make_shared<CombPopulationControl<Vector3D, Tessellation3D>>(tess, maxPhotonsPerCell, 5);

    std::vector<Particle3D> initialParticles;
    size_t initialParticlesPerCell = 0;
    auto mcStep = std::make_shared<RadiationMCStep>(
        tess, cells, extensives, physics, popControl, boundaryCond,
        initialParticles, initialParticlesPerCell, withHydro
    );
    sim.addPhysics(mcStep);
    sim.SetTimeStep(dt);

    std::cout << "Densmore 2012 heterogeneous step-opacity (serial MC + RW)"
              << "\n  Nx=" << Nx << ", G=" << G
              << ", new/cell=" << newPhotonsPerCell
              << ", max/cell=" << maxPhotonsPerCell
              << "\n  dt=" << dt << " s, t_final=" << tf << " s"
              << ", iterations=" << iterations
              << std::endl;

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

        if(i % 10 == 0 || i + 1 == iterations)
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
    }

    auto endWall = std::chrono::high_resolution_clock::now();
    double wallSec = std::chrono::duration<double>(endWall - startWall).count();
    std::cout << "Total wall time: " << wallSec << "s" << std::endl;

    size_t nPoints = tess.GetPointNo();
    std::string const caseDir = fs::path(__FILE__).parent_path().string();
    std::string const profilePath = caseDir + "/desmore2012_mc_serial_profile.txt";
    std::ofstream out(profilePath);
    out << "# Densmore2012 serial MC+RW  t=" << simTime << "  Nx=" << Nx << "\n";
    out << "# x(cm)  T(K)\n";

    std::vector<std::pair<double, double>> cellData(nPoints);
    for(size_t i = 0; i < nPoints; i++)
        cellData[i] = {tess.GetMeshPoint(i).x, cells[i].temperature};

    std::sort(cellData.begin(), cellData.end());
    for(auto const &[x, T] : cellData)
        out << x << " " << T << "\n";
    out.close();
    std::cout << "Wrote " << profilePath << std::endl;

  }
  catch(const UniversalError &e)
  {
      std::cerr << "=== UniversalError ===" << std::endl;
      reportError(e);
      return 1;
  }
  catch(const std::exception &e)
  {
      std::cerr << "=== std::exception: " << e.what() << " ===" << std::endl;
      return 1;
  }

    return 0;
}
