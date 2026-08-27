#ifndef IMC_POSTPROCESS_TDE_RUNTIME_HPP
#define IMC_POSTPROCESS_TDE_RUNTIME_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "PostProcessConfig.hpp"

#include "source/3D/output/read3D.hpp"
#include "source/3D/output/Snapshot3D.hpp"
#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/3D/monte/MonteCarloManager3D.hpp"
#include "source/3D/radiation/RadiationIMC.hpp"
#include "source/3D/radiation/SphericalObserver.hpp"
#include "source/monte/boundary/Vacuum.hpp"
#include "source/monte/population/NoPopulationControl.hpp"
#include "source/newtonian/three_dimensional/OndrejEOS.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"
#include "source/newtonian/three_dimensional/conserved_3d.hpp"
#include "source/Radiation/MultigroupDiffusionCoefficientCalculator.hpp"
#include "source/Radiation/STAgreyOpacity.hpp"
#include "source/Radiation/Diffusion.hpp"
#include "CMMC/src/units/units.hpp"
#include "CMMC/src/planck_integral/planck_integral.hpp"
#include "source/misc/universal_error.hpp"
#include "source/misc/simple_io.hpp"
#include "source/misc/utils.hpp"

namespace imc_postprocess_tde {

class STAMGopacityMC : public OpacityCalculator
{
private:
    std::vector<double> rho_, T_;
    std::vector<std::vector<std::vector<double>>> planck_, scatter_;
    std::unordered_map<size_t, double> rosselandScale_;

public:
    void SetRosselandScaleFactors(std::unordered_map<size_t, double> factors)
    {
        rosselandScale_ = std::move(factors);
    }

    STAMGopacityMC(std::string const& file_directory)
    {
        energy_groups_boundary = read_vector(file_directory + "frequency_edges.txt");
        for (double& Egb : energy_groups_boundary)
            Egb *= 11604.5 * CG::boltzmann_constant;
        energy_groups_center.resize(energy_groups_boundary.size() - 1);
        for (size_t i = 0; i < energy_groups_boundary.size() - 1; ++i)
            energy_groups_center[i] = std::sqrt(energy_groups_boundary[i] * energy_groups_boundary[i + 1]);
        size_t const Ng = energy_groups_boundary.size() - 1;

        T_ = read_vector(file_directory + "T.txt");
        for (size_t i = 0; i < T_.size(); ++i)
        {
            T_[i] *= 11604.5;
            T_[i] = std::log(T_[i]);
        }
        size_t const Nt = T_.size();

        rho_ = read_vector(file_directory + "rho.txt");
        size_t const Nrho = rho_.size();
        for (size_t i = 0; i < Nrho; ++i)
            rho_[i] = std::log(rho_[i]);

        planck_.resize(Ng);
        scatter_.resize(Ng);
        for (size_t i = 0; i < Ng; ++i)
        {
            auto temp_planck = read_vector(file_directory + "sigma_absorption_rossland_" + std::to_string(i + 1) + ".txt");
            auto temp_scatter = read_vector(file_directory + "sigma_scattering_planck_" + std::to_string(i + 1) + ".txt");
            planck_[i].resize(Nrho);
            scatter_[i].resize(Nrho);
            for (size_t j = 0; j < Nrho; ++j)
            {
                planck_[i][j].resize(Nt);
                scatter_[i][j].resize(Nt);
                for (size_t k = 0; k < Nt; ++k)
                {
                    planck_[i][j][k] = std::log(temp_planck[j * Nt + k]) + rho_[j];
                    scatter_[i][j][k] = std::log(temp_scatter[j * Nt + k]) + rho_[j];
                }
            }
        }
    }

    double CalcAbsorptionOpacity(ComputationalCell3D const& cell, double energy) const override
    {
        size_t const group = findGroup(energy);
        double T = std::log(cell.temperature);
        double d = std::log(cell.density);
        double d_ratio = 1;
        double T_ratio = 1;
        if (d < rho_[0])
        {
            d_ratio = cell.density / std::exp(rho_[0]);
            d = rho_[0];
        }
        if (d > rho_.back())
        {
            d_ratio = cell.density / std::exp(rho_.back());
            d = rho_.back();
        }
        if (T < T_[0])
            T = T_[0];
        if (T > T_.back())
        {
            T_ratio = std::pow(cell.temperature / std::exp(T_.back()), -1.5);
            T = T_.back();
        }
        double result = std::exp(BiLinearInterpolation(rho_, T_, planck_[group], d, T)) * d_ratio * T_ratio;
        if (!rosselandScale_.empty()) {
            auto it = rosselandScale_.find(cell.ID);
            if (it != rosselandScale_.end())
                result *= it->second;
            else
                throw UniversalError("Cell ID not found in rosselandScale_");
        }
        return result;
    }

    double CalcScatteringOpacity(ComputationalCell3D const& cell, double energy) const override
    {
        size_t const group = findGroup(energy);
        double T = std::log(cell.temperature);
        double d = std::log(cell.density);
        double d_ratio = 1;
        if (d < rho_[0])
        {
            d_ratio = cell.density / std::exp(rho_[0]);
            d = rho_[0];
        }
        if (d > rho_.back())
        {
            d_ratio = cell.density / std::exp(rho_.back());
            d = rho_.back();
        }
        if (T < T_[0])
            T = T_[0];
        if (T > T_.back())
            T = T_.back();
        return std::exp(BiLinearInterpolation(rho_, T_, scatter_[group], d, T)) * d_ratio;
    }

    double CalcScatteringOpacity(ComputationalCell3D const& cell) const override
    {
        double avg = 0.0;
        for (size_t g = 0; g < energy_groups_center.size(); ++g)
            avg += CalcScatteringOpacity(cell, energy_groups_center[g]);
        return avg / static_cast<double>(energy_groups_center.size());
    }

    double CalcPlanckOpacity(ComputationalCell3D const& cell) const override
    {
        double kT = CG::boltzmann_constant * cell.temperature;
        double weightedSum = 0.0;
        double totalWeight = 0.0;
        for (size_t g = 0; g < energy_groups_center.size(); ++g)
        {
            double nu = energy_groups_center[g];
            double x = nu / kT;
            double planckWeight = (x > 0.0 && x < 500.0) ? x * x * x / std::expm1(x) : 0.0;
            weightedSum += CalcAbsorptionOpacity(cell, nu) * planckWeight;
            totalWeight += planckWeight;
        }
        return (totalWeight > 0.0) ? weightedSum / totalWeight : 0.0;
    }
};


using VacuumBoundary3D = VacuumBoundaryCondition<Vector3D, Tessellation3D>;
using NoPopulationControl3D = STORM::NoPopulationControl<Vector3D, Tessellation3D>;

inline std::shared_ptr<MonteCarloManager3D> CreateMonteCarloManager(
    Config const& config,
    Tessellation3D const& tessellation,
    std::shared_ptr<MonteCarloPhysics<Vector3D, Tessellation3D>> const& physics,
    std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> const& population,
    std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> const& boundary)
{
#ifdef RICH_MPI
    if (config.communication == MonteCarloCommunication::TwoSided) {
        return std::make_shared<TwoSidedMonteCarloManager3D>(
            tessellation, physics, population, boundary);
    }
    return std::make_shared<RDMAMonteCarloManager3D>(
        tessellation, physics, population, boundary);
#else
    (void)config;
    return std::make_shared<MonteCarloManagerSerial3D>(
        tessellation, physics, population, boundary);
#endif
}

class PostProcessSession
{
public:
    PostProcessSession(
        int rankIn,
        int mpiSizeIn,
        Voronoi3D& tessIn,
        std::vector<ComputationalCell3D>& cellsIn,
        std::vector<Conserved3D>& extensivesIn,
        std::shared_ptr<EquationOfState> eosIn,
        std::shared_ptr<OpacityCalculator> opacityIn,
        std::shared_ptr<OpacityCalculator> greyOpacityIn,
        std::shared_ptr<SphericalObserver> observerIn,
        std::shared_ptr<VacuumBoundary3D> boundaryIn,
        std::shared_ptr<NoPopulationControl3D> popControlIn,
        std::shared_ptr<RadiationIMC> physicsIn,
        std::shared_ptr<MonteCarloManager3D> managerIn,
        RadiationIMCParameters paramsIn,
        size_t nCellsIn,
        double snapshotTimeIn,
        int snapshotCycleIn,
        ComputationalCell3D dummyCellIn,
        std::vector<double> fldLuminosityIn,
        double totalFldLuminosityIn,
        std::function<void(std::unordered_map<size_t, double>)>
            applyOpacityScaleFactorsIn)
        : rank(rankIn), mpiSize(mpiSizeIn), tess(tessIn), cells(cellsIn),
          extensives(extensivesIn), eos(std::move(eosIn)),
          opacity(std::move(opacityIn)), greyOpacity(std::move(greyOpacityIn)),
          observer(std::move(observerIn)), boundary(std::move(boundaryIn)),
          popControl(std::move(popControlIn)), physics(std::move(physicsIn)),
          manager(std::move(managerIn)), params(std::move(paramsIn)),
          nCells(nCellsIn), snapshotTime(snapshotTimeIn),
          snapshotCycle(snapshotCycleIn), dummyCell(std::move(dummyCellIn)),
          fldLuminosity(std::move(fldLuminosityIn)),
          totalFldLuminosity(totalFldLuminosityIn),
          applyOpacityScaleFactors(std::move(applyOpacityScaleFactorsIn))
    {}

    PostProcessSession(PostProcessSession const&) = delete;
    PostProcessSession& operator=(PostProcessSession const&) = delete;
    PostProcessSession(PostProcessSession&&) = delete;
    PostProcessSession& operator=(PostProcessSession&&) = delete;

    int rank;
    int mpiSize;
    Voronoi3D& tess;
    std::vector<ComputationalCell3D>& cells;
    std::vector<Conserved3D>& extensives;
    std::shared_ptr<EquationOfState> eos;
    std::shared_ptr<OpacityCalculator> opacity;
    std::shared_ptr<OpacityCalculator> greyOpacity;
    std::shared_ptr<SphericalObserver> observer;
    std::shared_ptr<VacuumBoundary3D> boundary;
    std::shared_ptr<NoPopulationControl3D> popControl;
    std::shared_ptr<RadiationIMC> physics;
    std::shared_ptr<MonteCarloManager3D> manager;
    RadiationIMCParameters params;
    size_t nCells;
    double snapshotTime;
    int snapshotCycle;
    ComputationalCell3D dummyCell;
    std::vector<double> fldLuminosity;
    double totalFldLuminosity;
    std::function<void(std::unordered_map<size_t, double>)>
        applyOpacityScaleFactors;
    bool fluxSourceEnabled = false;
    double fluxSourceTau = 0.0;
    std::vector<Vector3D> fluxSourceDirections;
    std::vector<double> fluxSourceRadius;
    std::vector<int> fluxSourceRadiusDirectlyResolved;
    double fluxSourceDirectlyResolvedFraction = 0.0;
    double fluxSourceInjectedLuminosity = 0.0;
    double fluxSourceNetLuminosity = 0.0;
    double fluxSourceInwardLuminosity = 0.0;
    uint64_t fluxSourceBoundaryFaceCount = 0;
    uint64_t fluxSourceEmittingFaceCount = 0;
};

using PostprocessRuntime = PostProcessSession;

struct ForwardPostprocessResult
{
    bool ran = false;
    bool usesVelocity = false;
    bool usesDDMC = false;
    bool usesPolarization = false;
    bool usesCompton = false;
    std::vector<double> fleckFactors;
    double sourceLuminosity = 0.0;
    double emittedLuminosity = 0.0;
    double crossingLuminosity = 0.0;
    double crossingLuminosityStderr = 0.0;
    double emittedEnergy = 0.0;
    double timedOutFraction = 0.0;
    double luminosityWeightedPolarizationDegree = 0.0;
    uint64_t polarizedObserverCount = 0;
};

} // namespace imc_postprocess_tde

#endif // IMC_POSTPROCESS_TDE_RUNTIME_HPP
