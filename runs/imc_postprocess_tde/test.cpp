#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "source/3D/output/read3D.hpp"
#include "source/3D/output/Snapshot3D.hpp"
#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/3D/monte/MonteCarloManager3D.hpp"
#include "source/3D/radiation/RadiationIMC.hpp"
#include "source/3D/radiation/SphericalObserver.hpp"
#include "source/3D/radiation/IMCMeasuredLoadBalance.hpp"
#include "source/3D/radiation/IMCStepCounterCostCalculator.hpp"
#include "source/monte/boundary/Vacuum.hpp"
#include "source/monte/population/NoControl.hpp"
#include "source/newtonian/three_dimensional/OndrejEOS.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"
#include "source/newtonian/three_dimensional/conserved_3d.hpp"
#include "source/Radiation/MultigroupDiffusionCoefficientCalculator.hpp"
#include "source/Radiation/STAgreyOpacity.hpp"
#include "source/Radiation/Diffusion.hpp"
#include "source/Radiation/CMMC/src/units/units.hpp"
#include "source/Radiation/CMMC/src/planck_integral/planck_integral.hpp"
#include "source/misc/universal_error.hpp"
#include "source/misc/simple_io.hpp"
#include "source/misc/utils.hpp"
#include "source/3D/radiation/ReverseAdjointTransport3D.hpp"
#include "source/3D/radiation/ReverseEstimatorConfig.hpp"
#include "source/3D/radiation/FleckFactorHelper.hpp"

#include <fstream>
#include <iomanip>
#include <limits>
#include <unordered_map>

#ifdef RICH_MPI
#include <mpi.h>
#include "source/mpi/mpi_commands.hpp"
#endif

namespace {

// ============================================================
// STA multigroup opacity from BaseTDECompton
// ============================================================
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

// ============================================================
// CLI Config
// ============================================================
enum class OpacityScaleMode { None, Rosseland, Planck };

struct Config
{
    std::string inputPath = "/home/elads/TDEMG/R0.47M0.5BH1e+06beta1S50n1.5Compton/snap_full_136.h5";
    std::string outputPath = "tde_postprocess_output.h5";
    std::string vtkOutput;
    std::string opacityDir = "/home/elads/RICH/data/STA/MG/";
    std::string greyOpacityDir;
    std::string eosDir = "/home/elads/RICH/data/EOS/";
    double radius = 5e14;
    size_t nObservers = 256;
    double sourceDt = 1.0;
    double transportTime = 0.0;
    Vector3D center = Vector3D(0, 0, 0);
    size_t photonsPerCell = 100;
    bool compton = false;
    size_t comptonSamples = 200000;
    bool comptonAngleDependent = true;
    size_t nGenerations = 1;
    bool ddmc = true;
    bool randomWalk = true;
    bool useCellVelocities = true;
    bool polarization = true;
    int polarizationManualScatterings = 128;
    double polarizationDepolarizationScatterings = 0.5;
    std::string polarizationClosure = "damped_last_scatterings";
    bool measuredLoadBalance = true;
    OpacityScaleMode opacityScaleMode = OpacityScaleMode::Planck;
    bool adaptiveSourceCells = false;
    size_t adaptiveSourceBurnin = 3;
    double adaptiveSourceStrength = 0.95;
    double adaptiveSourceEma = 0.5;
    double adaptiveSourceMinEscapedFrac = 1e-12;
    double adaptiveSourceMaxFactor = 1000.0;
    size_t adaptiveSourceBurninPhotonMultiplier = 2;
    double adaptiveSourceLearnedReserveFrac = 0.25;
    double adaptiveSourceLearnedMinFactor = 20.0;
    bool adaptiveObserverEquity = true;
    double adaptiveObserverExtraBudgetFrac = 2;
    double adaptiveObserverTargetNeff = 1000000;
    double adaptiveObserverTargetPolSnr = 10.0;
    double adaptiveObserverDeficitMax = 100.0;
    double adaptiveObserverDeficitEma = 0.8;
    double measuredLBWeightCompression = -1.0;
    double adaptiveLBImbalanceThreshold = 2.0;
    size_t adaptiveLBCooldownGenerations = 2;
    size_t adaptiveLBMaxRebalances = 6;

    bool adaptiveGroupQuality = false;
    bool adaptiveGroupSourceCells = false;
    bool adaptiveGroupFrequencySampling = false;
    bool adaptiveGroupHistory = true;
    double adaptiveGroupTargetNeff = 1e4;
    double adaptiveGroupTargetPolSnr = 10.0;
    double adaptiveGroupDeficitMax = 100.0;
    size_t adaptiveGroupMinCrossings = 3;
    double adaptiveGroupMinLuminosity = 0.0;
    double adaptiveGroupMinLuminosityFracOfGroupMax = 0.01;
    double adaptiveGroupIneligiblePriorityCap = 2.0;
    double adaptiveGroupRetainPriorityFloor = 3.0;
    std::string adaptiveGroupLuminosityNormalization = "mixed";
    double adaptiveGroupLuminosityGlobalWeight = 0.5;
    double adaptiveGroupLuminosityPower = 1.0;
    double adaptiveGroupPolarizationPower = 1.0;
    double adaptiveGroupLuminosityWeight = 0.5;
    double adaptiveGroupPolarizationWeight = 0.5;
    double adaptiveGroupPolarizationFloor = 0.02;
    double adaptiveGroupHistoryEma = 0.35;
    double adaptiveGroupLatestWeight = 0.25;
    double adaptiveGroupCumulativeWeight = 0.50;
    double adaptiveGroupEmaWeight = 0.25;
    double adaptiveGroupScoreEma = 0.35;
    double adaptiveGroupStrength = 0.75;
    double adaptiveGroupPdfFloor = 0.02;
    double adaptiveGroupMaxBias = 100.0;
    double adaptiveGroupMaxWeightCorrection = 100.0;
    size_t adaptiveGroupMaxLocalStats = 200000;
    size_t adaptiveGroupStatMinCount = 1;
    double adaptiveGroupStatPriorityKeep = 2.0;
    bool adaptiveGroupFallbackToIntegratedOnOverflow = true;
    bool adaptiveDiagnosticsVerbose = false;

    PostProcessEstimatorMode estimatorMode = PostProcessEstimatorMode::Forward;
    size_t reversePacketsPerObserverGroup = 10000;
    uint64_t reverseSeed = 12345;
    std::string reverseOutputPrefix = "reverse_tally";
    bool reverseMeasuredLB = false;
    size_t reverseLBPilotPacketsPerObserverGroup = 0;
    double reverseLBWeightCompression = -1.0;
    double reverseProgressIntervalSec = 5.0;
    size_t reverseMaxEvents = 500000;
    double reverseDDMCMinCellOpticalDepth = 15.0;
    double reverseDDMCMinParticleOpticalDepth = 5.0;
    bool reverseDDMCObserverExclusion = true;
    bool reverseDDMCPhotosphereExclusion = true;
    double reverseDDMCPhotosphereOpticalDepth = 5.0;
};

void printUsage(int rank)
{
    if (rank != 0) return;
    std::cerr << "Usage: rich [options]\n"
              << "Options:\n"
              << "  --input PATH             Input HDF5 snapshot (default: TDE snap_full_136.h5)\n"
              << "  --output PATH            Output HDF5 file (default: tde_postprocess_output.h5)\n"
              << "  --opacity-dir PATH       STA opacity table directory\n"
              << "  --grey-opacity-dir PATH  Grey STA opacity directory (default: parent of opacity-dir)\n"
              << "  --eos-dir PATH           EOS table directory\n"
              << "  --radius R               Observer sphere radius [cm] (default: 5e14)\n"
              << "  --n-observers N          Observer patches (default: 256)\n"
              << "  --source-dt DT           Source emission dt [s] (default: 1.0)\n"
              << "  --transport-time T       Max transport time [s] (default: 2*R/c)\n"
              << "  --center X Y Z           Observer sphere center [cm]\n"
              << "  --photons-per-cell N     Packets per cell (default: 100)\n"
              << "  --compton                Enable Compton (requires multigroup build)\n"
              << "  --compton-samples N      Compton matrix samples (default: 200000)\n"
              << "  --n-generations N        Split transport into N generations (default: 1)\n"
              << "  --no-ddmc                Disable DDMC thick-cell acceleration\n"
              << "  --no-random-walk         Disable random-walk thick-cell acceleration\n"
              << "  --no-velocity            Ignore cell velocities (no Doppler shifts)\n"
              << "  --polarization           Enable postprocess linear polarization\n"
              << "  --polarization-manual-scatterings N\n"
              << "  --polarization-depolarization-scatterings N\n"
              << "  --polarization-closure NAME\n"
              << "  --no-measured-lb         Disable first-generation measured load balance\n"
              << "  --no-opacity-scale       Disable MG absorption scaling\n"
              << "  --opacity-scale-mode M   Normalization mode: planck (default) or rosseland\n"
              << "  --adaptive-source-cells  Learn escaping source cells across generations\n"
              << "  --no-adaptive-source-cells\n"
              << "  --adaptive-source-burnin N       Legacy flag accepted; fixed adaptive cadence ignores it\n"
              << "  --adaptive-source-strength F     Learned-score allocation fraction (default: 0.75)\n"
              << "  --adaptive-source-ema F          Learned score EMA update factor (default: 0.5)\n"
              << "  --adaptive-source-min-escaped-frac F (default: 1e-10)\n"
              << "  --adaptive-source-max-factor F   Max photons/cell boost over base (default: 1000)\n"
              << "  --adaptive-source-burnin-photon-multiplier N Legacy flag accepted; fixed cadence ignores it\n"
              << "  --adaptive-source-learned-reserve-frac F (default: 0.25)\n"
              << "  --adaptive-source-learned-min-factor F (default: 20)\n"
              << "  --adaptive-observer-equity       Boost cells feeding low-stat observers (default)\n"
              << "  --no-adaptive-observer-equity\n"
              << "  --adaptive-observer-extra-budget-frac F (default: 0.25)\n"
              << "  --adaptive-observer-target-neff F (default: 100000)\n"
              << "  --adaptive-observer-target-pol-snr F (default: 5)\n"
              << "  --adaptive-observer-deficit-max F (default: 10)\n"
              << "  --adaptive-observer-deficit-ema F (default: 0.5)\n"
              << "  --measured-lb-weight-compression F (default: adaptive=1, non-adaptive=0.5)\n"
              << "  --adaptive-lb-imbalance-threshold F Legacy flag accepted; fixed 5-step cadence ignores it\n"
              << "  --adaptive-lb-cooldown-gens N Legacy flag accepted; fixed 5-step cadence ignores it\n"
              << "  --adaptive-lb-max-rebalances N Legacy flag accepted; fixed 5-step cadence ignores it\n"
              << "\n  [Adaptive group statistics]\n"
              << "  --adaptive-group-quality                  Build observer/group quality diagnostics\n"
              << "  --no-adaptive-group-quality\n"
              << "  --adaptive-group-source-cells             Learn source-cell/group scores; requires --adaptive-source-cells and --adaptive-group-quality\n"
              << "  --no-adaptive-group-source-cells\n"
              << "  --adaptive-group-frequency-sampling       Bias source group sampling with p/q correction; requires group source cells and MG, unsupported with --compton\n"
              << "  --no-adaptive-group-frequency-sampling\n"
              << "  --adaptive-group-history                  Use latest/cumulative/EMA predictor history (default)\n"
              << "  --no-adaptive-group-history\n"
              << "  --adaptive-group-target-neff F            Target observer/group Neff (default: 1e4)\n"
              << "  --adaptive-group-target-pol-snr F         Target observer/group polarization SNR (default: 10)\n"
              << "  --adaptive-group-deficit-max F            Max deficit multiplier (default: 100)\n"
              << "  --adaptive-group-min-crossings N          Minimum crossings for raw bin priority (default: 3)\n"
              << "  --adaptive-group-min-luminosity F         Absolute luminosity eligibility floor (default: 0)\n"
              << "  --adaptive-group-min-luminosity-frac F    Group-relative luminosity eligibility floor (default: 0.01)\n"
              << "  --adaptive-group-min-luminosity-frac-of-group-max F (alias)\n"
              << "  --adaptive-group-ineligible-priority-cap F (default: 2)\n"
              << "  --adaptive-group-retain-priority-floor F  Retain historically important bins (default: 3)\n"
              << "  --adaptive-group-luminosity-normalization global|per-group|mixed\n"
              << "  --adaptive-group-luminosity-global-weight F (default: 0.5)\n"
              << "  --adaptive-group-luminosity-power F       Non-negative luminosity priority power (default: 1)\n"
              << "  --adaptive-group-polarization-power F     Non-negative polarization priority power (default: 1)\n"
              << "  --adaptive-group-luminosity-weight F      Science priority luminosity weight (default: 0.5)\n"
              << "  --adaptive-group-polarization-weight F    Science priority polarization weight (default: 0.5)\n"
              << "  --adaptive-group-polarization-floor F     Polarization importance floor (default: 0.02)\n"
              << "  --adaptive-group-history-ema F            EMA factor in [0,1] (default: 0.35)\n"
              << "  --adaptive-group-latest-weight F          Predictor latest weight (renormalized with cumulative/EMA)\n"
              << "  --adaptive-group-cumulative-weight F      Predictor cumulative weight\n"
              << "  --adaptive-group-ema-weight F             Predictor EMA weight\n"
              << "  --adaptive-group-score-ema F              Source-cell/group score EMA factor in [0,1]\n"
              << "  --adaptive-group-strength F               Group sampling blend strength in [0,1]\n"
              << "  --adaptive-group-pdf-floor F              Best-effort proposal PDF floor in [0,1]\n"
              << "  --adaptive-group-max-bias F               Max learned/physical proposal ratio (default: 100)\n"
              << "  --adaptive-group-max-weight-correction F  Max p/q before physical fallback (default: 100)\n"
              << "  --adaptive-group-max-local-stats N        Max local source-cell/group stats before pruning/fallback\n"
              << "  --adaptive-group-stat-min-count N         Min local count retained before MPI exchange\n"
              << "  --adaptive-group-stat-priority-keep F     Retain low-count stats from high-priority bins\n"
              << "  --adaptive-group-fallback-to-integrated-on-overflow\n"
              << "  --no-adaptive-group-fallback-to-integrated-on-overflow\n"
              << "  --adaptive-diagnostics-verbose\n"
              << "\n  [Reverse estimator]\n"
              << "  --postprocess-estimator forward|reverse|both  (default: forward)\n"
              << "  --reverse-packets-per-observer-group N  (default: 10000)\n"
              << "  --reverse-seed N                        (default: 12345)\n"
              << "  --reverse-output-prefix PREFIX           (default: reverse_tally)\n"
              << "  --reverse-progress-interval-sec X        (default: 5)\n"
              << "  --reverse-max-events N                   (default: 500000)\n"
              << "  --reverse-ddmc-min-cell-optical-depth X  (default: 15)\n"
              << "  --reverse-ddmc-min-particle-optical-depth X (default: 5)\n"
              << "  --reverse-ddmc-no-observer-exclusion\n"
              << "  --reverse-ddmc-no-photosphere-exclusion\n"
              << "  --reverse-ddmc-photosphere-optical-depth X (default: 5)\n"
              << "  --reverse-measured-lb                    Run pilot reverse pass and repartition\n"
              << "  --reverse-lb-pilot-packets-per-observer-group N (default: production/100)\n"
              << "  --reverse-lb-weight-compression X        (default: forward measured-LB default)\n";
}

bool parseArgs(int argc, char* argv[], Config &cfg, int rank)
{
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) { cfg.inputPath = argv[++i]; }
        else if (arg == "--output" && i + 1 < argc) { cfg.outputPath = argv[++i]; }
        else if (arg == "--opacity-dir" && i + 1 < argc) { cfg.opacityDir = argv[++i]; }
        else if (arg == "--grey-opacity-dir" && i + 1 < argc) { cfg.greyOpacityDir = argv[++i]; }
        else if (arg == "--eos-dir" && i + 1 < argc) { cfg.eosDir = argv[++i]; }
        else if (arg == "--radius" && i + 1 < argc) { cfg.radius = std::atof(argv[++i]); }
        else if (arg == "--n-observers" && i + 1 < argc) { cfg.nObservers = static_cast<size_t>(std::atoi(argv[++i])); }
        else if (arg == "--source-dt" && i + 1 < argc) { cfg.sourceDt = std::atof(argv[++i]); }
        else if (arg == "--transport-time" && i + 1 < argc) { cfg.transportTime = std::atof(argv[++i]); }
        else if (arg == "--center" && i + 3 < argc) {
            double x = std::atof(argv[++i]);
            double y = std::atof(argv[++i]);
            double z = std::atof(argv[++i]);
            cfg.center = Vector3D(x, y, z);
        }
        else if (arg == "--photons-per-cell" && i + 1 < argc) { cfg.photonsPerCell = static_cast<size_t>(std::atoi(argv[++i])); }
        else if (arg == "--compton") { cfg.compton = true; }
        else if (arg == "--compton-samples" && i + 1 < argc) { cfg.comptonSamples = static_cast<size_t>(std::atoi(argv[++i])); }
        else if (arg == "--compton-angle-dependent" && i + 1 < argc) { cfg.comptonAngleDependent = (std::atoi(argv[++i]) != 0); }
        else if (arg == "--vtk-output" && i + 1 < argc) { cfg.vtkOutput = argv[++i]; }
        else if (arg == "--n-generations" && i + 1 < argc) { cfg.nGenerations = static_cast<size_t>(std::atoi(argv[++i])); }
        else if (arg == "--no-ddmc") { cfg.ddmc = false; }
        else if (arg == "--no-random-walk") { cfg.randomWalk = false; }
        else if (arg == "--no-velocity") { cfg.useCellVelocities = false; }
        else if (arg == "--polarization") { cfg.polarization = true; }
        else if (arg == "--polarization-manual-scatterings" && i + 1 < argc) { cfg.polarizationManualScatterings = std::atoi(argv[++i]); }
        else if (arg == "--polarization-depolarization-scatterings" && i + 1 < argc) { cfg.polarizationDepolarizationScatterings = std::atof(argv[++i]); }
        else if (arg == "--polarization-closure" && i + 1 < argc) { cfg.polarizationClosure = argv[++i]; }
        else if (arg == "--no-measured-lb") { cfg.measuredLoadBalance = false; }
        else if (arg == "--peel-off" || arg == "--no-peel-off") {
            if (rank == 0)
                std::cerr << "Error: " << arg << " was removed from this branch\n";
            return false;
        }
        else if (arg == "--adaptive-source-cells") { cfg.adaptiveSourceCells = true; }
        else if (arg == "--no-adaptive-source-cells") { cfg.adaptiveSourceCells = false; }
        else if (arg == "--adaptive-source-burnin" && i + 1 < argc) { cfg.adaptiveSourceBurnin = static_cast<size_t>(std::atoi(argv[++i])); }
        else if (arg == "--adaptive-source-strength" && i + 1 < argc) { cfg.adaptiveSourceStrength = std::atof(argv[++i]); }
        else if (arg == "--adaptive-source-ema" && i + 1 < argc) { cfg.adaptiveSourceEma = std::atof(argv[++i]); }
        else if (arg == "--adaptive-source-min-escaped-frac" && i + 1 < argc) { cfg.adaptiveSourceMinEscapedFrac = std::atof(argv[++i]); }
        else if (arg == "--adaptive-source-max-factor" && i + 1 < argc) { cfg.adaptiveSourceMaxFactor = std::atof(argv[++i]); }
        else if (arg == "--adaptive-source-burnin-photon-multiplier" && i + 1 < argc) { cfg.adaptiveSourceBurninPhotonMultiplier = static_cast<size_t>(std::atoi(argv[++i])); }
        else if (arg == "--adaptive-source-learned-reserve-frac" && i + 1 < argc) { cfg.adaptiveSourceLearnedReserveFrac = std::atof(argv[++i]); }
        else if (arg == "--adaptive-source-learned-min-factor" && i + 1 < argc) { cfg.adaptiveSourceLearnedMinFactor = std::atof(argv[++i]); }
        else if (arg == "--adaptive-observer-equity") { cfg.adaptiveObserverEquity = true; }
        else if (arg == "--no-adaptive-observer-equity") { cfg.adaptiveObserverEquity = false; }
        else if (arg == "--adaptive-observer-extra-budget-frac" && i + 1 < argc) { cfg.adaptiveObserverExtraBudgetFrac = std::atof(argv[++i]); }
        else if (arg == "--adaptive-observer-target-neff" && i + 1 < argc) { cfg.adaptiveObserverTargetNeff = std::atof(argv[++i]); }
        else if (arg == "--adaptive-observer-target-pol-snr" && i + 1 < argc) { cfg.adaptiveObserverTargetPolSnr = std::atof(argv[++i]); }
        else if (arg == "--adaptive-observer-deficit-max" && i + 1 < argc) { cfg.adaptiveObserverDeficitMax = std::atof(argv[++i]); }
        else if (arg == "--adaptive-observer-deficit-ema" && i + 1 < argc) { cfg.adaptiveObserverDeficitEma = std::atof(argv[++i]); }
        else if (arg == "--measured-lb-weight-compression" && i + 1 < argc) { cfg.measuredLBWeightCompression = std::atof(argv[++i]); }
        else if (arg == "--adaptive-lb-imbalance-threshold" && i + 1 < argc) { cfg.adaptiveLBImbalanceThreshold = std::atof(argv[++i]); }
        else if (arg == "--adaptive-lb-cooldown-gens" && i + 1 < argc) { cfg.adaptiveLBCooldownGenerations = static_cast<size_t>(std::atoi(argv[++i])); }
        else if (arg == "--adaptive-lb-max-rebalances" && i + 1 < argc) { cfg.adaptiveLBMaxRebalances = static_cast<size_t>(std::atoi(argv[++i])); }
        else if (arg == "--adaptive-group-quality") { cfg.adaptiveGroupQuality = true; }
        else if (arg == "--no-adaptive-group-quality") { cfg.adaptiveGroupQuality = false; }
        else if (arg == "--adaptive-group-source-cells") { cfg.adaptiveGroupSourceCells = true; }
        else if (arg == "--no-adaptive-group-source-cells") { cfg.adaptiveGroupSourceCells = false; }
        else if (arg == "--adaptive-group-frequency-sampling") { cfg.adaptiveGroupFrequencySampling = true; }
        else if (arg == "--no-adaptive-group-frequency-sampling") { cfg.adaptiveGroupFrequencySampling = false; }
        else if (arg == "--adaptive-group-history") { cfg.adaptiveGroupHistory = true; }
        else if (arg == "--no-adaptive-group-history") { cfg.adaptiveGroupHistory = false; }
        else if (arg == "--adaptive-group-target-neff" && i + 1 < argc) { cfg.adaptiveGroupTargetNeff = std::atof(argv[++i]); }
        else if (arg == "--adaptive-group-target-pol-snr" && i + 1 < argc) { cfg.adaptiveGroupTargetPolSnr = std::atof(argv[++i]); }
        else if (arg == "--adaptive-group-deficit-max" && i + 1 < argc) { cfg.adaptiveGroupDeficitMax = std::atof(argv[++i]); }
        else if (arg == "--adaptive-group-min-crossings" && i + 1 < argc) { cfg.adaptiveGroupMinCrossings = static_cast<size_t>(std::atoi(argv[++i])); }
        else if (arg == "--adaptive-group-min-luminosity" && i + 1 < argc) { cfg.adaptiveGroupMinLuminosity = std::atof(argv[++i]); }
        else if ((arg == "--adaptive-group-min-luminosity-frac" ||
                  arg == "--adaptive-group-min-luminosity-frac-of-group-max") &&
                 i + 1 < argc) { cfg.adaptiveGroupMinLuminosityFracOfGroupMax = std::atof(argv[++i]); }
        else if (arg == "--adaptive-group-ineligible-priority-cap" && i + 1 < argc) { cfg.adaptiveGroupIneligiblePriorityCap = std::atof(argv[++i]); }
        else if (arg == "--adaptive-group-retain-priority-floor" && i + 1 < argc) { cfg.adaptiveGroupRetainPriorityFloor = std::atof(argv[++i]); }
        else if (arg == "--adaptive-group-luminosity-normalization" && i + 1 < argc) { cfg.adaptiveGroupLuminosityNormalization = argv[++i]; }
        else if (arg == "--adaptive-group-luminosity-global-weight" && i + 1 < argc) { cfg.adaptiveGroupLuminosityGlobalWeight = std::atof(argv[++i]); }
        else if (arg == "--adaptive-group-luminosity-power" && i + 1 < argc) { cfg.adaptiveGroupLuminosityPower = std::atof(argv[++i]); }
        else if (arg == "--adaptive-group-polarization-power" && i + 1 < argc) { cfg.adaptiveGroupPolarizationPower = std::atof(argv[++i]); }
        else if (arg == "--adaptive-group-luminosity-weight" && i + 1 < argc) { cfg.adaptiveGroupLuminosityWeight = std::atof(argv[++i]); }
        else if (arg == "--adaptive-group-polarization-weight" && i + 1 < argc) { cfg.adaptiveGroupPolarizationWeight = std::atof(argv[++i]); }
        else if (arg == "--adaptive-group-polarization-floor" && i + 1 < argc) { cfg.adaptiveGroupPolarizationFloor = std::atof(argv[++i]); }
        else if (arg == "--adaptive-group-history-ema" && i + 1 < argc) { cfg.adaptiveGroupHistoryEma = std::atof(argv[++i]); }
        else if (arg == "--adaptive-group-latest-weight" && i + 1 < argc) { cfg.adaptiveGroupLatestWeight = std::atof(argv[++i]); }
        else if (arg == "--adaptive-group-cumulative-weight" && i + 1 < argc) { cfg.adaptiveGroupCumulativeWeight = std::atof(argv[++i]); }
        else if (arg == "--adaptive-group-ema-weight" && i + 1 < argc) { cfg.adaptiveGroupEmaWeight = std::atof(argv[++i]); }
        else if (arg == "--adaptive-group-score-ema" && i + 1 < argc) { cfg.adaptiveGroupScoreEma = std::atof(argv[++i]); }
        else if (arg == "--adaptive-group-strength" && i + 1 < argc) { cfg.adaptiveGroupStrength = std::atof(argv[++i]); }
        else if (arg == "--adaptive-group-pdf-floor" && i + 1 < argc) { cfg.adaptiveGroupPdfFloor = std::atof(argv[++i]); }
        else if (arg == "--adaptive-group-max-bias" && i + 1 < argc) { cfg.adaptiveGroupMaxBias = std::atof(argv[++i]); }
        else if (arg == "--adaptive-group-max-weight-correction" && i + 1 < argc) { cfg.adaptiveGroupMaxWeightCorrection = std::atof(argv[++i]); }
        else if (arg == "--adaptive-group-max-local-stats" && i + 1 < argc) { cfg.adaptiveGroupMaxLocalStats = static_cast<size_t>(std::atoll(argv[++i])); }
        else if (arg == "--adaptive-group-stat-min-count" && i + 1 < argc) { cfg.adaptiveGroupStatMinCount = static_cast<size_t>(std::atoi(argv[++i])); }
        else if (arg == "--adaptive-group-stat-priority-keep" && i + 1 < argc) { cfg.adaptiveGroupStatPriorityKeep = std::atof(argv[++i]); }
        else if (arg == "--adaptive-group-fallback-to-integrated-on-overflow") { cfg.adaptiveGroupFallbackToIntegratedOnOverflow = true; }
        else if (arg == "--no-adaptive-group-fallback-to-integrated-on-overflow") { cfg.adaptiveGroupFallbackToIntegratedOnOverflow = false; }
        else if (arg == "--adaptive-diagnostics-verbose") { cfg.adaptiveDiagnosticsVerbose = true; }
        else if (arg == "--no-rosseland-scale" || arg == "--no-opacity-scale") { cfg.opacityScaleMode = OpacityScaleMode::None; }
        else if (arg == "--opacity-scale-mode" && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "planck") cfg.opacityScaleMode = OpacityScaleMode::Planck;
            else if (m == "rosseland") cfg.opacityScaleMode = OpacityScaleMode::Rosseland;
            else if (m == "none") cfg.opacityScaleMode = OpacityScaleMode::None;
            else { if (rank == 0) std::cerr << "Unknown opacity-scale-mode: " << m << " (planck|rosseland|none)\n"; return false; }
        }
        else if (arg == "--postprocess-estimator" && i + 1 < argc) {
            std::string m = argv[++i];
            if (m == "forward") cfg.estimatorMode = PostProcessEstimatorMode::Forward;
            else if (m == "reverse") cfg.estimatorMode = PostProcessEstimatorMode::Reverse;
            else if (m == "both") cfg.estimatorMode = PostProcessEstimatorMode::Both;
            else { if (rank == 0) std::cerr << "Unknown --postprocess-estimator: " << m << " (forward|reverse|both)\n"; return false; }
        }
        else if (arg == "--reverse-packets-per-observer-group" && i + 1 < argc) { cfg.reversePacketsPerObserverGroup = static_cast<size_t>(std::atoi(argv[++i])); }
        else if (arg == "--reverse-seed" && i + 1 < argc) { cfg.reverseSeed = static_cast<uint64_t>(std::atoll(argv[++i])); }
        else if (arg == "--reverse-output-prefix" && i + 1 < argc) { cfg.reverseOutputPrefix = argv[++i]; }
        else if (arg == "--reverse-progress-interval-sec" && i + 1 < argc) { cfg.reverseProgressIntervalSec = std::atof(argv[++i]); }
        else if (arg == "--reverse-max-events" && i + 1 < argc) { cfg.reverseMaxEvents = static_cast<size_t>(std::atoll(argv[++i])); }
        else if (arg == "--reverse-ddmc-min-cell-optical-depth" && i + 1 < argc) { cfg.reverseDDMCMinCellOpticalDepth = std::atof(argv[++i]); }
        else if (arg == "--reverse-ddmc-min-particle-optical-depth" && i + 1 < argc) { cfg.reverseDDMCMinParticleOpticalDepth = std::atof(argv[++i]); }
        else if (arg == "--reverse-ddmc-no-observer-exclusion") { cfg.reverseDDMCObserverExclusion = false; }
        else if (arg == "--reverse-ddmc-no-photosphere-exclusion") { cfg.reverseDDMCPhotosphereExclusion = false; }
        else if (arg == "--reverse-ddmc-photosphere-optical-depth" && i + 1 < argc) { cfg.reverseDDMCPhotosphereOpticalDepth = std::atof(argv[++i]); }
        else if (arg == "--reverse-measured-lb") { cfg.reverseMeasuredLB = true; }
        else if (arg == "--no-reverse-measured-lb") { cfg.reverseMeasuredLB = false; }
        else if (arg == "--reverse-lb-pilot-packets-per-observer-group" && i + 1 < argc) { cfg.reverseLBPilotPacketsPerObserverGroup = static_cast<size_t>(std::atoi(argv[++i])); }
        else if (arg == "--reverse-lb-weight-compression" && i + 1 < argc) { cfg.reverseLBWeightCompression = std::atof(argv[++i]); }
        else { if (rank == 0) std::cerr << "Unknown argument: " << arg << "\n"; return false; }
    }

    if (cfg.radius <= 0.0) { if (rank == 0) std::cerr << "--radius must be positive\n"; return false; }
    if (cfg.nObservers == 0) { if (rank == 0) std::cerr << "--n-observers must be > 0\n"; return false; }
    if (cfg.sourceDt <= 0.0) { if (rank == 0) std::cerr << "--source-dt must be positive\n"; return false; }
    if (cfg.photonsPerCell == 0) { if (rank == 0) std::cerr << "--photons-per-cell must be > 0\n"; return false; }
    if (cfg.nGenerations == 0) { if (rank == 0) std::cerr << "--n-generations must be >= 1\n"; return false; }
    if (cfg.adaptiveSourceStrength < 0.0 || cfg.adaptiveSourceStrength > 1.0 || !std::isfinite(cfg.adaptiveSourceStrength)) { if (rank == 0) std::cerr << "--adaptive-source-strength must be finite in [0,1]\n"; return false; }
    if (cfg.adaptiveSourceEma <= 0.0 || cfg.adaptiveSourceEma > 1.0 || !std::isfinite(cfg.adaptiveSourceEma)) { if (rank == 0) std::cerr << "--adaptive-source-ema must be finite in (0,1]\n"; return false; }
    if (cfg.adaptiveSourceMinEscapedFrac < 0.0 || !std::isfinite(cfg.adaptiveSourceMinEscapedFrac)) { if (rank == 0) std::cerr << "--adaptive-source-min-escaped-frac must be finite and nonnegative\n"; return false; }
    if (cfg.adaptiveSourceMaxFactor < 1.0 || !std::isfinite(cfg.adaptiveSourceMaxFactor)) { if (rank == 0) std::cerr << "--adaptive-source-max-factor must be finite and >= 1\n"; return false; }
    if (cfg.adaptiveSourceLearnedReserveFrac < 0.0 || cfg.adaptiveSourceLearnedReserveFrac > 1.0 || !std::isfinite(cfg.adaptiveSourceLearnedReserveFrac)) { if (rank == 0) std::cerr << "--adaptive-source-learned-reserve-frac must be finite in [0,1]\n"; return false; }
    if (cfg.adaptiveSourceLearnedMinFactor < 1.0 || !std::isfinite(cfg.adaptiveSourceLearnedMinFactor)) { if (rank == 0) std::cerr << "--adaptive-source-learned-min-factor must be finite and >= 1\n"; return false; }
    if (cfg.adaptiveObserverExtraBudgetFrac < 0.0 || !std::isfinite(cfg.adaptiveObserverExtraBudgetFrac)) { if (rank == 0) std::cerr << "--adaptive-observer-extra-budget-frac must be finite and nonnegative\n"; return false; }
    if (cfg.adaptiveObserverTargetNeff <= 0.0 || !std::isfinite(cfg.adaptiveObserverTargetNeff)) { if (rank == 0) std::cerr << "--adaptive-observer-target-neff must be finite and positive\n"; return false; }
    if (cfg.adaptiveObserverTargetPolSnr <= 0.0 || !std::isfinite(cfg.adaptiveObserverTargetPolSnr)) { if (rank == 0) std::cerr << "--adaptive-observer-target-pol-snr must be finite and positive\n"; return false; }
    if (cfg.adaptiveObserverDeficitMax < 1.0 || !std::isfinite(cfg.adaptiveObserverDeficitMax)) { if (rank == 0) std::cerr << "--adaptive-observer-deficit-max must be finite and >= 1\n"; return false; }
    if (cfg.adaptiveObserverDeficitEma <= 0.0 || cfg.adaptiveObserverDeficitEma > 1.0 || !std::isfinite(cfg.adaptiveObserverDeficitEma)) { if (rank == 0) std::cerr << "--adaptive-observer-deficit-ema must be finite in (0,1]\n"; return false; }
    if (cfg.measuredLBWeightCompression != -1.0 && (cfg.measuredLBWeightCompression <= 0.0 || !std::isfinite(cfg.measuredLBWeightCompression))) { if (rank == 0) std::cerr << "--measured-lb-weight-compression must be finite and > 0\n"; return false; }
    if (cfg.reversePacketsPerObserverGroup == 0) { if (rank == 0) std::cerr << "--reverse-packets-per-observer-group must be > 0\n"; return false; }
    if (cfg.reverseProgressIntervalSec <= 0.0 || !std::isfinite(cfg.reverseProgressIntervalSec)) { if (rank == 0) std::cerr << "--reverse-progress-interval-sec must be finite and > 0\n"; return false; }
    if (cfg.reverseMaxEvents == 0) { if (rank == 0) std::cerr << "--reverse-max-events must be > 0\n"; return false; }
    if (cfg.reverseDDMCMinCellOpticalDepth <= 0.0 || !std::isfinite(cfg.reverseDDMCMinCellOpticalDepth)) { if (rank == 0) std::cerr << "--reverse-ddmc-min-cell-optical-depth must be finite and > 0\n"; return false; }
    if (cfg.reverseDDMCMinParticleOpticalDepth <= 0.0 || !std::isfinite(cfg.reverseDDMCMinParticleOpticalDepth)) { if (rank == 0) std::cerr << "--reverse-ddmc-min-particle-optical-depth must be finite and > 0\n"; return false; }
    if (cfg.reverseDDMCPhotosphereOpticalDepth <= 0.0 || !std::isfinite(cfg.reverseDDMCPhotosphereOpticalDepth)) { if (rank == 0) std::cerr << "--reverse-ddmc-photosphere-optical-depth must be finite and > 0\n"; return false; }
    if (cfg.reverseLBWeightCompression != -1.0 && (cfg.reverseLBWeightCompression <= 0.0 || !std::isfinite(cfg.reverseLBWeightCompression))) { if (rank == 0) std::cerr << "--reverse-lb-weight-compression must be finite and > 0\n"; return false; }

    if (cfg.greyOpacityDir.empty()) {
        std::string d = cfg.opacityDir;
        if (d.size() > 3 && d.substr(d.size() - 3) == "MG/")
            d = d.substr(0, d.size() - 3);
        else if (d.size() > 2 && d.substr(d.size() - 2) == "MG")
            d = d.substr(0, d.size() - 2);
        cfg.greyOpacityDir = d;
    }

    if (cfg.transportTime <= 0.0)
        cfg.transportTime = 2.0 * cfg.radius / units::clight;

#if ENERGY_GROUPS_NUM <= 1
    if (cfg.compton) { if (rank == 0) std::cerr << "--compton requires ENERGY_GROUPS_NUM > 1\n"; return false; }
#endif
    if (cfg.compton && cfg.adaptiveSourceCells) {
        if (rank == 0) std::cerr << "--adaptive-source-cells does not support --compton yet\n";
        return false;
    }

    if (cfg.adaptiveGroupSourceCells && !cfg.adaptiveSourceCells) {
        if (rank == 0) std::cerr << "--adaptive-group-source-cells requires --adaptive-source-cells\n";
        return false;
    }
    if (cfg.adaptiveGroupSourceCells && !cfg.adaptiveGroupQuality) {
        if (rank == 0) std::cerr << "--adaptive-group-source-cells requires --adaptive-group-quality\n";
        return false;
    }
    if (cfg.adaptiveGroupFrequencySampling && !cfg.adaptiveGroupSourceCells) {
        if (rank == 0) std::cerr << "--adaptive-group-frequency-sampling requires --adaptive-group-source-cells\n";
        return false;
    }
#if ENERGY_GROUPS_NUM <= 1
    if (cfg.adaptiveGroupFrequencySampling) {
        if (rank == 0) std::cerr << "--adaptive-group-frequency-sampling requires ENERGY_GROUPS_NUM > 1\n";
        return false;
    }
#endif
    if (cfg.adaptiveGroupFrequencySampling && cfg.compton) {
        if (rank == 0) std::cerr << "--adaptive-group-frequency-sampling with --compton is not supported\n";
        return false;
    }

    if (cfg.adaptiveGroupQuality) {
        auto failGroupValidation = [&](std::string const& msg) {
            if (rank == 0) std::cerr << msg << "\n";
            return false;
        };
        auto requireRange = [&](double value, double lo, double hi, std::string const& name) {
            if (!std::isfinite(value) || value < lo || value > hi)
                return failGroupValidation(name + " must be in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
            return true;
        };
        auto requireNonNegative = [&](double value, std::string const& name) {
            if (!std::isfinite(value) || value < 0.0)
                return failGroupValidation(name + " must be non-negative");
            return true;
        };
        auto requirePositive = [&](double value, std::string const& name) {
            if (!std::isfinite(value) || value <= 0.0)
                return failGroupValidation(name + " must be positive");
            return true;
        };

        if (cfg.adaptiveGroupLuminosityNormalization != "global" &&
            cfg.adaptiveGroupLuminosityNormalization != "per-group" &&
            cfg.adaptiveGroupLuminosityNormalization != "mixed")
            return failGroupValidation("--adaptive-group-luminosity-normalization must be global, per-group, or mixed");
        if (!requirePositive(cfg.adaptiveGroupTargetNeff, "--adaptive-group-target-neff") ||
            !requirePositive(cfg.adaptiveGroupTargetPolSnr, "--adaptive-group-target-pol-snr") ||
            !requirePositive(cfg.adaptiveGroupDeficitMax, "--adaptive-group-deficit-max") ||
            !requireNonNegative(cfg.adaptiveGroupMinLuminosity, "--adaptive-group-min-luminosity") ||
            !requireRange(cfg.adaptiveGroupMinLuminosityFracOfGroupMax, 0.0, 1.0, "--adaptive-group-min-luminosity-frac") ||
            !requireNonNegative(cfg.adaptiveGroupIneligiblePriorityCap, "--adaptive-group-ineligible-priority-cap") ||
            !requireNonNegative(cfg.adaptiveGroupRetainPriorityFloor, "--adaptive-group-retain-priority-floor") ||
            !requireRange(cfg.adaptiveGroupLuminosityGlobalWeight, 0.0, 1.0, "--adaptive-group-luminosity-global-weight") ||
            !requireNonNegative(cfg.adaptiveGroupLuminosityPower, "--adaptive-group-luminosity-power") ||
            !requireNonNegative(cfg.adaptiveGroupPolarizationPower, "--adaptive-group-polarization-power") ||
            !requireNonNegative(cfg.adaptiveGroupLuminosityWeight, "--adaptive-group-luminosity-weight") ||
            !requireNonNegative(cfg.adaptiveGroupPolarizationWeight, "--adaptive-group-polarization-weight") ||
            !requireNonNegative(cfg.adaptiveGroupPolarizationFloor, "--adaptive-group-polarization-floor") ||
            !requireRange(cfg.adaptiveGroupHistoryEma, 0.0, 1.0, "--adaptive-group-history-ema") ||
            !requireRange(cfg.adaptiveGroupScoreEma, 0.0, 1.0, "--adaptive-group-score-ema") ||
            !requireRange(cfg.adaptiveGroupStrength, 0.0, 1.0, "--adaptive-group-strength") ||
            !requirePositive(cfg.adaptiveGroupMaxBias, "--adaptive-group-max-bias") ||
            !requirePositive(cfg.adaptiveGroupMaxWeightCorrection, "--adaptive-group-max-weight-correction") ||
            !requireNonNegative(cfg.adaptiveGroupStatPriorityKeep, "--adaptive-group-stat-priority-keep"))
            return false;
        if (!std::isfinite(cfg.adaptiveGroupPdfFloor) ||
            cfg.adaptiveGroupPdfFloor < 0.0 ||
            cfg.adaptiveGroupPdfFloor >= 1.0)
            return failGroupValidation("--adaptive-group-pdf-floor must be in [0, 1)");
        if (cfg.adaptiveGroupDeficitMax < 1.0)
            return failGroupValidation("--adaptive-group-deficit-max must be >= 1");
        if (cfg.adaptiveGroupMaxBias < 1.0)
            return failGroupValidation("--adaptive-group-max-bias must be >= 1");
        if (cfg.adaptiveGroupMaxWeightCorrection < 1.0)
            return failGroupValidation("--adaptive-group-max-weight-correction must be >= 1");
        if (cfg.adaptiveGroupMaxLocalStats == 0 && cfg.adaptiveGroupSourceCells)
            return failGroupValidation("--adaptive-group-max-local-stats must be > 0 when group source cells are enabled");

        double wsum = cfg.adaptiveGroupLatestWeight + cfg.adaptiveGroupCumulativeWeight + cfg.adaptiveGroupEmaWeight;
        if (!std::isfinite(wsum) ||
            cfg.adaptiveGroupLatestWeight < 0.0 ||
            cfg.adaptiveGroupCumulativeWeight < 0.0 ||
            cfg.adaptiveGroupEmaWeight < 0.0)
            return failGroupValidation("adaptive group predictor weights must be finite and non-negative");
        if (!(wsum > 0.0))
            return failGroupValidation("adaptive group predictor weights must not all be zero");
        if (wsum > 0.0 && std::abs(wsum - 1.0) > 1e-6) {
            cfg.adaptiveGroupLatestWeight /= wsum;
            cfg.adaptiveGroupCumulativeWeight /= wsum;
            cfg.adaptiveGroupEmaWeight /= wsum;
            if (rank == 0) std::cout << "ADAPTIVE_GROUP: predictor weights renormalized to sum=1\n";
        }
    }

    return true;
}

struct AdaptiveSourceState
{
    std::unordered_map<size_t, double> scoreByCellID;
    std::vector<double> observerDeficitByIndex;
    double observerBudgetMultiplier = 1.0;
    bool burninCompletePrinted = false;
    bool postAdaptiveMeasuredLBDone = false;
    size_t adaptiveMeasuredLBCount = 0;
    size_t lastAdaptiveMeasuredLBGeneration = std::numeric_limits<size_t>::max();
};

struct ObserverQualityDiagnostics
{
    bool enabled = false;
    bool polarizationMode = false;
    size_t observerCount = 0;
    size_t weakObservers = 0;
    size_t zeroStatObservers = 0;
    double budgetMultiplier = 1.0;
    double deficitMin = 1.0;
    double deficitAvg = 1.0;
    double deficitMax = 1.0;
    double neffP05 = 0.0;
    double neffMedian = 0.0;
    double neffP95 = 0.0;
    double snrP05 = 0.0;
    double snrMedian = 0.0;
    double snrP95 = 0.0;
    std::vector<double> deficitByObserver;
    std::vector<double> neffByObserver;
    std::vector<double> snrByObserver;
    std::vector<unsigned long long> crossingsByObserver;
};

constexpr double MEASURED_LB_MAX_CELL_IMBALANCE = 2.5;

double EffectiveMeasuredLBWeightCompression(Config const& cfg)
{
    if (cfg.measuredLBWeightCompression > 0.0)
        return cfg.measuredLBWeightCompression;
    return cfg.adaptiveSourceCells ? 1.0 : 0.5;
}

struct RankStepImbalance
{
    unsigned long long localSteps = 0;
    unsigned long long globalSteps = 0;
    double meanRankSteps = 0.0;
    double maxRankSteps = 0.0;
    double maxOverMean = 0.0;
};

RankStepImbalance ComputeRankStepImbalance(
    std::string const& label,
    size_t gen,
    std::vector<size_t> const& localSteps,
    int rank)
{
    RankStepImbalance out;
    for (size_t s : localSteps)
        out.localSteps += static_cast<unsigned long long>(s);

#ifdef RICH_MPI
    unsigned long long globalSteps = out.localSteps;
    MPI_Allreduce(MPI_IN_PLACE, &globalSteps, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    out.globalSteps = globalSteps;

    double localStepsD = static_cast<double>(out.localSteps);
    MPI_Allreduce(&localStepsD, &out.maxRankSteps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    int mpiSize = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    out.meanRankSteps = static_cast<double>(out.globalSteps) / std::max(mpiSize, 1);
#else
    out.globalSteps = out.localSteps;
    out.maxRankSteps = static_cast<double>(out.localSteps);
    out.meanRankSteps = static_cast<double>(out.localSteps);
#endif

    out.maxOverMean = (out.meanRankSteps > 0.0)
        ? out.maxRankSteps / out.meanRankSteps : 0.0;
    if (rank == 0) {
        std::cout << label << " rank_step_imbalance after generation " << (gen + 1)
                  << ": global_steps=" << out.globalSteps
                  << " mean_rank_steps=" << out.meanRankSteps
                  << " max_rank_steps=" << out.maxRankSteps
                  << " max_over_mean=" << out.maxOverMean
                  << std::endl;
    }
    return out;
}

bool AdaptiveLBCooldownSatisfied(AdaptiveSourceState const& state,
                                 Config const& cfg,
                                 size_t gen)
{
    if (state.lastAdaptiveMeasuredLBGeneration == std::numeric_limits<size_t>::max())
        return true;
    return gen >= state.lastAdaptiveMeasuredLBGeneration + cfg.adaptiveLBCooldownGenerations;
}

void AppendZeroVtkScalar(std::ofstream& file, std::string const& name, size_t n)
{
    file << "SCALARS " << name << " double 1\n"
         << "LOOKUP_TABLE default\n";
    for (size_t i = 0; i < n; ++i)
        file << 0.0 << "\n";
}

struct PackedSourceEscapeStat
{
    unsigned long long cellID = 0;
    unsigned long long observerIndex = 0;
    double weightSq = 0.0;
    double maxWeight = 0.0;
    double energy = 0.0;
    unsigned long long count = 0;
};

struct PackedAdaptiveScoreDelta
{
    unsigned long long cellID = 0;
    double delta = 0.0;
};

struct PackedSourceGroupEscapeStat
{
    unsigned long long cellID = 0;
    unsigned long long observerIndex = 0;
    unsigned long long groupIndex = 0;
    double weightSq = 0.0;
    double maxWeight = 0.0;
    double energy = 0.0;
    unsigned long long count = 0;
};

struct PackedAdaptiveCellGroupScoreDelta
{
    unsigned long long cellID = 0;
    unsigned long long groupIndex = 0;
    double delta = 0.0;
};

struct AdaptiveGroupHistory
{
    bool initialized = false;
    size_t observerCount = 0;
    size_t groupCount = 0;
    size_t updateCount = 0;

    std::vector<std::vector<double>> emaPriority;
    std::vector<std::vector<double>> emaDeficit;

    std::vector<std::vector<double>> cumulativeEnergy;
    std::vector<std::vector<double>> cumulativeWeightSq;
    std::vector<std::vector<double>> cumulativeStokesQ;
    std::vector<std::vector<double>> cumulativeStokesU;
    std::vector<std::vector<double>> cumulativeSumWQ2;
    std::vector<std::vector<double>> cumulativeSumWU2;
    std::vector<std::vector<size_t>> cumulativeCrossings;
};

struct ObserverGroupQualityDiagnostics
{
    bool enabled = false;
    bool polarizationMode = false;
    size_t observerCount = 0;
    size_t groupCount = 0;

    std::vector<std::vector<double>> luminosity;
    std::vector<std::vector<double>> neff;
    std::vector<std::vector<double>> polarizationDegree;
    std::vector<std::vector<double>> polarizationSnr;
    std::vector<std::vector<double>> latestPriority;
    std::vector<std::vector<double>> cumulativePriority;
    std::vector<std::vector<double>> predictedPriority;
    std::vector<std::vector<double>> deficit;
    std::vector<std::vector<size_t>> crossings;
    size_t activeBins = 0;
    size_t highPriorityBins = 0;
    double neffP05 = 0.0;
    double neffMedian = 0.0;
    double neffP95 = 0.0;
    double polSnrP05 = 0.0;
    double polSnrMedian = 0.0;
    double polSnrP95 = 0.0;
};

struct AdaptiveGroupSourceState
{
    std::unordered_map<size_t, std::vector<double>> scoreByCellGroup;
    std::unordered_map<size_t, double> cellScoreFromGroups;
};

struct AdaptiveSourceUpdateSummary
{
    double totalEscapedEnergy = 0.0;
    unsigned long long totalCrossings = 0;
    size_t sourceCellObserverPairs = 0;
    size_t observersWithCrossings = 0;
    size_t passedCells = 0;
    size_t newCells = 0;
    size_t retainedCells = 0;
    size_t decayedCells = 0;
    unsigned long long maxLocalSourcePairs = 0;
    unsigned long long maxReceivedShardPairs = 0;
    unsigned long long maxPackedBytes = 0;
    size_t scoreDeltaCells = 0;
    size_t scoreMapCells = 0;
    std::vector<SphericalObserver::SourceCellEscapeStat> topStats;
};

struct AdaptiveGroupSourceUpdateSummary
{
    bool fallbackToIntegratedPath = false;
    std::string fallbackReason = "none";
    unsigned long long localStatsInput = 0;
    unsigned long long localStatsAfterPrune = 0;
    unsigned long long localStatsDropped = 0;
    unsigned long long maxLocalSourceGroupStats = 0;
    unsigned long long maxReceivedShardStats = 0;
    unsigned long long mpiStatsExchanged = 0;
    unsigned long long maxPackedBytes = 0;
    size_t passedStats = 0;
    size_t scoreDeltaCells = 0;
    size_t scoreMapCells = 0;
};

void AccumulateAdaptiveGroupSourceSummary(
    AdaptiveGroupSourceUpdateSummary& total,
    AdaptiveGroupSourceUpdateSummary const& gen)
{
    total.localStatsAfterPrune += gen.localStatsAfterPrune;
    total.localStatsDropped += gen.localStatsDropped;
    total.mpiStatsExchanged += gen.mpiStatsExchanged;
    total.maxReceivedShardStats =
        std::max(total.maxReceivedShardStats, gen.maxReceivedShardStats);
    total.maxPackedBytes = std::max(total.maxPackedBytes, gen.maxPackedBytes);
}

uint64_t SplitMix64(uint64_t x);

struct AdaptivePairKey
{
    size_t observerIndex = 0;
    size_t cellID = 0;

    bool operator==(AdaptivePairKey const& other) const
    {
        return observerIndex == other.observerIndex && cellID == other.cellID;
    }
};

struct AdaptivePairKeyHash
{
    size_t operator()(AdaptivePairKey const& key) const
    {
        uint64_t x = static_cast<uint64_t>(key.cellID);
        x ^= static_cast<uint64_t>(key.observerIndex) + 0x9e3779b97f4a7c15ULL +
             (x << 6) + (x >> 2);
        return static_cast<size_t>(SplitMix64(x));
    }
};

struct AdaptiveSourceGroupKey
{
    size_t observerIndex = 0;
    size_t groupIndex = 0;
    size_t cellID = 0;

    bool operator==(AdaptiveSourceGroupKey const& other) const
    {
        return observerIndex == other.observerIndex
            && groupIndex == other.groupIndex
            && cellID == other.cellID;
    }
};

struct AdaptiveSourceGroupKeyHash
{
    size_t operator()(AdaptiveSourceGroupKey const& key) const
    {
        uint64_t x = static_cast<uint64_t>(key.cellID);
        x ^= static_cast<uint64_t>(key.observerIndex) + 0x9e3779b97f4a7c15ULL +
             (x << 6) + (x >> 2);
        x ^= static_cast<uint64_t>(key.groupIndex) + 0x9e3779b97f4a7c15ULL +
             (x << 6) + (x >> 2);
        return static_cast<size_t>(SplitMix64(x));
    }
};

uint64_t SplitMix64(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

int AdaptiveCellOwner(size_t cellID, int ranks)
{
    if (ranks <= 1)
        return 0;
    return static_cast<int>(SplitMix64(static_cast<uint64_t>(cellID)) %
                            static_cast<uint64_t>(ranks));
}

int CheckedByteCount(size_t count, size_t elementSize, std::string const& label)
{
    unsigned long long bytes =
        static_cast<unsigned long long>(count) * static_cast<unsigned long long>(elementSize);
    if (bytes > static_cast<unsigned long long>(std::numeric_limits<int>::max()))
        throw UniversalError(label + " too large for MPI byte count");
    return static_cast<int>(bytes);
}

int CheckedByteTotal(unsigned long long bytes, std::string const& label)
{
    if (bytes > static_cast<unsigned long long>(std::numeric_limits<int>::max()))
        throw UniversalError(label + " total too large for MPI byte displacements");
    return static_cast<int>(bytes);
}

PackedSourceEscapeStat PackSourceEscapeStat(SphericalObserver::SourceCellEscapeStat const& s)
{
    PackedSourceEscapeStat p;
    p.cellID = static_cast<unsigned long long>(s.cellID);
    p.observerIndex = static_cast<unsigned long long>(s.observerIndex);
    p.energy = s.energy;
    p.count = static_cast<unsigned long long>(s.count);
    p.weightSq = s.weightSq;
    p.maxWeight = s.maxWeight;
    return p;
}

SphericalObserver::SourceCellEscapeStat UnpackSourceEscapeStat(PackedSourceEscapeStat const& p)
{
    SphericalObserver::SourceCellEscapeStat s;
    s.cellID = static_cast<size_t>(p.cellID);
    s.observerIndex = static_cast<size_t>(p.observerIndex);
    s.energy = p.energy;
    s.count = static_cast<size_t>(p.count);
    s.weightSq = p.weightSq;
    s.maxWeight = p.maxWeight;
    return s;
}

PackedSourceGroupEscapeStat PackSourceGroupEscapeStat(
    SphericalObserver::SourceCellGroupEscapeStat const& s)
{
    PackedSourceGroupEscapeStat p;
    p.cellID = static_cast<unsigned long long>(s.cellID);
    p.observerIndex = static_cast<unsigned long long>(s.observerIndex);
    p.groupIndex = static_cast<unsigned long long>(s.groupIndex);
    p.energy = s.energy;
    p.count = static_cast<unsigned long long>(s.count);
    p.weightSq = s.weightSq;
    p.maxWeight = s.maxWeight;
    return p;
}

std::vector<PackedSourceGroupEscapeStat>
ExchangeSourceGroupStatsByCellOwner(
    std::vector<SphericalObserver::SourceCellGroupEscapeStat> const& localStats,
    AdaptiveGroupSourceUpdateSummary& summary)
{
    summary.maxLocalSourceGroupStats =
        static_cast<unsigned long long>(localStats.size());

#ifdef RICH_MPI
    int ranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    std::vector<size_t> sendElements(static_cast<size_t>(ranks), 0);
    size_t sendTotal = 0;
    for (auto const& s : localStats) {
        if (!(s.energy > 0.0) || s.count == 0 || !std::isfinite(s.energy))
            continue;
        int owner = AdaptiveCellOwner(s.cellID, ranks);
        ++sendElements[static_cast<size_t>(owner)];
        ++sendTotal;
    }

    std::vector<int> sendCounts(static_cast<size_t>(ranks), 0);
    std::vector<int> recvCounts(static_cast<size_t>(ranks), 0);
    for (int r = 0; r < ranks; ++r) {
        sendCounts[static_cast<size_t>(r)] =
            CheckedByteCount(sendElements[static_cast<size_t>(r)],
                             sizeof(PackedSourceGroupEscapeStat),
                             "Adaptive source-group shard");
    }
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT,
                 MPI_COMM_WORLD);

    std::vector<int> sendDispls(static_cast<size_t>(ranks), 0);
    std::vector<int> recvDispls(static_cast<size_t>(ranks), 0);
    unsigned long long totalSendBytes64 = 0;
    unsigned long long totalRecvBytes64 = 0;
    for (int r = 0; r < ranks; ++r) {
        sendDispls[static_cast<size_t>(r)] =
            CheckedByteTotal(totalSendBytes64, "Adaptive source-group shard send");
        recvDispls[static_cast<size_t>(r)] =
            CheckedByteTotal(totalRecvBytes64, "Adaptive source-group shard receive");
        totalSendBytes64 += static_cast<unsigned long long>(sendCounts[static_cast<size_t>(r)]);
        totalRecvBytes64 += static_cast<unsigned long long>(recvCounts[static_cast<size_t>(r)]);
    }
    int const totalSendBytes = CheckedByteTotal(totalSendBytes64,
                                                "Adaptive source-group shard send");
    int const totalRecvBytes = CheckedByteTotal(totalRecvBytes64,
                                                "Adaptive source-group shard receive");

    std::vector<PackedSourceGroupEscapeStat> sendData(sendTotal);
    std::vector<size_t> nextSendIndex(static_cast<size_t>(ranks), 0);
    for (int r = 0; r < ranks; ++r)
        nextSendIndex[static_cast<size_t>(r)] =
            static_cast<size_t>(sendDispls[static_cast<size_t>(r)]) /
            sizeof(PackedSourceGroupEscapeStat);
    for (auto const& s : localStats) {
        if (!(s.energy > 0.0) || s.count == 0 || !std::isfinite(s.energy))
            continue;
        int owner = AdaptiveCellOwner(s.cellID, ranks);
        sendData[nextSendIndex[static_cast<size_t>(owner)]++] =
            PackSourceGroupEscapeStat(s);
    }

    std::vector<PackedSourceGroupEscapeStat> recvData(
        static_cast<size_t>(totalRecvBytes) / sizeof(PackedSourceGroupEscapeStat));
    MPI_Alltoallv(sendData.empty() ? nullptr : sendData.data(), sendCounts.data(),
                  sendDispls.data(), MPI_BYTE,
                  recvData.empty() ? nullptr : recvData.data(), recvCounts.data(),
                  recvDispls.data(), MPI_BYTE, MPI_COMM_WORLD);

    unsigned long long localPairs = static_cast<unsigned long long>(localStats.size());
    unsigned long long recvPairs = static_cast<unsigned long long>(recvData.size());
    unsigned long long packedBytes = totalSendBytes64 + totalRecvBytes64;
    MPI_Allreduce(&localPairs, &summary.maxLocalSourceGroupStats, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&recvPairs, &summary.maxReceivedShardStats, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&recvPairs, &summary.mpiStatsExchanged, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&packedBytes, &summary.maxPackedBytes, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    (void)totalSendBytes;
    return recvData;
#else
    std::vector<PackedSourceGroupEscapeStat> result;
    result.reserve(localStats.size());
    for (auto const& s : localStats) {
        if (!(s.energy > 0.0) || s.count == 0 || !std::isfinite(s.energy))
            continue;
        result.push_back(PackSourceGroupEscapeStat(s));
    }
    summary.maxReceivedShardStats = static_cast<unsigned long long>(result.size());
    summary.mpiStatsExchanged = static_cast<unsigned long long>(result.size());
    summary.maxPackedBytes =
        static_cast<unsigned long long>(result.size()) *
        sizeof(PackedSourceGroupEscapeStat);
    return result;
#endif
}

std::vector<PackedAdaptiveCellGroupScoreDelta>
AllgatherAdaptiveCellGroupScoreDeltas(
    std::vector<PackedAdaptiveCellGroupScoreDelta> const& localDeltas)
{
#ifdef RICH_MPI
    int ranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int localBytes = CheckedByteCount(localDeltas.size(),
                                      sizeof(PackedAdaptiveCellGroupScoreDelta),
                                      "Adaptive source-group score delta packet");
    std::vector<int> counts(static_cast<size_t>(ranks), 0);
    MPI_Allgather(&localBytes, 1, MPI_INT, counts.data(), 1, MPI_INT,
                  MPI_COMM_WORLD);

    std::vector<int> displs(static_cast<size_t>(ranks), 0);
    unsigned long long totalBytes64 = 0;
    for (int r = 0; r < ranks; ++r) {
        displs[static_cast<size_t>(r)] =
            CheckedByteTotal(totalBytes64, "Adaptive source-group score delta");
        totalBytes64 += static_cast<unsigned long long>(counts[static_cast<size_t>(r)]);
    }
    int const totalBytes =
        CheckedByteTotal(totalBytes64, "Adaptive source-group score delta");

    std::vector<PackedAdaptiveCellGroupScoreDelta> result(
        static_cast<size_t>(totalBytes) / sizeof(PackedAdaptiveCellGroupScoreDelta));
    MPI_Allgatherv(localDeltas.empty() ? nullptr : localDeltas.data(), localBytes,
                   MPI_BYTE, result.empty() ? nullptr : result.data(),
                   counts.data(), displs.data(), MPI_BYTE, MPI_COMM_WORLD);
    return result;
#else
    return localDeltas;
#endif
}

std::vector<PackedSourceEscapeStat>
ExchangeSourceStatsByCellOwner(std::vector<SphericalObserver::SourceCellEscapeStat> const& localStats,
                               AdaptiveSourceUpdateSummary& summary)
{
    summary.maxLocalSourcePairs = static_cast<unsigned long long>(localStats.size());

#ifdef RICH_MPI
    int ranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    std::vector<size_t> sendElements(static_cast<size_t>(ranks), 0);
    size_t sendTotal = 0;
    for (auto const& s : localStats) {
        if (!(s.energy > 0.0) || s.count == 0 || !std::isfinite(s.energy))
            continue;
        int owner = AdaptiveCellOwner(s.cellID, ranks);
        ++sendElements[static_cast<size_t>(owner)];
        ++sendTotal;
    }

    std::vector<int> sendCounts(static_cast<size_t>(ranks), 0);
    std::vector<int> recvCounts(static_cast<size_t>(ranks), 0);
    for (int r = 0; r < ranks; ++r) {
        sendCounts[static_cast<size_t>(r)] =
            CheckedByteCount(sendElements[static_cast<size_t>(r)],
                             sizeof(PackedSourceEscapeStat),
                             "Adaptive source shard");
    }
    MPI_Alltoall(sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> sendDispls(static_cast<size_t>(ranks), 0);
    std::vector<int> recvDispls(static_cast<size_t>(ranks), 0);
    unsigned long long totalSendBytes64 = 0;
    unsigned long long totalRecvBytes64 = 0;
    for (int r = 0; r < ranks; ++r) {
        sendDispls[static_cast<size_t>(r)] =
            CheckedByteTotal(totalSendBytes64, "Adaptive source shard send");
        recvDispls[static_cast<size_t>(r)] =
            CheckedByteTotal(totalRecvBytes64, "Adaptive source shard receive");
        totalSendBytes64 += static_cast<unsigned long long>(sendCounts[static_cast<size_t>(r)]);
        totalRecvBytes64 += static_cast<unsigned long long>(recvCounts[static_cast<size_t>(r)]);
    }
    int const totalSendBytes = CheckedByteTotal(totalSendBytes64, "Adaptive source shard send");
    int const totalRecvBytes = CheckedByteTotal(totalRecvBytes64, "Adaptive source shard receive");

    std::vector<PackedSourceEscapeStat> sendData(sendTotal);
    std::vector<size_t> nextSendIndex(static_cast<size_t>(ranks), 0);
    for (int r = 0; r < ranks; ++r)
        nextSendIndex[static_cast<size_t>(r)] =
            static_cast<size_t>(sendDispls[static_cast<size_t>(r)]) / sizeof(PackedSourceEscapeStat);
    for (auto const& s : localStats) {
        if (!(s.energy > 0.0) || s.count == 0 || !std::isfinite(s.energy))
            continue;
        int owner = AdaptiveCellOwner(s.cellID, ranks);
        sendData[nextSendIndex[static_cast<size_t>(owner)]++] = PackSourceEscapeStat(s);
    }
    std::vector<PackedSourceEscapeStat> recvData(static_cast<size_t>(totalRecvBytes) /
                                                 sizeof(PackedSourceEscapeStat));

    MPI_Alltoallv(sendData.empty() ? nullptr : sendData.data(), sendCounts.data(),
                  sendDispls.data(), MPI_BYTE,
                  recvData.empty() ? nullptr : recvData.data(), recvCounts.data(),
                  recvDispls.data(), MPI_BYTE, MPI_COMM_WORLD);

    unsigned long long localPairs = static_cast<unsigned long long>(localStats.size());
    unsigned long long recvPairs = static_cast<unsigned long long>(recvData.size());
    unsigned long long packedBytes = totalSendBytes64 + totalRecvBytes64;
    MPI_Allreduce(&localPairs, &summary.maxLocalSourcePairs, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&recvPairs, &summary.maxReceivedShardPairs, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&packedBytes, &summary.maxPackedBytes, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX, MPI_COMM_WORLD);
    return recvData;
#else
    std::vector<PackedSourceEscapeStat> result;
    result.reserve(localStats.size());
    for (auto const& s : localStats) {
        if (!(s.energy > 0.0) || s.count == 0 || !std::isfinite(s.energy))
            continue;
        result.push_back(PackSourceEscapeStat(s));
    }
    summary.maxReceivedShardPairs = static_cast<unsigned long long>(result.size());
    summary.maxPackedBytes =
        static_cast<unsigned long long>(result.size()) * sizeof(PackedSourceEscapeStat);
    return result;
#endif
}

std::vector<PackedAdaptiveScoreDelta>
AllgatherAdaptiveScoreDeltas(std::vector<PackedAdaptiveScoreDelta> const& localDeltas)
{
#ifdef RICH_MPI
    int ranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int localBytes = CheckedByteCount(localDeltas.size(), sizeof(PackedAdaptiveScoreDelta),
                                      "Adaptive source score delta packet");
    std::vector<int> counts(static_cast<size_t>(ranks), 0);
    MPI_Allgather(&localBytes, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> displs(static_cast<size_t>(ranks), 0);
    unsigned long long totalBytes64 = 0;
    for (int r = 0; r < ranks; ++r) {
        displs[static_cast<size_t>(r)] =
            CheckedByteTotal(totalBytes64, "Adaptive source score delta");
        totalBytes64 += static_cast<unsigned long long>(counts[static_cast<size_t>(r)]);
    }
    int const totalBytes = CheckedByteTotal(totalBytes64, "Adaptive source score delta");

    std::vector<PackedAdaptiveScoreDelta> result(static_cast<size_t>(totalBytes) /
                                                 sizeof(PackedAdaptiveScoreDelta));
    MPI_Allgatherv(localDeltas.empty() ? nullptr : localDeltas.data(), localBytes, MPI_BYTE,
                   result.empty() ? nullptr : result.data(), counts.data(), displs.data(),
                   MPI_BYTE, MPI_COMM_WORLD);
    return result;
#else
    return localDeltas;
#endif
}

std::vector<SphericalObserver::SourceCellEscapeStat>
GatherTopSourceStats(std::vector<SphericalObserver::SourceCellEscapeStat> const& localStats)
{
    constexpr size_t TOP_N = 10;

    std::vector<SphericalObserver::SourceCellEscapeStat> localTop;
    localTop.reserve(TOP_N);
    for (auto const& s : localStats) {
        if (localTop.size() < TOP_N) {
            localTop.push_back(s);
            continue;
        }

        auto minIt = std::min_element(localTop.begin(), localTop.end(),
                                      [](auto const& a, auto const& b) {
                                          return a.energy < b.energy;
                                      });
        if (minIt != localTop.end() && s.energy > minIt->energy)
            *minIt = s;
    }
    std::sort(localTop.begin(), localTop.end(),
              [](auto const& a, auto const& b) { return a.energy > b.energy; });

#ifdef RICH_MPI
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    std::vector<PackedSourceEscapeStat> send(TOP_N);
    for (size_t i = 0; i < localTop.size(); ++i)
        send[i] = PackSourceEscapeStat(localTop[i]);

    std::vector<PackedSourceEscapeStat> recv;
    if (rank == 0)
        recv.resize(TOP_N * static_cast<size_t>(ranks));
    MPI_Gather(send.data(), static_cast<int>(TOP_N * sizeof(PackedSourceEscapeStat)), MPI_BYTE,
               rank == 0 ? recv.data() : nullptr,
               static_cast<int>(TOP_N * sizeof(PackedSourceEscapeStat)), MPI_BYTE,
               0, MPI_COMM_WORLD);

    std::vector<SphericalObserver::SourceCellEscapeStat> result;
    if (rank == 0) {
        for (auto const& p : recv) {
            if (p.count == 0 || !(p.energy > 0.0) || !std::isfinite(p.energy))
                continue;
            result.push_back(UnpackSourceEscapeStat(p));
        }
        std::sort(result.begin(), result.end(),
                  [](auto const& a, auto const& b) { return a.energy > b.energy; });
        if (result.size() > TOP_N)
            result.resize(TOP_N);
    }
    return result;
#else
    return localTop;
#endif
}

void ReduceDoubleVector(std::vector<double>& values)
{
#ifdef RICH_MPI
    if (!values.empty())
        MPI_Allreduce(MPI_IN_PLACE, values.data(), static_cast<int>(values.size()),
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
    (void)values;
#endif
}

void ReduceUnsignedLongLongVector(std::vector<unsigned long long>& values)
{
#ifdef RICH_MPI
    if (!values.empty())
        MPI_Allreduce(MPI_IN_PLACE, values.data(), static_cast<int>(values.size()),
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#else
    (void)values;
#endif
}

SphericalObserver::ObserverQualitySnapshot
CollectGlobalObserverQuality(SphericalObserver::ObserverQualitySnapshot local)
{
    ReduceDoubleVector(local.energy);
    ReduceDoubleVector(local.energyWeightSq);
    ReduceUnsignedLongLongVector(local.crossingCount);
    ReduceDoubleVector(local.stokesQ);
    ReduceDoubleVector(local.stokesU);
    ReduceDoubleVector(local.polarizationWeightSq);
    ReduceDoubleVector(local.sumWQ2);
    ReduceDoubleVector(local.sumWU2);
#ifdef RICH_MPI
    int polEnabled = local.polarizationEnabled ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &polEnabled, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    local.polarizationEnabled = (polEnabled != 0);
#endif
    return local;
}

double Percentile(std::vector<double> values, double p)
{
    values.erase(std::remove_if(values.begin(), values.end(),
                 [](double x) { return !std::isfinite(x); }), values.end());
    if (values.empty())
        return 0.0;
    std::sort(values.begin(), values.end());
    double const pos = std::clamp(p, 0.0, 1.0) *
                       static_cast<double>(values.size() - 1);
    size_t const lo = static_cast<size_t>(std::floor(pos));
    size_t const hi = std::min(values.size() - 1, lo + 1);
    double const t = pos - static_cast<double>(lo);
    return values[lo] * (1.0 - t) + values[hi] * t;
}

ObserverQualityDiagnostics BuildObserverQualityDiagnostics(
    SphericalObserver::ObserverQualitySnapshot const& snap,
    Config const& cfg,
    AdaptiveSourceState& state)
{
    ObserverQualityDiagnostics diag;
    diag.enabled = cfg.adaptiveSourceCells && cfg.adaptiveObserverEquity;
    diag.polarizationMode = diag.enabled && cfg.polarization && snap.polarizationEnabled;
    diag.observerCount = snap.energy.size();
    diag.budgetMultiplier = 1.0;

    if (!diag.enabled || diag.observerCount == 0) {
        state.observerBudgetMultiplier = 1.0;
        return diag;
    }

    std::vector<double> rawDeficit(diag.observerCount, 1.0);
    diag.neffByObserver.assign(diag.observerCount, 0.0);
    diag.snrByObserver.assign(diag.observerCount, 0.0);
    diag.crossingsByObserver = snap.crossingCount;

    for (size_t i = 0; i < diag.observerCount; ++i) {
        double const energy = snap.energy[i];
        double const w2 = diag.polarizationMode
            ? ((i < snap.polarizationWeightSq.size()) ? snap.polarizationWeightSq[i] : 0.0)
            : ((i < snap.energyWeightSq.size()) ? snap.energyWeightSq[i] : 0.0);
        double neff = 0.0;
        if (energy > 0.0 && w2 > 0.0 && std::isfinite(energy) && std::isfinite(w2))
            neff = energy * energy / w2;
        diag.neffByObserver[i] = neff;

        double deficit = 1.0;
        if (neff > 0.0)
            deficit = std::max(deficit, cfg.adaptiveObserverTargetNeff / neff);
        else
            deficit = cfg.adaptiveObserverDeficitMax;

        if (diag.polarizationMode && energy > 0.0 && w2 > 0.0) {
            double const q = (i < snap.stokesQ.size()) ? snap.stokesQ[i] / energy : 0.0;
            double const u = (i < snap.stokesU.size()) ? snap.stokesU[i] / energy : 0.0;
            double varQ = (i < snap.sumWQ2.size()) ? snap.sumWQ2[i] / energy - q * q : 0.0;
            double varU = (i < snap.sumWU2.size()) ? snap.sumWU2[i] / energy - u * u : 0.0;
            varQ = std::max(0.0, varQ);
            varU = std::max(0.0, varU);
            double const sigQ = (neff > 0.0) ? std::sqrt(varQ / neff) : 0.0;
            double const sigU = (neff > 0.0) ? std::sqrt(varU / neff) : 0.0;
            double const sigP = std::sqrt(sigQ * sigQ + sigU * sigU);
            double const polDegree = std::sqrt(q * q + u * u);
            double const snr = (sigP > 0.0) ? polDegree / sigP : 0.0;
            diag.snrByObserver[i] = snr;
            if (snr > 0.0)
                deficit = std::max(deficit, cfg.adaptiveObserverTargetPolSnr / snr);
            else
                deficit = cfg.adaptiveObserverDeficitMax;
        }

        rawDeficit[i] = std::clamp(deficit, 1.0, cfg.adaptiveObserverDeficitMax);
    }

    if (state.observerDeficitByIndex.size() != diag.observerCount)
        state.observerDeficitByIndex.assign(diag.observerCount, 1.0);
    diag.deficitByObserver.resize(diag.observerCount, 1.0);

    double deficitSum = 0.0;
    diag.deficitMin = std::numeric_limits<double>::max();
    diag.deficitMax = 1.0;
    for (size_t i = 0; i < diag.observerCount; ++i) {
        double const oldDeficit = state.observerDeficitByIndex[i];
        double const smooth = oldDeficit * (1.0 - cfg.adaptiveObserverDeficitEma)
                            + rawDeficit[i] * cfg.adaptiveObserverDeficitEma;
        double const finalDeficit = std::clamp(smooth, 1.0, cfg.adaptiveObserverDeficitMax);
        state.observerDeficitByIndex[i] = finalDeficit;
        diag.deficitByObserver[i] = finalDeficit;
        deficitSum += finalDeficit;
        diag.deficitMin = std::min(diag.deficitMin, finalDeficit);
        diag.deficitMax = std::max(diag.deficitMax, finalDeficit);
        if (finalDeficit > 1.0001)
            ++diag.weakObservers;
        unsigned long long const crossings =
            (i < diag.crossingsByObserver.size()) ? diag.crossingsByObserver[i] : 0ULL;
        if (diag.neffByObserver[i] <= 0.0 || crossings == 0)
            ++diag.zeroStatObservers;
    }
    if (diag.deficitMin == std::numeric_limits<double>::max())
        diag.deficitMin = 1.0;
    diag.deficitAvg = deficitSum / static_cast<double>(diag.observerCount);

    diag.neffP05 = Percentile(diag.neffByObserver, 0.05);
    diag.neffMedian = Percentile(diag.neffByObserver, 0.50);
    diag.neffP95 = Percentile(diag.neffByObserver, 0.95);
    diag.snrP05 = Percentile(diag.snrByObserver, 0.05);
    diag.snrMedian = Percentile(diag.snrByObserver, 0.50);
    diag.snrP95 = Percentile(diag.snrByObserver, 0.95);

    double const weakFrac = static_cast<double>(diag.weakObservers) /
                        static_cast<double>(diag.observerCount);

// The old weakFrac-only driver barely responds when only a few observers are
// terrible.  Use deficit severity as well, so low-SNR / low-Neff tails get
// meaningful extra budget.
double const deficitTail95 = Percentile(diag.deficitByObserver, 0.95);

double deficitDriver = 0.0;
deficitDriver = std::max(deficitDriver, diag.deficitAvg - 1.0);
deficitDriver = std::max(deficitDriver, 0.25 * (deficitTail95 - 1.0));
deficitDriver = std::max(deficitDriver, 0.10 * (diag.deficitMax - 1.0));

// Keep a small weak-fraction term so many mildly weak observers still increase
// budget, but do not rely on it for a few pathological observers.
deficitDriver = std::max(deficitDriver, weakFrac);

diag.budgetMultiplier =
    1.0 + cfg.adaptiveObserverExtraBudgetFrac * std::max(0.0, deficitDriver);

// Hard safety cap.  Increase this only if you are prepared for the memory/runtime
// cost.  With --adaptive-observer-extra-budget-frac 2, this can still get large
// when deficitMax is 100.
diag.budgetMultiplier = std::min(diag.budgetMultiplier, 10.0);

state.observerBudgetMultiplier = diag.budgetMultiplier;
return diag;
}

::RadiationIMC::SourceAllocationSummary
ReduceSourceAllocationSummary(::RadiationIMC::SourceAllocationSummary local)
{
#ifdef RICH_MPI
    unsigned long long const localSourceCells = local.sourceCells;
    unsigned long long const localLearnedCells = local.learnedCells;
    unsigned long long sums[8] = {
        local.totalPhotons,
        local.sourceCells,
        local.boostedCells,
        local.learnedCells,
        local.learnedBoostedCells,
        local.learnedPhotons,
        local.learnedExtraPhotons,
        0
    };
    MPI_Allreduce(MPI_IN_PLACE, sums, 8, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    local.totalPhotons = sums[0];
    local.sourceCells = sums[1];
    local.boostedCells = sums[2];
    local.learnedCells = sums[3];
    local.learnedBoostedCells = sums[4];
    local.learnedPhotons = sums[5];
    local.learnedExtraPhotons = sums[6];

    unsigned long long minPhotons = localSourceCells > 0
        ? static_cast<unsigned long long>(local.minPhotons)
        : ULLONG_MAX;
    unsigned long long maxPhotons = static_cast<unsigned long long>(local.maxPhotons);
    unsigned long long learnedMinPhotons = localLearnedCells > 0
        ? static_cast<unsigned long long>(local.learnedMinPhotons)
        : ULLONG_MAX;
    unsigned long long learnedMaxPhotons = static_cast<unsigned long long>(local.learnedMaxPhotons);
    MPI_Allreduce(MPI_IN_PLACE, &minPhotons, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxPhotons, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &learnedMinPhotons, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &learnedMaxPhotons, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    local.minPhotons = (minPhotons == ULLONG_MAX) ? 0 : static_cast<size_t>(minPhotons);
    local.maxPhotons = static_cast<size_t>(maxPhotons);
    local.learnedMinPhotons = (learnedMinPhotons == ULLONG_MAX) ? 0 : static_cast<size_t>(learnedMinPhotons);
    local.learnedMaxPhotons = static_cast<size_t>(learnedMaxPhotons);

    MPI_Allreduce(MPI_IN_PLACE, &local.adaptiveScoreSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    int adaptive = local.adaptiveEnabled ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &adaptive, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    local.adaptiveEnabled = adaptive != 0;
#endif
    return local;
}

::RadiationIMC::GroupSamplingDiagnostics
ReduceGroupSamplingDiagnostics(::RadiationIMC::GroupSamplingDiagnostics local)
{
#ifdef RICH_MPI
    size_t const localWeightCorrectionCount = local.weightCorrectionCount;
    unsigned long long sums[7] = {
        static_cast<unsigned long long>(local.totalSampled),
        static_cast<unsigned long long>(local.weightCorrectionCount),
        static_cast<unsigned long long>(local.weightCorrectionCapped),
        static_cast<unsigned long long>(local.weightCorrectionFallback),
        static_cast<unsigned long long>(local.invalidPdfFallback),
        static_cast<unsigned long long>(local.invalidPdfFallbackPackets),
        local.estimatorPotentiallyBiased ? 1ULL : 0ULL
    };
    MPI_Allreduce(MPI_IN_PLACE, sums, 7, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
    local.totalSampled = static_cast<size_t>(sums[0]);
    local.weightCorrectionCount = static_cast<size_t>(sums[1]);
    local.weightCorrectionCapped = static_cast<size_t>(sums[2]);
    local.weightCorrectionFallback = static_cast<size_t>(sums[3]);
    local.invalidPdfFallback = static_cast<size_t>(sums[4]);
    local.invalidPdfFallbackPackets = static_cast<size_t>(sums[5]);
    local.estimatorPotentiallyBiased = sums[6] > 0;

    unsigned long long cellsWithGroupScores =
        static_cast<unsigned long long>(local.cellsWithGroupScores);
    MPI_Allreduce(MPI_IN_PLACE, &cellsWithGroupScores, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    local.cellsWithGroupScores = static_cast<size_t>(cellsWithGroupScores);

    double doubleSums[3] = {
        local.weightCorrectionSum,
        local.sampledEnergy,
        local.cappedEnergy
    };
    MPI_Allreduce(MPI_IN_PLACE, doubleSums, 3, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    local.weightCorrectionSum = doubleSums[0];
    local.sampledEnergy = doubleSums[1];
    local.cappedEnergy = doubleSums[2];

    double minCorr = localWeightCorrectionCount > 0
        ? local.weightCorrectionMin
        : std::numeric_limits<double>::infinity();
    double maxCorr = localWeightCorrectionCount > 0
        ? local.weightCorrectionMax
        : 1.0;
    MPI_Allreduce(MPI_IN_PLACE, &minCorr, 1, MPI_DOUBLE, MPI_MIN,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxCorr, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    local.weightCorrectionMin = std::isfinite(minCorr) ? minCorr : 1.0;
    local.weightCorrectionMax = local.weightCorrectionCount > 0 ? maxCorr : 1.0;
#endif
    local.cappedEnergyFraction = local.sampledEnergy > 0.0
        ? local.cappedEnergy / local.sampledEnergy
        : 0.0;
    local.estimatorPotentiallyBiased =
        local.estimatorPotentiallyBiased ||
        local.weightCorrectionCapped > 0 ||
        local.cappedEnergy > 0.0;
    return local;
}

void AccumulateGroupSamplingDiagnostics(
    ::RadiationIMC::GroupSamplingDiagnostics& total,
    ::RadiationIMC::GroupSamplingDiagnostics const& gen)
{
    if (gen.weightCorrectionCount > 0) {
        if (total.weightCorrectionCount == 0) {
            total.weightCorrectionMin = gen.weightCorrectionMin;
            total.weightCorrectionMax = gen.weightCorrectionMax;
        } else {
            total.weightCorrectionMin =
                std::min(total.weightCorrectionMin, gen.weightCorrectionMin);
            total.weightCorrectionMax =
                std::max(total.weightCorrectionMax, gen.weightCorrectionMax);
        }
    }

    total.totalSampled += gen.totalSampled;
    total.cellsWithGroupScores =
        std::max(total.cellsWithGroupScores, gen.cellsWithGroupScores);
    total.weightCorrectionSum += gen.weightCorrectionSum;
    total.weightCorrectionCount += gen.weightCorrectionCount;
    total.weightCorrectionCapped += gen.weightCorrectionCapped;
    total.weightCorrectionFallback += gen.weightCorrectionFallback;
    total.invalidPdfFallback += gen.invalidPdfFallback;
    total.invalidPdfFallbackPackets += gen.invalidPdfFallbackPackets;
    total.sampledEnergy += gen.sampledEnergy;
    total.cappedEnergy += gen.cappedEnergy;
    total.estimatorPotentiallyBiased =
        total.estimatorPotentiallyBiased || gen.estimatorPotentiallyBiased;
    total.cappedEnergyFraction = total.sampledEnergy > 0.0
        ? total.cappedEnergy / total.sampledEnergy
        : 0.0;
}

AdaptiveSourceUpdateSummary UpdateAdaptiveSourceScoresDistributed(
    std::vector<SphericalObserver::SourceCellEscapeStat> const& localStats,
    Config const& cfg,
    AdaptiveSourceState& state,
    ObserverQualityDiagnostics const& observerQuality,
    bool decayExistingScores)
{
    AdaptiveSourceUpdateSummary summary;
    size_t const nObs = cfg.nObservers;
    std::vector<double> energyByObserver(nObs, 0.0);
    std::vector<double> weightSqByObserver(nObs, 0.0);
    std::vector<unsigned long long> crossingsByObserver(nObs, 0ULL);

    for (auto const& s : localStats) {
        if (s.observerIndex >= nObs || !(s.energy > 0.0) || !std::isfinite(s.energy))
            continue;

        energyByObserver[s.observerIndex] += s.energy;
        double const w2 = (s.weightSq > 0.0 && std::isfinite(s.weightSq))
            ? s.weightSq
            : s.energy * s.energy;
        weightSqByObserver[s.observerIndex] += w2;
        crossingsByObserver[s.observerIndex] += static_cast<unsigned long long>(s.count);
    }

#ifdef RICH_MPI
    if (!energyByObserver.empty()) {
        MPI_Allreduce(MPI_IN_PLACE, energyByObserver.data(), static_cast<int>(energyByObserver.size()),
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, weightSqByObserver.data(), static_cast<int>(weightSqByObserver.size()),
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, crossingsByObserver.data(), static_cast<int>(crossingsByObserver.size()),
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    }
#endif

    for (size_t obs = 0; obs < nObs; ++obs) {
        summary.totalEscapedEnergy += energyByObserver[obs];
        summary.totalCrossings += crossingsByObserver[obs];
        if (crossingsByObserver[obs] > 0)
            ++summary.observersWithCrossings;
    }

    if (!cfg.adaptiveSourceCells) {
        summary.scoreMapCells = state.scoreByCellID.size();
        return summary;
    }

    if (decayExistingScores) {
        double const decay = 1.0 - cfg.adaptiveSourceEma;
        summary.decayedCells = state.scoreByCellID.size();
        for (auto& kv : state.scoreByCellID) {
            kv.second *= decay;
        }
    }

    std::vector<PackedSourceEscapeStat> received = ExchangeSourceStatsByCellOwner(localStats, summary);
    std::unordered_map<AdaptivePairKey, SphericalObserver::SourceCellEscapeStat, AdaptivePairKeyHash> byPair;
    byPair.reserve(received.size());
    for (auto const& p : received) {
        if (p.count == 0 || !(p.energy > 0.0) || !std::isfinite(p.energy))
            continue;
        size_t const observerIndex = static_cast<size_t>(p.observerIndex);
        size_t const cellID = static_cast<size_t>(p.cellID);
        AdaptivePairKey const key{observerIndex, cellID};
        auto& s = byPair[key];
        s.cellID = cellID;
        s.observerIndex = observerIndex;
        s.energy += p.energy;
        s.weightSq += p.weightSq;
        s.maxWeight = std::max(s.maxWeight, p.maxWeight);
        s.count += static_cast<size_t>(p.count);
    }
    std::vector<PackedSourceEscapeStat>().swap(received);

    std::vector<SphericalObserver::SourceCellEscapeStat> ownedStats;
    ownedStats.reserve(byPair.size());
    for (auto const& kv : byPair)
        ownedStats.push_back(kv.second);
    std::unordered_map<AdaptivePairKey, SphericalObserver::SourceCellEscapeStat,
                       AdaptivePairKeyHash>().swap(byPair);
    summary.topStats = GatherTopSourceStats(ownedStats);

    unsigned long long localPairCount = static_cast<unsigned long long>(ownedStats.size());
    unsigned long long globalPairCount = localPairCount;
#ifdef RICH_MPI
    MPI_Allreduce(&localPairCount, &globalPairCount, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
#endif
    summary.sourceCellObserverPairs = static_cast<size_t>(globalPairCount);

    std::unordered_map<size_t, double> deltaByCell;
    std::unordered_set<size_t> touchedNewCells;
    for (auto const& s : ownedStats) {
        if (s.observerIndex >= energyByObserver.size() ||
            !(energyByObserver[s.observerIndex] > 0.0) ||
            !std::isfinite(energyByObserver[s.observerIndex]))
            continue;

        double observerBoost = 1.0;
        if (observerQuality.enabled &&
            s.observerIndex < observerQuality.deficitByObserver.size() &&
            observerQuality.deficitByObserver[s.observerIndex] > 0.0 &&
            std::isfinite(observerQuality.deficitByObserver[s.observerIndex]))
        {
            observerBoost = observerQuality.deficitByObserver[s.observerIndex];
        }

        double const eFrac = s.energy / energyByObserver[s.observerIndex];

        double const w2 = (s.weightSq > 0.0 && std::isfinite(s.weightSq))
            ? s.weightSq
            : s.energy * s.energy;

        double const w2Total = weightSqByObserver[s.observerIndex];

        double const w2Frac = (w2Total > 0.0 && std::isfinite(w2Total))
            ? w2 / w2Total
            : eFrac;

        // In polarization mode, variance matters much more than energy:
        // a cell emitting a few huge packets can dominate Q/U uncertainty.
        double const varianceMix = observerQuality.polarizationMode ? 0.85 : 0.35;

        double const sourceQualityScore =
            (1.0 - varianceMix) * eFrac + varianceMix * w2Frac;

        double const minScore =
            cfg.adaptiveSourceMinEscapedFrac / std::max(1.0, observerBoost);

        if (!(sourceQualityScore >= minScore) || !std::isfinite(sourceQualityScore))
            continue;

        ++summary.passedCells;

        auto it = state.scoreByCellID.find(s.cellID);
        bool existed = (it != state.scoreByCellID.end() && it->second > 0.0);
        bool alreadyTouched = touchedNewCells.find(s.cellID) != touchedNewCells.end();
        if (existed || alreadyTouched)
            ++summary.retainedCells;
        else {
            ++summary.newCells;
            touchedNewCells.insert(s.cellID);
        }

        deltaByCell[s.cellID] += cfg.adaptiveSourceEma * observerBoost * sourceQualityScore;
    }

    std::vector<PackedAdaptiveScoreDelta> localDeltas;
    localDeltas.reserve(deltaByCell.size());
    for (auto const& kv : deltaByCell) {
        if (!(kv.second > 0.0) || !std::isfinite(kv.second))
            continue;
        PackedAdaptiveScoreDelta p;
        p.cellID = static_cast<unsigned long long>(kv.first);
        p.delta = kv.second;
        localDeltas.push_back(p);
    }
    std::unordered_map<size_t, double>().swap(deltaByCell);
    std::unordered_set<size_t>().swap(touchedNewCells);
    std::vector<SphericalObserver::SourceCellEscapeStat>().swap(ownedStats);

    size_t localDeltaCells = localDeltas.size();
    std::vector<PackedAdaptiveScoreDelta> allDeltas = AllgatherAdaptiveScoreDeltas(localDeltas);
    std::vector<PackedAdaptiveScoreDelta>().swap(localDeltas);
    for (auto const& p : allDeltas) {
        if (!(p.delta > 0.0) || !std::isfinite(p.delta))
            continue;
        state.scoreByCellID[static_cast<size_t>(p.cellID)] += p.delta;
    }
    size_t const scoreDeltaCells = allDeltas.size();
    std::vector<PackedAdaptiveScoreDelta>().swap(allDeltas);

    for (auto it = state.scoreByCellID.begin(); it != state.scoreByCellID.end(); ) {
        if (!(it->second > 0.0) || !std::isfinite(it->second))
            it = state.scoreByCellID.erase(it);
        else
            ++it;
    }

    unsigned long long localPassed = static_cast<unsigned long long>(summary.passedCells);
    unsigned long long localNew = static_cast<unsigned long long>(summary.newCells);
    unsigned long long localRetained = static_cast<unsigned long long>(summary.retainedCells);
    unsigned long long globalPassed = localPassed;
    unsigned long long globalNew = localNew;
    unsigned long long globalRetained = localRetained;
#ifdef RICH_MPI
    MPI_Allreduce(&localPassed, &globalPassed, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localNew, &globalNew, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localRetained, &globalRetained, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
    summary.passedCells = static_cast<size_t>(globalPassed);
    summary.newCells = static_cast<size_t>(globalNew);
    summary.retainedCells = static_cast<size_t>(globalRetained);
    summary.scoreDeltaCells = scoreDeltaCells;
    summary.scoreMapCells = state.scoreByCellID.size();

    unsigned long long maxLocalDeltaCells = static_cast<unsigned long long>(localDeltaCells);
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &maxLocalDeltaCells, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
#endif
    summary.maxPackedBytes = std::max(
        summary.maxPackedBytes,
        (maxLocalDeltaCells + static_cast<unsigned long long>(scoreDeltaCells)) *
            static_cast<unsigned long long>(sizeof(PackedAdaptiveScoreDelta)));

    return summary;
}

void PrintAdaptiveGenerationStart(
    std::string const& label,
    Config const& cfg,
    AdaptiveSourceState const& state,
    size_t gen,
    size_t totalGenerations,
    size_t burninGenerations,
    bool adaptiveActive,
    int rank)
{
    if (rank != 0 || !cfg.adaptiveSourceCells)
        return;
    std::string mode = "burn-in";
    if (adaptiveActive)
        mode = (gen == burninGenerations) ? "first adaptive" : "adaptive";
    std::string postAdaptiveLB = "disabled";
    if (cfg.measuredLoadBalance)
        postAdaptiveLB = state.postAdaptiveMeasuredLBDone ? "done" : "pending";
    size_t burninRemaining = (gen < burninGenerations)
        ? burninGenerations - gen : 0;
    std::cout << label << " adaptive generation state: gen " << (gen + 1)
              << "/" << totalGenerations
              << " mode=" << mode
              << " burnin_remaining=" << burninRemaining
              << " learned_cells=" << state.scoreByCellID.size()
              << " post_adaptive_LB="
              << postAdaptiveLB
              << std::endl;
    if (gen == burninGenerations && !state.burninCompletePrinted)
        std::cout << label << " adaptive source weights active for first time" << std::endl;
}

void PrintAdaptiveGenerationStats(
    std::string const& label,
    Config const& cfg,
    AdaptiveSourceState const& state,
    AdaptiveSourceUpdateSummary const& update,
    ::RadiationIMC::SourceAllocationSummary allocation,
    ObserverQualityDiagnostics const& observerQuality,
    size_t gen,
    int rank)
{
    if (rank != 0 || !cfg.adaptiveSourceCells)
        return;
    double avgPhotons = allocation.sourceCells > 0
        ? static_cast<double>(allocation.totalPhotons) / static_cast<double>(allocation.sourceCells)
        : 0.0;
    double learnedAvgPhotons = allocation.learnedCells > 0
        ? static_cast<double>(allocation.learnedPhotons) / static_cast<double>(allocation.learnedCells)
        : 0.0;
    double learnedPhotonFrac = allocation.totalPhotons > 0
        ? static_cast<double>(allocation.learnedPhotons) / static_cast<double>(allocation.totalPhotons)
        : 0.0;
    std::cout << label << " adaptive stats after generation " << (gen + 1)
              << ": crossing_energy=" << update.totalEscapedEnergy
              << " crossing_count=" << update.totalCrossings
              << " source_cell_observer_pairs=" << update.sourceCellObserverPairs
              << " observers_with_crossings=" << update.observersWithCrossings
              << " cells_passing_filter=" << update.passedCells
              << " learned_cells=" << state.scoreByCellID.size()
              << " new=" << update.newCells
              << " retained=" << update.retainedCells
              << " decayed=" << update.decayedCells << "\n"
              << label << " source allocation used: adaptive="
              << (allocation.adaptiveEnabled ? "yes" : "no")
              << " total_photons=" << allocation.totalPhotons
              << " boosted_cells=" << allocation.boostedCells
              << " learned_cells_allocated=" << allocation.learnedCells
              << " learned_boosted_cells=" << allocation.learnedBoostedCells
              << " learned_photons=" << allocation.learnedPhotons
              << " learned_photon_frac=" << learnedPhotonFrac
              << " learned_extra_photons=" << allocation.learnedExtraPhotons
              << " photons/cell min/avg/max=" << allocation.minPhotons
              << "/" << avgPhotons
              << "/" << allocation.maxPhotons
              << " learned photons/cell min/avg/max=" << allocation.learnedMinPhotons
              << "/" << learnedAvgPhotons
              << "/" << allocation.learnedMaxPhotons
              << std::endl;

    std::cout << label << " adaptive tally memory: max_local_pairs="
              << update.maxLocalSourcePairs
              << " max_received_shard_pairs=" << update.maxReceivedShardPairs
              << " score_delta_cells=" << update.scoreDeltaCells
              << " score_map_cells=" << update.scoreMapCells
              << " max_packed_bytes=" << update.maxPackedBytes
              << std::endl;

    if (observerQuality.enabled) {
        std::cout << label << " observer-equity stats: mode="
                  << (observerQuality.polarizationMode ? "polarization" : "luminosity")
                  << " weak_observers=" << observerQuality.weakObservers
                  << "/" << observerQuality.observerCount
                  << " zero_stat_observers=" << observerQuality.zeroStatObservers
                  << " deficit min/avg/max=" << observerQuality.deficitMin
                  << "/" << observerQuality.deficitAvg
                  << "/" << observerQuality.deficitMax
                  << " neff p05/med/p95=" << observerQuality.neffP05
                  << "/" << observerQuality.neffMedian
                  << "/" << observerQuality.neffP95;
        if (observerQuality.polarizationMode)
            std::cout << " pol_snr p05/med/p95=" << observerQuality.snrP05
                      << "/" << observerQuality.snrMedian
                      << "/" << observerQuality.snrP95;
        std::cout << " next_adaptive_budget_multiplier="
                  << observerQuality.budgetMultiplier
                  << std::endl;

        std::vector<size_t> order(observerQuality.deficitByObserver.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](size_t a, size_t b) {
                      return observerQuality.deficitByObserver[a] >
                             observerQuality.deficitByObserver[b];
                  });
        size_t const topWeak = std::min<size_t>(10, order.size());
        if (topWeak > 0) {
            std::cout << label << " weakest observers:" << std::endl;
            for (size_t j = 0; j < topWeak; ++j) {
                size_t const obs = order[j];
                double const neff = (obs < observerQuality.neffByObserver.size())
                    ? observerQuality.neffByObserver[obs] : 0.0;
                double const snr = (obs < observerQuality.snrByObserver.size())
                    ? observerQuality.snrByObserver[obs] : 0.0;
                unsigned long long crossings =
                    (obs < observerQuality.crossingsByObserver.size())
                    ? observerQuality.crossingsByObserver[obs] : 0ULL;
                std::cout << "  observer=" << obs
                          << " deficit=" << observerQuality.deficitByObserver[obs]
                          << " crossing_count=" << crossings
                          << " neff=" << neff;
                if (observerQuality.polarizationMode)
                    std::cout << " pol_snr=" << snr;
                std::cout << std::endl;
            }
        }
    }

    auto const& stats = update.topStats;
    size_t const topN = std::min<size_t>(10, stats.size());
    if (topN > 0) {
        std::cout << label << " top escaping source cells:" << std::endl;
        for (size_t i = 0; i < topN; ++i) {
            auto const& s = stats[i];
            double frac = (update.totalEscapedEnergy > 0.0) ? s.energy / update.totalEscapedEnergy : 0.0;
            auto it = state.scoreByCellID.find(s.cellID);
            double score = (it != state.scoreByCellID.end()) ? it->second : 0.0;
            double const sourceNeff = (s.weightSq > 0.0)
                ? s.energy * s.energy / s.weightSq
                : 0.0;
            double const avgWeight = (s.count > 0)
                ? s.energy / static_cast<double>(s.count)
                : 0.0;
            std::cout << "  observer=" << s.observerIndex
                << " cellID=" << s.cellID
                << " escaped_energy=" << s.energy
                << " escaped_frac=" << frac
                << " crossings=" << s.count
                << " source_neff=" << sourceNeff
                << " avg_weight=" << avgWeight
                << " max_weight=" << s.maxWeight
                << " weightSq=" << s.weightSq
                << " adaptive_score=" << score
                << std::endl;
        }
    }
}

// --- GROUP-AWARE ADAPTIVE FUNCTIONS ---

void CollectGlobalObserverGroupQuality(
    SphericalObserver::ObserverGroupQualitySnapshot& snap)
{
#ifdef RICH_MPI
    size_t const nObs = snap.observerCount;
    size_t const nGrp = snap.groupCount;
    size_t const flat = nObs * nGrp;
    if (flat == 0) return;

    auto flattenD = [&](std::vector<std::vector<double>>& mat) {
        std::vector<double> buf(flat, 0.0);
        for (size_t o = 0; o < nObs; ++o)
            for (size_t g = 0; g < nGrp; ++g)
                buf[o * nGrp + g] = mat[o][g];
        MPI_Allreduce(MPI_IN_PLACE, buf.data(), static_cast<int>(flat), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        for (size_t o = 0; o < nObs; ++o)
            for (size_t g = 0; g < nGrp; ++g)
                mat[o][g] = buf[o * nGrp + g];
    };

    auto flattenSz = [&](std::vector<std::vector<size_t>>& mat) {
        std::vector<unsigned long long> buf(flat, 0ULL);
        for (size_t o = 0; o < nObs; ++o)
            for (size_t g = 0; g < nGrp; ++g)
                buf[o * nGrp + g] = static_cast<unsigned long long>(mat[o][g]);
        MPI_Allreduce(MPI_IN_PLACE, buf.data(), static_cast<int>(flat), MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        for (size_t o = 0; o < nObs; ++o)
            for (size_t g = 0; g < nGrp; ++g)
                mat[o][g] = static_cast<size_t>(buf[o * nGrp + g]);
    };

    flattenD(snap.energy);
    flattenD(snap.energyWeightSq);
    flattenSz(snap.crossingCount);
    if (snap.polarizationEnabled) {
        flattenD(snap.stokesQ);
        flattenD(snap.stokesU);
        flattenD(snap.sumWQ2);
        flattenD(snap.sumWU2);
    }
#else
    (void)snap;
#endif
}

ObserverGroupQualityDiagnostics BuildObserverGroupQualityDiagnosticsFromSnapshot(
    SphericalObserver::ObserverGroupQualitySnapshot const& snap,
    Config const& cfg,
    AdaptiveGroupHistory& history,
    double sourceDt)
{
    ObserverGroupQualityDiagnostics diag;
    diag.enabled = cfg.adaptiveGroupQuality;
    if (!diag.enabled) return diag;

    size_t const nObs = snap.observerCount;
    size_t const nGrp = snap.groupCount;
    diag.observerCount = nObs;
    diag.groupCount = nGrp;
    diag.polarizationMode = snap.polarizationEnabled;

    auto make2d = [&](double val = 0.0) {
        return std::vector<std::vector<double>>(nObs, std::vector<double>(nGrp, val));
    };

    diag.luminosity = make2d();
    diag.neff = make2d();
    diag.polarizationDegree = make2d();
    diag.polarizationSnr = make2d();
    diag.latestPriority = make2d();
    diag.cumulativePriority = make2d();
    diag.predictedPriority = make2d(1.0);
    diag.deficit = make2d(1.0);
    diag.crossings.assign(nObs, std::vector<size_t>(nGrp, 0));

    double const eps = 1e-30;
    double invDt = (sourceDt > 0.0) ? 1.0 / sourceDt : 0.0;

    double maxLumGlobal = 0.0;
    std::vector<double> maxLumPerGroup(nGrp, 0.0);
    for (size_t o = 0; o < nObs; ++o) {
        for (size_t g = 0; g < nGrp; ++g) {
            double L = snap.energy[o][g] * invDt;
            diag.luminosity[o][g] = L;
            maxLumGlobal = std::max(maxLumGlobal, L);
            maxLumPerGroup[g] = std::max(maxLumPerGroup[g], L);
            diag.crossings[o][g] = snap.crossingCount[o][g];

            double E = snap.energy[o][g];
            double W2 = snap.energyWeightSq[o][g];
            diag.neff[o][g] = (W2 > eps) ? (E * E) / W2 : 0.0;
        }
    }

    if (snap.polarizationEnabled) {
        for (size_t o = 0; o < nObs; ++o) {
            for (size_t g = 0; g < nGrp; ++g) {
                double E = snap.energy[o][g];
                if (E <= eps) continue;
                double q = snap.stokesQ[o][g] / E;
                double u = snap.stokesU[o][g] / E;
                double p = std::sqrt(q * q + u * u);
                diag.polarizationDegree[o][g] = p;

                double varQ = snap.sumWQ2[o][g] / E - q * q;
                double varU = snap.sumWU2[o][g] / E - u * u;
                double neff = diag.neff[o][g];
                double sigQ = std::sqrt(std::max(varQ, 0.0) / std::max(neff, eps));
                double sigU = std::sqrt(std::max(varU, 0.0) / std::max(neff, eps));
                double sigP = std::sqrt(sigQ * sigQ + sigU * sigU);
                diag.polarizationSnr[o][g] = (sigP > eps) ? p / sigP : 0.0;
            }
        }
    }

    double const gw = cfg.adaptiveGroupLuminosityGlobalWeight;
    for (size_t o = 0; o < nObs; ++o) {
        for (size_t g = 0; g < nGrp; ++g) {
            double L = diag.luminosity[o][g];
            double normGlobal = (maxLumGlobal > 0.0) ? L / maxLumGlobal : 0.0;
            double normGroup = (maxLumPerGroup[g] > 0.0) ? L / maxLumPerGroup[g] : 0.0;
            double lumImportance = 0.0;
            if (cfg.adaptiveGroupLuminosityNormalization == "global")
                lumImportance = normGlobal;
            else if (cfg.adaptiveGroupLuminosityNormalization == "per-group")
                lumImportance = normGroup;
            else
                lumImportance = gw * normGlobal + (1.0 - gw) * normGroup;

            double lumComponent = cfg.adaptiveGroupLuminosityWeight
                * std::pow(lumImportance, cfg.adaptiveGroupLuminosityPower);

            double polComponent = 0.0;
            if (snap.polarizationEnabled) {
                double polNorm = std::max(diag.polarizationDegree[o][g], cfg.adaptiveGroupPolarizationFloor);
                polComponent = cfg.adaptiveGroupPolarizationWeight
                    * std::pow(polNorm, cfg.adaptiveGroupPolarizationPower);
            }

            double science = lumComponent + polComponent;

            double defNeff = (diag.neff[o][g] > eps)
                ? cfg.adaptiveGroupTargetNeff / diag.neff[o][g] : cfg.adaptiveGroupDeficitMax;
            double defPol = 1.0;
            if (snap.polarizationEnabled &&
                diag.polarizationDegree[o][g] >= cfg.adaptiveGroupPolarizationFloor &&
                snap.crossingCount[o][g] >= cfg.adaptiveGroupMinCrossings) {
                if (diag.polarizationSnr[o][g] > eps)
                    defPol = cfg.adaptiveGroupTargetPolSnr / diag.polarizationSnr[o][g];
                else
                    defPol = cfg.adaptiveGroupDeficitMax;
            }
            double deficitRaw = std::max(defNeff, defPol);
            deficitRaw = std::clamp(deficitRaw, 1.0, cfg.adaptiveGroupDeficitMax);
            diag.deficit[o][g] = deficitRaw;

            double priority = science * deficitRaw;

            bool const luminosityEligible =
                (cfg.adaptiveGroupMinLuminosity > 0.0 &&
                 L >= cfg.adaptiveGroupMinLuminosity) ||
                (cfg.adaptiveGroupMinLuminosityFracOfGroupMax > 0.0 &&
                 maxLumPerGroup[g] > 0.0 &&
                 L >= cfg.adaptiveGroupMinLuminosityFracOfGroupMax * maxLumPerGroup[g]);
            bool const retainedHistoryPriority =
                history.initialized &&
                o < history.emaPriority.size() &&
                g < history.emaPriority[o].size() &&
                history.emaPriority[o][g] >= cfg.adaptiveGroupRetainPriorityFloor;
            bool eligible = (snap.crossingCount[o][g] >= cfg.adaptiveGroupMinCrossings)
                || luminosityEligible
                || retainedHistoryPriority;
            if (!eligible)
                priority = std::min(priority, cfg.adaptiveGroupIneligiblePriorityCap);

            diag.latestPriority[o][g] = priority;
        }
    }

    if (!history.initialized) {
        history.initialized = true;
        history.observerCount = nObs;
        history.groupCount = nGrp;
        history.updateCount = 0;
        history.emaPriority = diag.latestPriority;
        history.emaDeficit = diag.deficit;
        history.cumulativeEnergy = std::vector<std::vector<double>>(nObs, std::vector<double>(nGrp, 0.0));
        history.cumulativeWeightSq = std::vector<std::vector<double>>(nObs, std::vector<double>(nGrp, 0.0));
        history.cumulativeStokesQ = std::vector<std::vector<double>>(nObs, std::vector<double>(nGrp, 0.0));
        history.cumulativeStokesU = std::vector<std::vector<double>>(nObs, std::vector<double>(nGrp, 0.0));
        history.cumulativeSumWQ2 = std::vector<std::vector<double>>(nObs, std::vector<double>(nGrp, 0.0));
        history.cumulativeSumWU2 = std::vector<std::vector<double>>(nObs, std::vector<double>(nGrp, 0.0));
        history.cumulativeCrossings.assign(nObs, std::vector<size_t>(nGrp, 0));
    }

    for (size_t o = 0; o < nObs; ++o) {
        for (size_t g = 0; g < nGrp; ++g) {
            history.cumulativeEnergy[o][g] += snap.energy[o][g];
            history.cumulativeWeightSq[o][g] += snap.energyWeightSq[o][g];
            if (snap.polarizationEnabled) {
                history.cumulativeStokesQ[o][g] += snap.stokesQ[o][g];
                history.cumulativeStokesU[o][g] += snap.stokesU[o][g];
                history.cumulativeSumWQ2[o][g] += snap.sumWQ2[o][g];
                history.cumulativeSumWU2[o][g] += snap.sumWU2[o][g];
            }
            history.cumulativeCrossings[o][g] += snap.crossingCount[o][g];
        }
    }

    double maxCumLumGlobal = 0.0;
    std::vector<double> maxCumLumPerGroup(nGrp, 0.0);
    for (size_t o = 0; o < nObs; ++o) {
        for (size_t g = 0; g < nGrp; ++g) {
            double const cumL = history.cumulativeEnergy[o][g];
            maxCumLumGlobal = std::max(maxCumLumGlobal, cumL);
            maxCumLumPerGroup[g] = std::max(maxCumLumPerGroup[g], cumL);
        }
    }

    for (size_t o = 0; o < nObs; ++o) {
        for (size_t g = 0; g < nGrp; ++g) {
            double cumE = history.cumulativeEnergy[o][g];
            double cumW2 = history.cumulativeWeightSq[o][g];
            double cumNeff = (cumW2 > eps) ? (cumE * cumE) / cumW2 : 0.0;
            double cumDefNeff = (cumNeff > eps) ? cfg.adaptiveGroupTargetNeff / cumNeff : cfg.adaptiveGroupDeficitMax;
            double cumDefPol = 1.0;
            double cumPolDegree = 0.0;
            double cumPolSnr = 0.0;
            if (snap.polarizationEnabled && cumE > eps) {
                double cq = history.cumulativeStokesQ[o][g] / cumE;
                double cu = history.cumulativeStokesU[o][g] / cumE;
                cumPolDegree = std::sqrt(cq * cq + cu * cu);
                double cvQ = history.cumulativeSumWQ2[o][g] / cumE - cq * cq;
                double cvU = history.cumulativeSumWU2[o][g] / cumE - cu * cu;
                double csigQ = std::sqrt(std::max(cvQ, 0.0) / std::max(cumNeff, eps));
                double csigU = std::sqrt(std::max(cvU, 0.0) / std::max(cumNeff, eps));
                double csigP = std::sqrt(csigQ * csigQ + csigU * csigU);
                cumPolSnr = (csigP > eps) ? cumPolDegree / csigP : 0.0;
                if (cumPolDegree >= cfg.adaptiveGroupPolarizationFloor &&
                    history.cumulativeCrossings[o][g] >= cfg.adaptiveGroupMinCrossings) {
                    cumDefPol = (cumPolSnr > eps)
                        ? cfg.adaptiveGroupTargetPolSnr / cumPolSnr
                        : cfg.adaptiveGroupDeficitMax;
                }
            }
            double cumDef = std::clamp(std::max(cumDefNeff, cumDefPol), 1.0, cfg.adaptiveGroupDeficitMax);
            double normGlobal = (maxCumLumGlobal > 0.0) ? cumE / maxCumLumGlobal : 0.0;
            double normGroup = (maxCumLumPerGroup[g] > 0.0) ? cumE / maxCumLumPerGroup[g] : 0.0;
            double cumLumImportance = 0.0;
            if (cfg.adaptiveGroupLuminosityNormalization == "global")
                cumLumImportance = normGlobal;
            else if (cfg.adaptiveGroupLuminosityNormalization == "per-group")
                cumLumImportance = normGroup;
            else
                cumLumImportance = gw * normGlobal + (1.0 - gw) * normGroup;
            double const cumLumComponent = cfg.adaptiveGroupLuminosityWeight
                * std::pow(cumLumImportance, cfg.adaptiveGroupLuminosityPower);
            double cumPolComponent = 0.0;
            if (snap.polarizationEnabled) {
                double const polNorm = std::max(cumPolDegree, cfg.adaptiveGroupPolarizationFloor);
                cumPolComponent = cfg.adaptiveGroupPolarizationWeight
                    * std::pow(polNorm, cfg.adaptiveGroupPolarizationPower);
            }
            diag.cumulativePriority[o][g] = (cumLumComponent + cumPolComponent) * cumDef;
        }
    }

    if (cfg.adaptiveGroupHistory && history.updateCount > 0) {
        double alpha = cfg.adaptiveGroupHistoryEma;
        for (size_t o = 0; o < nObs; ++o) {
            for (size_t g = 0; g < nGrp; ++g) {
                double combined = cfg.adaptiveGroupLatestWeight * diag.latestPriority[o][g]
                    + cfg.adaptiveGroupCumulativeWeight * diag.cumulativePriority[o][g]
                    + cfg.adaptiveGroupEmaWeight * history.emaPriority[o][g];
                history.emaPriority[o][g] = (1.0 - alpha) * history.emaPriority[o][g] + alpha * combined;
                history.emaDeficit[o][g] = (1.0 - alpha) * history.emaDeficit[o][g] + alpha * diag.deficit[o][g];
                diag.predictedPriority[o][g] = std::clamp(history.emaPriority[o][g], 1.0, cfg.adaptiveGroupDeficitMax);
            }
        }
    } else {
        for (size_t o = 0; o < nObs; ++o)
            for (size_t g = 0; g < nGrp; ++g)
                diag.predictedPriority[o][g] = diag.latestPriority[o][g];
    }
    ++history.updateCount;

    std::vector<double> allNeff, allPolSnr;
    for (size_t o = 0; o < nObs; ++o) {
        for (size_t g = 0; g < nGrp; ++g) {
            if (snap.crossingCount[o][g] > 0) {
                allNeff.push_back(diag.neff[o][g]);
                if (snap.polarizationEnabled)
                    allPolSnr.push_back(diag.polarizationSnr[o][g]);
                ++diag.activeBins;
                if (diag.predictedPriority[o][g] > 5.0)
                    ++diag.highPriorityBins;
            }
        }
    }
    auto percentile = [](std::vector<double>& v, double p) -> double {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        size_t idx = static_cast<size_t>(p * static_cast<double>(v.size() - 1));
        return v[std::min(idx, v.size() - 1)];
    };
    diag.neffP05 = percentile(allNeff, 0.05);
    diag.neffMedian = percentile(allNeff, 0.50);
    diag.neffP95 = percentile(allNeff, 0.95);
    diag.polSnrP05 = percentile(allPolSnr, 0.05);
    diag.polSnrMedian = percentile(allPolSnr, 0.50);
    diag.polSnrP95 = percentile(allPolSnr, 0.95);

    return diag;
}

double PredictedGroupPriority(
    ObserverGroupQualityDiagnostics const& groupQuality,
    size_t observerIndex,
    size_t groupIndex)
{
    if (observerIndex < groupQuality.predictedPriority.size() &&
        groupIndex < groupQuality.predictedPriority[observerIndex].size() &&
        std::isfinite(groupQuality.predictedPriority[observerIndex][groupIndex]) &&
        groupQuality.predictedPriority[observerIndex][groupIndex] > 0.0)
        return groupQuality.predictedPriority[observerIndex][groupIndex];
    return 1.0;
}

double PreMpiGroupStatScore(
    SphericalObserver::SourceCellGroupEscapeStat const& s,
    ObserverGroupQualityDiagnostics const& groupQuality)
{
    double const priority =
        PredictedGroupPriority(groupQuality, s.observerIndex, s.groupIndex);
    double const energy = (s.energy > 0.0 && std::isfinite(s.energy)) ? s.energy : 0.0;
    double const weightSq = (s.weightSq > 0.0 && std::isfinite(s.weightSq))
        ? s.weightSq
        : energy * energy;
    return priority * (energy + std::sqrt(std::max(weightSq, 0.0)));
}

std::vector<SphericalObserver::SourceCellGroupEscapeStat>
PruneLocalSourceGroupStats(
    std::vector<SphericalObserver::SourceCellGroupEscapeStat> const& localStats,
    ObserverGroupQualityDiagnostics const& groupQuality,
    Config const& cfg,
    AdaptiveGroupSourceUpdateSummary& summary)
{
    summary.localStatsInput = static_cast<unsigned long long>(localStats.size());
    std::vector<SphericalObserver::SourceCellGroupEscapeStat> pruned;
    pruned.reserve(std::min(localStats.size(), cfg.adaptiveGroupMaxLocalStats));

    for (auto const& s : localStats) {
        if (s.observerIndex >= groupQuality.observerCount ||
            s.groupIndex >= groupQuality.groupCount ||
            !(s.energy > 0.0) ||
            !std::isfinite(s.energy))
            continue;
        double const priority =
            PredictedGroupPriority(groupQuality, s.observerIndex, s.groupIndex);
        if (s.count >= cfg.adaptiveGroupStatMinCount ||
            priority >= cfg.adaptiveGroupStatPriorityKeep)
            pruned.push_back(s);
    }

    if (pruned.size() > cfg.adaptiveGroupMaxLocalStats) {
        if (cfg.adaptiveGroupFallbackToIntegratedOnOverflow) {
            summary.fallbackToIntegratedPath = true;
            summary.fallbackReason = "local_group_stats_overflow";
            summary.localStatsDropped =
                static_cast<unsigned long long>(localStats.size());
            return {};
        }

        std::sort(pruned.begin(), pruned.end(),
            [&](auto const& a, auto const& b) {
                return PreMpiGroupStatScore(a, groupQuality) >
                       PreMpiGroupStatScore(b, groupQuality);
            });
        pruned.resize(cfg.adaptiveGroupMaxLocalStats);
    }

    summary.localStatsAfterPrune = static_cast<unsigned long long>(pruned.size());
    summary.localStatsDropped =
        summary.localStatsInput > summary.localStatsAfterPrune
            ? summary.localStatsInput - summary.localStatsAfterPrune
            : 0ULL;
    return pruned;
}

void DecayAndPruneGroupSourceState(
    AdaptiveGroupSourceState& groupState,
    double decay,
    double pruneThreshold)
{
    for (auto it = groupState.scoreByCellGroup.begin();
         it != groupState.scoreByCellGroup.end(); ) {
        double sum = 0.0;
        double maxAbs = 0.0;
        for (auto& v : it->second) {
            v *= decay;
            if (!std::isfinite(v) || v < 0.0)
                v = 0.0;
            sum += v;
            maxAbs = std::max(maxAbs, std::abs(v));
        }
        if (maxAbs < pruneThreshold) {
            groupState.cellScoreFromGroups.erase(it->first);
            it = groupState.scoreByCellGroup.erase(it);
        } else {
            groupState.cellScoreFromGroups[it->first] = sum;
            ++it;
        }
    }

    for (auto it = groupState.cellScoreFromGroups.begin();
         it != groupState.cellScoreFromGroups.end(); ) {
        if (!(it->second > pruneThreshold) || !std::isfinite(it->second))
            it = groupState.cellScoreFromGroups.erase(it);
        else
            ++it;
    }
}

AdaptiveGroupSourceUpdateSummary UpdateAdaptiveSourceGroupScores(
    std::vector<SphericalObserver::SourceCellGroupEscapeStat> const& localGroupStats,
    ObserverGroupQualityDiagnostics const& groupQuality,
    Config const& cfg,
    AdaptiveGroupSourceState& groupState,
    int rank,
    [[maybe_unused]] int mpiSize)
{
    AdaptiveGroupSourceUpdateSummary summary;
    if (!cfg.adaptiveGroupSourceCells || !groupQuality.enabled)
        return summary;

    size_t const nObs = groupQuality.observerCount;
    size_t const nGrp = groupQuality.groupCount;

    std::vector<std::vector<double>> totalEnergyByOG(nObs, std::vector<double>(nGrp, 0.0));
    std::vector<std::vector<double>> totalW2ByOG(nObs, std::vector<double>(nGrp, 0.0));

    for (auto const& s : localGroupStats) {
        if (s.observerIndex < nObs && s.groupIndex < nGrp) {
            totalEnergyByOG[s.observerIndex][s.groupIndex] += s.energy;
            totalW2ByOG[s.observerIndex][s.groupIndex] +=
                (s.weightSq > 0.0 && std::isfinite(s.weightSq))
                    ? s.weightSq
                    : s.energy * s.energy;
        }
    }

#ifdef RICH_MPI
    size_t flat = nObs * nGrp;
    std::vector<double> flatE(flat, 0.0), flatW2(flat, 0.0);
    for (size_t o = 0; o < nObs; ++o)
        for (size_t g = 0; g < nGrp; ++g) {
            flatE[o * nGrp + g] = totalEnergyByOG[o][g];
            flatW2[o * nGrp + g] = totalW2ByOG[o][g];
        }
    MPI_Allreduce(MPI_IN_PLACE, flatE.data(), static_cast<int>(flat), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, flatW2.data(), static_cast<int>(flat), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (size_t o = 0; o < nObs; ++o)
        for (size_t g = 0; g < nGrp; ++g) {
            totalEnergyByOG[o][g] = flatE[o * nGrp + g];
            totalW2ByOG[o][g] = flatW2[o * nGrp + g];
        }
#endif

    double const eps = 1e-30;
    bool const polMode = groupQuality.polarizationMode;
    double const varianceMix = polMode ? 0.85 : 0.50;
    double const ema = cfg.adaptiveGroupScoreEma;
    double const decay = 1.0 - ema;

    auto prunedLocalStats = PruneLocalSourceGroupStats(
        localGroupStats, groupQuality, cfg, summary);
#ifdef RICH_MPI
    int fallbackLocal = summary.fallbackToIntegratedPath ? 1 : 0;
    int fallbackGlobal = fallbackLocal;
    MPI_Allreduce(&fallbackLocal, &fallbackGlobal, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    if (fallbackGlobal != 0) {
        summary.fallbackToIntegratedPath = true;
        summary.fallbackReason = "local_group_stats_overflow";
    }
#endif
    if (summary.fallbackToIntegratedPath) {
        groupState.scoreByCellGroup.clear();
        groupState.cellScoreFromGroups.clear();
        return summary;
    }

    DecayAndPruneGroupSourceState(groupState, decay, 1e-20);

    std::vector<PackedSourceGroupEscapeStat> received =
        ExchangeSourceGroupStatsByCellOwner(prunedLocalStats, summary);
    std::vector<SphericalObserver::SourceCellGroupEscapeStat>().swap(prunedLocalStats);

    std::unordered_map<AdaptiveSourceGroupKey,
                       SphericalObserver::SourceCellGroupEscapeStat,
                       AdaptiveSourceGroupKeyHash> byGroupCell;
    byGroupCell.reserve(received.size());
    for (auto const& p : received) {
        if (p.count == 0 || !(p.energy > 0.0) || !std::isfinite(p.energy))
            continue;
        size_t const observerIndex = static_cast<size_t>(p.observerIndex);
        size_t const groupIndex = static_cast<size_t>(p.groupIndex);
        size_t const cellID = static_cast<size_t>(p.cellID);
        AdaptiveSourceGroupKey const key{observerIndex, groupIndex, cellID};
        auto& s = byGroupCell[key];
        s.cellID = cellID;
        s.observerIndex = observerIndex;
        s.groupIndex = groupIndex;
        s.energy += p.energy;
        s.weightSq += p.weightSq;
        s.maxWeight = std::max(s.maxWeight, p.maxWeight);
        s.count += static_cast<size_t>(p.count);
    }
    std::vector<PackedSourceGroupEscapeStat>().swap(received);

    std::unordered_map<size_t, std::vector<double>> deltaByCellGroup;
    for (auto const& kv : byGroupCell) {
        auto const& s = kv.second;
        if (s.observerIndex >= nObs || s.groupIndex >= nGrp)
            continue;
        if (!(s.energy > 0.0) || s.count < cfg.adaptiveGroupStatMinCount)
            continue;

        double const totE = totalEnergyByOG[s.observerIndex][s.groupIndex];
        double const totW2 = totalW2ByOG[s.observerIndex][s.groupIndex];
        double const eFrac = (totE > eps) ? s.energy / totE : 0.0;
        double const statW2 = (s.weightSq > 0.0 && std::isfinite(s.weightSq))
            ? s.weightSq
            : s.energy * s.energy;
        double const w2Frac = (totW2 > eps) ? statW2 / totW2 : eFrac;
        double const sourceQuality = (1.0 - varianceMix) * eFrac + varianceMix * w2Frac;

        double const priority =
            PredictedGroupPriority(groupQuality, s.observerIndex, s.groupIndex);

        double delta = ema * priority * sourceQuality;
        if (!(delta > 0.0) || !std::isfinite(delta))
            continue;

        auto& gvec = deltaByCellGroup[s.cellID];
        if (gvec.empty()) gvec.assign(nGrp, 0.0);
        gvec[s.groupIndex] += delta;
        ++summary.passedStats;
    }
    std::unordered_map<AdaptiveSourceGroupKey,
                       SphericalObserver::SourceCellGroupEscapeStat,
                       AdaptiveSourceGroupKeyHash>().swap(byGroupCell);

    std::vector<PackedAdaptiveCellGroupScoreDelta> localDeltas;
    for (auto const& kv : deltaByCellGroup) {
        size_t const cellID = kv.first;
        for (size_t g = 0; g < kv.second.size(); ++g) {
            double const delta = kv.second[g];
            if (!(delta > 0.0) || !std::isfinite(delta))
                continue;
            PackedAdaptiveCellGroupScoreDelta p;
            p.cellID = static_cast<unsigned long long>(cellID);
            p.groupIndex = static_cast<unsigned long long>(g);
            p.delta = delta;
            localDeltas.push_back(p);
        }
    }
    std::unordered_map<size_t, std::vector<double>>().swap(deltaByCellGroup);

    size_t const localDeltaCells = localDeltas.size();
    std::vector<PackedAdaptiveCellGroupScoreDelta> allDeltas =
        AllgatherAdaptiveCellGroupScoreDeltas(localDeltas);
    std::vector<PackedAdaptiveCellGroupScoreDelta>().swap(localDeltas);

    for (auto const& p : allDeltas) {
        if (!(p.delta > 0.0) || !std::isfinite(p.delta))
            continue;
        size_t const cellID = static_cast<size_t>(p.cellID);
        size_t const groupIndex = static_cast<size_t>(p.groupIndex);
        if (groupIndex >= nGrp)
            continue;
        auto& gvec = groupState.scoreByCellGroup[cellID];
        if (gvec.empty()) gvec.assign(nGrp, 0.0);
        gvec[groupIndex] += p.delta;
        groupState.cellScoreFromGroups[cellID] += p.delta;
    }
    summary.scoreDeltaCells = allDeltas.size();
    std::vector<PackedAdaptiveCellGroupScoreDelta>().swap(allDeltas);

    DecayAndPruneGroupSourceState(groupState, 1.0, 1e-20);
    summary.scoreMapCells = groupState.scoreByCellGroup.size();

    unsigned long long maxLocalDeltaCells =
        static_cast<unsigned long long>(localDeltaCells);
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &maxLocalDeltaCells, 1, MPI_UNSIGNED_LONG_LONG,
                  MPI_MAX, MPI_COMM_WORLD);
#endif
    summary.maxPackedBytes = std::max(
        summary.maxPackedBytes,
        (maxLocalDeltaCells + static_cast<unsigned long long>(summary.scoreDeltaCells)) *
            static_cast<unsigned long long>(sizeof(PackedAdaptiveCellGroupScoreDelta)));

    if (rank == 0 && cfg.adaptiveDiagnosticsVerbose) {
        std::cout << "GROUP_ADAPT source_group_state: cells_with_group_scores="
                  << groupState.scoreByCellGroup.size()
                  << " cells_with_cell_score=" << groupState.cellScoreFromGroups.size()
                  << " dropped_stats=" << summary.localStatsDropped
                  << " mpi_bytes_max=" << summary.maxPackedBytes
                  << std::endl;
    }
    return summary;
}

void PrintAdaptiveGroupGenerationStats(
    ObserverGroupQualityDiagnostics const& gq,
    ::RadiationIMC::GroupSamplingDiagnostics const& gsd,
    size_t gen,
    int rank)
{
    if (rank != 0 || !gq.enabled) return;
    std::cout << "GROUP_ADAPT gen=" << (gen + 1)
              << " groups=" << gq.groupCount
              << " active_bins=" << gq.activeBins
              << " high_priority_bins=" << gq.highPriorityBins
              << " neff_p05/med/p95=" << gq.neffP05 << "/" << gq.neffMedian << "/" << gq.neffP95;
    if (gq.polarizationMode)
        std::cout << " pol_snr_p05/med/p95=" << gq.polSnrP05 << "/" << gq.polSnrMedian << "/" << gq.polSnrP95;
    if (gsd.totalSampled > 0 || gsd.weightCorrectionFallback > 0 || gsd.invalidPdfFallback > 0) {
        double avgCorr = (gsd.weightCorrectionCount > 0) ? gsd.weightCorrectionSum / gsd.weightCorrectionCount : 1.0;
        double cappedFrac = (gsd.weightCorrectionCount > 0)
            ? static_cast<double>(gsd.weightCorrectionCapped) / static_cast<double>(gsd.weightCorrectionCount)
            : 0.0;
        std::cout << " weight_corr_min/avg/max=" << gsd.weightCorrectionMin << "/" << avgCorr << "/" << gsd.weightCorrectionMax
                  << " capped_frac=" << cappedFrac
                  << " capped_energy_frac=" << gsd.cappedEnergyFraction
                  << " fallback=" << gsd.weightCorrectionFallback
                  << " invalid_pdf_cells=" << gsd.invalidPdfFallback
                  << " biased=" << (gsd.estimatorPotentiallyBiased ? 1 : 0);
    }
    std::cout << std::endl;
}

std::unordered_map<size_t, std::array<double, ENERGY_GROUPS_NUM>>
BuildGroupScoresForIMC(
    AdaptiveGroupSourceState const& groupState,
    std::vector<ComputationalCell3D> const& localCells,
    size_t nGroups)
{
    std::unordered_set<size_t> localCellIDs;
    localCellIDs.reserve(localCells.size());
    for (auto const& cell : localCells)
        localCellIDs.insert(cell.ID);

    std::unordered_map<size_t, std::array<double, ENERGY_GROUPS_NUM>> result;
    for (auto const& kv : groupState.scoreByCellGroup) {
        if (localCellIDs.find(kv.first) == localCellIDs.end())
            continue;
        std::array<double, ENERGY_GROUPS_NUM> arr{};
        size_t const copyLen = std::min(kv.second.size(), static_cast<size_t>(ENERGY_GROUPS_NUM));
        for (size_t g = 0; g < copyLen; ++g)
            arr[g] = kv.second[g];
        (void)nGroups;
        result[kv.first] = arr;
    }
    return result;
}

std::unordered_map<size_t, double>
BuildCombinedSourceScoresForIMC(
    AdaptiveSourceState const& integratedState,
    AdaptiveGroupSourceState const& groupState)
{
    std::unordered_map<size_t, double> combined = integratedState.scoreByCellID;
    for (auto const& kv : groupState.cellScoreFromGroups) {
        if (kv.second > 0.0 && std::isfinite(kv.second))
            combined[kv.first] += kv.second;
    }
    return combined;
}

// Rosseland weight fraction for a single group with dimensionless boundaries [a, b].
// Uses: integral_a^b x^4 e^x/(e^x-1)^2 dx = a^4/(e^a-1) - b^4/(e^b-1) + 4*(pi^4/15)*planck_integral(a,b)
// Normalized by the full-spectrum integral 4*pi^4/15.
double RosselandWeightFraction(double a, double b)
{
    double const fullIntegral = 4.0 * std::pow(M_PI, 4) / 15.0;
    double boundaryTerm = 0.0;
    if (a > 0.0 && a < 500.0)
        boundaryTerm += std::pow(a, 4) / std::expm1(a);
    if (b > 0.0 && b < 500.0)
        boundaryTerm -= std::pow(b, 4) / std::expm1(b);
    double planckTerm = 4.0 * (std::pow(M_PI, 4) / 15.0) * ::planck_integral::planck_integral(a, b);
    return (boundaryTerm + planckTerm) / fullIntegral;
}

// Solve for alpha such that:
//   sum_g [ f_g / (alpha * sigmaA[g] + sigmaS[g]) ] = 1/kappaRGrey
// F(alpha) is monotonically decreasing, so bisection converges.
double SolveRosselandAlpha(
    std::vector<double> const& sigmaA,
    std::vector<double> const& sigmaS,
    std::vector<double> const& fRoss,
    double kappaRGrey)
{
    double const target = 1.0 / kappaRGrey;

    auto evalF = [&](double alpha) {
        double sum = 0.0;
        for (size_t g = 0; g < sigmaA.size(); ++g) {
            double denom = alpha * sigmaA[g] + sigmaS[g];
            if (denom > 0.0)
                sum += fRoss[g] / denom;
        }
        return sum - target;
    };

    // F is decreasing: F(0) = sum(f_g/sigmaS_g) - target (large positive if scattering << grey)
    //                   F(inf) → -target (negative)
    // Find bracket
    double lo = 0.0, hi = 1.0;
    while (evalF(hi) > 0.0 && hi < 1e10)
        hi *= 2.0;

    if (evalF(lo) <= 0.0)
        return lo;
    if (evalF(hi) >= 0.0)
        return hi;

    for (int iter = 0; iter < 100; ++iter) {
        double mid = 0.5 * (lo + hi);
        if (evalF(mid) > 0.0)
            lo = mid;
        else
            hi = mid;
        if ((hi - lo) < 1e-12 * (lo + hi + 1e-300))
            break;
    }
    return 0.5 * (lo + hi);
}

// Planck weight fraction for a single group with dimensionless boundaries [a, b].
// Returns the fraction of the Planck spectrum integral_a^b x^3/(e^x-1) dx
// normalised by the full-spectrum integral pi^4/15.
double PlanckWeightFraction(double a, double b)
{
    return ::planck_integral::planck_integral(a, b);
}

// Solve for alpha such that the Planck-weighted MG absorption matches the grey Planck:
//   sum_g [ f_planck_g * alpha * sigmaA[g] ] = kappaPGrey
// This is a simple ratio (no iteration needed).
double SolvePlanckAlpha(
    std::vector<double> const& sigmaA,
    std::vector<double> const& fPlanck,
    double kappaPGrey)
{
    double mgPlanck = 0.0;
    for (size_t g = 0; g < sigmaA.size(); ++g)
        mgPlanck += fPlanck[g] * sigmaA[g];
    if (mgPlanck <= 0.0)
        return 1.0;
    return kappaPGrey / mgPlanck;
}

template <class GreyOpacityT>
void RecomputeOpacityScaleFactors(
    STAMGopacityMC& opacity,
    GreyOpacityT const& greyOpacity,
    std::vector<ComputationalCell3D> const& cells,
    size_t const Ncells,
    int const rank,
    OpacityScaleMode mode,
    std::string const& label)
{
  // The scale-factor map is rank-local and keyed by cell.ID.  Clear it first so
  // CalcAbsorptionOpacity below samples the unscaled MG opacity, then rebuild it
  // for the cells currently owned by this rank.
  opacity.SetRosselandScaleFactors(std::unordered_map<size_t, double>());

  size_t const Ng = opacity.energy_groups_boundary.size() - 1;
  std::unordered_map<size_t, double> scaleFactors;
  scaleFactors.reserve(Ncells);

  double alphaMin = std::numeric_limits<double>::max();
  double alphaMax = 0.0;
  double alphaSum = 0.0;
  size_t alphaCount = 0;
  size_t alphaOutliers = 0;

  bool const usePlanck = (mode == OpacityScaleMode::Planck);

  for (size_t i = 0; i < Ncells; ++i) {
    double const kT = CG::boltzmann_constant * cells[i].temperature;
    if (kT <= 0.0 || !std::isfinite(kT)) continue;

    std::vector<double> fWeight(Ng);
    double fTotal = 0.0;
    for (size_t g = 0; g < Ng; ++g) {
      double a = opacity.energy_groups_boundary[g] / kT;
      double b = opacity.energy_groups_boundary[g + 1] / kT;
      if (a >= b || a > 500.0) {
        fWeight[g] = 0.0;
        continue;
      }
      b = std::min(b, 500.0);
      fWeight[g] = usePlanck ? PlanckWeightFraction(a, b)
                             : RosselandWeightFraction(a, b);
      fTotal += fWeight[g];
    }
    if (fTotal <= 0.0) continue;
    for (double& f : fWeight) f /= fTotal;

    std::vector<double> sigA(Ng);
    for (size_t g = 0; g < Ng; ++g)
      sigA[g] = opacity.CalcAbsorptionOpacity(cells[i], opacity.energy_groups_center[g]);

    double alpha;
    if (usePlanck) {
      double const kappaPGrey = greyOpacity.CalcPlanckOpacity(cells[i]);
      if (kappaPGrey <= 0.0 || !std::isfinite(kappaPGrey)) continue;
      alpha = 30 * SolvePlanckAlpha(sigA, fWeight, kappaPGrey); // 2 is ad hoc factor to match the gray luminosity
    } else {
      std::vector<double> sigS(Ng);
      for (size_t g = 0; g < Ng; ++g)
        sigS[g] = opacity.CalcScatteringOpacity(cells[i], opacity.energy_groups_center[g]);
      double const D_grey = greyOpacity.CalcDiffusionCoefficient(cells[i]);
      if (D_grey <= 0.0 || !std::isfinite(D_grey)) continue;
      double const kappaRGrey = CG::speed_of_light / (3.0 * D_grey);
      alpha = SolveRosselandAlpha(sigA, sigS, fWeight, kappaRGrey);
    }

    scaleFactors[cells[i].ID] = alpha;

    alphaMin = std::min(alphaMin, alpha);
    alphaMax = std::max(alphaMax, alpha);
    alphaSum += alpha;
    ++alphaCount;
    if (alpha < 0.5 || alpha > 2.0) ++alphaOutliers;
  }

  opacity.SetRosselandScaleFactors(std::move(scaleFactors));

  double globalMin = 0.0, globalMax = 0.0, globalSum = 0.0;
  size_t globalCount = 0, globalOutliers = 0;
#ifdef RICH_MPI
  MPI_Reduce(&alphaMin, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&alphaMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&alphaSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&alphaCount, &globalCount, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&alphaOutliers, &globalOutliers, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
#else
  globalMin = alphaMin;
  globalMax = alphaMax;
  globalSum = alphaSum;
  globalCount = alphaCount;
  globalOutliers = alphaOutliers;
#endif

  if (rank == 0) {
    char const* modeStr = usePlanck ? "Planck" : "Rosseland";
    double const alphaMean = (globalCount > 0) ? globalSum / static_cast<double>(globalCount) : 1.0;
    std::cout << modeStr << " scale " << label << ": alpha min=" << globalMin
              << " max=" << globalMax << " mean=" << alphaMean
              << " outliers(>2x)=" << globalOutliers << "/" << globalCount
              << std::endl;
  }
}

bool MeasuredLBDebugMemory()
{
    static int cached = -1;
    if (cached < 0) {
        char const* val = std::getenv("RICH_MEASURED_LB_DEBUG_MEMORY");
        cached = (val != nullptr && std::string(val) != "0" && std::string(val) != "false") ? 1 : 0;
    }
    return cached != 0;
}

void PrintVmRSS(std::string const& label, int rank) {
    if (!MeasuredLBDebugMemory())
        return;
#ifdef __linux__
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::cerr << "MEMORY_RSS rank=" << rank
                      << " label=" << label
                      << " " << line << "\n";
            break;
        }
    }
#endif
}

} // anonymous namespace

int main(int argc, char* argv[])
{
#ifdef RICH_MPI
    MPI_Init(&argc, &argv);
    int rank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
#else
    int rank = 0, mpiSize = 1;
#endif

    try {
        Config cfg;
        if (!parseArgs(argc, argv, cfg, rank)) {
            printUsage(rank);
#ifdef RICH_MPI
            MPI_Finalize();
#endif
            return 1;
        }

        if (rank == 0) {
            std::cout << "=== TDE IMC Post-Processing ===\n"
                      << "Input:           " << cfg.inputPath << "\n"
                      << "Output:          " << cfg.outputPath << "\n"
                      << "Opacity dir:     " << cfg.opacityDir << "\n"
                      << "Grey opacity:    " << cfg.greyOpacityDir << "\n"
                      << "EOS dir:         " << cfg.eosDir << "\n"
                      << "Radius:          " << cfg.radius << " cm\n"
                      << "Observers:       " << cfg.nObservers << "\n"
                      << "Source dt:       " << cfg.sourceDt << " s\n"
                      << "Transport time:  " << cfg.transportTime << " s\n"
                      << "Photons/cell:    " << cfg.photonsPerCell << "\n"
                      << "Center:          (" << cfg.center.x << ", " << cfg.center.y << ", " << cfg.center.z << ")\n"
                      << "Compton:         " << (cfg.compton ? "yes" : "no") << "\n"
                      << "DDMC:            " << (cfg.ddmc ? "yes" : "no") << "\n"
                      << "Cell velocities: " << (cfg.useCellVelocities ? "yes" : "no") << "\n"
                      << "Polarization:    " << (cfg.polarization ? "yes" : "no") << "\n"
                      << "Measured LB:     " << (cfg.measuredLoadBalance ? "requested" : "disabled") << "\n"
                      << "  weight compression: " << EffectiveMeasuredLBWeightCompression(cfg) << "\n"
                      << "  max cell imbalance: " << MEASURED_LB_MAX_CELL_IMBALANCE << "\n"
                      << "  adaptive cadence: learned-only probe LB, then learned-final step 5 and 10 LB only\n"
                      << "Opacity scale:   " << (cfg.opacityScaleMode == OpacityScaleMode::Planck ? "planck" :
                                                  cfg.opacityScaleMode == OpacityScaleMode::Rosseland ? "rosseland" : "disabled") << "\n"
                      << "Adaptive source: " << (cfg.adaptiveSourceCells ? "enabled" : "disabled") << "\n"
                      << "  MG schedule:   1 exact-1 burn-in, 14 exact-3 burn-in, learned-only exact-75 probe, LB, "
                      << (10 * cfg.nGenerations + 20) << " learned-only final steps (min=500 max=2000)\n"
                      << "  final LB cadence: learned-final steps 5 and 10 only\n"
                      << "  min esc frac:  " << cfg.adaptiveSourceMinEscapedFrac << "\n"
                      << "  strength:      " << cfg.adaptiveSourceStrength << "\n"
                      << "  EMA:           " << cfg.adaptiveSourceEma << "\n"
                      << "  max factor:    " << cfg.adaptiveSourceMaxFactor << "\n"
                      << "  learned reserve frac:      " << cfg.adaptiveSourceLearnedReserveFrac << "\n"
                      << "  learned min factor:        " << cfg.adaptiveSourceLearnedMinFactor << "\n"
                      << "  observer equity:           " << ((cfg.adaptiveSourceCells && cfg.adaptiveObserverEquity) ? "enabled" : "disabled") << "\n"
                      << "  observer target neff:      " << cfg.adaptiveObserverTargetNeff << "\n"
                      << "  observer target pol SNR:   " << cfg.adaptiveObserverTargetPolSnr << "\n"
                      << "  observer deficit max/EMA:  " << cfg.adaptiveObserverDeficitMax << "/" << cfg.adaptiveObserverDeficitEma << "\n"
                      << "  observer extra budget max: " << cfg.adaptiveObserverExtraBudgetFrac << "\n"
                      << "  burnin/adapt LB: " << ((cfg.adaptiveSourceCells && cfg.measuredLoadBalance) ? "requested" : "disabled") << "\n"
                      << "Requested generations: " << cfg.nGenerations << "\n"
                      << "MPI ranks:       " << mpiSize << "\n"
                      << std::endl;
        }

        // ============================================================
        // Code-unit scale factors (for snapshot → CGS conversion)
        // ============================================================
        double const lscale = 7e10;   // cm
        double const mscale = 2e33;   // g
        double const tscale = 1603;   // s

        double const rho_factor = mscale / (lscale * lscale * lscale);
        double const vel_factor = lscale / tscale;
        double const energy_factor = lscale * lscale / (tscale * tscale);

        // ============================================================
        // Load EOS (identity scales: inputs will already be CGS)
        // ============================================================
        auto eos = std::make_shared<OndrejEOS>(
            cfg.eosDir + "density.txt",
            cfg.eosDir + "Pfile.txt",
            cfg.eosDir + "csfile.txt",
            cfg.eosDir + "Sfile.txt",
            cfg.eosDir + "Ufile.txt",
            cfg.eosDir + "Tfile.txt",
            cfg.eosDir + "CVfile.txt",
            1.0, 1.0, 1.0);

        if (rank == 0)
            std::cout << "EOS loaded (CGS mode, lscale=" << lscale << " mscale=" << mscale << " tscale=" << tscale << ")." << std::endl;

        // ============================================================
        // Load STA multigroup opacity
        // ============================================================
        auto opacity = std::make_shared<STAMGopacityMC>(cfg.opacityDir);

#if ENERGY_GROUPS_NUM > 1
        if (opacity->energy_groups_boundary.size() != ENERGY_GROUPS_NUM + 1)
        {
            UniversalError eo("STA opacity table group count does not match ENERGY_GROUPS_NUM");
            eo.addEntry("Table boundaries", opacity->energy_groups_boundary.size());
            eo.addEntry("Expected boundaries", ENERGY_GROUPS_NUM + 1);
            throw eo;
        }
        for (size_t g = 0; g <= ENERGY_GROUPS_NUM; ++g)
            ComputationalCell3D::energyBoundaries[g] = opacity->energy_groups_boundary[g];
#endif

        if (rank == 0)
        {
            std::cout << "Opacity loaded: " << opacity->energy_groups_center.size() << " groups." << std::endl;
            std::cout << "ENERGY_GROUPS_NUM=" << ENERGY_GROUPS_NUM << "  energyBoundaries:";
            for (size_t g = 0; g <= ENERGY_GROUPS_NUM; ++g)
                std::cout << " " << ComputationalCell3D::energyBoundaries[g];
            std::cout << std::endl;
        }

        // ============================================================
        // Read snapshot (MPI-written)
        // ============================================================
        if (rank == 0)
            std::cout << "Reading snapshot..." << std::endl;

        Snapshot3D snapshot;
#ifdef RICH_MPI
        snapshot = ReadSnapshot3DParallel(cfg.inputPath);
        int const fileRanks = GetNumberOfRanksInHDF(cfg.inputPath);
        if (mpiSize < fileRanks && rank == 0) {
            for (int fileRank = mpiSize; fileRank < fileRanks; ++fileRank) {
                Snapshot3D extra = ReadSnapshot3DParallel(cfg.inputPath, fileRank);
                snapshot.cells.insert(snapshot.cells.end(),
                                      extra.cells.begin(), extra.cells.end());
                snapshot.mesh_points.insert(snapshot.mesh_points.end(),
                                            extra.mesh_points.begin(),
                                            extra.mesh_points.end());
            }
        }
#else
        snapshot = ReadSnapshot3D(cfg.inputPath);
#endif

        if (snapshot.mesh_points.empty()) {
            if (rank == 0) std::cerr << "Empty snapshot\n";
#ifdef RICH_MPI
            MPI_Finalize();
#endif
            return 1;
        }

        if (rank == 0)
            std::cout << "Snapshot read: " << snapshot.mesh_points.size() << " points, time=" << snapshot.time << std::endl;

        // ============================================================
        // Convert snapshot from code units to CGS
        // ============================================================
        snapshot.ll = snapshot.ll * lscale;
        snapshot.ur = snapshot.ur * lscale;
        snapshot.time *= tscale;

        for (auto& pt : snapshot.mesh_points)
            pt = pt * lscale;

        for (auto& c : snapshot.cells) {
            c.density *= rho_factor;
            c.pressure *= mscale / (tscale * tscale * lscale);
            c.internal_energy *= energy_factor;
            c.velocity = c.velocity * vel_factor;
            c.Erad *= energy_factor;
            for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
                c.Eg[g] *= energy_factor;
        }

        if (rank == 0)
            std::cout << "Converted snapshot to CGS." << std::endl;

        // ============================================================
        // Rebuild tessellation (two-pass for weighted load balancing)
        // ============================================================
#ifdef RICH_MPI
        ComputationalCell3D dummyCell;
        std::vector<double> lbWeights;
        std::vector<Vector3D> localPoints;
        {
            Voronoi3D tempTess(snapshot.ll, snapshot.ur);
            tempTess.BuildParallel(snapshot.mesh_points);
            MPI_exchange_data(tempTess, snapshot.cells, false, 1, &dummyCell);

            size_t N = tempTess.GetPointNo();

            std::vector<double> tauScatVec(N, 0.0);
            std::vector<double> tauPlanckLocal(N, 0.0);
            for (size_t i = 0; i < N; ++i)
            {
                double charLen = std::cbrt(tempTess.GetVolume(i));
                tauPlanckLocal[i] = opacity->CalcPlanckOpacity(snapshot.cells[i]) * charLen;
                tauScatVec[i] = opacity->CalcScatteringOpacity(snapshot.cells[i]) * charLen;
            }

            MPI_exchange_data(tempTess, tauScatVec, true);
            size_t Ntot = tauScatVec.size();

            lbWeights.resize(N);
            localPoints.resize(N);
            for (size_t i = 0; i < N; ++i)
            {
                localPoints[i] = tempTess.GetMeshPoint(i);

                std::vector<size_t> neighbors = tempTess.GetNeighbors(i);
                double avgTauScat = 0.0;
                size_t count = 0;
                avgTauScat += tauScatVec[i];
                ++count;

                for (size_t nb : neighbors)
                {
                    if (nb < N || !tempTess.IsPointOutsideBox(nb)) 
                    {
                        avgTauScat += 0.5 * (tauScatVec[nb] + tauScatVec[i]);
                        ++count;
                    }
                }
                avgTauScat /= static_cast<double>(count);

                double const tauPlanck = tauPlanckLocal[i];
                double maxweight = 3;
                if (avgTauScat > 5.0)
                {
                    maxweight = std::max(maxweight, 0.05 * avgTauScat);
                    lbWeights[i] = 1.0 + (tauPlanck < tauScatVec[i] * 0.1 ? std::min(maxweight, avgTauScat) : 0.0);
                }
            }
        }
        snapshot.mesh_points.clear();
        snapshot.mesh_points.shrink_to_fit();

        Voronoi3D tess(snapshot.ll, snapshot.ur);
        tess.BuildParallel(localPoints, lbWeights);

        localPoints.clear();
        localPoints.shrink_to_fit();
        lbWeights.clear();
        lbWeights.shrink_to_fit();

        MPI_exchange_data(tess, snapshot.cells, false, 1, &dummyCell);
#else
        Voronoi3D tess(snapshot.ll, snapshot.ur);
        tess.Build(snapshot.mesh_points);
#endif

        size_t Ncells = tess.GetPointNo();
        std::vector<ComputationalCell3D> &cells = snapshot.cells;
        if (cells.size() < Ncells)
            throw UniversalError("Snapshot cell count is smaller than tessellation cell count");

        if (rank == 0)
            std::cout << "Tessellation built: " << Ncells << " local cells." << std::endl;

        // Validate cells
        size_t badCells = 0;
        for (size_t i = 0; i < Ncells; ++i) {
            if (cells[i].density <= 0.0 || !std::isfinite(cells[i].density) ||
                cells[i].temperature <= 0.0 || !std::isfinite(cells[i].temperature)) {
                ++badCells;
            }
        }
        if (badCells > 0) {
            UniversalError eo("Invalid density or temperature in post-process snapshot");
            eo.addEntry("Invalid local cells", badCells);
            throw eo;
        }

        // ============================================================
        // Compute grey FLD luminosity per observer patch
        // ============================================================
        auto greyOpacity = std::make_shared<STAgreyOpacity>(cfg.greyOpacityDir);
        if (rank == 0)
            std::cout << "Grey opacity loaded for FLD luminosity." << std::endl;

        // ============================================================
        // Scale MG absorption to match grey mean
        // ============================================================
#if ENERGY_GROUPS_NUM > 1
        if (cfg.opacityScaleMode != OpacityScaleMode::None) {
            RecomputeOpacityScaleFactors(
                *opacity, *greyOpacity, cells, Ncells, rank, cfg.opacityScaleMode, "initial");
        }
#endif

        // Per-cell radiation energy density, diffusion coefficient, and FLD flux vector
        std::vector<double> Er_vol(Ncells);
        std::vector<double> D_cell(Ncells);
        std::vector<Vector3D> fldFlux(Ncells);

        for (size_t i = 0; i < Ncells; ++i) {
            Er_vol[i] = cells[i].density * cells[i].Erad;
            D_cell[i] = greyOpacity->CalcDiffusionCoefficient(cells[i]);
        }

#ifdef RICH_MPI
        MPI_exchange_data(tess, Er_vol, true);
#endif

        // Green-Gauss gradient of E_r
        std::vector<Vector3D> gradEr(Ncells);
        {
            std::vector<size_t> neighbors;
            for (size_t i = 0; i < Ncells; ++i) {
                auto const& faces = tess.GetCellFaces(i);
                tess.GetNeighbors(i, neighbors);
                Vector3D grad(0, 0, 0);
                for (size_t j = 0; j < neighbors.size(); ++j) {
                    size_t nb = neighbors[j];
                    if (tess.IsPointOutsideBox(nb))
                        continue;
                    double Er_face = 0.5 * (Er_vol[i] + Er_vol[nb]);
                    Vector3D r_ij = tess.GetMeshPoint(nb) - tess.GetMeshPoint(i);
                    double dist = fastabs(r_ij);
                    if (dist < 1e-200)
                        continue;
                    Vector3D nhat = r_ij * (1.0 / dist);
                    grad += nhat * (tess.GetArea(faces[j]) * Er_face);
                }
                double vol = tess.GetVolume(i);
                if (vol > 0.0)
                    grad *= 1.0 / vol;
                gradEr[i] = grad;
            }
        }

        // Apply flux limiter and compute FLD flux vector F = -lambda * D * grad(Er)
        for (size_t i = 0; i < Ncells; ++i) {
            double lambda = CG::CalcSingleFluxLimiter(gradEr[i], D_cell[i], Er_vol[i]);
            fldFlux[i] = gradEr[i] * (-lambda * D_cell[i]);
        }

        if (rank == 0)
            std::cout << "FLD flux computed for " << Ncells << " cells." << std::endl;

        // ============================================================
        // Build extensives
        // ============================================================
        std::vector<Conserved3D> extensives(Ncells);
        for (size_t i = 0; i < Ncells; ++i)
            PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);

        // ============================================================
        // Energy group boundaries
        // ============================================================
        std::vector<double> groupBoundaries;
#if ENERGY_GROUPS_NUM > 1
        groupBoundaries.resize(ENERGY_GROUPS_NUM + 1);
        for (size_t g = 0; g <= ENERGY_GROUPS_NUM; ++g)
            groupBoundaries[g] = ComputationalCell3D::energyBoundaries[g];
#endif

        // ============================================================
        // Construct observer
        // ============================================================
        auto observer = std::make_shared<SphericalObserver>(
            cfg.center, cfg.radius, cfg.nObservers, groupBoundaries);

        // ============================================================
        // Map FLD flux to observer patches
        // ============================================================
        size_t nObs = observer->getNumObservers();
        std::vector<Vector3D> const& obsDirections = observer->getDirections();
        std::vector<double> const& obsSolidAngles = observer->getObserverSolidAngles();

        std::vector<double> fldLuminosity(nObs, 0.0);
        std::vector<double> patchMinDist(nObs, std::numeric_limits<double>::max());

        for (size_t p = 0; p < nObs; ++p) {
            Vector3D spherePoint = cfg.center + obsDirections[p] * cfg.radius;
            double patchArea_p = obsSolidAngles[p] * cfg.radius * cfg.radius;
            for (size_t i = 0; i < Ncells; ++i) {
                double dist = fastabs(tess.GetMeshPoint(i) - spherePoint);
                if (dist < patchMinDist[p]) {
                    patchMinDist[p] = dist;
                    double radialFlux = ScalarProd(fldFlux[i], obsDirections[p]);
                    fldLuminosity[p] = std::max(0.0, radialFlux) * patchArea_p;
                }
            }
        }

#ifdef RICH_MPI
        {
            struct DistVal { double dist; int rank; };
            std::vector<DistVal> localDV(nObs), globalDV(nObs);
            for (size_t p = 0; p < nObs; ++p) {
                localDV[p].dist = patchMinDist[p];
                localDV[p].rank = rank;
            }
            MPI_Allreduce(localDV.data(), globalDV.data(),
                          static_cast<int>(nObs), MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
            for (size_t p = 0; p < nObs; ++p) {
                if (globalDV[p].rank != rank)
                    fldLuminosity[p] = 0.0;
            }
            MPI_Allreduce(MPI_IN_PLACE, fldLuminosity.data(),
                          static_cast<int>(nObs), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
#endif

        // Free FLD intermediates no longer needed
        Er_vol.clear(); Er_vol.shrink_to_fit();
        D_cell.clear(); D_cell.shrink_to_fit();
        gradEr.clear(); gradEr.shrink_to_fit();
        fldFlux.clear(); fldFlux.shrink_to_fit();
        patchMinDist.clear(); patchMinDist.shrink_to_fit();

        double totalFldLum = 0.0;
        for (size_t p = 0; p < nObs; ++p)
            totalFldLum += fldLuminosity[p];

        if (rank == 0)
            std::cout << "FLD luminosity mapped to " << nObs << " patches, total = "
                      << totalFldLum << " erg/s" << std::endl;

        // ============================================================
        // Construct boundary condition
        // ============================================================
        auto boundary = std::make_shared<VacuumBoundaryCondition<Vector3D, Tessellation3D>>(tess);

        // ============================================================
        // Construct RadiationIMC
        // ============================================================
        STORM::RadiationIMCParameters<ENERGY_GROUPS_NUM> params;
        size_t genPhotonsPerCell = std::max<size_t>(1, cfg.photonsPerCell / cfg.nGenerations);
        params.newPhotonsPerCell = genPhotonsPerCell;
        params.withHydro = false;
        params.noHydroFeedback = true;
        params.withRandomWalk = cfg.randomWalk;
        params.rwMinCellOpticalDepth = 15;
        params.withDDMC = cfg.ddmc;
        params.ddmcMinCellOpticalDepth = 15;
        params.ddmcMinParticleOpticalDepth = 5;
        params.ddmcUseMultigroupPGRW = true;
        params.MMC = false;
        params.diffusionPressureGradient = false;
#if ENERGY_GROUPS_NUM > 1
        params.withMultigroupOpacity = true;
#else
        params.withMultigroupOpacity = false;
#endif
        params.withCompton = cfg.compton;
        params.comptonMatrixSamples = cfg.comptonSamples;
        params.comptonAngleDependent = cfg.comptonAngleDependent;

        params.postProcess.enabled = true;
        params.postProcess.sourceDt = cfg.sourceDt;
        params.postProcess.transportTime = cfg.transportTime;
        params.postProcess.useCellVelocities = cfg.useCellVelocities;
        params.postProcess.polarization.enabled = cfg.polarization;
        params.postProcess.polarization.manualScatteringsAfterAcceleration = cfg.polarizationManualScatterings;
        params.postProcess.polarization.depolarizationScatterings = cfg.polarizationDepolarizationScatterings;
        params.postProcess.polarization.acceleratedClosure = cfg.polarizationClosure;

        auto physics = std::make_shared<::RadiationIMC>(
            tess, boundary, cells, extensives, eos, opacity, params);
        physics->setObserver(observer);

        auto popControl = std::make_shared<NoPopulationControl<Vector3D, Tessellation3D>>(tess);

        // ============================================================
        // Construct manager
        // ============================================================
        std::shared_ptr<MonteCarloManager3D> manager;
#ifdef RICH_MPI
        manager = std::make_shared<RDMAMonteCarloManager3D>(
            tess, physics, popControl, boundary);
#else
        manager = std::make_shared<MonteCarloManagerSerial3D>(
            tess, physics, popControl, boundary);
#endif

        if (rank == 0)
        {
            std::cout << "Starting transport..." << std::endl;
            size_t diagCells = std::min<size_t>(5, cells.size());
            for (size_t ci = 0; ci < diagCells; ++ci)
            {
                auto const& cc = cells[ci];
                double vol = tess.GetVolume(ci);
                double charLen = std::cbrt(vol);
                double sigAbs = opacity->CalcAbsorptionOpacity(cc, opacity->energy_groups_center[0]);
                double sigScat = opacity->CalcScatteringOpacity(cc, opacity->energy_groups_center[0]);
                double sigScatGrey = opacity->CalcScatteringOpacity(cc);
                double tauAbs = sigAbs * charLen;
                double tauScat = sigScat * charLen;
                std::cerr << "[DIAG] cell " << ci
                          << ": rho=" << cc.density
                          << " T=" << cc.temperature
                          << " vol=" << vol
                          << " L=" << charLen
                          << " sig_abs(g0)=" << sigAbs
                          << " sig_scat(g0)=" << sigScat
                          << " sig_scat_grey=" << sigScatGrey
                          << " tau_abs=" << tauAbs
                          << " tau_scat=" << tauScat
                          << "\n";
            }
            std::cerr << std::flush;
        }

        // ============================================================
        // Run transport (generation loop)
        // ============================================================
        using Particle3D = MonteCarloParticle<Vector3D, Tessellation3D>;

        bool const measuredLBActive =
            cfg.measuredLoadBalance && (cfg.adaptiveSourceCells || cfg.nGenerations > 1);
#if ENERGY_GROUPS_NUM > 1
        bool const isMultigroup = true;
#else
        bool const isMultigroup = false;
#endif

        if (rank == 0)
            std::cout << "Measured LB active: " << (measuredLBActive ? "yes" : "no") << std::endl;

        imc_measured_lb::Parameters measuredLBParams;
        measuredLBParams.floorCost = 1.0;
        measuredLBParams.stepWeight = 1.0;
        measuredLBParams.particleWeight = cfg.adaptiveSourceCells ? 1.0 : 0.0;
        measuredLBParams.medianClampFactor = isMultigroup ? 30.0 : 20.0;
        measuredLBParams.missingCellCost = isMultigroup ? 5.0 : 2.0;
        measuredLBParams.grayZeroStepInflation = 2.0;
        measuredLBParams.multigroupZeroStepInflation = 5.0;
        measuredLBParams.maxCellImbalance = MEASURED_LB_MAX_CELL_IMBALANCE;
        measuredLBParams.useMedianClamp = true;
        double const measuredLBWeightCompression = EffectiveMeasuredLBWeightCompression(cfg);

        AdaptiveSourceState mgAdaptive;
        AdaptiveGroupHistory mgGroupHistory;
        AdaptiveGroupSourceState mgGroupSourceState;
        ::RadiationIMC::GroupSamplingDiagnostics mgLastGroupSamplingDiag;
        ::RadiationIMC::GroupSamplingDiagnostics mgFinalGroupSamplingDiag;
        AdaptiveGroupSourceUpdateSummary mgFinalGroupSourceSummary;
        bool mgGroupFallbackToIntegrated = false;
        std::string mgGroupFallbackReason = "none";
        observer->clearGenerationStatistics();
        size_t mgIncludedFinalGenerations = 0;
        size_t mgDiscardedBurninGenerations = 0;

        const bool runForwardPostprocess =
            cfg.estimatorMode == PostProcessEstimatorMode::Forward ||
            cfg.estimatorMode == PostProcessEstimatorMode::Both;

        size_t const mgInitialBurninGenerations = cfg.adaptiveSourceCells ? 1 : 0;
        size_t const mgUniformBurninGenerations = cfg.adaptiveSourceCells ? 14 : 0;
        size_t const mgBurninGenerations = mgInitialBurninGenerations + mgUniformBurninGenerations;
        size_t const mgLearnedProbeGenerations = cfg.adaptiveSourceCells ? 1 : 0;
        size_t const mgFinalStartGeneration = mgBurninGenerations + mgLearnedProbeGenerations;
        size_t const mgFinalGenerations = cfg.adaptiveSourceCells
            ? 10 * cfg.nGenerations + 20
            : cfg.nGenerations;
        size_t const mgTotalGenerations = cfg.adaptiveSourceCells
            ? mgFinalStartGeneration + mgFinalGenerations
            : cfg.nGenerations;
        if (runForwardPostprocess)
        {
        for (size_t gen = 0; gen < mgTotalGenerations; ++gen)
        {
            bool const firstBurninThisGen =
                cfg.adaptiveSourceCells && gen < mgInitialBurninGenerations;
            bool const uniformBurninThisGen =
                cfg.adaptiveSourceCells &&
                gen >= mgInitialBurninGenerations &&
                gen < mgBurninGenerations;
            bool const burninThisGen = firstBurninThisGen || uniformBurninThisGen;
            bool const learnedProbeThisGen =
                cfg.adaptiveSourceCells &&
                gen >= mgBurninGenerations &&
                gen < mgFinalStartGeneration;
            bool const finalThisGen =
                !cfg.adaptiveSourceCells || gen >= mgFinalStartGeneration;
            size_t const finalGenerationIndex = finalThisGen
                ? (cfg.adaptiveSourceCells ? gen - mgFinalStartGeneration : gen)
                : 0;
            bool const adaptiveActiveThisGen =
                cfg.adaptiveSourceCells &&
                (learnedProbeThisGen || finalThisGen) &&
                !mgAdaptive.scoreByCellID.empty();
            size_t const photonsThisGen = firstBurninThisGen ? 1
                : (uniformBurninThisGen ? 3
                   : (learnedProbeThisGen ? 75
                      : (cfg.adaptiveSourceCells ? 1 : genPhotonsPerCell)));
            std::string phase = "final";
            if (firstBurninThisGen)
                phase = "burnin_exact1";
            else if (uniformBurninThisGen)
                phase = "burnin_exact3";
            else if (learnedProbeThisGen)
                phase = "learned_only_probe_exact75";
            else if (cfg.adaptiveSourceCells)
                phase = "learned_only_final";
            physics->setNewPhotonsPerCell(photonsThisGen);
            if (rank == 0)
                std::cout << "Generation " << (gen + 1) << "/" << mgTotalGenerations
                          << " phase=" << phase
                          << " photons_per_cell_this_gen=" << photonsThisGen;
            if (rank == 0 && finalThisGen)
                std::cout << " final_step=" << (finalGenerationIndex + 1)
                          << "/" << mgFinalGenerations;
            if (rank == 0)
                std::cout << std::endl;
            PrintAdaptiveGenerationStart("MG", cfg, mgAdaptive, gen, mgTotalGenerations,
                                         mgBurninGenerations,
                                         adaptiveActiveThisGen, rank);

            if (adaptiveActiveThisGen) {
                auto combinedSourceScores =
                    BuildCombinedSourceScoresForIMC(mgAdaptive, mgGroupSourceState);
                physics->setAdaptiveSourceCellScores(
                    std::move(combinedSourceScores),
                    cfg.adaptiveSourceStrength,
                    cfg.adaptiveSourceMaxFactor,
                    cfg.adaptiveSourceLearnedReserveFrac,
                    cfg.adaptiveSourceLearnedMinFactor,
                    mgAdaptive.observerBudgetMultiplier);
            } else {
                physics->clearAdaptiveSourceCellScores();
            }
            if (firstBurninThisGen)
                physics->setSourceEmissionControl(false, true, 1);
            else if (uniformBurninThisGen)
                physics->setSourceEmissionControl(false, true, 3);
            else if (learnedProbeThisGen)
                physics->setSourceEmissionControl(true, true, 75);
            else if (cfg.adaptiveSourceCells && finalThisGen)
                physics->setSourceEmissionControl(true, false, 1, 1000, 5000);
            else
                physics->clearSourceEmissionControl();
            observer->resetGenerationSourceCellEscapeStats();
            observer->resetGenerationSourceCellGroupEscapeStats();
            observer->setGenerationSourceCellGroupStatsEnabled(
                cfg.adaptiveGroupSourceCells && cfg.adaptiveGroupQuality &&
                ENERGY_GROUPS_NUM > 1 && !burninThisGen);

            if (adaptiveActiveThisGen && cfg.adaptiveGroupFrequencySampling &&
                !mgGroupSourceState.scoreByCellGroup.empty()) {
                auto groupScoresForIMC = BuildGroupScoresForIMC(
                    mgGroupSourceState, cells, static_cast<size_t>(ENERGY_GROUPS_NUM));
                physics->setAdaptiveSourceCellGroupScores(
                    std::move(groupScoresForIMC),
                    cfg.adaptiveGroupStrength,
                    cfg.adaptiveGroupPdfFloor,
                    cfg.adaptiveGroupMaxBias,
                    cfg.adaptiveGroupMaxWeightCorrection);
            } else {
                physics->clearAdaptiveSourceCellGroupScores();
            }

            physics->reseedRNG(static_cast<uint64_t>(rank+12345678) * mgTotalGenerations + gen);

            std::vector<Particle3D> empty;
            auto remaining = manager->step(std::move(empty), cells, cfg.transportTime);
            (void)remaining;

            auto mgAllocation = ReduceSourceAllocationSummary(
                physics->getLastSourceAllocationSummary());
            auto mgGroupSamplingDiag = ReduceGroupSamplingDiagnostics(
                physics->getLastGroupSamplingDiagnostics());
            mgLastGroupSamplingDiag = mgGroupSamplingDiag;
            auto mgSourceStats = observer->getGenerationSourceCellEscapeStats();
            observer->resetGenerationSourceCellEscapeStats();
            auto mgGroupSourceStats = observer->getGenerationSourceCellGroupEscapeStats();
            observer->resetGenerationSourceCellGroupEscapeStats();
            ObserverQualityDiagnostics mgObserverQuality;
            if (cfg.adaptiveSourceCells && cfg.adaptiveObserverEquity) {
                mgObserverQuality = BuildObserverQualityDiagnostics(
                    CollectGlobalObserverQuality(observer->getObserverQualitySnapshot()),
                    cfg, mgAdaptive);
            }

            ObserverGroupQualityDiagnostics mgGroupQuality;
            if (cfg.adaptiveGroupQuality && ENERGY_GROUPS_NUM > 1) {
                auto groupSnap = observer->getObserverGroupQualitySnapshot();
                CollectGlobalObserverGroupQuality(groupSnap);
                if (cfg.adaptiveGroupFrequencySampling &&
                    groupSnap.groupCount != static_cast<size_t>(ENERGY_GROUPS_NUM)) {
                    throw UniversalError(
                        "--adaptive-group-frequency-sampling requires observer group count to match ENERGY_GROUPS_NUM");
                }
                mgGroupQuality = BuildObserverGroupQualityDiagnosticsFromSnapshot(
                    groupSnap, cfg, mgGroupHistory, cfg.sourceDt);
            }

            auto mgUpdate = UpdateAdaptiveSourceScoresDistributed(
                mgSourceStats, cfg, mgAdaptive, mgObserverQuality,
                !burninThisGen);
            std::vector<SphericalObserver::SourceCellEscapeStat>().swap(mgSourceStats);

            AdaptiveGroupSourceUpdateSummary mgGroupUpdate;
            if (cfg.adaptiveGroupSourceCells && mgGroupQuality.enabled && !burninThisGen) {
                mgGroupUpdate = UpdateAdaptiveSourceGroupScores(
                    mgGroupSourceStats, mgGroupQuality, cfg,
                    mgGroupSourceState, rank, mpiSize);
                if (mgGroupUpdate.fallbackToIntegratedPath) {
                    mgGroupFallbackToIntegrated = true;
                    mgGroupFallbackReason = mgGroupUpdate.fallbackReason;
                }
            }
            std::vector<SphericalObserver::SourceCellGroupEscapeStat>().swap(mgGroupSourceStats);

            PrintAdaptiveGenerationStats(
                "MG", cfg, mgAdaptive, mgUpdate, mgAllocation,
                mgObserverQuality, gen, rank);
            PrintAdaptiveGroupGenerationStats(mgGroupQuality, mgGroupSamplingDiag, gen, rank);
            if (rank == 0 && cfg.adaptiveSourceCells)
                std::cout << "MG learned cells after iteration " << (gen + 1)
                          << ": " << mgAdaptive.scoreByCellID.size() << std::endl;
            observer->addBoxEscapeEnergy(boundary->getEscapedEnergy());
            boundary->resetEscapedEnergy();
            observer->mpiReduceToRank0();
            bool const includeGenerationInFinal = finalThisGen;
            if (includeGenerationInFinal) {
                if (rank == 0)
                    observer->accumulateCurrentTalliesForStatistics(cfg.sourceDt);
                AccumulateGroupSamplingDiagnostics(
                    mgFinalGroupSamplingDiag, mgGroupSamplingDiag);
                AccumulateAdaptiveGroupSourceSummary(
                    mgFinalGroupSourceSummary, mgGroupUpdate);
                ++mgIncludedFinalGenerations;
            } else {
                ++mgDiscardedBurninGenerations;
            }
            observer->resetTallies();
            if (cfg.adaptiveSourceCells && !mgAdaptive.burninCompletePrinted &&
                gen + 1 == mgBurninGenerations)
            {
                if (rank == 0)
                    std::cout << "MG adaptive source burn-in complete" << std::endl;
                mgAdaptive.burninCompletePrinted = true;
            }

#ifdef RICH_MPI
            RankStepImbalance const mgStepImbalance =
                ComputeRankStepImbalance("MG", gen, manager->GetCellsStepsCounters(), rank);
            bool const doInitialMeasuredLB =
                (gen == 0 && measuredLBActive && !cfg.adaptiveSourceCells);
            bool const doPostAdaptiveMeasuredLB =
                measuredLBActive &&
                cfg.adaptiveSourceCells &&
                learnedProbeThisGen &&
                !mgAdaptive.postAdaptiveMeasuredLBDone;
            bool const doAdaptivePeriodicMeasuredLB =
                measuredLBActive &&
                cfg.adaptiveSourceCells &&
                finalThisGen &&
                finalGenerationIndex + 1 < mgFinalGenerations &&
                finalGenerationIndex + 1 <= 10 &&
                (finalGenerationIndex + 1) % 5 == 0;
            std::string const mgLBLabel = doPostAdaptiveMeasuredLB
                ? "MEASURED_LB_ADAPTIVE"
                : (doAdaptivePeriodicMeasuredLB
                    ? "MEASURED_LB_ADAPTIVE_PERIODIC"
                    : "MEASURED_LB");
            if (rank == 0 && doPostAdaptiveMeasuredLB)
                std::cout << "MG learned-only probe complete; running measured LB before final calculation" << std::endl;
            if (rank == 0 && doAdaptivePeriodicMeasuredLB)
                std::cout << "MG periodic final measured LB after final step "
                          << (finalGenerationIndex + 1)
                          << ": rank_step_imbalance="
                          << mgStepImbalance.maxOverMean
                          << std::endl;
            // The just-finished generation has already been recorded into the
            // final-output statistics accumulator when it is eligible. Its step
            // counters are used here only to repartition later generations.
            // Repartition assumptions:
            //   - Each generation is independent (no census particles carried between them).
            //   - noHydroFeedback is true: gas state is not modified by MC generations.
            //   - Extensives can be rebuilt from cell primitives after repartition.
            //   - The observer statistics accumulator and adaptive scores are preserved.
            if (doInitialMeasuredLB ||
                doPostAdaptiveMeasuredLB || doAdaptivePeriodicMeasuredLB) {
                if (!params.noHydroFeedback) {
                    throw UniversalError("Measured load balance repartition requires noHydroFeedback=true");
                }

                PrintVmRSS("before_measured_lb", rank);

                std::vector<double> measuredWeightsForExchange;

                {
                    auto const& localSteps = manager->GetCellsStepsCounters();

                    std::vector<size_t> cellIDs(Ncells);
                    for (size_t i = 0; i < Ncells; ++i)
                        cellIDs[i] = cells[i].ID;

                    auto localMeasurements = imc_measured_lb::BuildLocalMeasurements(
                        cellIDs, localSteps, physics->getLastSourcePhotonsPerCell());

                    uint64_t localTotalSteps = 0;
                    for (auto const& m : localMeasurements)
                        localTotalSteps += static_cast<uint64_t>(m.stepCount);

                    uint64_t globalTotalSteps = 0;
                    MPI_Allreduce(&localTotalSteps, &globalTotalSteps, 1,
                                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

                    if (globalTotalSteps == 0) {
                        if (rank == 0)
                            std::cerr << mgLBLabel
                                      << ": measured generation had zero total steps, skipping repartition\n";
                    } else {
                        // Global-mean-based clamp via MPI_Allreduce (no per-cell allgather).
                        auto localCostByCellID = imc_measured_lb::BuildMeasuredCosts(
                            localMeasurements, measuredLBParams, isMultigroup, MPI_COMM_WORLD);

                        imc_measured_lb::PrintMeasuredLBDiagnosticsDistributed(
                            localMeasurements, localCostByCellID, isMultigroup, MPI_COMM_WORLD);

                        PrintVmRSS("after_local_costs", rank);

                        size_t localMissingCosts = 0;
                        for (size_t i = 0; i < Ncells; ++i) {
                            if (localCostByCellID.find(cells[i].ID) == localCostByCellID.end())
                                ++localMissingCosts;
                        }
                        uint64_t localMissing64 = static_cast<uint64_t>(localMissingCosts);
                        uint64_t globalMissingCosts = 0;
                        MPI_Allreduce(&localMissing64, &globalMissingCosts, 1,
                                      MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
                        if (rank == 0 && globalMissingCosts != 0) {
                            std::cerr << "MEASURED_LB_WARNING missing local measured costs before repartition: "
                                      << globalMissingCosts << "\n";
                        }

                        IMCStepCounterCostCalculator::Parameters costCalcParams;
                        costCalcParams.floorCost = measuredLBParams.floorCost;
                        costCalcParams.missingCellCost = measuredLBParams.missingCellCost;
                        IMCStepCounterCostCalculator measuredCostCalc(std::move(localCostByCellID), costCalcParams);

                        std::vector<Vector3D> currentPoints(Ncells);
                        for (size_t i = 0; i < Ncells; ++i)
                            currentPoints[i] = tess.GetMeshPoint(i);

                        auto lbWeightsNew = measuredCostCalc.CalculateCost(tess, cells);

                        if (lbWeightsNew.size() != Ncells) {
                            throw UniversalError("Measured LB weight count mismatch before BuildParallel");
                        }

                        measuredWeightsForExchange = lbWeightsNew;

                        for (auto& w : lbWeightsNew)
                            w = std::pow(w, measuredLBWeightCompression);

                        tess.BuildParallel(currentPoints, lbWeightsNew);
                    }
                }

                PrintVmRSS("after_build_parallel", rank);

                if (!measuredWeightsForExchange.empty()) {
                    MPI_exchange_data(tess, cells, false, 1, &dummyCell);

                    double dummyWeight = measuredLBParams.missingCellCost;
                    MPI_exchange_data(tess, measuredWeightsForExchange, false, 1, &dummyWeight);

                    Ncells = tess.GetPointNo();

                    if (cells.size() != Ncells) {
                        UniversalError eo("Cell count mismatch after measured LB repartition");
                        eo.addEntry("cells.size()", static_cast<double>(cells.size()));
                        eo.addEntry("Ncells", static_cast<double>(Ncells));
                        throw eo;
                    }

                    if (measuredWeightsForExchange.size() != Ncells) {
                        UniversalError eo("Measured weight count mismatch after repartition");
                        eo.addEntry("weights.size()", static_cast<double>(measuredWeightsForExchange.size()));
                        eo.addEntry("Ncells", static_cast<double>(Ncells));
                        throw eo;
                    }

                    imc_measured_lb::PrintPostRepartitionDiagnosticsFromWeights(
                        measuredWeightsForExchange, measuredLBWeightCompression,
                        isMultigroup, MPI_COMM_WORLD);

                    PrintVmRSS("after_exchange", rank);

                    extensives.resize(Ncells);
                    for (size_t i = 0; i < Ncells; ++i)
                        PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);

#if ENERGY_GROUPS_NUM > 1
                    if (cfg.opacityScaleMode != OpacityScaleMode::None) {
                        // Measured LB changes which cell IDs are local to each MPI rank.
                        // The alpha table is local and ID-keyed, so refresh it
                        // before rebuilding physics and before the next transport step.
                        RecomputeOpacityScaleFactors(
                            *opacity, *greyOpacity, cells, Ncells, rank, cfg.opacityScaleMode, "after measured LB repartition");
                    }
#endif

                    boundary = std::make_shared<VacuumBoundaryCondition<Vector3D, Tessellation3D>>(tess);

                    physics = std::make_shared<::RadiationIMC>(
                        tess, boundary, cells, extensives, eos, opacity, params);
                    physics->setObserver(observer);

                    popControl = std::make_shared<NoPopulationControl<Vector3D, Tessellation3D>>(tess);

                    manager = std::make_shared<RDMAMonteCarloManager3D>(
                        tess, physics, popControl, boundary);

                    if (rank == 0)
                        std::cout << mgLBLabel
                                  << ": repartitioned, new local cells=" << Ncells << std::endl;

                    PrintVmRSS("after_rebuild_physics", rank);
                }
                if (doPostAdaptiveMeasuredLB) {
                    mgAdaptive.postAdaptiveMeasuredLBDone = true;
                    mgAdaptive.adaptiveMeasuredLBCount += 1;
                    mgAdaptive.lastAdaptiveMeasuredLBGeneration = gen;
                    if (rank == 0)
                        std::cout << "MG post-adaptive measured load balance complete" << std::endl;
                }
                if (doAdaptivePeriodicMeasuredLB) {
                    mgAdaptive.adaptiveMeasuredLBCount += 1;
                    mgAdaptive.lastAdaptiveMeasuredLBGeneration = gen;
                    if (rank == 0)
                        std::cout << "MG periodic measured load balance complete" << std::endl;
                }
            }
#endif // RICH_MPI
        }

        // ============================================================
        // Finish diagnostics
        // ============================================================
        if (rank == 0)
            observer->loadStatisticalMeanTallies();

        if (rank == 0) {
            SphericalObserver::Diagnostics diag;
            diag.sourceDt = cfg.sourceDt;
            diag.transportTime = cfg.transportTime;
            diag.mpiRanks = mpiSize;
            diag.comptonEnabled = cfg.compton ? 1 : 0;
            diag.emittedEnergy = observer->getEmittedEnergy();
            diag.absorbedEnergy = observer->getAbsorbedEnergy();
            diag.boxEscapeEnergy = observer->getBoxEscapeEnergy();
            diag.timedOutEnergy = observer->getTimedOutEnergy();
            diag.cutoffEnergy = observer->getCutoffEnergy();
            diag.snapshotTime = snapshot.time;
            diag.snapshotCycle = snapshot.cycle;
            diag.nGenerations = static_cast<int>(mgFinalGenerations);
            diag.includedFinalGenerations = static_cast<int>(mgIncludedFinalGenerations);
            diag.discardedBurninGenerations = static_cast<int>(mgDiscardedBurninGenerations);
            diag.adaptiveOnlyFinalOutput = cfg.adaptiveSourceCells ? 1 : 0;
            diag.adaptiveGroupQualityEnabled = cfg.adaptiveGroupQuality ? 1 : 0;
            diag.adaptiveGroupSourceCellsEnabled = cfg.adaptiveGroupSourceCells ? 1 : 0;
            diag.adaptiveGroupFrequencySamplingEnabled = cfg.adaptiveGroupFrequencySampling ? 1 : 0;
            diag.adaptiveGroupHistoryEnabled = cfg.adaptiveGroupHistory ? 1 : 0;
            diag.adaptiveGroupLuminosityNormalization = cfg.adaptiveGroupLuminosityNormalization;
            diag.adaptiveGroupTargetNeff = cfg.adaptiveGroupTargetNeff;
            diag.adaptiveGroupTargetPolSnr = cfg.adaptiveGroupTargetPolSnr;
            diag.adaptiveGroupDeficitMax = cfg.adaptiveGroupDeficitMax;
            diag.adaptiveGroupMinCrossings = static_cast<int>(cfg.adaptiveGroupMinCrossings);
            diag.adaptiveGroupMinLuminosity = cfg.adaptiveGroupMinLuminosity;
            diag.adaptiveGroupMinLuminosityFracOfGroupMax =
                cfg.adaptiveGroupMinLuminosityFracOfGroupMax;
            diag.adaptiveGroupLatestWeight = cfg.adaptiveGroupLatestWeight;
            diag.adaptiveGroupCumulativeWeight = cfg.adaptiveGroupCumulativeWeight;
            diag.adaptiveGroupEmaWeight = cfg.adaptiveGroupEmaWeight;
            diag.adaptiveGroupSamplingStrength = cfg.adaptiveGroupStrength;
            diag.adaptiveGroupSamplingPdfFloor = cfg.adaptiveGroupPdfFloor;
            diag.adaptiveGroupSamplingMaxBias = cfg.adaptiveGroupMaxBias;
            diag.adaptiveGroupSamplingMaxWeightCorrection = cfg.adaptiveGroupMaxWeightCorrection;
            diag.adaptiveGroupSamplingTotalSampled =
                static_cast<unsigned long long>(mgFinalGroupSamplingDiag.totalSampled);
            diag.adaptiveGroupWeightCorrectionMin =
                mgFinalGroupSamplingDiag.weightCorrectionMin;
            diag.adaptiveGroupWeightCorrectionMean =
                (mgFinalGroupSamplingDiag.weightCorrectionCount > 0)
                    ? mgFinalGroupSamplingDiag.weightCorrectionSum /
                      static_cast<double>(mgFinalGroupSamplingDiag.weightCorrectionCount)
                    : 1.0;
            diag.adaptiveGroupWeightCorrectionMax =
                mgFinalGroupSamplingDiag.weightCorrectionMax;
            diag.adaptiveGroupWeightCorrectionCappedFraction =
                (mgFinalGroupSamplingDiag.weightCorrectionCount > 0)
                    ? static_cast<double>(mgFinalGroupSamplingDiag.weightCorrectionCapped) /
                      static_cast<double>(mgFinalGroupSamplingDiag.weightCorrectionCount)
                    : 0.0;
            diag.adaptiveGroupWeightCorrectionFallbackCount =
                static_cast<unsigned long long>(mgFinalGroupSamplingDiag.weightCorrectionFallback);
            diag.adaptiveGroupInvalidPdfFallbackCount =
                static_cast<unsigned long long>(mgFinalGroupSamplingDiag.invalidPdfFallback);
            diag.adaptiveGroupInvalidPdfFallbackPacketCount =
                static_cast<unsigned long long>(mgFinalGroupSamplingDiag.invalidPdfFallbackPackets);
            diag.adaptiveGroupCappedEnergyFraction =
                mgFinalGroupSamplingDiag.cappedEnergyFraction;
            diag.adaptiveGroupEstimatorPotentiallyBiased =
                mgFinalGroupSamplingDiag.estimatorPotentiallyBiased ? 1 : 0;
            diag.adaptiveGroupFallbackToIntegratedPath =
                mgGroupFallbackToIntegrated ? 1 : 0;
            diag.adaptiveGroupFallbackReason = mgGroupFallbackReason;
            diag.adaptiveGroupSourceLocalStatsAfterPrune =
                mgFinalGroupSourceSummary.localStatsAfterPrune;
            diag.adaptiveGroupSourceLocalStatsDropped =
                mgFinalGroupSourceSummary.localStatsDropped;
            diag.adaptiveGroupSourceMpiStatsExchanged =
                mgFinalGroupSourceSummary.mpiStatsExchanged;
            diag.adaptiveGroupSourceMpiPackedBytes =
                mgFinalGroupSourceSummary.maxPackedBytes;

            observer->writeHDF5(cfg.outputPath, diag);

            if (!cfg.vtkOutput.empty()) {
                observer->writeVTK(cfg.vtkOutput, cfg.sourceDt);

                // Append FLD luminosity scalars to the VTK file
                std::ofstream vtkAppend(cfg.vtkOutput, std::ios::app);
                if (vtkAppend.is_open()) {
                    vtkAppend << std::scientific << std::setprecision(12);

                    double fourPi = 4.0 * M_PI;

                    vtkAppend << "SCALARS fld_surface_luminosity double 1\n"
                              << "LOOKUP_TABLE default\n";
                    for (size_t p = 0; p < nObs; ++p)
                        vtkAppend << fldLuminosity[p] << "\n";

                    vtkAppend << "SCALARS fld_surface_isotropic_equivalent_luminosity double 1\n"
                              << "LOOKUP_TABLE default\n";
                    for (size_t p = 0; p < nObs; ++p) {
                        double isoEquiv = (obsSolidAngles[p] > 0.0)
                            ? fldLuminosity[p] * fourPi / obsSolidAngles[p] : 0.0;
                        vtkAppend << isoEquiv << "\n";
                    }

                    vtkAppend << "SCALARS log10_fld_surface_luminosity double 1\n"
                              << "LOOKUP_TABLE default\n";
                    for (size_t p = 0; p < nObs; ++p) {
                        double val = (fldLuminosity[p] > 0.0) ? std::log10(fldLuminosity[p]) : -99.0;
                        vtkAppend << val << "\n";
                    }

                    vtkAppend << "SCALARS fld_surface_flux double 1\n"
                              << "LOOKUP_TABLE default\n";
                    for (size_t p = 0; p < nObs; ++p) {
                        double patchArea_p = obsSolidAngles[p] * cfg.radius * cfg.radius;
                        double flux = (patchArea_p > 0.0) ? fldLuminosity[p] / patchArea_p : 0.0;
                        vtkAppend << flux << "\n";
                    }
                    AppendZeroVtkScalar(vtkAppend, "fld_surface_luminosity_stderr_gen", nObs);
                    AppendZeroVtkScalar(vtkAppend, "fld_surface_luminosity_relerr_gen", nObs);
                    AppendZeroVtkScalar(vtkAppend, "fld_surface_luminosity_stderr_packet", nObs);
                    AppendZeroVtkScalar(vtkAppend, "fld_surface_luminosity_relerr_packet", nObs);
                    AppendZeroVtkScalar(vtkAppend, "fld_surface_luminosity_neff", nObs);
                    AppendZeroVtkScalar(vtkAppend, "fld_surface_isotropic_equivalent_luminosity_stderr_gen", nObs);
                    AppendZeroVtkScalar(vtkAppend, "fld_surface_isotropic_equivalent_luminosity_relerr_gen", nObs);
                    AppendZeroVtkScalar(vtkAppend, "fld_surface_flux_stderr_gen", nObs);
                    AppendZeroVtkScalar(vtkAppend, "fld_surface_flux_relerr_gen", nObs);
                    AppendZeroVtkScalar(vtkAppend, "log10_fld_surface_luminosity_stderr_gen", nObs);
                    AppendZeroVtkScalar(vtkAppend, "log10_fld_surface_luminosity_relerr_gen", nObs);
                }
            }

            double totalLum = observer->getTotalCrossingEnergy() / cfg.sourceDt;
            double residual = diag.emittedEnergy - diag.absorbedEnergy
                            - diag.boxEscapeEnergy - diag.timedOutEnergy - diag.cutoffEnergy;
            double timedOutFrac = (diag.emittedEnergy > 0.0)
                ? diag.timedOutEnergy / diag.emittedEnergy : 0.0;

            std::cout << "\n=== TDE Post-Processing Results ===\n"
                      << "Generations:              " << mgFinalGenerations << "\n"
                      << "Final included generations: " << mgIncludedFinalGenerations << "\n"
                      << "Discarded burn-in generations: " << mgDiscardedBurninGenerations << "\n"
                      << "Final average policy:     " << (cfg.adaptiveSourceCells ? "adaptive_only" : "all_generations") << "\n"
                      << "Photons/cell/gen:         " << genPhotonsPerCell << "\n"
                      << "Total crossing luminosity: " << totalLum << " +/- "
                      << observer->getTotalLuminosityStderrGen(cfg.sourceDt)
                      << " erg/s (rel=" << observer->getTotalLuminosityRelErrGen(cfg.sourceDt) << ")\n"
                      << "Total FLD luminosity:     " << totalFldLum << " erg/s\n"
                      << "Emitted energy:           " << diag.emittedEnergy << " erg\n"
                      << "Absorbed energy:          " << diag.absorbedEnergy << " erg\n"
                      << "Box escape energy:        " << diag.boxEscapeEnergy << " erg\n"
                      << "Timed-out energy:         " << diag.timedOutEnergy << " erg\n"
                      << "Cutoff energy:            " << diag.cutoffEnergy << " erg\n"
                      << "Sink residual:            " << residual << " erg\n"
                      << "Timed-out fraction:       " << timedOutFrac << "\n"
                      << "Output written to:        " << cfg.outputPath << "\n"
                      << std::endl;
        }
        } // end if (runForwardPostprocess)

        // ============================================================
        // Reverse estimator (if requested)
        // ============================================================
        if (cfg.estimatorMode == PostProcessEstimatorMode::Reverse ||
            cfg.estimatorMode == PostProcessEstimatorMode::Both)
        {
            if (rank == 0)
                std::cout << "\n=== Running Reverse Adjoint Estimator ===\n";

            ReverseEstimatorConfig rcfg;
            rcfg.packetsPerObserverGroup = cfg.reversePacketsPerObserverGroup;
            rcfg.seed = cfg.reverseSeed;
            rcfg.outputPrefix = cfg.reverseOutputPrefix;
            rcfg.estimatorMode = cfg.estimatorMode;
            rcfg.progressIntervalSec = cfg.reverseProgressIntervalSec;
            rcfg.maxEvents = cfg.reverseMaxEvents;
            rcfg.ddmcMinCellOpticalDepth = cfg.reverseDDMCMinCellOpticalDepth;
            rcfg.ddmcMinParticleOpticalDepth = cfg.reverseDDMCMinParticleOpticalDepth;
            rcfg.ddmcObserverExclusion = cfg.reverseDDMCObserverExclusion;
            rcfg.ddmcPhotosphereExclusion = cfg.reverseDDMCPhotosphereExclusion;
            rcfg.ddmcPhotosphereOpticalDepth = cfg.reverseDDMCPhotosphereOpticalDepth;

            std::vector<double> reverseFleckFactors;
            bool fleckFromForward = false;

            auto refreshReverseFleckFactors = [&]() {
                if (cfg.estimatorMode == PostProcessEstimatorMode::Both &&
                    runForwardPostprocess && physics && !physics->getFactorFleck().empty())
                {
                    reverseFleckFactors = physics->getFactorFleck();
                    fleckFromForward = true;
                }
                else
                {
                    reverseFleckFactors = fleck_helper::computeFleckFactors(
                        cells, *eos, *opacity, cfg.sourceDt);
                    fleckFromForward = false;
                }
            };
            refreshReverseFleckFactors();

            if (cfg.reverseMeasuredLB)
            {
                if (!params.noHydroFeedback) {
                    throw UniversalError("Reverse measured load balance requires noHydroFeedback=true");
                }

                size_t const pilotPackets =
                    ResolveReverseLBPilotPacketsPerObserverGroup(
                        cfg.reversePacketsPerObserverGroup,
                        cfg.reverseLBPilotPacketsPerObserverGroup);

                if (pilotPackets == 0) {
                    if (rank == 0)
                        std::cerr << "REVERSE_MEASURED_LB_SKIP"
                                  << " reason=production_packets_too_small"
                                  << " production_packets_per_observer_group="
                                  << cfg.reversePacketsPerObserverGroup
                                  << "\n" << std::flush;
                }
                else {
                if (rank == 0)
                    std::cerr << "REVERSE_MEASURED_LB_PILOT"
                              << " packets_per_observer_group=" << pilotPackets
                              << " production_packets_per_observer_group="
                              << cfg.reversePacketsPerObserverGroup
                              << "\n" << std::flush;

                ReverseEstimatorConfig pilotCfg = rcfg;
                pilotCfg.packetsPerObserverGroup = pilotPackets;
                pilotCfg.outputPrefix = cfg.reverseOutputPrefix + "_pilot";

                ReverseAdjointTransport3D pilotEstimator(
                    tess, cells, opacity, observer, pilotCfg,
                    cfg.sourceDt, cfg.transportTime, reverseFleckFactors,
                    cfg.useCellVelocities, cfg.ddmc, cfg.polarization,
                    cfg.compton);
                pilotEstimator.setFleckFromForwardVector(fleckFromForward);
                pilotEstimator.run();

                auto toSizeTVector = [](std::vector<uint64_t> const& in) {
                    std::vector<size_t> out(in.size(), 0);
                    for (size_t i = 0; i < in.size(); ++i)
                        out[i] = static_cast<size_t>(in[i]);
                    return out;
                };

                std::vector<size_t> cellIDs(Ncells);
                for (size_t i = 0; i < Ncells; ++i)
                    cellIDs[i] = cells[i].ID;

                auto localMeasurements = imc_measured_lb::BuildLocalMeasurements(
                    cellIDs,
                    toSizeTVector(pilotEstimator.localCellStepCounts()),
                    toSizeTVector(pilotEstimator.localCellPacketEntries()));

                uint64_t localTotalSteps = 0;
                for (auto const& m : localMeasurements)
                    localTotalSteps += static_cast<uint64_t>(m.stepCount);

                uint64_t globalTotalSteps = localTotalSteps;
#ifdef RICH_MPI
                MPI_Allreduce(&localTotalSteps, &globalTotalSteps, 1,
                              MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif

                if (globalTotalSteps == 0) {
                    if (rank == 0)
                        std::cerr << "REVERSE_MEASURED_LB_SKIP reason=zero_pilot_steps\n"
                                  << std::flush;
                }
                else {
#ifdef RICH_MPI
                    auto localCostByCellID = imc_measured_lb::BuildMeasuredCosts(
                        localMeasurements, measuredLBParams, isMultigroup, MPI_COMM_WORLD);

                    imc_measured_lb::PrintMeasuredLBDiagnosticsDistributed(
                        localMeasurements, localCostByCellID, isMultigroup, MPI_COMM_WORLD);

                    IMCStepCounterCostCalculator::Parameters costCalcParams;
                    costCalcParams.floorCost = measuredLBParams.floorCost;
                    costCalcParams.missingCellCost = measuredLBParams.missingCellCost;
                    IMCStepCounterCostCalculator measuredCostCalc(std::move(localCostByCellID), costCalcParams);

                    std::vector<Vector3D> currentPoints(Ncells);
                    for (size_t i = 0; i < Ncells; ++i)
                        currentPoints[i] = tess.GetMeshPoint(i);

                    auto lbWeightsNew = measuredCostCalc.CalculateCost(tess, cells);
                    if (lbWeightsNew.size() != Ncells) {
                        throw UniversalError("Reverse measured LB weight count mismatch before BuildParallel");
                    }

                    std::vector<double> measuredWeightsForExchange = lbWeightsNew;
                    std::vector<double> forwardFleckForExchange = reverseFleckFactors;
                    bool const exchangeForwardFleck =
                        fleckFromForward && forwardFleckForExchange.size() == Ncells;

                    double const reverseLBWeightCompression =
                        (cfg.reverseLBWeightCompression > 0.0)
                            ? cfg.reverseLBWeightCompression
                            : measuredLBWeightCompression;
                    for (auto& w : lbWeightsNew)
                        w = std::pow(w, reverseLBWeightCompression);

                    tess.BuildParallel(currentPoints, lbWeightsNew);

                    MPI_exchange_data(tess, cells, false, 1, &dummyCell);
                    double dummyWeight = measuredLBParams.missingCellCost;
                    MPI_exchange_data(tess, measuredWeightsForExchange, false, 1, &dummyWeight);
                    if (exchangeForwardFleck) {
                        double dummyFleck = 1.0;
                        MPI_exchange_data(tess, forwardFleckForExchange, false, 1, &dummyFleck);
                    }

                    Ncells = tess.GetPointNo();
                    if (cells.size() != Ncells) {
                        UniversalError eo("Cell count mismatch after reverse measured LB repartition");
                        eo.addEntry("cells.size()", static_cast<double>(cells.size()));
                        eo.addEntry("Ncells", static_cast<double>(Ncells));
                        throw eo;
                    }
                    if (measuredWeightsForExchange.size() != Ncells) {
                        UniversalError eo("Reverse measured weight count mismatch after repartition");
                        eo.addEntry("weights.size()", static_cast<double>(measuredWeightsForExchange.size()));
                        eo.addEntry("Ncells", static_cast<double>(Ncells));
                        throw eo;
                    }

                    imc_measured_lb::PrintPostRepartitionDiagnosticsFromWeights(
                        measuredWeightsForExchange, reverseLBWeightCompression,
                        isMultigroup, MPI_COMM_WORLD);

                    extensives.resize(Ncells);
                    for (size_t i = 0; i < Ncells; ++i)
                        PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);

#if ENERGY_GROUPS_NUM > 1
                    if (cfg.opacityScaleMode != OpacityScaleMode::None) {
                        RecomputeOpacityScaleFactors(
                            *opacity, *greyOpacity, cells, Ncells, rank,
                            cfg.opacityScaleMode, "after reverse measured LB repartition");
                    }
#endif

                    boundary = std::make_shared<VacuumBoundaryCondition<Vector3D, Tessellation3D>>(tess);

                    physics = std::make_shared<::RadiationIMC>(
                        tess, boundary, cells, extensives, eos, opacity, params);
                    physics->setObserver(observer);

                    popControl = std::make_shared<NoPopulationControl<Vector3D, Tessellation3D>>(tess);
                    manager = std::make_shared<RDMAMonteCarloManager3D>(
                        tess, physics, popControl, boundary);

                    if (exchangeForwardFleck && forwardFleckForExchange.size() == Ncells) {
                        reverseFleckFactors = std::move(forwardFleckForExchange);
                        fleckFromForward = true;
                    }
                    else {
                        reverseFleckFactors = fleck_helper::computeFleckFactors(
                            cells, *eos, *opacity, cfg.sourceDt);
                        fleckFromForward = false;
                    }

                    if (rank == 0)
                        std::cerr << "REVERSE_MEASURED_LB_DONE"
                                  << " new_local_cells=" << Ncells
                                  << " weight_compression=" << reverseLBWeightCompression
                                  << "\n" << std::flush;
#else
                    if (rank == 0)
                        std::cerr << "REVERSE_MEASURED_LB_SERIAL"
                                  << " pilot_steps=" << globalTotalSteps
                                  << " repartition=skipped\n" << std::flush;
#endif
                }
                }
            }

            auto reverseEstimator = std::make_unique<ReverseAdjointTransport3D>(
                tess, cells, opacity, observer, rcfg,
                cfg.sourceDt, cfg.transportTime, reverseFleckFactors,
                cfg.useCellVelocities, cfg.ddmc, cfg.polarization,
                cfg.compton);
            reverseEstimator->setFleckFromForwardVector(fleckFromForward);

            reverseEstimator->run();

            // Write comparison BEFORE reverse outputs so metadata is correct
            if (cfg.estimatorMode == PostProcessEstimatorMode::Both && rank == 0
                && observer)
            {
                std::string cp = "/postprocess_comparison";
                size_t nObs = observer->getNumObservers();

                auto fwdLum = observer->getLuminosity(cfg.sourceDt);
                std::vector<double> lumRev(nObs), lumDelta(nObs), lumRelDelta(nObs);
                double totalFwdLum = 0.0;
                for (size_t p = 0; p < nObs; ++p)
                {
                    lumRev[p] = reverseEstimator->tallies().getObsI(p);
                    lumDelta[p] = lumRev[p] - fwdLum[p];
                    totalFwdLum += std::abs(fwdLum[p]);
                }
                double floor = totalFwdLum * 1e-12;
                for (size_t p = 0; p < nObs; ++p)
                    lumRelDelta[p] = lumDelta[p] / std::max(std::abs(fwdLum[p]), floor);

                std::vector<double> fwdQ(nObs, 0.0), fwdU(nObs, 0.0);
                std::vector<std::vector<double>> fwdGroupQ, fwdGroupU;
#ifdef MONTECARLO_POLARIZATION
                auto const &obsQ = observer->getObserverStokesQ();
                auto const &obsU = observer->getObserverStokesU();
                for (size_t p = 0; p < nObs && p < obsQ.size(); ++p)
                {
                    fwdQ[p] = obsQ[p];
                    fwdU[p] = obsU[p];
                }
                fwdGroupQ = observer->getGroupStokesQ();
                fwdGroupU = observer->getGroupStokesU();
#endif
                auto fwdGroupLum = observer->getGroupLuminosity(cfg.sourceDt);

                reverseEstimator->writeComparisonOutputs(cfg.outputPath, cp,
                    fwdLum, lumRev, lumDelta, lumRelDelta, fwdQ, fwdU,
                    fwdGroupLum, fwdGroupQ, fwdGroupU);

                if (rank == 0)
                    std::cout << "Comparison outputs written to: " << cfg.outputPath << std::endl;
            }

            std::string reverseOutputFile = cfg.reverseOutputPrefix + ".h5";
            reverseEstimator->writeOutputs(reverseOutputFile);

            if (rank == 0)
                std::cout << "Reverse estimator output: " << reverseOutputFile << std::endl;
        }

        // ============================================================
        // Release MG objects before grey run to avoid OOM
        // ============================================================
        manager.reset();
        physics.reset();
        popControl.reset();
        observer.reset();
        boundary.reset();
        opacity.reset();
        mgAdaptive = AdaptiveSourceState{};
        if (rank == 0)
            std::cout << "MG resources released." << std::endl;

        // ============================================================
        // Grey IMC run (half generations)
        // ============================================================
        size_t nGreyGens = cfg.adaptiveSourceCells
            ? 10 * cfg.nGenerations
            : std::max<size_t>(1, cfg.nGenerations / 2);
        size_t greyPhotonsPerCell = std::max<size_t>(1, cfg.photonsPerCell / (2 * nGreyGens));

        if (rank == 0)
            std::cout << "\n=== Starting Grey IMC run (" << nGreyGens << " generations) ===" << std::endl;

        auto greyObserver = std::make_shared<SphericalObserver>(
            cfg.center, cfg.radius, cfg.nObservers, std::vector<double>());

        auto greyBoundary = std::make_shared<VacuumBoundaryCondition<Vector3D, Tessellation3D>>(tess);

        STORM::RadiationIMCParameters<ENERGY_GROUPS_NUM> greyParams;
        greyParams.newPhotonsPerCell = greyPhotonsPerCell;
        greyParams.withHydro = false;
        greyParams.noHydroFeedback = true;
        greyParams.withRandomWalk = cfg.randomWalk;
        greyParams.rwMinCellOpticalDepth = 15;
        greyParams.withDDMC = cfg.ddmc;
        greyParams.ddmcMinCellOpticalDepth = 15;
        greyParams.ddmcMinParticleOpticalDepth = 5;
        greyParams.ddmcUseMultigroupPGRW = false;
        greyParams.MMC = false;
        greyParams.diffusionPressureGradient = false;
        greyParams.withMultigroupOpacity = false;
        greyParams.withCompton = false;
        greyParams.postProcess.enabled = true;
        greyParams.postProcess.sourceDt = cfg.sourceDt;
        greyParams.postProcess.transportTime = cfg.transportTime;
        greyParams.postProcess.useCellVelocities = cfg.useCellVelocities;
        greyParams.postProcess.polarization.enabled = cfg.polarization;
        greyParams.postProcess.polarization.manualScatteringsAfterAcceleration = cfg.polarizationManualScatterings;
        greyParams.postProcess.polarization.depolarizationScatterings = cfg.polarizationDepolarizationScatterings;
        greyParams.postProcess.polarization.acceleratedClosure = cfg.polarizationClosure;

        auto greyPhysics = std::make_shared<::RadiationIMC>(
            tess, greyBoundary, cells, extensives, eos, greyOpacity, greyParams);
        greyPhysics->setObserver(greyObserver);

        auto greyPopControl = std::make_shared<NoPopulationControl<Vector3D, Tessellation3D>>(tess);

        std::shared_ptr<MonteCarloManager3D> greyManager;
#ifdef RICH_MPI
        greyManager = std::make_shared<RDMAMonteCarloManager3D>(
            tess, greyPhysics, greyPopControl, greyBoundary);
#else
        greyManager = std::make_shared<MonteCarloManagerSerial3D>(
            tess, greyPhysics, greyPopControl, greyBoundary);
#endif

        bool const greyMeasuredLBActive =
            cfg.measuredLoadBalance && (cfg.adaptiveSourceCells || nGreyGens > 1);

        if (rank == 0)
            std::cout << "Grey measured LB active: " << (greyMeasuredLBActive ? "yes" : "no") << std::endl;

        imc_measured_lb::Parameters greyLBParams;
        greyLBParams.floorCost = 1.0;
        greyLBParams.stepWeight = 1.0;
        greyLBParams.particleWeight = cfg.adaptiveSourceCells ? 1.0 : 0.0;
        greyLBParams.medianClampFactor = 30.0;
        greyLBParams.missingCellCost = 2.0;
        greyLBParams.grayZeroStepInflation = 2.0;
        greyLBParams.multigroupZeroStepInflation = 5.0;
        greyLBParams.maxCellImbalance = MEASURED_LB_MAX_CELL_IMBALANCE;
        greyLBParams.useMedianClamp = true;
        double const greyLBWeightCompression = EffectiveMeasuredLBWeightCompression(cfg);

        AdaptiveSourceState greyAdaptive;
        bool greyBurninMeasuredLBDone = false;
        bool greyFirstNonBurninMeasuredLBDone = false;
        greyObserver->clearGenerationStatistics();
        size_t greyIncludedFinalGenerations = 0;
        size_t greyDiscardedBurninGenerations = 0;
        size_t const greyInitialBurninGenerations = cfg.adaptiveSourceCells ? 1 : 0;
        size_t const greyUniformBurninGenerations = cfg.adaptiveSourceCells ? 14 : 0;
        size_t const greyBurninGenerations = greyInitialBurninGenerations + greyUniformBurninGenerations;
        size_t const greyLearnedProbeGenerations = cfg.adaptiveSourceCells ? 1 : 0;
        size_t const greyFinalStartGeneration = greyBurninGenerations + greyLearnedProbeGenerations;
        size_t const greyTotalGenerations = cfg.adaptiveSourceCells
            ? greyFinalStartGeneration + nGreyGens
            : nGreyGens;
        for (size_t gen = 0; gen < greyTotalGenerations; ++gen) {
            bool const greyFirstBurninThisGen =
                cfg.adaptiveSourceCells && gen < greyInitialBurninGenerations;
            bool const greyUniformBurninThisGen =
                cfg.adaptiveSourceCells &&
                gen >= greyInitialBurninGenerations &&
                gen < greyBurninGenerations;
            bool const greyBurninThisGen =
                greyFirstBurninThisGen || greyUniformBurninThisGen;
            bool const greyLearnedProbeThisGen =
                cfg.adaptiveSourceCells &&
                gen >= greyBurninGenerations &&
                gen < greyFinalStartGeneration;
            bool const greyFinalThisGen =
                !cfg.adaptiveSourceCells || gen >= greyFinalStartGeneration;
            size_t const greyFinalGenerationIndex = greyFinalThisGen
                ? (cfg.adaptiveSourceCells ? gen - greyFinalStartGeneration : gen)
                : 0;
            bool const greyAdaptiveActiveThisGen =
                cfg.adaptiveSourceCells &&
                (greyLearnedProbeThisGen || greyFinalThisGen) &&
                !greyAdaptive.scoreByCellID.empty();
            size_t const greyPhotonsThisGen = greyFirstBurninThisGen ? 1
                : (greyUniformBurninThisGen ? 3
                   : (greyLearnedProbeThisGen ? 75
                      : (cfg.adaptiveSourceCells ? 1 : greyPhotonsPerCell)));
            std::string greyPhase = "final";
            if (greyFirstBurninThisGen)
                greyPhase = "burnin_exact1";
            else if (greyUniformBurninThisGen)
                greyPhase = "burnin_exact3";
            else if (greyLearnedProbeThisGen)
                greyPhase = "learned_only_probe_exact75";
            else if (cfg.adaptiveSourceCells)
                greyPhase = "learned_only_final";
            greyPhysics->setNewPhotonsPerCell(greyPhotonsThisGen);
            if (rank == 0)
                std::cout << "Grey generation " << (gen + 1) << "/" << greyTotalGenerations
                          << " phase=" << greyPhase
                          << " photons_per_cell_this_gen=" << greyPhotonsThisGen;
            if (rank == 0 && greyFinalThisGen)
                std::cout << " final_step=" << (greyFinalGenerationIndex + 1)
                          << "/" << nGreyGens;
            if (rank == 0)
                std::cout << std::endl;
            PrintAdaptiveGenerationStart("Grey", cfg, greyAdaptive, gen, greyTotalGenerations,
                                         greyBurninGenerations,
                                         greyAdaptiveActiveThisGen, rank);

            if (greyAdaptiveActiveThisGen)
                greyPhysics->setAdaptiveSourceCellScores(
                    greyAdaptive.scoreByCellID,
                    cfg.adaptiveSourceStrength,
                    cfg.adaptiveSourceMaxFactor,
                    cfg.adaptiveSourceLearnedReserveFrac,
                    cfg.adaptiveSourceLearnedMinFactor,
                    greyAdaptive.observerBudgetMultiplier);
            else
                greyPhysics->clearAdaptiveSourceCellScores();
            if (greyFirstBurninThisGen)
                greyPhysics->setSourceEmissionControl(false, true, 1);
            else if (greyUniformBurninThisGen)
                greyPhysics->setSourceEmissionControl(false, true, 3);
            else if (greyLearnedProbeThisGen)
                greyPhysics->setSourceEmissionControl(true, true, 75);
            else if (cfg.adaptiveSourceCells && greyFinalThisGen)
                greyPhysics->setSourceEmissionControl(true, false, 1, 100, 2000);
            else
                greyPhysics->clearSourceEmissionControl();
            greyObserver->resetGenerationSourceCellEscapeStats();

            greyPhysics->reseedRNG(static_cast<uint64_t>(rank + 87654321) * greyTotalGenerations + gen);

            std::vector<Particle3D> empty;
            auto remaining = greyManager->step(std::move(empty), cells, cfg.transportTime);
            (void)remaining;

            auto greyAllocation = ReduceSourceAllocationSummary(
                greyPhysics->getLastSourceAllocationSummary());
            auto greySourceStats = greyObserver->getGenerationSourceCellEscapeStats();
            greyObserver->resetGenerationSourceCellEscapeStats();
            ObserverQualityDiagnostics greyObserverQuality;
            if (cfg.adaptiveSourceCells && cfg.adaptiveObserverEquity) {
                greyObserverQuality = BuildObserverQualityDiagnostics(
                    CollectGlobalObserverQuality(greyObserver->getObserverQualitySnapshot()),
                    cfg, greyAdaptive);
            }
            auto greyUpdate = UpdateAdaptiveSourceScoresDistributed(
                greySourceStats, cfg, greyAdaptive, greyObserverQuality,
                !greyBurninThisGen);
            std::vector<SphericalObserver::SourceCellEscapeStat>().swap(greySourceStats);
            PrintAdaptiveGenerationStats(
                "Grey", cfg, greyAdaptive, greyUpdate, greyAllocation,
                greyObserverQuality, gen, rank);
            if (rank == 0 && cfg.adaptiveSourceCells)
                std::cout << "Grey learned cells after iteration " << (gen + 1)
                          << ": " << greyAdaptive.scoreByCellID.size() << std::endl;
            greyObserver->addBoxEscapeEnergy(greyBoundary->getEscapedEnergy());
            greyBoundary->resetEscapedEnergy();
            greyObserver->mpiReduceToRank0();
            bool const greyIncludeGenerationInFinal = greyFinalThisGen;
            if (greyIncludeGenerationInFinal) {
                if (rank == 0)
                    greyObserver->accumulateCurrentTalliesForStatistics(cfg.sourceDt);
                ++greyIncludedFinalGenerations;
            } else {
                ++greyDiscardedBurninGenerations;
            }
            greyObserver->resetTallies();
            if (cfg.adaptiveSourceCells && !greyAdaptive.burninCompletePrinted &&
                greyBurninGenerations > 0 &&
                gen + 1 == greyBurninGenerations)
            {
                if (rank == 0)
                    std::cout << "Grey adaptive source burn-in complete" << std::endl;
                greyAdaptive.burninCompletePrinted = true;
            }

#ifdef RICH_MPI
            RankStepImbalance const greyStepImbalance =
                ComputeRankStepImbalance("Grey", gen, greyManager->GetCellsStepsCounters(), rank);
            bool const greyDoBurninMeasuredLB =
                greyMeasuredLBActive &&
                cfg.adaptiveSourceCells &&
                greyFirstBurninThisGen &&
                !greyBurninMeasuredLBDone;
            bool const greyDoInitialMeasuredLB =
                (gen == 0 && greyMeasuredLBActive && !cfg.adaptiveSourceCells);
            bool const greyDoPostAdaptiveMeasuredLB =
                greyMeasuredLBActive &&
                cfg.adaptiveSourceCells &&
                !greyBurninThisGen &&
                !greyFirstNonBurninMeasuredLBDone;
            bool const greyDoAdaptivePeriodicMeasuredLB =
                greyMeasuredLBActive &&
                cfg.adaptiveSourceCells &&
                greyFinalThisGen &&
                greyFinalGenerationIndex + 1 < nGreyGens &&
                (greyFinalGenerationIndex + 1) % 50 == 0;
            std::string const greyLBLabel = greyDoBurninMeasuredLB
                ? "MEASURED_LB_GREY_BURNIN"
                : (greyDoPostAdaptiveMeasuredLB
                    ? "MEASURED_LB_GREY_FIRST_NON_BURNIN"
                    : (greyDoAdaptivePeriodicMeasuredLB
                        ? "MEASURED_LB_GREY_ADAPTIVE_PERIODIC"
                        : "MEASURED_LB_GREY"));
            if (rank == 0 && greyDoBurninMeasuredLB)
                std::cout << "Grey first burn-in step complete; running measured LB" << std::endl;
            if (rank == 0 && greyDoPostAdaptiveMeasuredLB)
                std::cout << "Grey first non-burn-in step complete; running measured LB" << std::endl;
            if (rank == 0 && greyDoAdaptivePeriodicMeasuredLB)
                std::cout << "Grey periodic final measured LB after final step "
                          << (greyFinalGenerationIndex + 1)
                          << ": rank_step_imbalance="
                          << greyStepImbalance.maxOverMean
                          << std::endl;
            if (greyDoInitialMeasuredLB || greyDoBurninMeasuredLB ||
                greyDoPostAdaptiveMeasuredLB || greyDoAdaptivePeriodicMeasuredLB) {
                if (!greyParams.noHydroFeedback) {
                    throw UniversalError("Grey measured load balance repartition requires noHydroFeedback=true");
                }

                PrintVmRSS("grey_before_measured_lb", rank);

                std::vector<double> greyWeightsForExchange;

                {
                    auto const& greyLocalSteps = greyManager->GetCellsStepsCounters();

                    std::vector<size_t> greyCellIDs(Ncells);
                    for (size_t i = 0; i < Ncells; ++i)
                        greyCellIDs[i] = cells[i].ID;

                    auto greyLocalMeas = imc_measured_lb::BuildLocalMeasurements(
                        greyCellIDs, greyLocalSteps, greyPhysics->getLastSourcePhotonsPerCell());

                    uint64_t greyLocalTotalSteps = 0;
                    for (auto const& m : greyLocalMeas)
                        greyLocalTotalSteps += static_cast<uint64_t>(m.stepCount);

                    uint64_t greyGlobalTotalSteps = 0;
                    MPI_Allreduce(&greyLocalTotalSteps, &greyGlobalTotalSteps, 1,
                                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

                    if (greyGlobalTotalSteps == 0) {
                        if (rank == 0)
                            std::cerr << greyLBLabel
                                      << ": measured generation had zero total steps, skipping repartition\n";
                    } else {
                        auto greyCostByCellID = imc_measured_lb::BuildMeasuredCosts(
                            greyLocalMeas, greyLBParams, false, MPI_COMM_WORLD);

                        imc_measured_lb::PrintMeasuredLBDiagnosticsDistributed(
                            greyLocalMeas, greyCostByCellID, false, MPI_COMM_WORLD);

                        IMCStepCounterCostCalculator::Parameters greyCostCalcParams;
                        greyCostCalcParams.floorCost = greyLBParams.floorCost;
                        greyCostCalcParams.missingCellCost = greyLBParams.missingCellCost;
                        IMCStepCounterCostCalculator greyCostCalc(std::move(greyCostByCellID), greyCostCalcParams);

                        std::vector<Vector3D> greyCurrentPoints(Ncells);
                        for (size_t i = 0; i < Ncells; ++i)
                            greyCurrentPoints[i] = tess.GetMeshPoint(i);

                        auto greyLBWeights = greyCostCalc.CalculateCost(tess, cells);

                        if (greyLBWeights.size() != Ncells) {
                            throw UniversalError("Grey measured LB weight count mismatch before BuildParallel");
                        }

                        greyWeightsForExchange = greyLBWeights;

                        for (auto& w : greyLBWeights)
                            w = std::pow(w, greyLBWeightCompression);

                        tess.BuildParallel(greyCurrentPoints, greyLBWeights);
                    }
                }

                if (!greyWeightsForExchange.empty()) {
                    MPI_exchange_data(tess, cells, false, 1, &dummyCell);

                    double greyDummyWeight = greyLBParams.missingCellCost;
                    MPI_exchange_data(tess, greyWeightsForExchange, false, 1, &greyDummyWeight);

                    Ncells = tess.GetPointNo();

                    if (cells.size() != Ncells) {
                        UniversalError eo("Cell count mismatch after grey measured LB repartition");
                        eo.addEntry("cells.size()", static_cast<double>(cells.size()));
                        eo.addEntry("Ncells", static_cast<double>(Ncells));
                        throw eo;
                    }

                    if (greyWeightsForExchange.size() != Ncells) {
                        UniversalError eo("Grey measured weight count mismatch after repartition");
                        eo.addEntry("weights.size()", static_cast<double>(greyWeightsForExchange.size()));
                        eo.addEntry("Ncells", static_cast<double>(Ncells));
                        throw eo;
                    }

                    imc_measured_lb::PrintPostRepartitionDiagnosticsFromWeights(
                        greyWeightsForExchange, greyLBWeightCompression,
                        false, MPI_COMM_WORLD);

                    extensives.resize(Ncells);
                    for (size_t i = 0; i < Ncells; ++i)
                        PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);

                    greyObserver->addBoxEscapeEnergy(greyBoundary->getEscapedEnergy());
                    greyBoundary = std::make_shared<VacuumBoundaryCondition<Vector3D, Tessellation3D>>(tess);

                    greyPhysics = std::make_shared<::RadiationIMC>(
                        tess, greyBoundary, cells, extensives, eos, greyOpacity, greyParams);
                    greyPhysics->setObserver(greyObserver);

                    greyPopControl = std::make_shared<NoPopulationControl<Vector3D, Tessellation3D>>(tess);

                    greyManager = std::make_shared<RDMAMonteCarloManager3D>(
                        tess, greyPhysics, greyPopControl, greyBoundary);

                    if (rank == 0)
                        std::cout << greyLBLabel
                                  << ": repartitioned, new local cells=" << Ncells << std::endl;

                    PrintVmRSS("grey_after_rebuild_physics", rank);
                }
                if (greyDoBurninMeasuredLB) {
                    greyBurninMeasuredLBDone = true;
                    if (rank == 0)
                        std::cout << "Grey burn-in measured load balance complete" << std::endl;
                }
                if (greyDoPostAdaptiveMeasuredLB) {
                    greyFirstNonBurninMeasuredLBDone = true;
                    greyAdaptive.postAdaptiveMeasuredLBDone = true;
                    greyAdaptive.adaptiveMeasuredLBCount += 1;
                    greyAdaptive.lastAdaptiveMeasuredLBGeneration = gen;
                    if (rank == 0)
                        std::cout << "Grey post-adaptive measured load balance complete" << std::endl;
                }
                if (greyDoAdaptivePeriodicMeasuredLB) {
                    greyAdaptive.adaptiveMeasuredLBCount += 1;
                    greyAdaptive.lastAdaptiveMeasuredLBGeneration = gen;
                    if (rank == 0)
                        std::cout << "Grey periodic measured load balance complete" << std::endl;
                }
            }
#endif // RICH_MPI
        }
        if (rank == 0)
            greyObserver->loadStatisticalMeanTallies();

        if (rank == 0) {
            std::string greyVtk;
            if (!cfg.vtkOutput.empty()) {
                size_t dotPos = cfg.vtkOutput.rfind('.');
                if (dotPos != std::string::npos)
                    greyVtk = cfg.vtkOutput.substr(0, dotPos) + "_grey" + cfg.vtkOutput.substr(dotPos);
                else
                    greyVtk = cfg.vtkOutput + "_grey";
            }

            if (!greyVtk.empty()) {
                greyObserver->writeVTK(greyVtk, cfg.sourceDt);

                std::vector<double> const& greySolidAngles = greyObserver->getObserverSolidAngles();
                std::ofstream vtkAppend(greyVtk, std::ios::app);
                if (vtkAppend.is_open()) {
                    vtkAppend << std::scientific << std::setprecision(12);

                    double fourPi = 4.0 * M_PI;

                    vtkAppend << "SCALARS fld_surface_luminosity double 1\n"
                              << "LOOKUP_TABLE default\n";
                    for (size_t p = 0; p < nObs; ++p)
                        vtkAppend << fldLuminosity[p] << "\n";

                    vtkAppend << "SCALARS fld_surface_isotropic_equivalent_luminosity double 1\n"
                              << "LOOKUP_TABLE default\n";
                    for (size_t p = 0; p < nObs; ++p) {
                        double isoEquiv = (greySolidAngles[p] > 0.0)
                            ? fldLuminosity[p] * fourPi / greySolidAngles[p] : 0.0;
                        vtkAppend << isoEquiv << "\n";
                    }

                    vtkAppend << "SCALARS log10_fld_surface_luminosity double 1\n"
                              << "LOOKUP_TABLE default\n";
                    for (size_t p = 0; p < nObs; ++p) {
                        double val = (fldLuminosity[p] > 0.0) ? std::log10(fldLuminosity[p]) : -99.0;
                        vtkAppend << val << "\n";
                    }

                    vtkAppend << "SCALARS fld_surface_flux double 1\n"
                              << "LOOKUP_TABLE default\n";
                    for (size_t p = 0; p < nObs; ++p) {
                        double patchArea_p = greySolidAngles[p] * cfg.radius * cfg.radius;
                        double flux = (patchArea_p > 0.0) ? fldLuminosity[p] / patchArea_p : 0.0;
                        vtkAppend << flux << "\n";
                    }
                    AppendZeroVtkScalar(vtkAppend, "fld_surface_luminosity_stderr_gen", nObs);
                    AppendZeroVtkScalar(vtkAppend, "fld_surface_luminosity_relerr_gen", nObs);
                    AppendZeroVtkScalar(vtkAppend, "fld_surface_luminosity_stderr_packet", nObs);
                    AppendZeroVtkScalar(vtkAppend, "fld_surface_luminosity_relerr_packet", nObs);
                    AppendZeroVtkScalar(vtkAppend, "fld_surface_luminosity_neff", nObs);
                    AppendZeroVtkScalar(vtkAppend, "fld_surface_isotropic_equivalent_luminosity_stderr_gen", nObs);
                    AppendZeroVtkScalar(vtkAppend, "fld_surface_isotropic_equivalent_luminosity_relerr_gen", nObs);
                    AppendZeroVtkScalar(vtkAppend, "fld_surface_flux_stderr_gen", nObs);
                    AppendZeroVtkScalar(vtkAppend, "fld_surface_flux_relerr_gen", nObs);
                    AppendZeroVtkScalar(vtkAppend, "log10_fld_surface_luminosity_stderr_gen", nObs);
                    AppendZeroVtkScalar(vtkAppend, "log10_fld_surface_luminosity_relerr_gen", nObs);
                }
            }

            double greyTotalLum = greyObserver->getTotalCrossingEnergy() / cfg.sourceDt;
            double greyEmitted = greyObserver->getEmittedEnergy();
            double greyAbsorbed = greyObserver->getAbsorbedEnergy();
            double greyBoxEscape = greyObserver->getBoxEscapeEnergy();
            double greyTimedOut = greyObserver->getTimedOutEnergy();
            double greyCutoff = greyObserver->getCutoffEnergy();
            double greyResidual = greyEmitted - greyAbsorbed - greyBoxEscape - greyTimedOut - greyCutoff;
            double greyTimedOutFrac = (greyEmitted > 0.0) ? greyTimedOut / greyEmitted : 0.0;

            std::cout << "\n=== Grey IMC Results ===\n"
                      << "Generations:              " << nGreyGens << "\n"
                      << "Final included generations: " << greyIncludedFinalGenerations << "\n"
                      << "Discarded burn-in generations: " << greyDiscardedBurninGenerations << "\n";
            if (cfg.adaptiveSourceCells)
                std::cout << "Schedule burn-in/probe/final: " << greyBurninGenerations
                          << "/" << greyLearnedProbeGenerations
                          << "/" << nGreyGens << "\n";
            std::cout << "Final average policy:     " << (cfg.adaptiveSourceCells ? "adaptive_only" : "all_generations") << "\n"
                      << "Photons/cell/gen:         " << greyPhotonsPerCell << "\n"
                      << "Total crossing luminosity: " << greyTotalLum << " +/- "
                      << greyObserver->getTotalLuminosityStderrGen(cfg.sourceDt)
                      << " erg/s (rel=" << greyObserver->getTotalLuminosityRelErrGen(cfg.sourceDt) << ")\n"
                      << "Total FLD luminosity:     " << totalFldLum << " erg/s\n"
                      << "Emitted energy:           " << greyEmitted << " erg\n"
                      << "Absorbed energy:          " << greyAbsorbed << " erg\n"
                      << "Box escape energy:        " << greyBoxEscape << " erg\n"
                      << "Timed-out energy:         " << greyTimedOut << " erg\n"
                      << "Cutoff energy:            " << greyCutoff << " erg\n"
                      << "Sink residual:            " << greyResidual << " erg\n"
                      << "Timed-out fraction:       " << greyTimedOutFrac << "\n"
                      << "Grey VTK written to:      " << greyVtk << "\n"
                      << std::endl;
        }

    } catch (UniversalError const& eo) {
        std::cerr << "UniversalError on rank " << rank << ":\n";
        reportError(eo, std::cerr);
#ifdef RICH_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    } catch (std::exception const& e) {
        std::cerr << "Exception on rank " << rank << ": " << e.what() << "\n";
#ifdef RICH_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }

#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return 0;
}
