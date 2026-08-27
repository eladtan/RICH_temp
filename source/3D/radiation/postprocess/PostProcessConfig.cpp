#include "PostProcessConfig.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include "CMMC/src/units/units.hpp"

namespace imc_postprocess_tde {

#if 0
void printLegacyUsage(int rank)
{
    if (rank != 0) return;
    std::cerr << "Usage: rich [options]\n"
              << "Options:\n"
              << "  --input PATH             Input HDF5 snapshot (default: TDE snap_full_136.h5)\n"
              << "  --output PATH            Output HDF5 file (default: tde_postprocess_output.h5)\n"
              << "  --vtk-output PATH        Base VTK surface map path (default: output path with .vtk)\n"
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
              << "  --no-photosphere         Disable observer photosphere postprocessing\n"
              << "  --flux-source-compare    Run MG and grey transport from one FLD-normalized CER source\n"
              << "  --flux-source-tau TAU    Grey effective optical depth of CER (default: 5)\n"
              << "  --flux-source-rays N     Inward rays used only to construct the CER\n"
              << "                           (default: use --n-observers)\n"
              << "  --flux-source-ddmc-face-tau TAU\n"
              << "                           Minimum DDMC optical depth from exterior cell center to CER face (default: 5)\n"
              << "                           Requires MG, positive snapshot Erad, and non-Compton transport\n"
              << "                           RW is disabled; DDMC uses native CER thermalization with face-local IMC fallback\n"
              << "  --polarization           Enable postprocess linear polarization\n"
              << "  --no-polarization        Disable postprocess linear polarization\n"
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
              << "  --adaptive-source-learned-min-photons N  Min photons/cell for learned cells (default: 200)\n"
              << "  --adaptive-source-learned-max-photons N  Max photons/cell for top-scoring learned cells (default: 5000)\n"
              << "  --adaptive-source-score-power F          Score shaping power for photon allocation (default: 2)\n"
              << "  --adaptive-source-weight-score-frac F    Weight-squared fraction in learned score (default: 1)\n"
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
              << "  --adaptive-diagnostics-verbose\n";
}
#endif

std::string ReplaceExtension(std::string const& path, std::string const& newExt)
{
    if (path.empty())
        return "output" + newExt;
    size_t const slash = path.find_last_of("/\\");
    size_t const dot = path.find_last_of('.');
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash))
        return path.substr(0, dot) + newExt;
    return path + newExt;
}

std::string InsertSuffixBeforeExtension(
    std::string const& path, std::string const& suffix)
{
    size_t const slash = path.find_last_of("/\\");
    size_t const dot = path.find_last_of('.');
    if (dot != std::string::npos && (slash == std::string::npos || dot > slash))
        return path.substr(0, dot) + suffix + path.substr(dot);
    return path + suffix;
}

std::string BaseVtkOutputPath(Config const& cfg)
{
    if (!cfg.writeVtk)
        return std::string();
    if (!cfg.vtkOutput.empty())
        return cfg.vtkOutput;
    return ReplaceExtension(cfg.outputPath, ".vtk");
}

std::string GreyVtkOutputPath(Config const& cfg)
{
    return InsertSuffixBeforeExtension(BaseVtkOutputPath(cfg), "_grey");
}

bool ValidateConfig(Config &cfg, int rank)
{
    if (cfg.inputPath.empty()) {
        if (rank == 0) std::cerr << "--input.snapshot is required\n";
        return false;
    }
    if (cfg.opacityDir.empty()) {
        if (rank == 0)
            std::cerr << "--input.multigroup-opacity-directory is required\n";
        return false;
    }
    if (cfg.eosDir.empty()) {
        if (rank == 0) std::cerr << "--input.eos-directory is required\n";
        return false;
    }
    if (cfg.nGenerations == 0) {
        if (rank == 0)
            std::cerr << "--transport.generations must be positive\n";
        return false;
    }
    if (cfg.adaptiveSourceCells && cfg.adaptiveSourceBurnin < 2) {
        if (rank == 0)
            std::cerr << "--adaptive.source.burnin-generations must be at least 2 when adaptive source sampling is enabled\n";
        return false;
    }
#if 0
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
        else if (arg == "--no-photosphere") { cfg.photosphere = false; }
        else if (arg == "--flux-source-compare") { cfg.fluxSourceCompare = true; }
        else if (arg == "--flux-source-tau" && i + 1 < argc) { cfg.fluxSourceThermalizationTau = std::atof(argv[++i]); }
        else if (arg == "--flux-source-rays" && i + 1 < argc) { cfg.fluxSourceRays = static_cast<size_t>(std::atoll(argv[++i])); }
        else if (arg == "--flux-source-ddmc-face-tau" && i + 1 < argc) { cfg.fluxSourceDDMCFaceOpticalDepth = std::atof(argv[++i]); }
        else if (arg == "--polarization") { cfg.polarization = true; }
        else if (arg == "--no-polarization") { cfg.polarization = false; }
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
        else if (arg == "--adaptive-source-learned-min-photons" && i + 1 < argc) { cfg.adaptiveSourceLearnedMinPhotons = static_cast<size_t>(std::atoll(argv[++i])); }
        else if (arg == "--adaptive-source-learned-max-photons" && i + 1 < argc) { cfg.adaptiveSourceLearnedMaxPhotons = static_cast<size_t>(std::atoll(argv[++i])); }
        else if (arg == "--adaptive-source-score-power" && i + 1 < argc) { cfg.adaptiveSourceScorePower = std::atof(argv[++i]); }
        else if (arg == "--adaptive-source-weight-score-frac" && i + 1 < argc) { cfg.adaptiveSourceWeightScoreFrac = std::atof(argv[++i]); }
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
        else { if (rank == 0) std::cerr << "Unknown argument: " << arg << "\n"; return false; }
    }

#endif
    if (cfg.radius <= 0.0) { if (rank == 0) std::cerr << "--observer.radius must be positive\n"; return false; }
    if (cfg.nObservers == 0) { if (rank == 0) std::cerr << "--n-observers must be > 0\n"; return false; }
    if (cfg.fluxSourceRays > static_cast<size_t>(INT_MAX)) {
        if (rank == 0)
            std::cerr << "--flux-source-rays must be 0 or at most INT_MAX\n";
        return false;
    }
    if (cfg.sourceDt <= 0.0) { if (rank == 0) std::cerr << "--source-dt must be positive\n"; return false; }
    if (cfg.photonsPerCell == 0) { if (rank == 0) std::cerr << "--photons-per-cell must be > 0\n"; return false; }
    if (cfg.nGenerations == 0) { if (rank == 0) std::cerr << "--n-generations must be >= 1\n"; return false; }
    if (!(cfg.fluxSourceThermalizationTau > 0.0) ||
        !std::isfinite(cfg.fluxSourceThermalizationTau)) {
        if (rank == 0) std::cerr << "--flux-source-tau must be finite and positive\n";
        return false;
    }
    if (!(cfg.fluxSourceDDMCFaceOpticalDepth > 0.0) ||
        !std::isfinite(cfg.fluxSourceDDMCFaceOpticalDepth)) {
        if (rank == 0)
            std::cerr << "--flux-source-ddmc-face-tau must be finite and positive\n";
        return false;
    }
    if (cfg.fluxSourceCompare && cfg.compton) {
        if (rank == 0) std::cerr << "--flux-source-compare does not support --compton yet\n";
        return false;
    }
#if ENERGY_GROUPS_NUM <= 1
    if (cfg.fluxSourceCompare) {
        if (rank == 0) std::cerr << "--flux-source-compare requires ENERGY_GROUPS_NUM > 1\n";
        return false;
    }
#endif
    if (cfg.adaptiveSourceStrength < 0.0 || cfg.adaptiveSourceStrength > 1.0 || !std::isfinite(cfg.adaptiveSourceStrength)) { if (rank == 0) std::cerr << "--adaptive-source-strength must be finite in [0,1]\n"; return false; }
    if (cfg.adaptiveSourceEma <= 0.0 || cfg.adaptiveSourceEma > 1.0 || !std::isfinite(cfg.adaptiveSourceEma)) { if (rank == 0) std::cerr << "--adaptive-source-ema must be finite in (0,1]\n"; return false; }
    if (cfg.adaptiveSourceMinEscapedFrac < 0.0 || !std::isfinite(cfg.adaptiveSourceMinEscapedFrac)) { if (rank == 0) std::cerr << "--adaptive-source-min-escaped-frac must be finite and nonnegative\n"; return false; }
    if (cfg.adaptiveSourceMaxFactor < 1.0 || !std::isfinite(cfg.adaptiveSourceMaxFactor)) { if (rank == 0) std::cerr << "--adaptive-source-max-factor must be finite and >= 1\n"; return false; }
    if (cfg.adaptiveSourceLearnedReserveFrac < 0.0 || cfg.adaptiveSourceLearnedReserveFrac > 1.0 || !std::isfinite(cfg.adaptiveSourceLearnedReserveFrac)) { if (rank == 0) std::cerr << "--adaptive-source-learned-reserve-frac must be finite in [0,1]\n"; return false; }
    if (cfg.adaptiveSourceLearnedMinFactor < 1.0 || !std::isfinite(cfg.adaptiveSourceLearnedMinFactor)) { if (rank == 0) std::cerr << "--adaptive-source-learned-min-factor must be finite and >= 1\n"; return false; }
    if (cfg.adaptiveSourceLearnedMinPhotons == 0) { if (rank == 0) std::cerr << "--adaptive-source-learned-min-photons must be > 0\n"; return false; }
    if (cfg.adaptiveSourceLearnedMaxPhotons <= cfg.adaptiveSourceLearnedMinPhotons) { if (rank == 0) std::cerr << "--adaptive-source-learned-max-photons must be > --adaptive-source-learned-min-photons\n"; return false; }
    if (cfg.adaptiveSourceScorePower < 0.0 || !std::isfinite(cfg.adaptiveSourceScorePower)) { if (rank == 0) std::cerr << "--adaptive-source-score-power must be finite and >= 0\n"; return false; }
    if (cfg.adaptiveSourceWeightScoreFrac < 0.0 || cfg.adaptiveSourceWeightScoreFrac > 1.0 || !std::isfinite(cfg.adaptiveSourceWeightScoreFrac)) { if (rank == 0) std::cerr << "--adaptive-source-weight-score-frac must be finite in [0,1]\n"; return false; }
    if (cfg.adaptiveObserverExtraBudgetFrac < 0.0 || !std::isfinite(cfg.adaptiveObserverExtraBudgetFrac)) { if (rank == 0) std::cerr << "--adaptive-observer-extra-budget-frac must be finite and nonnegative\n"; return false; }
    if (cfg.adaptiveObserverTargetNeff <= 0.0 || !std::isfinite(cfg.adaptiveObserverTargetNeff)) { if (rank == 0) std::cerr << "--adaptive-observer-target-neff must be finite and positive\n"; return false; }
    if (cfg.adaptiveObserverTargetPolSnr <= 0.0 || !std::isfinite(cfg.adaptiveObserverTargetPolSnr)) { if (rank == 0) std::cerr << "--adaptive-observer-target-pol-snr must be finite and positive\n"; return false; }
    if (cfg.adaptiveObserverDeficitMax < 1.0 || !std::isfinite(cfg.adaptiveObserverDeficitMax)) { if (rank == 0) std::cerr << "--adaptive-observer-deficit-max must be finite and >= 1\n"; return false; }
    if (cfg.adaptiveObserverDeficitEma <= 0.0 || cfg.adaptiveObserverDeficitEma > 1.0 || !std::isfinite(cfg.adaptiveObserverDeficitEma)) { if (rank == 0) std::cerr << "--adaptive-observer-deficit-ema must be finite in (0,1]\n"; return false; }
    if (cfg.measuredLBWeightCompression != -1.0 && (cfg.measuredLBWeightCompression <= 0.0 || !std::isfinite(cfg.measuredLBWeightCompression))) { if (rank == 0) std::cerr << "--measured-lb-weight-compression must be finite and > 0\n"; return false; }
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


} // namespace imc_postprocess_tde
