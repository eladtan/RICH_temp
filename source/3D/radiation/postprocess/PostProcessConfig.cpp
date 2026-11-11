#include "PostProcessConfig.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <filesystem>
#include <system_error>
#include <iostream>

#include "CMMC/src/units/units.hpp"

namespace imc_postprocess_tde {


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
    // Empty when VTK output is off, like BaseVtkOutputPath (a bare "_grey"
    // would be written to the working directory).
    std::string const base = BaseVtkOutputPath(cfg);
    return base.empty() ? base : InsertSuffixBeforeExtension(base, "_grey");
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
    if (cfg.radius <= 0.0) { if (rank == 0) std::cerr << "--observer.radius must be positive\n"; return false; }
    if (cfg.nObservers == 0) { if (rank == 0) std::cerr << "--n-observers must be > 0\n"; return false; }
    if (cfg.fluxSourceRays > static_cast<size_t>(INT_MAX)) {
        if (rank == 0)
            std::cerr << "--flux-source-rays must be 0 or at most INT_MAX\n";
        return false;
    }
    if (cfg.sourceDt <= 0.0) { if (rank == 0) std::cerr << "--source-dt must be positive\n"; return false; }
    if (cfg.photonsPerCell == 0) { if (rank == 0) std::cerr << "--photons-per-cell must be > 0\n"; return false; }
    if (cfg.nGenerations == 0 && cfg.restartFrom.empty()) { if (rank == 0) std::cerr << "--transport.generations must be >= 1 (0 is allowed only with --restart.from)\n"; return false; }
    if (!cfg.restartFrom.empty()) {
        std::error_code ec;
        std::filesystem::path const source =
            std::filesystem::weakly_canonical(cfg.restartFrom, ec);
        if (ec || !std::filesystem::is_regular_file(source)) {
            if (rank == 0) std::cerr << "--restart.from file not found: " << cfg.restartFrom << "\n";
            return false;
        }
        std::filesystem::path const output =
            std::filesystem::weakly_canonical(cfg.outputPath, ec);
        if (!ec && output == source && !cfg.restartInPlace) {
            if (rank == 0)
                std::cerr << "--output.stem names the --restart.from file " << cfg.restartFrom
                          << "; pass --restart.in-place true to replace it, or choose another stem\n";
            return false;
        }
    }
    if (!(cfg.fluxSourceThermalizationTau > 0.0) ||
        !std::isfinite(cfg.fluxSourceThermalizationTau)) {
        if (rank == 0) std::cerr << "--flux-source-tau must be finite and positive\n";
        return false;
    }
    if (!(cfg.splitObserverFraction >= 0.0) || !std::isfinite(cfg.splitObserverFraction)) {
        if (rank == 0) std::cerr << "--transport.split-observer-fraction must be finite and >= 0 (0 = off)\n";
        return false;
    }
    if (!(cfg.splitEscapeReference > 0.0) || cfg.splitEscapeReference > 1.0) {
        if (rank == 0) std::cerr << "--transport.split-escape-reference must lie in (0, 1]\n";
        return false;
    }
#ifdef STORM_WITH_GPU
    if (cfg.splitObserverFraction > 0.0) {
        // Device-transported packets do not pass the host step that splits.
        if (rank == 0) std::cerr << "--transport.split-observer-fraction is not supported with STORM GPU transport\n";
        return false;
    }
#endif
    if (cfg.splitObserverFraction > 0.0 && (!cfg.adaptiveSourceCells || !cfg.fluxSourceCompare)) {
        // The importance is the learned escaped energy over the emitted
        // energy of the flux-source and volume emission.
        if (rank == 0) std::cerr << "--transport.split-observer-fraction requires adaptive.source.enabled and flux-source.enabled\n";
        return false;
    }
    if (cfg.splitObserverFraction > 0.0 && cfg.splitCopyBudget < 1) {
        if (rank == 0) std::cerr << "--transport.split-copy-budget must be >= 1\n";
        return false;
    }
    if (cfg.splitMaxCopies < 2) {
        if (rank == 0) std::cerr << "--transport.split-max-copies must be >= 2\n";
        return false;
    }
    if (!(cfg.splitGroupScale > 0.0) || !(cfg.splitGroupScale <= 1.0)) {
        if (rank == 0) std::cerr << "--transport.split-group-scale must lie in (0, 1]\n";
        return false;
    }
    if (!(cfg.splitGroupFloor > 0.0) || !(cfg.splitGroupFloor <= 1.0)) {
        if (rank == 0) std::cerr << "--transport.split-group-floor must lie in (0, 1]\n";
        return false;
    }
    if (!(cfg.splitImportanceFloor >= 0.0) || !(cfg.splitImportanceFloor <= 1.0)) {
        if (rank == 0) std::cerr << "--transport.split-importance-floor must lie in [0, 1]\n";
        return false;
    }
    if (cfg.splitImportanceFloor > 0.0 && !cfg.splitPerGroup) {
        if (rank == 0) std::cerr << "--transport.split-importance-floor requires --transport.split-per-group true\n";
        return false;
    }
    if ((cfg.splitPerGroup || cfg.packetProvenance) && !(cfg.splitObserverFraction > 0.0)) {
        // Both are built on the scalar window and its cell importance.
        if (rank == 0)
            std::cerr << "--transport.split-per-group and --diagnostics.packet-provenance require "
                         "--transport.split-observer-fraction > 0\n";
        return false;
    }
#if ENERGY_GROUPS_NUM <= 1 || ENERGY_GROUPS_NUM > 254
    if (cfg.splitPerGroup || cfg.packetProvenance) {
        if (rank == 0)
            std::cerr << "--transport.split-per-group and --diagnostics.packet-provenance require "
                         "1 < ENERGY_GROUPS_NUM <= 254\n";
        return false;
    }
#endif
    if (!(cfg.ddmcMaxCellAdvectionPeclet > 0.0)) {
        if (rank == 0)
            std::cerr << "--transport.ddmc-max-cell-advection-peclet must be positive\n";
        return false;
    }
    if (!(cfg.ddmcCompressionMaxDynamicalTimes >= 0.0)) {
        if (rank == 0)
            std::cerr << "--transport.ddmc-compression-max-dynamical-times must be >= 0 or inf\n";
        return false;
    }
#ifdef STORM_WITH_GPU
    if (std::isfinite(cfg.ddmcCompressionMaxDynamicalTimes) && cfg.ddmc && cfg.useCellVelocities) {
        if (rank == 0)
            std::cerr << "--transport.ddmc-compression-max-dynamical-times is not supported with STORM GPU transport (use inf)\n";
        return false;
    }
#endif
    if (!(cfg.fluxSourceDDMCFaceOpticalDepth > 0.0) ||
        !std::isfinite(cfg.fluxSourceDDMCFaceOpticalDepth)) {
        if (rank == 0)
            std::cerr << "--flux-source-ddmc-face-tau must be finite and positive\n";
        return false;
    }
    if (!(cfg.measuredLBClampFactor == 0.0 || cfg.measuredLBClampFactor >= 1.0) ||
        !std::isfinite(cfg.measuredLBClampFactor)) {
        if (rank == 0) std::cerr << "--load-balance.clamp-factor must be 0 (built-in) or >= 1\n";
        return false;
    }
    if (!(cfg.weightCutoffFraction > 0.0) || !(cfg.weightCutoffFraction < 1.0)) {
        if (rank == 0) std::cerr << "--transport.weight-cutoff-fraction must lie in (0, 1)\n";
        return false;
    }
    if (!(cfg.volumeEmissionCoolingLimitCfl > 0.0) || !(cfg.volumeEmissionCoolingLimitCfl < 1.0)) {
        if (rank == 0) std::cerr << "--volume-emission.cooling-limit-cfl must lie in (0, 1)\n";
        return false;
    }
    if (!(cfg.volumeEmissionCoolingLimitDt >= 0.0) || !std::isfinite(cfg.volumeEmissionCoolingLimitDt)) {
        if (rank == 0) std::cerr << "--volume-emission.cooling-limit-dt must be finite and >= 0 (0 = Courant step)\n";
        return false;
    }
    if (!(cfg.fluxSourceSmoothingFloor >= 0.0) || cfg.fluxSourceSmoothingFloor > 1.0) {
        if (rank == 0) std::cerr << "--flux-source.smoothing-floor must lie in [0, 1]\n";
        return false;
    }
    if (!(cfg.fluxSourceMappingMaxDepthFactor == 0.0 || cfg.fluxSourceMappingMaxDepthFactor >= 1.0) ||
        !std::isfinite(cfg.fluxSourceMappingMaxDepthFactor)) {
        if (rank == 0)
            std::cerr << "--flux-source.mapping-max-depth-factor must be 0 (no floor) or >= 1 "
                         "(the floor is this many times the thermalization tau)\n";
        return false;
    }
    if (cfg.fluxSourceMappingNeighbours > 64) {
        if (rank == 0)
            std::cerr << "--flux-source.mapping-neighbours must be at most 64 (6 is one ring of "
                         "neighbouring rays; many more lowers every ray to the minimum over a wide cone)\n";
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
    if (cfg.adaptiveSourceLearnedMinPhotons < 2) { if (rank == 0) std::cerr << "--adaptive-source-learned-min-photons must be >= 2\n"; return false; }
    if (cfg.adaptiveSourceLearnedMaxPhotons <= cfg.adaptiveSourceLearnedMinPhotons) { if (rank == 0) std::cerr << "--adaptive-source-learned-max-photons must be > --adaptive-source-learned-min-photons\n"; return false; }
    if (cfg.adaptiveSourceLearnedPhotonsPerCellBudget < cfg.adaptiveSourceLearnedMinPhotons || cfg.adaptiveSourceLearnedPhotonsPerCellBudget > cfg.adaptiveSourceLearnedMaxPhotons) { if (rank == 0) std::cerr << "--adaptive-source-learned-photons-per-cell-budget must lie within [learned-min-photons, learned-max-photons]\n"; return false; }
    if (cfg.adaptiveSourceScorePower < 0.0 || !std::isfinite(cfg.adaptiveSourceScorePower)) { if (rank == 0) std::cerr << "--adaptive-source-score-power must be finite and >= 0\n"; return false; }
    if (!(cfg.adaptiveSourceRampStartPhotons >= 0.0) || !std::isfinite(cfg.adaptiveSourceRampStartPhotons)) { if (rank == 0) std::cerr << "--adaptive.source.ramp-start-photons must be finite and >= 0 (0 disables the ramp)\n"; return false; }
    if (cfg.adaptiveSourceRampStartPhotons > 0.0 && (!(cfg.adaptiveSourceRampFactor >= 1.1) || !std::isfinite(cfg.adaptiveSourceRampFactor))) { if (rank == 0) std::cerr << "--adaptive.source.ramp-factor must be finite and >= 1.1 (2 doubles the budget every ramp generation)\n"; return false; }
    if (cfg.adaptiveSourceWeightScoreFrac < 0.0 || cfg.adaptiveSourceWeightScoreFrac > 1.0 || !std::isfinite(cfg.adaptiveSourceWeightScoreFrac)) { if (rank == 0) std::cerr << "--adaptive-source-weight-score-frac must be finite in [0,1]\n"; return false; }
    if (cfg.adaptiveObserverExtraBudgetFrac < 0.0 || !std::isfinite(cfg.adaptiveObserverExtraBudgetFrac)) { if (rank == 0) std::cerr << "--adaptive-observer-extra-budget-frac must be finite and nonnegative\n"; return false; }
    if (cfg.adaptiveObserverTargetNeff <= 0.0 || !std::isfinite(cfg.adaptiveObserverTargetNeff)) { if (rank == 0) std::cerr << "--adaptive-observer-target-neff must be finite and positive\n"; return false; }
    if (cfg.adaptiveObserverTargetPolSnr <= 0.0 || !std::isfinite(cfg.adaptiveObserverTargetPolSnr)) { if (rank == 0) std::cerr << "--adaptive-observer-target-pol-snr must be finite and positive\n"; return false; }
    if (cfg.adaptiveObserverTargetPolSigma < 0.0 || !std::isfinite(cfg.adaptiveObserverTargetPolSigma)) { if (rank == 0) std::cerr << "--adaptive-observer-target-pol-sigma must be finite and >= 0\n"; return false; }
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
    }

    return true;
}


} // namespace imc_postprocess_tde
