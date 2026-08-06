#include "PostProcessCommandLine.hpp"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace PostProcessIMC
{
namespace
{

struct Option
{
    std::string name;
    std::string type;
    std::string help;
    std::function<bool(std::string const&, std::string&)> set;
    std::function<std::string()> value;
};

bool ParseBool(std::string const& text, bool& value)
{
    if (text == "true") {
        value = true;
        return true;
    }
    if (text == "false") {
        value = false;
        return true;
    }
    return false;
}

bool ParseDouble(std::string const& text, double& value)
{
    errno = 0;
    char* end = nullptr;
    double const parsed = std::strtod(text.c_str(), &end);
    if (errno == ERANGE || end == text.c_str() || *end != '\0' ||
        !std::isfinite(parsed))
        return false;
    value = parsed;
    return true;
}

bool ParseSize(std::string const& text, size_t& value)
{
    if (text.empty() || text[0] == '-')
        return false;
    errno = 0;
    char* end = nullptr;
    unsigned long long const parsed = std::strtoull(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || *end != '\0' ||
        parsed > std::numeric_limits<size_t>::max())
        return false;
    value = static_cast<size_t>(parsed);
    return true;
}

bool ParseInt(std::string const& text, int& value)
{
    errno = 0;
    char* end = nullptr;
    long const parsed = std::strtol(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || *end != '\0' ||
        parsed < std::numeric_limits<int>::min() ||
        parsed > std::numeric_limits<int>::max())
        return false;
    value = static_cast<int>(parsed);
    return true;
}

std::string FormatDouble(double value)
{
    std::ostringstream stream;
    stream << std::setprecision(17) << value;
    return stream.str();
}

void AddString(std::vector<Option>& options, std::string name,
               std::string help, std::string& target)
{
    options.push_back(Option{
        std::move(name), "string", std::move(help),
        [&target](std::string const& text, std::string&) {
            target = text;
            return true;
        },
        [&target]() { return target; }});
}

void AddBool(std::vector<Option>& options, std::string name,
             std::string help, bool& target)
{
    options.push_back(Option{
        std::move(name), "bool", std::move(help),
        [&target](std::string const& text, std::string& error) {
            if (!ParseBool(text, target)) {
                error = "expected true or false";
                return false;
            }
            return true;
        },
        [&target]() { return target ? "true" : "false"; }});
}

void AddDouble(std::vector<Option>& options, std::string name,
               std::string help, double& target)
{
    options.push_back(Option{
        std::move(name), "number", std::move(help),
        [&target](std::string const& text, std::string& error) {
            if (!ParseDouble(text, target)) {
                error = "expected a finite number";
                return false;
            }
            return true;
        },
        [&target]() { return FormatDouble(target); }});
}

void AddSize(std::vector<Option>& options, std::string name,
             std::string help, size_t& target)
{
    options.push_back(Option{
        std::move(name), "integer", std::move(help),
        [&target](std::string const& text, std::string& error) {
            if (!ParseSize(text, target)) {
                error = "expected a non-negative integer";
                return false;
            }
            return true;
        },
        [&target]() { return std::to_string(target); }});
}

void AddInt(std::vector<Option>& options, std::string name,
            std::string help, int& target)
{
    options.push_back(Option{
        std::move(name), "integer", std::move(help),
        [&target](std::string const& text, std::string& error) {
            if (!ParseInt(text, target)) {
                error = "expected an integer";
                return false;
            }
            return true;
        },
        [&target]() { return std::to_string(target); }});
}

void AddOpacityMode(std::vector<Option>& options, OpacityScaleMode& target)
{
    options.push_back(Option{
        "opacity-scaling.mode", "none|rosseland|planck",
        "Multigroup absorption normalization.",
        [&target](std::string const& text, std::string& error) {
            if (text == "none")
                target = OpacityScaleMode::None;
            else if (text == "rosseland")
                target = OpacityScaleMode::Rosseland;
            else if (text == "planck")
                target = OpacityScaleMode::Planck;
            else {
                error = "expected none, rosseland, or planck";
                return false;
            }
            return true;
        },
        [&target]() {
            if (target == OpacityScaleMode::None)
                return std::string("none");
            if (target == OpacityScaleMode::Rosseland)
                return std::string("rosseland");
            return std::string("planck");
        }});
}

std::vector<Option> MakeRegistry(PostProcessConfig& c)
{
    std::vector<Option> options;
    options.reserve(72);

    AddString(options, "input.snapshot", "Snapshot3D HDF5 input.", c.input.snapshot);
    AddString(options, "input.multigroup-opacity-directory", "STA multigroup opacity tables.", c.input.multigroupOpacityDirectory);
    AddString(options, "input.grey-opacity-directory", "STA grey opacity tables.", c.input.greyOpacityDirectory);
    AddString(options, "input.eos-directory", "Equation-of-state tables.", c.input.eosDirectory);
    AddString(options, "output.stem", "Output path without .h5 or .vtk.", c.output.stem);
    AddBool(options, "output.write-vtk", "Write the observer VTK map.", c.output.writeVtk);

    AddDouble(options, "observer.center-x", "Observer sphere center x [cm].", c.observer.center.x);
    AddDouble(options, "observer.center-y", "Observer sphere center y [cm].", c.observer.center.y);
    AddDouble(options, "observer.center-z", "Observer sphere center z [cm].", c.observer.center.z);
    AddDouble(options, "observer.radius", "Observer sphere radius [cm].", c.observer.radius);
    AddSize(options, "observer.count", "Number of observer directions.", c.observer.count);

    AddDouble(options, "transport.source-dt", "Source emission interval [s].", c.transport.sourceDt);
    AddDouble(options, "transport.duration", "Transport duration [s], or zero for the snapshot default.", c.transport.duration);
    AddSize(options, "transport.photons-per-cell", "Base packet count per source cell.", c.transport.photonsPerCell);
    AddSize(options, "transport.generations", "Number of learning/transport generations.", c.transport.generations);
    AddBool(options, "transport.ddmc", "Enable DDMC acceleration.", c.transport.ddmc);
    AddBool(options, "transport.random-walk", "Enable random-walk acceleration.", c.transport.randomWalk);
    AddBool(options, "transport.use-cell-velocities", "Include snapshot cell velocities.", c.transport.useCellVelocities);
    AddBool(options, "transport.compton.enabled", "Enable Compton transport.", c.transport.compton.enabled);
    AddSize(options, "transport.compton.matrix-samples", "Compton redistribution samples.", c.transport.compton.matrixSamples);
    AddBool(options, "transport.compton.angle-dependent", "Use angle-dependent Compton scattering.", c.transport.compton.angleDependent);

    AddBool(options, "polarization.enabled", "Track Stokes polarization.", c.polarization.enabled);
    AddInt(options, "polarization.manual-scatterings-after-acceleration", "Explicit scatterings after acceleration.", c.polarization.manualScatteringsAfterAcceleration);
    AddDouble(options, "polarization.depolarization-scatterings", "Depolarization scattering scale.", c.polarization.depolarizationScatterings);
    AddString(options, "polarization.accelerated-closure", "Accelerated-polarization closure name.", c.polarization.acceleratedClosure);
    AddBool(options, "photosphere.enabled", "Calculate observer photospheres.", c.photosphere.enabled);

    AddBool(options, "flux-source.enabled", "Compare against an FLD boundary source.", c.fluxSource.enabled);
    AddDouble(options, "flux-source.thermalization-tau", "Flux-source thermalization optical depth.", c.fluxSource.thermalizationTau);
    AddSize(options, "flux-source.construction-rays", "Rays used to construct the source surface.", c.fluxSource.constructionRays);
    AddDouble(options, "flux-source.ddmc-face-optical-depth", "DDMC face optical-depth threshold.", c.fluxSource.ddmcFaceOpticalDepth);
    AddOpacityMode(options, c.opacityScaling.mode);

    AddBool(options, "adaptive.source.enabled", "Learn source-cell allocation.", c.adaptive.source.enabled);
    AddSize(options, "adaptive.source.burnin-generations", "Compatibility burn-in generation count.", c.adaptive.source.burninGenerations);
    AddDouble(options, "adaptive.source.strength", "Learned-score allocation fraction.", c.adaptive.source.strength);
    AddDouble(options, "adaptive.source.ema", "Source-score exponential averaging factor.", c.adaptive.source.ema);
    AddDouble(options, "adaptive.source.min-escaped-fraction", "Minimum learned escaped-energy fraction.", c.adaptive.source.minEscapedFraction);
    AddDouble(options, "adaptive.source.max-factor", "Maximum per-cell packet boost.", c.adaptive.source.maxFactor);
    AddSize(options, "adaptive.source.burnin-photon-multiplier", "Compatibility burn-in packet multiplier.", c.adaptive.source.burninPhotonMultiplier);
    AddDouble(options, "adaptive.source.learned-reserve-fraction", "Packet fraction reserved for learned cells.", c.adaptive.source.learnedReserveFraction);
    AddDouble(options, "adaptive.source.learned-min-factor", "Minimum learned-cell boost.", c.adaptive.source.learnedMinFactor);
    AddSize(options, "adaptive.source.learned-min-photons", "Minimum packets per learned cell.", c.adaptive.source.learnedMinPhotons);
    AddSize(options, "adaptive.source.learned-max-photons", "Maximum packets per learned cell.", c.adaptive.source.learnedMaxPhotons);
    AddDouble(options, "adaptive.source.score-power", "Learned source score exponent.", c.adaptive.source.scorePower);
    AddDouble(options, "adaptive.source.weight-score-fraction", "Escaped-weight contribution to source score.", c.adaptive.source.weightScoreFraction);

    AddBool(options, "adaptive.observer.equity", "Allocate for observer-direction deficits.", c.adaptive.observer.equity);
    AddDouble(options, "adaptive.observer.extra-budget-fraction", "Extra packet budget for observer equity.", c.adaptive.observer.extraBudgetFraction);
    AddDouble(options, "adaptive.observer.target-effective-packets", "Target effective packets per observer.", c.adaptive.observer.targetEffectivePackets);
    AddDouble(options, "adaptive.observer.target-polarization-snr", "Target observer polarization SNR.", c.adaptive.observer.targetPolarizationSnr);
    AddDouble(options, "adaptive.observer.max-deficit", "Maximum observer allocation deficit.", c.adaptive.observer.maxDeficit);
    AddDouble(options, "adaptive.observer.deficit-ema", "Observer-deficit exponential averaging factor.", c.adaptive.observer.deficitEma);

    AddBool(options, "adaptive.group.quality", "Enable observer/group quality targeting.", c.adaptive.group.quality);
    AddBool(options, "adaptive.group.source-cells", "Enable source-cell/group allocation.", c.adaptive.group.sourceCells);
    AddBool(options, "adaptive.group.frequency-sampling", "Bias emitted group sampling.", c.adaptive.group.frequencySampling);
    AddBool(options, "adaptive.group.history", "Accumulate group diagnostics across generations.", c.adaptive.group.history);
    AddDouble(options, "adaptive.group.target-effective-packets", "Target effective packets per observer/group.", c.adaptive.group.targetEffectivePackets);
    AddDouble(options, "adaptive.group.target-polarization-snr", "Target polarization SNR per observer/group.", c.adaptive.group.targetPolarizationSnr);
    AddDouble(options, "adaptive.group.max-deficit", "Maximum observer/group deficit.", c.adaptive.group.maxDeficit);
    AddSize(options, "adaptive.group.min-crossings", "Minimum crossings for an eligible group bin.", c.adaptive.group.minCrossings);
    AddDouble(options, "adaptive.group.min-luminosity", "Absolute group luminosity eligibility floor.", c.adaptive.group.minLuminosity);
    AddDouble(options, "adaptive.group.min-luminosity-fraction-of-group-max", "Relative group luminosity eligibility floor.", c.adaptive.group.minLuminosityFractionOfGroupMax);
    AddDouble(options, "adaptive.group.ineligible-priority-cap", "Priority cap for ineligible group bins.", c.adaptive.group.ineligiblePriorityCap);
    AddDouble(options, "adaptive.group.retain-priority-floor", "Priority floor retained during pruning.", c.adaptive.group.retainPriorityFloor);
    AddString(options, "adaptive.group.luminosity-normalization", "Group luminosity normalization mode.", c.adaptive.group.luminosityNormalization);
    AddDouble(options, "adaptive.group.luminosity-global-weight", "Global contribution to mixed luminosity normalization.", c.adaptive.group.luminosityGlobalWeight);
    AddDouble(options, "adaptive.group.luminosity-power", "Luminosity score exponent.", c.adaptive.group.luminosityPower);
    AddDouble(options, "adaptive.group.polarization-power", "Polarization score exponent.", c.adaptive.group.polarizationPower);
    AddDouble(options, "adaptive.group.luminosity-weight", "Luminosity contribution to group score.", c.adaptive.group.luminosityWeight);
    AddDouble(options, "adaptive.group.polarization-weight", "Polarization contribution to group score.", c.adaptive.group.polarizationWeight);
    AddDouble(options, "adaptive.group.polarization-floor", "Polarization fraction floor.", c.adaptive.group.polarizationFloor);
    AddDouble(options, "adaptive.group.history-ema", "Historical diagnostic averaging factor.", c.adaptive.group.historyEma);
    AddDouble(options, "adaptive.group.latest-weight", "Latest-generation score weight.", c.adaptive.group.latestWeight);
    AddDouble(options, "adaptive.group.cumulative-weight", "Cumulative score weight.", c.adaptive.group.cumulativeWeight);
    AddDouble(options, "adaptive.group.ema-weight", "EMA score weight.", c.adaptive.group.emaWeight);
    AddDouble(options, "adaptive.group.score-ema", "Group allocation score averaging factor.", c.adaptive.group.scoreEma);
    AddDouble(options, "adaptive.group.strength", "Adaptive frequency PDF mixture strength.", c.adaptive.group.strength);
    AddDouble(options, "adaptive.group.pdf-floor", "Minimum adaptive frequency PDF component.", c.adaptive.group.pdfFloor);
    AddDouble(options, "adaptive.group.max-bias", "Maximum frequency-sampling bias.", c.adaptive.group.maxBias);
    AddDouble(options, "adaptive.group.max-weight-correction", "Maximum inverse-PDF weight correction.", c.adaptive.group.maxWeightCorrection);
    AddSize(options, "adaptive.group.max-local-stats", "Maximum local source-cell/group statistics.", c.adaptive.group.maxLocalStats);
    AddSize(options, "adaptive.group.stat-min-count", "Minimum local statistic count retained.", c.adaptive.group.statMinCount);
    AddDouble(options, "adaptive.group.stat-priority-keep", "Priority threshold retaining sparse statistics.", c.adaptive.group.statPriorityKeep);
    AddBool(options, "adaptive.group.fallback-to-integrated-on-overflow", "Fall back to integrated allocation on statistics overflow.", c.adaptive.group.fallbackToIntegratedOnOverflow);

    AddBool(options, "load-balance.measured", "Use measured first-generation load balancing.", c.loadBalance.measured);
    AddDouble(options, "load-balance.weight-compression", "Measured load-weight compression, or -1 for automatic.", c.loadBalance.weightCompression);
    AddDouble(options, "load-balance.imbalance-threshold", "Imbalance ratio that triggers adaptive repartitioning.", c.loadBalance.imbalanceThreshold);
    AddSize(options, "load-balance.cooldown-generations", "Generations between adaptive rebalances.", c.loadBalance.cooldownGenerations);
    AddSize(options, "load-balance.max-rebalances", "Maximum adaptive rebalances.", c.loadBalance.maxRebalances);
    AddBool(options, "diagnostics.verbose-adaptive", "Print detailed adaptive diagnostics.", c.diagnostics.verboseAdaptive);

    return options;
}

} // namespace

CommandLineAction ParseCommandLine(
    int argc, char* argv[], PostProcessConfig& config, int rank)
{
    std::vector<Option> options = MakeRegistry(config);
    std::unordered_map<std::string, size_t> byName;
    for (size_t i = 0; i < options.size(); ++i)
        byName.emplace(options[i].name, i);

    CommandLineAction action = CommandLineAction::Run;
    for (int i = 1; i < argc; ++i) {
        std::string argument(argv[i]);
        if (argument == "--help") {
            action = CommandLineAction::Help;
            continue;
        }
        if (argument == "--print-effective-config") {
            action = CommandLineAction::PrintEffectiveConfig;
            continue;
        }
        if (argument.compare(0, 2, "--") != 0) {
            if (rank == 0)
                std::cerr << "Unexpected positional argument: " << argument << "\n";
            return CommandLineAction::Error;
        }
        std::string const name = argument.substr(2);
        auto const found = byName.find(name);
        if (found == byName.end()) {
            if (rank == 0)
                std::cerr << "Unknown option: " << argument << "\n";
            return CommandLineAction::Error;
        }
        if (i + 1 >= argc) {
            if (rank == 0)
                std::cerr << "Missing value for " << argument << "\n";
            return CommandLineAction::Error;
        }
        std::string error;
        if (!options[found->second].set(argv[++i], error)) {
            if (rank == 0)
                std::cerr << "Invalid value for " << argument << ": " << error << "\n";
            return CommandLineAction::Error;
        }
    }
    return action;
}

void PrintCommandLineHelp(PostProcessConfig const& defaults, int rank)
{
    if (rank != 0)
        return;
    PostProcessConfig mutableDefaults = defaults;
    std::vector<Option> const options = MakeRegistry(mutableDefaults);
    std::cout << "Usage: rich [options]\n\n"
              << "Options use --<section>.<field> <value>. Booleans require true or false.\n"
              << "  --help\n"
              << "  --print-effective-config\n";
    for (Option const& option : options)
        std::cout << "  --" << option.name << " <" << option.type << ">\n"
                  << "      " << option.help << " Default: " << option.value() << "\n";
}

void PrintEffectiveConfig(PostProcessConfig const& config, int rank)
{
    if (rank != 0)
        return;
    PostProcessConfig mutableConfig = config;
    std::vector<Option> const options = MakeRegistry(mutableConfig);
    for (Option const& option : options)
        std::cout << option.name << " = " << option.value() << "\n";
}

std::vector<std::pair<std::string, std::string>> EffectiveConfigEntries(
    PostProcessConfig const& config)
{
    PostProcessConfig mutableConfig = config;
    std::vector<Option> const options = MakeRegistry(mutableConfig);
    std::vector<std::pair<std::string, std::string>> entries;
    entries.reserve(options.size());
    for (Option const& option : options)
        entries.emplace_back(option.name, option.value());
    return entries;
}

} // namespace PostProcessIMC
