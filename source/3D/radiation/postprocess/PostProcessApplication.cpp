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
#include <utility>
#include <vector>

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
#include "source/misc/mesh_generator3D.hpp"

#include <fstream>
#include <iomanip>
#include <limits>
#include <unordered_map>

#ifdef RICH_MPI
#include <mpi.h>
#include "source/mpi/mpi_commands.hpp"
#endif

#include "IMCPostProcess.hpp"
#include "PostProcessCommandLine.hpp"
#include "PostProcessConfig.hpp"
#include "PostProcessResultWriter.hpp"
#include "PostProcessRuntime.hpp"
#include "AdaptiveStatistics.hpp"
#include "PhotosphereCalculation.hpp"
#include "ForwardCalculation.hpp"
#include "GreyCalculation.hpp"
#include "FluxSourceCalculation.hpp"

using namespace imc_postprocess_tde;

namespace {

PostProcessIMC::PostProcessResult* embeddedResultSink = nullptr;

PostProcessIMC::PostProcessPassResult ToPublicResult(
    ForwardPostprocessResult const& value)
{
    PostProcessIMC::PostProcessPassResult result;
    result.ran = value.ran;
    result.usesVelocity = value.usesVelocity;
    result.usesDDMC = value.usesDDMC;
    result.usesPolarization = value.usesPolarization;
    result.usesCompton = value.usesCompton;
    result.sourceLuminosity = value.sourceLuminosity;
    result.emittedLuminosity = value.emittedLuminosity;
    result.crossingLuminosity = value.crossingLuminosity;
    result.crossingLuminosityStderr = value.crossingLuminosityStderr;
    result.emittedEnergy = value.emittedEnergy;
    result.timedOutFraction = value.timedOutFraction;
    result.luminosityWeightedPolarizationDegree =
        value.luminosityWeightedPolarizationDegree;
    result.polarizedObserverCount = value.polarizedObserverCount;
    return result;
}

void BroadcastPassResult(ForwardPostprocessResult& result)
{
#ifdef RICH_MPI
    int flags[5] = {
        result.ran ? 1 : 0,
        result.usesVelocity ? 1 : 0,
        result.usesDDMC ? 1 : 0,
        result.usesPolarization ? 1 : 0,
        result.usesCompton ? 1 : 0};
    double values[7] = {
        result.sourceLuminosity,
        result.emittedLuminosity,
        result.crossingLuminosity,
        result.crossingLuminosityStderr,
        result.emittedEnergy,
        result.timedOutFraction,
        result.luminosityWeightedPolarizationDegree};
    unsigned long long polarizedObservers =
        static_cast<unsigned long long>(result.polarizedObserverCount);
    MPI_Bcast(flags, 5, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(values, 7, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(
        &polarizedObservers, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    result.ran = flags[0] != 0;
    result.usesVelocity = flags[1] != 0;
    result.usesDDMC = flags[2] != 0;
    result.usesPolarization = flags[3] != 0;
    result.usesCompton = flags[4] != 0;
    result.sourceLuminosity = values[0];
    result.emittedLuminosity = values[1];
    result.crossingLuminosity = values[2];
    result.crossingLuminosityStderr = values[3];
    result.emittedEnergy = values[4];
    result.timedOutFraction = values[5];
    result.luminosityWeightedPolarizationDegree = values[6];
    result.polarizedObserverCount =
        static_cast<uint64_t>(polarizedObservers);
#else
    (void)result;
#endif
}

Config ToInternalConfig(PostProcessIMC::PostProcessConfig const& publicConfig)
{
    Config cfg;
    cfg.inputPath = publicConfig.input.snapshot;
    cfg.outputPath = ReplaceExtension(publicConfig.output.stem, ".h5");
    cfg.vtkOutput = ReplaceExtension(publicConfig.output.stem, ".vtk");
    cfg.writeVtk = publicConfig.output.writeVtk;
    cfg.opacityDir = publicConfig.input.multigroupOpacityDirectory;
    cfg.greyOpacityDir = publicConfig.input.greyOpacityDirectory;
    cfg.eosDir = publicConfig.input.eosDirectory;
    cfg.radius = publicConfig.observer.radius;
    cfg.nObservers = publicConfig.observer.count;
    cfg.sourceDt = publicConfig.transport.sourceDt;
    cfg.transportTime = publicConfig.transport.duration;
    cfg.center = publicConfig.observer.center;
    cfg.photonsPerCell = publicConfig.transport.photonsPerCell;
    cfg.compton = publicConfig.transport.compton.enabled;
    cfg.comptonSamples = publicConfig.transport.compton.matrixSamples;
    cfg.comptonAngleDependent = publicConfig.transport.compton.angleDependent;
    cfg.nGenerations = publicConfig.transport.generations;
    cfg.ddmc = publicConfig.transport.ddmc;
    cfg.randomWalk = publicConfig.transport.randomWalk;
    cfg.useCellVelocities = publicConfig.transport.useCellVelocities;
    cfg.polarization = publicConfig.polarization.enabled;
    cfg.photosphere = publicConfig.photosphere.enabled;
    cfg.fluxSourceCompare = publicConfig.fluxSource.enabled;
    cfg.fluxSourceThermalizationTau = publicConfig.fluxSource.thermalizationTau;
    cfg.fluxSourceRays = publicConfig.fluxSource.constructionRays;
    cfg.fluxSourceDDMCFaceOpticalDepth = publicConfig.fluxSource.ddmcFaceOpticalDepth;
    cfg.polarizationManualScatterings =
        publicConfig.polarization.manualScatteringsAfterAcceleration;
    cfg.polarizationDepolarizationScatterings =
        publicConfig.polarization.depolarizationScatterings;
    cfg.polarizationClosure = publicConfig.polarization.acceleratedClosure;
    cfg.measuredLoadBalance = publicConfig.loadBalance.measured;
    switch (publicConfig.opacityScaling.mode) {
    case PostProcessIMC::OpacityScaleMode::None:
        cfg.opacityScaleMode = OpacityScaleMode::None;
        break;
    case PostProcessIMC::OpacityScaleMode::Rosseland:
        cfg.opacityScaleMode = OpacityScaleMode::Rosseland;
        break;
    case PostProcessIMC::OpacityScaleMode::Planck:
        cfg.opacityScaleMode = OpacityScaleMode::Planck;
        break;
    }
    cfg.adaptiveSourceCells = publicConfig.adaptive.source.enabled;
    cfg.adaptiveSourceBurnin = publicConfig.adaptive.source.burninGenerations;
    cfg.adaptiveSourceStrength = publicConfig.adaptive.source.strength;
    cfg.adaptiveSourceEma = publicConfig.adaptive.source.ema;
    cfg.adaptiveSourceMinEscapedFrac = publicConfig.adaptive.source.minEscapedFraction;
    cfg.adaptiveSourceMaxFactor = publicConfig.adaptive.source.maxFactor;
    cfg.adaptiveSourceBurninPhotonMultiplier =
        publicConfig.adaptive.source.burninPhotonMultiplier;
    cfg.adaptiveSourceLearnedReserveFrac =
        publicConfig.adaptive.source.learnedReserveFraction;
    cfg.adaptiveSourceLearnedMinFactor = publicConfig.adaptive.source.learnedMinFactor;
    cfg.adaptiveSourceLearnedMinPhotons = publicConfig.adaptive.source.learnedMinPhotons;
    cfg.adaptiveSourceLearnedMaxPhotons = publicConfig.adaptive.source.learnedMaxPhotons;
    cfg.adaptiveSourceScorePower = publicConfig.adaptive.source.scorePower;
    cfg.adaptiveSourceWeightScoreFrac = publicConfig.adaptive.source.weightScoreFraction;
    cfg.adaptiveObserverEquity = publicConfig.adaptive.observer.equity;
    cfg.adaptiveObserverExtraBudgetFrac =
        publicConfig.adaptive.observer.extraBudgetFraction;
    cfg.adaptiveObserverTargetNeff =
        publicConfig.adaptive.observer.targetEffectivePackets;
    cfg.adaptiveObserverTargetPolSnr =
        publicConfig.adaptive.observer.targetPolarizationSnr;
    cfg.adaptiveObserverDeficitMax = publicConfig.adaptive.observer.maxDeficit;
    cfg.adaptiveObserverDeficitEma = publicConfig.adaptive.observer.deficitEma;
    cfg.measuredLBWeightCompression = publicConfig.loadBalance.weightCompression;
    cfg.adaptiveLBImbalanceThreshold = publicConfig.loadBalance.imbalanceThreshold;
    cfg.adaptiveLBCooldownGenerations = publicConfig.loadBalance.cooldownGenerations;
    cfg.adaptiveLBMaxRebalances = publicConfig.loadBalance.maxRebalances;
    cfg.adaptiveGroupQuality = publicConfig.adaptive.group.quality;
    cfg.adaptiveGroupSourceCells = publicConfig.adaptive.group.sourceCells;
    cfg.adaptiveGroupFrequencySampling = publicConfig.adaptive.group.frequencySampling;
    cfg.adaptiveGroupHistory = publicConfig.adaptive.group.history;
    cfg.adaptiveGroupTargetNeff = publicConfig.adaptive.group.targetEffectivePackets;
    cfg.adaptiveGroupTargetPolSnr = publicConfig.adaptive.group.targetPolarizationSnr;
    cfg.adaptiveGroupDeficitMax = publicConfig.adaptive.group.maxDeficit;
    cfg.adaptiveGroupMinCrossings = publicConfig.adaptive.group.minCrossings;
    cfg.adaptiveGroupMinLuminosity = publicConfig.adaptive.group.minLuminosity;
    cfg.adaptiveGroupMinLuminosityFracOfGroupMax =
        publicConfig.adaptive.group.minLuminosityFractionOfGroupMax;
    cfg.adaptiveGroupIneligiblePriorityCap =
        publicConfig.adaptive.group.ineligiblePriorityCap;
    cfg.adaptiveGroupRetainPriorityFloor = publicConfig.adaptive.group.retainPriorityFloor;
    cfg.adaptiveGroupLuminosityNormalization =
        publicConfig.adaptive.group.luminosityNormalization;
    cfg.adaptiveGroupLuminosityGlobalWeight =
        publicConfig.adaptive.group.luminosityGlobalWeight;
    cfg.adaptiveGroupLuminosityPower = publicConfig.adaptive.group.luminosityPower;
    cfg.adaptiveGroupPolarizationPower = publicConfig.adaptive.group.polarizationPower;
    cfg.adaptiveGroupLuminosityWeight = publicConfig.adaptive.group.luminosityWeight;
    cfg.adaptiveGroupPolarizationWeight = publicConfig.adaptive.group.polarizationWeight;
    cfg.adaptiveGroupPolarizationFloor = publicConfig.adaptive.group.polarizationFloor;
    cfg.adaptiveGroupHistoryEma = publicConfig.adaptive.group.historyEma;
    cfg.adaptiveGroupLatestWeight = publicConfig.adaptive.group.latestWeight;
    cfg.adaptiveGroupCumulativeWeight = publicConfig.adaptive.group.cumulativeWeight;
    cfg.adaptiveGroupEmaWeight = publicConfig.adaptive.group.emaWeight;
    cfg.adaptiveGroupScoreEma = publicConfig.adaptive.group.scoreEma;
    cfg.adaptiveGroupStrength = publicConfig.adaptive.group.strength;
    cfg.adaptiveGroupPdfFloor = publicConfig.adaptive.group.pdfFloor;
    cfg.adaptiveGroupMaxBias = publicConfig.adaptive.group.maxBias;
    cfg.adaptiveGroupMaxWeightCorrection = publicConfig.adaptive.group.maxWeightCorrection;
    cfg.adaptiveGroupMaxLocalStats = publicConfig.adaptive.group.maxLocalStats;
    cfg.adaptiveGroupStatMinCount = publicConfig.adaptive.group.statMinCount;
    cfg.adaptiveGroupStatPriorityKeep = publicConfig.adaptive.group.statPriorityKeep;
    cfg.adaptiveGroupFallbackToIntegratedOnOverflow =
        publicConfig.adaptive.group.fallbackToIntegratedOnOverflow;
    cfg.adaptiveDiagnosticsVerbose = publicConfig.diagnostics.verboseAdaptive;
    return cfg;
}

std::vector<double> IntegrateFldFluxOverObserverPatches(
    Voronoi3D const& tess,
    std::vector<Vector3D> const& fldFlux,
    SphericalObserver const& observer,
    int rank)
{
    size_t const nCells = tess.GetPointNo();
    size_t const nObservers = observer.getNumObservers();
    constexpr size_t samplesPerObserver = 32;

    if (fldFlux.size() < nCells)
        throw UniversalError("FLD surface integration received too few cell fluxes");
    if (nCells == 0 || nObservers == 0)
        throw UniversalError("FLD surface integration requires cells and observers");
    if (nObservers > static_cast<size_t>(INT_MAX))
        throw UniversalError("Too many FLD observer bins for MPI reduction");
    if (nObservers > std::numeric_limits<size_t>::max() / samplesPerObserver)
        throw UniversalError("FLD surface quadrature sample count overflow");

#ifndef RICH_MPI
    (void)rank;
#endif

    size_t const sampleCount = samplesPerObserver * nObservers;
    std::vector<Vector3D> const sampleDirections =
        fibonacci_sphere_directions(sampleCount);
    std::vector<Vector3D> const& observerDirections = observer.getDirections();
    std::vector<double> luminosity(nObservers, 0.0);
    unsigned long long localEvaluated = 0;

    double const radius = observer.getRadius();
    double const sampleArea = 4.0 * std::acos(-1.0) * radius * radius
                            / static_cast<double>(sampleCount);

    // Integrate the finite-volume FLD field as it is represented: a
    // piecewise-constant value in each Voronoi cell. Sub-sampling each angular
    // bin removes nearest-generator aliasing without smoothing across shocks,
    // opacity jumps, or resolution transitions.
    for (Vector3D const& direction : sampleDirections) {
        size_t observerIndex = 0;
        double bestAlignment = -std::numeric_limits<double>::infinity();
        for (size_t p = 0; p < nObservers; ++p) {
            double const alignment = ScalarProd(direction, observerDirections[p]);
            if (alignment > bestAlignment) {
                bestAlignment = alignment;
                observerIndex = p;
            }
        }

        Vector3D const point = observer.getCenter() + direction * radius;
#ifdef RICH_MPI
        if (tess.GetOwner(point) != rank)
            continue;
#endif
        size_t const cell = tess.GetContainingCell(point);
        if (cell >= fldFlux.size()) {
            UniversalError eo("FLD quadrature point did not map to an available cell");
            eo.addEntry("Cell index", cell);
            eo.addEntry("Available flux entries", fldFlux.size());
            throw eo;
        }

        double const radialFlux = ScalarProd(fldFlux[cell], direction);
        luminosity[observerIndex] += std::max(0.0, radialFlux) * sampleArea;
        ++localEvaluated;
    }

#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, luminosity.data(),
                  static_cast<int>(nObservers), MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    unsigned long long globalEvaluated = 0;
    MPI_Allreduce(&localEvaluated, &globalEvaluated, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#else
    unsigned long long const globalEvaluated = localEvaluated;
#endif

    if (globalEvaluated != static_cast<unsigned long long>(sampleCount)) {
        UniversalError eo("FLD surface quadrature did not evaluate every sample exactly once");
        eo.addEntry("Expected samples", sampleCount);
        eo.addEntry("Evaluated samples", globalEvaluated);
        throw eo;
    }

    return luminosity;
}

} // namespace

PostProcessIMC::PostProcessScenario PostProcessIMC::MakeStaSnapshotScenario(
    std::string name)
{
    PostProcessScenario scenario;
    scenario.name = std::move(name);

    scenario.factories.loadSnapshot = [](
        PostProcessConfig::Input const& input,
        ParallelContext const& parallel) {
#ifdef RICH_MPI
        Snapshot3D snapshot = ReadSnapshot3DParallel(input.snapshot);
        int const fileRanks = GetNumberOfRanksInHDF(input.snapshot);
        if (parallel.size < fileRanks && parallel.rank == 0) {
            for (int fileRank = parallel.size; fileRank < fileRanks; ++fileRank) {
                Snapshot3D extra = ReadSnapshot3DParallel(input.snapshot, fileRank);
                snapshot.cells.insert(snapshot.cells.end(),
                                      extra.cells.begin(), extra.cells.end());
                snapshot.mesh_points.insert(snapshot.mesh_points.end(),
                                            extra.mesh_points.begin(),
                                            extra.mesh_points.end());
            }
        }
        return snapshot;
#else
        (void)parallel;
        return ReadSnapshot3D(input.snapshot);
#endif
    };

    scenario.factories.transformSnapshot = [](
        Snapshot3D& snapshot, ParallelContext const&) {
        double const lengthScale = 7e10;
        double const massScale = 2e33;
        double const timeScale = 1603;
        double const densityScale =
            massScale / (lengthScale * lengthScale * lengthScale);
        double const velocityScale = lengthScale / timeScale;
        double const specificEnergyScale =
            lengthScale * lengthScale / (timeScale * timeScale);

        snapshot.ll = snapshot.ll * lengthScale;
        snapshot.ur = snapshot.ur * lengthScale;
        snapshot.time *= timeScale;
        for (Vector3D& point : snapshot.mesh_points)
            point = point * lengthScale;
        for (ComputationalCell3D& cell : snapshot.cells) {
            cell.density *= densityScale;
            cell.pressure *= massScale /
                (timeScale * timeScale * lengthScale);
            cell.internal_energy *= specificEnergyScale;
            cell.velocity = cell.velocity * velocityScale;
            cell.Erad *= specificEnergyScale;
            for (size_t group = 0; group < ENERGY_GROUPS_NUM; ++group)
                cell.Eg[group] *= specificEnergyScale;
        }
    };

    scenario.factories.createEquationOfState = [](
        PostProcessConfig::Input const& input) {
        return std::make_shared<OndrejEOS>(
            input.eosDirectory + "density.txt",
            input.eosDirectory + "Pfile.txt",
            input.eosDirectory + "csfile.txt",
            input.eosDirectory + "Sfile.txt",
            input.eosDirectory + "Ufile.txt",
            input.eosDirectory + "Tfile.txt",
            input.eosDirectory + "CVfile.txt",
            1.0, 1.0, 1.0);
    };
    scenario.factories.createMultigroupOpacity = [](
        PostProcessConfig::Input const& input) {
        return std::static_pointer_cast<OpacityCalculator>(
            std::make_shared<STAMGopacityMC>(
                input.multigroupOpacityDirectory));
    };
    scenario.factories.createGreyOpacity = [](
        PostProcessConfig::Input const& input) {
        return std::static_pointer_cast<OpacityCalculator>(
            std::make_shared<STAgreyOpacity>(input.greyOpacityDirectory));
    };
    scenario.factories.applyOpacityScaleFactors = [](
        OpacityCalculator& opacity,
        std::unordered_map<size_t, double> factors) {
        STAMGopacityMC* sta = dynamic_cast<STAMGopacityMC*>(&opacity);
        if (sta == nullptr)
            throw UniversalError(
                "STA scenario opacity scaling requires STAMGopacityMC");
        sta->SetRosselandScaleFactors(std::move(factors));
    };
    scenario.factories.createObserver = [](
        PostProcessConfig::Observer const& observer,
        std::vector<double> const& groupBoundaries) {
        return std::make_shared<SphericalObserver>(
            observer.center, observer.radius, observer.count,
            groupBoundaries);
    };
    scenario.factories.configureSource = [](
        SourceContext&, PostProcessConfig const&) {};
    return scenario;
}

int PostProcessIMC::RunPostProcessMain(
    int argc, char* argv[], PostProcessScenario scenario)
{
#ifdef RICH_MPI
    int mpiInitialized = 0;
    MPI_Initialized(&mpiInitialized);
    bool ownsMpi = (mpiInitialized == 0);
    if (ownsMpi)
        MPI_Init(&argc, &argv);
    int rank = 0, mpiSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
#else
    int rank = 0, mpiSize = 1;
#endif

    std::exception_ptr embeddedFailure;
    try {
        CommandLineAction const action =
            ParseCommandLine(argc, argv, scenario.defaults, rank);
        if (action == CommandLineAction::Help) {
            PrintCommandLineHelp(scenario.defaults, rank);
#ifdef RICH_MPI
            if (ownsMpi) MPI_Finalize();
#endif
            return 0;
        }
        if (action == CommandLineAction::Error) {
            if (rank == 0)
                std::cerr << "Use --help to list supported options.\n";
#ifdef RICH_MPI
            if (ownsMpi) MPI_Finalize();
#endif
            return 1;
        }

        Config cfg = ToInternalConfig(scenario.defaults);
        if (!ValidateConfig(cfg, rank)) {
#ifdef RICH_MPI
            if (ownsMpi) MPI_Finalize();
#endif
            return 1;
        }
#ifndef RICH_IMC_DDMC_ENABLED
        if (cfg.ddmc) {
            if (rank == 0)
                std::cerr << "Error: DDMC is enabled, but this executable was built without RadiationIMC_DDMC.cpp. "
                          << "Rebuild with forward DDMC support or pass --transport.ddmc false.\n";
#ifdef RICH_MPI
            if (ownsMpi) MPI_Finalize();
#endif
            return 1;
        }
#endif
#ifndef MONTECARLO_POLARIZATION
        if (cfg.polarization) {
            if (rank == 0)
                std::cerr
                    << "Error: polarization is enabled for imc_postprocess_tde, but this executable "
                    << "was built without MONTECARLO_POLARIZATION. Rebuild with "
                    << "--montecarlo-polarization or pass --polarization.enabled false.\n";
#ifdef RICH_MPI
            if (ownsMpi) MPI_Finalize();
#endif
            return 1;
        }
#endif
        scenario.defaults.input.greyOpacityDirectory = cfg.greyOpacityDir;
        scenario.defaults.transport.duration = cfg.transportTime;
        if (action == CommandLineAction::PrintEffectiveConfig) {
            PrintEffectiveConfig(scenario.defaults, rank);
#ifdef RICH_MPI
            if (ownsMpi) MPI_Finalize();
#endif
            return 0;
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
                      << "DDMC:            "
                      << (cfg.ddmc ? "yes" : "no")
                      << (cfg.fluxSourceCompare && cfg.ddmc
                          ? " (native thermalizing CER boundary)" : "") << "\n"
                      << "Cell velocities: " << (cfg.useCellVelocities ? "yes" : "no") << "\n"
                      << "Polarization:    " << (cfg.polarization ? "yes" : "no") << "\n"
                      << "Photosphere:     " << (cfg.photosphere ? "yes" : "no") << "\n"
                      << "Flux source test:" << (cfg.fluxSourceCompare ? " yes" : " no") << "\n"
                      << "Flux source tau: " << cfg.fluxSourceThermalizationTau << "\n"
                      << "Flux source DDMC face tau: "
                      << cfg.fluxSourceDDMCFaceOpticalDepth << "\n"
                      << "Flux source transport: "
                      << (cfg.fluxSourceCompare
                          ? (cfg.ddmc
                              ? "CER-aware DDMC with face-local explicit fallback"
                              : "explicit IMC (--no-ddmc)")
                          : "not applicable") << "\n"
                      << "Measured LB:     " << (cfg.measuredLoadBalance ? "requested" : "disabled") << "\n"
                      << "  weight compression: " << EffectiveMeasuredLBWeightCompression(cfg) << "\n"
                      << "  max cell imbalance: " << MEASURED_LB_MAX_CELL_IMBALANCE << "\n"
                      << "  adaptive cadence: learned-only probe LB, then every 10 learned-final steps before the last\n"
                      << "Opacity scale:   " << (cfg.opacityScaleMode == imc_postprocess_tde::OpacityScaleMode::Planck ? "planck" :
                                                  cfg.opacityScaleMode == imc_postprocess_tde::OpacityScaleMode::Rosseland ? "rosseland" : "disabled") << "\n"
                      << "Adaptive source: " << (cfg.adaptiveSourceCells ? "enabled" : "disabled") << "\n"
                      << "  MG schedule:   1 exact-1 burn-in, 19 exact-3 burn-in, learned-only exact-75 probe, LB, "
                      << cfg.nGenerations << " learned-only final steps (min=500 max=2000)\n"
                      << "  final LB cadence: every 10 learned-final steps before the last\n"
                      << "  min esc frac:  " << cfg.adaptiveSourceMinEscapedFrac << "\n"
                      << "  strength:      " << cfg.adaptiveSourceStrength << "\n"
                      << "  EMA:           " << cfg.adaptiveSourceEma << "\n"
                      << "  max factor:    " << cfg.adaptiveSourceMaxFactor << "\n"
                      << "  learned reserve frac:      " << cfg.adaptiveSourceLearnedReserveFrac << "\n"
                      << "  learned min factor:        " << cfg.adaptiveSourceLearnedMinFactor << "\n"
                      << "  learned photons/cell min:  " << cfg.adaptiveSourceLearnedMinPhotons << "\n"
                      << "  learned photons/cell max:  " << cfg.adaptiveSourceLearnedMaxPhotons << "\n"
                      << "  learned score power:       " << cfg.adaptiveSourceScorePower << "\n"
                      << "  weight score fraction:     " << cfg.adaptiveSourceWeightScoreFrac << "\n"
                      << "  observer equity:           " << ((cfg.adaptiveSourceCells && cfg.adaptiveObserverEquity) ? "enabled" : "disabled") << "\n"
                      << "  observer target neff:      " << cfg.adaptiveObserverTargetNeff << "\n"
                      << "  observer target pol SNR:   " << cfg.adaptiveObserverTargetPolSnr << "\n"
                      << "  polarization SNR scoring:  "
                      << ((cfg.adaptiveSourceCells && cfg.adaptiveObserverEquity && cfg.polarization)
                              ? "enabled" : "disabled") << "\n"
                      << "  observer deficit max/EMA:  " << cfg.adaptiveObserverDeficitMax << "/" << cfg.adaptiveObserverDeficitEma << "\n"
                      << "  observer extra budget max: " << cfg.adaptiveObserverExtraBudgetFrac << "\n"
                      << "  burnin/adapt LB: " << ((cfg.adaptiveSourceCells && cfg.measuredLoadBalance) ? "requested" : "disabled") << "\n"
                      << "Requested generations: " << cfg.nGenerations << "\n"
                      << "MPI ranks:       " << mpiSize << "\n"
                      << std::endl;
        }

        PostProcessConfig::Input effectiveInput;
        effectiveInput.snapshot = cfg.inputPath;
        effectiveInput.multigroupOpacityDirectory = cfg.opacityDir;
        effectiveInput.greyOpacityDirectory = cfg.greyOpacityDir;
        effectiveInput.eosDirectory = cfg.eosDir;
        ParallelContext const parallel{rank, mpiSize};

        if (!scenario.factories.createEquationOfState ||
            !scenario.factories.createMultigroupOpacity ||
            !scenario.factories.createGreyOpacity ||
            !scenario.factories.loadSnapshot ||
            !scenario.factories.transformSnapshot ||
            !scenario.factories.createObserver ||
            !scenario.factories.applyOpacityScaleFactors)
            throw UniversalError(
                "Post-process scenario has an incomplete factory set");

        std::shared_ptr<EquationOfState> eos =
            scenario.factories.createEquationOfState(effectiveInput);
        if (!eos)
            throw UniversalError("Post-process EOS factory returned null");

        if (rank == 0)
            std::cout << "EOS loaded for scenario " << scenario.name
                      << "." << std::endl;

        // ============================================================
        // Load STA multigroup opacity
        // ============================================================
        std::shared_ptr<OpacityCalculator> opacity =
            scenario.factories.createMultigroupOpacity(effectiveInput);
        if (!opacity)
            throw UniversalError(
                "Post-process multigroup opacity factory returned null");

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

        Snapshot3D snapshot =
            scenario.factories.loadSnapshot(effectiveInput, parallel);

        if (snapshot.mesh_points.empty()) {
            if (rank == 0) std::cerr << "Empty snapshot\n";
#ifdef RICH_MPI
            if (ownsMpi) MPI_Finalize();
#endif
            return 1;
        }

        if (rank == 0)
            std::cout << "Snapshot read: " << snapshot.mesh_points.size() << " points, time=" << snapshot.time << std::endl;

        scenario.factories.transformSnapshot(snapshot, parallel);

        if (rank == 0)
            std::cout << "Converted snapshot to CGS." << std::endl;

        // ============================================================
        // Rebuild tessellation (two-pass for weighted load balancing)
        // ============================================================
        ComputationalCell3D dummyCell;
#ifdef RICH_MPI
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
        std::shared_ptr<OpacityCalculator> greyOpacity =
            scenario.factories.createGreyOpacity(effectiveInput);
        if (!greyOpacity)
            throw UniversalError(
                "Post-process grey opacity factory returned null");
        if (rank == 0)
            std::cout << "Grey opacity loaded for FLD luminosity." << std::endl;

        // ============================================================
        // Scale MG absorption to match grey mean
        // ============================================================
#if ENERGY_GROUPS_NUM > 1
        if (cfg.opacityScaleMode !=
            imc_postprocess_tde::OpacityScaleMode::None) {
            RecomputeOpacityScaleFactors(
                *opacity, *greyOpacity, cells, Ncells, rank,
                cfg.opacityScaleMode,
                [&](std::unordered_map<size_t, double> factors) {
                    scenario.factories.applyOpacityScaleFactors(
                        *opacity, std::move(factors));
                },
                "initial");
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

#ifdef RICH_MPI
        MPI_exchange_data(tess, fldFlux, true);
#endif

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
        PostProcessConfig::Observer observerConfig;
        observerConfig.center = cfg.center;
        observerConfig.radius = cfg.radius;
        observerConfig.count = cfg.nObservers;
        std::shared_ptr<SphericalObserver> observer =
            scenario.factories.createObserver(observerConfig, groupBoundaries);
        if (!observer)
            throw UniversalError("Post-process observer factory returned null");

        // ============================================================
        // Map FLD flux to observer patches
        // ============================================================
        size_t const nObs = observer->getNumObservers();
        std::vector<double> fldLuminosity =
            IntegrateFldFluxOverObserverPatches(tess, fldFlux, *observer, rank);

        // Free FLD intermediates no longer needed
        Er_vol.clear(); Er_vol.shrink_to_fit();
        D_cell.clear(); D_cell.shrink_to_fit();
        gradEr.clear(); gradEr.shrink_to_fit();
        fldFlux.clear(); fldFlux.shrink_to_fit();

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
        RadiationIMCParameters params;
        size_t genPhotonsPerCell = std::max<size_t>(1, cfg.photonsPerCell / cfg.nGenerations);
        params.newPhotonsPerCell = genPhotonsPerCell;
        params.withHydro = false;
        params.noHydroFeedback = true;
        params.withRandomWalk = cfg.randomWalk && !cfg.fluxSourceCompare;
        params.rwMinCellOpticalDepth = 15;
        params.withDDMC = cfg.ddmc;
        params.ddmcMinCellOpticalDepth = 15;
        params.ddmcExternalSourceMinFaceOpticalDepth =
            cfg.fluxSourceDDMCFaceOpticalDepth;
        params.withMultigroupDDMC = true;
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

        auto physics = std::make_shared<RadiationIMC>(
            tess, boundary, cells, extensives, eos, opacity, params);
        physics->setObserver(observer);
        if (cfg.polarization &&
            !observer->getObserverQualitySnapshot().polarizationEnabled)
            throw UniversalError(
                "Polarization was requested but observer polarization tracking is disabled");

        auto popControl = std::make_shared<STORM::NoPopulationControl<Vector3D, Tessellation3D>>(tess);

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

        PostProcessSession runtime{
            rank, mpiSize, tess, cells, extensives, eos, opacity, greyOpacity,
            observer, boundary, popControl, physics, manager, params, Ncells,
            snapshot.time, snapshot.cycle, dummyCell, fldLuminosity,
            totalFldLum,
            [&](std::unordered_map<size_t, double> factors) {
                scenario.factories.applyOpacityScaleFactors(
                    *opacity, std::move(factors));
            }};

        if (scenario.factories.configureSource) {
            SourceContext sourceContext{
                tess, cells, *opacity, *greyOpacity, *observer, *physics,
                parallel};
            scenario.factories.configureSource(sourceContext, scenario.defaults);
        }

        if (cfg.photosphere) {
            observer->setPhotosphereData(ComputeObserverPhotospheres(cfg, runtime));
        }

        if (cfg.fluxSourceCompare) {
            InitializeFluxSourceSurface(cfg, runtime);
            ConfigureFluxSourceForCurrentDecomposition(cfg, runtime, *physics);
        }

        Config const passConfig = MakePassOutputConfig(cfg);
        ForwardPostprocessResult forwardResult =
            RunForwardPostprocess(passConfig, runtime);
        ForwardPostprocessResult greyResult =
            RunGreyPostprocess(passConfig, runtime, forwardResult);
        BroadcastPassResult(forwardResult);
        BroadcastPassResult(greyResult);
        FinalizeResultBundle(
            cfg, passConfig, scenario.defaults, scenario.name, runtime,
            forwardResult, greyResult);
        if (embeddedResultSink != nullptr) {
            embeddedResultSink->forward = ToPublicResult(forwardResult);
            embeddedResultSink->grey = ToPublicResult(greyResult);
            embeddedResultSink->hdf5Path = cfg.outputPath;
            embeddedResultSink->vtkPath = BaseVtkOutputPath(cfg);
        }
    } catch (UniversalError const& eo) {
        if (embeddedResultSink != nullptr) {
            embeddedFailure = std::current_exception();
        } else {
            std::cerr << "UniversalError on rank " << rank << ":\n";
            reportError(eo, std::cerr);
#ifdef RICH_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#endif
            return 1;
        }
    } catch (std::exception const& e) {
        if (embeddedResultSink != nullptr) {
            embeddedFailure = std::current_exception();
        } else {
            std::cerr << "Exception on rank " << rank << ": " << e.what() << "\n";
#ifdef RICH_MPI
            MPI_Abort(MPI_COMM_WORLD, 1);
#endif
            return 1;
        }
    }

    if (embeddedResultSink != nullptr) {
        int localFailure = embeddedFailure ? 1 : 0;
        int globalFailure = localFailure;
#ifdef RICH_MPI
        MPI_Allreduce(
            &localFailure, &globalFailure, 1, MPI_INT, MPI_MAX,
            MPI_COMM_WORLD);
#endif
        if (globalFailure != 0) {
            if (embeddedFailure)
                std::rethrow_exception(embeddedFailure);
            throw UniversalError("RunPostProcess failed on another MPI rank");
        }
    }

#ifdef RICH_MPI
    if (ownsMpi) MPI_Finalize();
#endif
    return 0;
}

PostProcessIMC::PostProcessResult PostProcessIMC::RunPostProcess(
    PostProcessConfig config,
    PostProcessScenario const& scenario,
    ParallelContext parallel)
{
    if (embeddedResultSink != nullptr)
        throw UniversalError("RunPostProcess does not support nested invocations");
#ifdef RICH_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized)
        throw UniversalError(
            "RunPostProcess requires an initialized MPI context; use RunPostProcessMain when MPI ownership is desired");
    int actualRank = 0, actualSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &actualRank);
    MPI_Comm_size(MPI_COMM_WORLD, &actualSize);
    if (parallel.rank != actualRank || parallel.size != actualSize)
        throw UniversalError("RunPostProcess ParallelContext does not match MPI_COMM_WORLD");
#else
    if (parallel.rank != 0 || parallel.size != 1)
        throw UniversalError("Serial RunPostProcess requires ParallelContext{0, 1}");
#endif

    PostProcessScenario configured = scenario;
    configured.defaults = std::move(config);
    PostProcessResult result;
    embeddedResultSink = &result;
    char programName[] = "postprocess";
    char* argv[] = {programName, nullptr};
    try {
        int const status = RunPostProcessMain(1, argv, std::move(configured));
        embeddedResultSink = nullptr;
        if (status != 0)
            throw UniversalError("RunPostProcess failed");
    } catch (...) {
        embeddedResultSink = nullptr;
        throw;
    }
    return result;
}
