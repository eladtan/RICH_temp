// Regression checks for the post-process weight window
// (transport.split-observer-fraction): a packet that starts a step in a cell whose
// measured escape probability is p splits into equal copies when heavier than
// w_max / min(1, p / p_ref), w_max = fraction x (escaping energy) / observers.
//
// The thick-core / thin-envelope snapshot of imc_postprocess_cooling_limit is
// run with a heavy exploration cap (0.05 of the escaping energy, about 2.4x an
// observer's share per generation) so that heavy packets reach the observers:
//   A  weight window off
//   B  weight window on (fraction 1e-4, p_ref 0.1)
//   C  weight window off, exploration fraction 3e-5, learned cells capped
//      too (volume-emission.cap-learned-weight, on by default)
//   D  as C without the learned cap
// Checks: B makes copies and A none; the emitted energy is the same; the grey
// and multigroup luminosities agree within their generation errors (the
// window is unbiased); the window does not make the errors worse. Every
// emitting cell here is learned and gets ~20 packets of ~1.6e39 erg, so at
// 3e-5 (a cap of ~1.4e39 erg) the cap raises some cells and leaves others:
// it adds packets in C and none in D, keeps the emitted energy, is unbiased
// and does not make the errors worse. Under MPI, the packet count predicted
// for the load balance before the probe matches the probe's own count.

#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "source/3D/radiation/postprocess/IMCPostProcess.hpp"
#include "source/Radiation/OpacityCalculator.hpp"
#include "source/Radiation/conj_grad_solve.hpp"
#include "source/misc/universal_error.hpp"
#include "source/newtonian/common/ideal_gas.hpp"

#ifdef RICH_MPI
#include <mpi.h>
#endif

static_assert(ENERGY_GROUPS_NUM == 4,
              "imc_postprocess_weight_split is built with --energy_groups_num=4");

namespace
{

constexpr double kHalfBox = 1.0e14;          // cm
constexpr double kCoreRadius = 0.35 * kHalfBox;
constexpr std::array<double, 4> kGroupAbsorptionFactor = {0.5, 1.0, 2.0, 4.0};
constexpr double kAbsorptionKappa = 1.0;     // cm^2/g
constexpr double kScatteringKappa = 0.2;     // cm^2/g
constexpr double kGamma = 5.0 / 3.0;
constexpr double kMuMp = 0.6 * 1.67262192e-24;
constexpr double kEnvelopeTemperature = 2e4;
constexpr double kHotTemperature = 15.0 * kEnvelopeTemperature;
constexpr double kColdTemperature = kEnvelopeTemperature / 3.0;
constexpr size_t kHotCells = 10;
constexpr size_t kColdCells = 6;
double SpecificHeatF() { return CG::boltzmann_constant / ((kGamma - 1.0) * kMuMp); }

class SyntheticOpacity final : public OpacityCalculator
{
public:
    explicit SyntheticOpacity(bool multigroup) : multigroup_(multigroup)
    {
        double const eV = 1.602176634e-12;
        if (multigroup_) {
            energy_groups_boundary = {1e-3 * eV, 3.0 * eV, 10.0 * eV, 30.0 * eV, 1e4 * eV};
            for (size_t g = 0; g + 1 < energy_groups_boundary.size(); ++g)
                energy_groups_center.push_back(
                    std::sqrt(energy_groups_boundary[g] * energy_groups_boundary[g + 1]));
        }
    }
    double CalcPlanckOpacity(ComputationalCell3D const& cell) const override
    { return CalcPlanckOpacityAtTemperature(cell, cell.temperature); }
    double CalcPlanckOpacityAtTemperature(ComputationalCell3D const& cell, double) const override
    { return kAbsorptionKappa * cell.density; }
    double CalcScatteringOpacity(ComputationalCell3D const& cell) const override
    { return kScatteringKappa * cell.density; }
    double CalcScatteringOpacityAtTemperature(ComputationalCell3D const& cell, double) const override
    { return kScatteringKappa * cell.density; }
    double CalcScatteringOpacity(ComputationalCell3D const& cell, double) const override
    { return kScatteringKappa * cell.density; }
    double CalcScatteringOpacityAtTemperature(ComputationalCell3D const& cell, double, double) const override
    { return kScatteringKappa * cell.density; }
    double CalcAbsorptionOpacity(ComputationalCell3D const& cell, double energy) const override
    { return CalcAbsorptionOpacityAtTemperature(cell, energy, cell.temperature); }
    double CalcAbsorptionOpacityAtTemperature(ComputationalCell3D const& cell, double energy, double) const override
    {
        double const factor = multigroup_ ? kGroupAbsorptionFactor[findGroup(energy)] : 1.0;
        return factor * kAbsorptionKappa * cell.density;
    }
    double CalcDiffusionCoefficient(ComputationalCell3D const& cell) const override
    { return CG::speed_of_light / (3.0 * (kAbsorptionKappa + kScatteringKappa) * cell.density); }
    double CalcDiffusionCoefficient(ComputationalCell3D const& cell, double energy) const override
    {
        return CG::speed_of_light /
            (3.0 * (CalcAbsorptionOpacity(cell, energy) + kScatteringKappa * cell.density));
    }

private:
    bool multigroup_;
};

enum class Planted { None, Hot, Cold };

struct SyntheticSnapshot
{
    Snapshot3D snapshot;
    std::vector<Planted> planted;   // per cell
};

// Envelope cells between 0.55 and 0.8 half-boxes from the centre, every
// seventh one, take the planted temperatures (radiation field unchanged).
SyntheticSnapshot MakeSnapshot(bool plant)
{
    size_t const n = 12;
    double const spacing = 2.0 * kHalfBox / static_cast<double>(n);
    std::mt19937_64 rng(20261010);
    std::uniform_real_distribution<double> jitter(-0.2, 0.2);
    double const a = CG::radiation_constant;

    SyntheticSnapshot result;
    Snapshot3D& snapshot = result.snapshot;
    snapshot.ll = Vector3D(-kHalfBox, -kHalfBox, -kHalfBox);
    snapshot.ur = Vector3D(kHalfBox, kHalfBox, kHalfBox);
    snapshot.time = 1.0;
    snapshot.cycle = 1;
    size_t candidates = 0, hot = 0, cold = 0;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            for (size_t k = 0; k < n; ++k) {
                Vector3D const point(
                    -kHalfBox + (static_cast<double>(i) + 0.5 + jitter(rng)) * spacing,
                    -kHalfBox + (static_cast<double>(j) + 0.5 + jitter(rng)) * spacing,
                    -kHalfBox + (static_cast<double>(k) + 0.5 + jitter(rng)) * spacing);
                double const r = abs(point);
                bool const core = r < kCoreRadius;
                ComputationalCell3D cell;
                cell.density = core ? 1e-9 : 1e-14 * std::pow(kCoreRadius / r, 2.0);
                double const radiationTemperature = core ? 2e5 : kEnvelopeTemperature;
                cell.temperature = radiationTemperature;
                Planted mark = Planted::None;
                if (plant && r > 0.55 * kHalfBox && r < 0.8 * kHalfBox && (candidates++ % 7) == 0) {
                    if (hot < kHotCells) {
                        mark = Planted::Hot;
                        cell.temperature = kHotTemperature;
                        ++hot;
                    } else if (cold < kColdCells) {
                        mark = Planted::Cold;
                        cell.temperature = kColdTemperature;
                        ++cold;
                    }
                }
                cell.internal_energy = SpecificHeatF() * cell.temperature;
                cell.pressure = (kGamma - 1.0) * cell.density * cell.internal_energy;
                cell.velocity = Vector3D(0.0, 0.0, 0.0);
                cell.Erad = a * std::pow(radiationTemperature, 4) / cell.density;
                // Grey snapshot (no E_g): planck opacity scaling.
                cell.ID = snapshot.cells.size();
                snapshot.mesh_points.push_back(point);
                snapshot.cells.push_back(cell);
                result.planted.push_back(mark);
            }
    if (plant && (hot != kHotCells || cold != kColdCells))
        throw UniversalError("imc_postprocess_cooling_limit: not enough envelope cells to plant");
    return result;
}

PostProcessIMC::PostProcessScenario MakeScenario(bool plant)
{
    using namespace PostProcessIMC;
    PostProcessScenario scenario = MakeStaSnapshotScenario("imc-postprocess-weight-split");
    scenario.factories.loadSnapshot = [plant](PostProcessConfig::Input const&, ParallelContext const& parallel) {
        Snapshot3D snapshot = MakeSnapshot(plant).snapshot;
        if (parallel.rank != 0) {   // rank 0 holds every point, as a 1-rank snapshot file would
            snapshot.mesh_points.clear();
            snapshot.cells.clear();
        }
        return snapshot;
    };
    scenario.factories.transformSnapshot = [](Snapshot3D&, ParallelContext const&) {};
    scenario.factories.createEquationOfState = [](PostProcessConfig::Input const&) {
        return std::static_pointer_cast<EquationOfState>(std::make_shared<IdealGas>(kGamma, SpecificHeatF(), 1.0, 0.0));
    };
    scenario.factories.createMultigroupOpacity = [](PostProcessConfig::Input const&) {
        return std::static_pointer_cast<OpacityCalculator>(std::make_shared<SyntheticOpacity>(true));
    };
    scenario.factories.createGreyOpacity = [](PostProcessConfig::Input const&) {
        return std::static_pointer_cast<OpacityCalculator>(std::make_shared<SyntheticOpacity>(false));
    };
    scenario.factories.applyOpacityScaleFactors =
        [](OpacityCalculator&, std::unordered_map<size_t, double>) {};

    PostProcessConfig& c = scenario.defaults;
    c.input.snapshot = "synthetic-core-envelope";
    c.input.multigroupOpacityDirectory = "synthetic";
    c.input.greyOpacityDirectory = "synthetic";
    c.input.eosDirectory = "synthetic";
    c.output.writeVtk = false;
    c.observer.radius = 0.9 * kHalfBox;
    c.observer.count = 48;
    c.transport.sourceDt = 100.0;
    c.transport.duration = 1.0e5;
    c.transport.photonsPerCell = 10;
    // 30 final generations: the generation-error check compares two
    // standard deviations, each from n - 1 = 29 degrees of freedom.
    c.transport.generations = 30;
    c.transport.useCellVelocities = true;
    c.transport.ddmc = true;
    c.transport.ddmcMinCellOpticalDepth = 5.0;
    c.transport.randomWalk = false;
    c.transport.compton.enabled = false;
    c.polarization.enabled = true;
    c.photosphere.enabled = false;
    c.fluxSource.enabled = true;
    c.fluxSource.thermalizationTau = 5.0;
    c.fluxSource.constructionRays = 256;
    c.fluxSource.ddmcFaceOpticalDepth = 5.0;
    c.fluxSource.surfaceMode = FluxSourceSurfaceMode::MultigroupInnermost;
    c.volumeEmission.enabled = true;
    c.volumeEmission.cutoffFraction = 1e-10;
    c.volumeEmission.burninPacketsTarget = 20000;
    c.volumeEmission.gateGroups = false;
    c.volumeEmission.burninExact = true;
    c.volumeEmission.explorationWeightFraction = 0.05;
    c.volumeEmission.coolingLimit = true;
    c.volumeEmission.learnedPhotonsPerCellBudget = 20;
    c.volumeEmission.learnedMinPhotons = 1;
    c.volumeEmission.learnedMaxPhotons = 200;
    // Exercises the alpha rebuild after the limit (the synthetic scenario's
    // scale hook is a no-op, so transport is unchanged).
    c.opacityScaling.mode = OpacityScaleMode::Planck;
    c.adaptive.source.enabled = true;
    c.adaptive.source.learnedPhotonsPerCellBudget = 20;
    c.adaptive.source.learnedMinPhotons = 2;
    c.adaptive.source.learnedMaxPhotons = 200;
    c.adaptive.source.scorePower = 0.5;
    c.adaptive.observer.equity = true;
    c.adaptive.observer.targetPolarizationSigma = 1e-2;
    c.adaptive.observer.normalizeDeficits = true;
    c.loadBalance.measured = false;
    return scenario;
}

std::string const kOutputDir = "imc_postprocess_weight_split_output";
// The synthetic envelope is thin and fully learned, so its packets weigh at
// most ~5e40 erg; the thick core's heavy packets never leave their cells.
// A window of 1e-4 of an observer's share (~1e38 erg) makes envelope packets
// split, which is what the unbiasedness checks need.
constexpr double kSplitFraction = 1e-4;
int gRank = 0;
int gSize = 1;

PostProcessIMC::PostProcessResult Run(std::string const& name, double splitFraction,
                                      bool capLearnedWeight = true,
                                      double explorationFraction = -1.0)
{
    PostProcessIMC::PostProcessScenario const scenario = MakeScenario(false);
    PostProcessIMC::PostProcessConfig config = scenario.defaults;
    config.output.stem = kOutputDir + "/" + name;
    config.volumeEmission.capLearnedWeight = capLearnedWeight;
    if (explorationFraction >= 0.0)
        config.volumeEmission.explorationWeightFraction = explorationFraction;
    config.transport.splitObserverFraction = splitFraction;
    config.transport.splitEscapeReference = 0.1;
    config.transport.splitMaxCopies = 1000;
    // Under MPI, rebalance every 2 final generations so the per-cell escape
    // statistics migrate with the cells (no effect in a serial build).
    config.loadBalance.measured = true;
    config.loadBalance.finalPeriod = 2;
    if (gRank == 0)
        std::cout << "\n##### run " << name << ": split_observer_fraction=" << splitFraction
                  << " cap_learned_weight=" << (capLearnedWeight ? "true" : "false")
                  << " exploration_weight_fraction=" << config.volumeEmission.explorationWeightFraction << std::endl;
    return PostProcessIMC::RunPostProcess(config, scenario, PostProcessIMC::ParallelContext{gRank, gSize});
}

double RelDiff(double a, double b)
{
    double const scale = std::max(std::abs(a), std::abs(b));
    return scale > 0.0 ? std::abs(a - b) / scale : 0.0;
}

} // namespace

int main(int argc, char* argv[])
{
#ifdef RICH_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &gRank);
    MPI_Comm_size(MPI_COMM_WORLD, &gSize);
#else
    (void)argc;
    (void)argv;
#endif
    if (gRank == 0) {
        std::filesystem::remove_all(kOutputDir);
        std::filesystem::create_directories(kOutputDir);
    }
#ifdef RICH_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    std::ofstream metrics;
    if (gRank == 0)
        metrics.open("imc_postprocess_weight_split_metrics.txt");
    metrics << std::setprecision(17);
    bool pass = true;
    auto check = [&](bool ok, std::string const& what) {
        if (gRank != 0)
            return;
        std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
        pass = pass && ok;
    };

    try {
        auto const A = Run("A", 0.0);
        auto const B = Run("B", kSplitFraction);
        constexpr double kLowExploration = 3e-5;
        auto const C = Run("C", 0.0, true, kLowExploration);
        auto const D = Run("D", 0.0, false, kLowExploration);
        if (gRank == 0) {
            check(A.forward.splitCopies == 0 && A.grey.splitCopies == 0, "window off: no copies");
            check(B.forward.splitCopies > 0 && B.grey.splitCopies > 0,
                  "window on: copies made (MG " + std::to_string(B.forward.splitCopies) +
                  ", grey " + std::to_string(B.grey.splitCopies) + ")");
            check(RelDiff(A.forward.emittedEnergy, B.forward.emittedEnergy) < 1e-9 &&
                  RelDiff(A.grey.emittedEnergy, B.grey.emittedEnergy) < 1e-9,
                  "emitted energy unchanged by the window");
            for (auto const& [label, a, b] : {std::make_tuple(std::string("grey"), A.grey, B.grey),
                                             std::make_tuple(std::string("MG"), A.forward, B.forward)}) {
                double const sigma = std::hypot(a.crossingLuminosityStderr, b.crossingLuminosityStderr);
                double const z = sigma > 0.0 ? std::abs(a.crossingLuminosity - b.crossingLuminosity) / sigma : 0.0;
                metrics << label << "_L A " << a.crossingLuminosity << " +- " << a.crossingLuminosityStderr
                        << " B " << b.crossingLuminosity << " +- " << b.crossingLuminosityStderr
                        << " z " << z << " copies " << b.splitCopies << "\n";
                check(sigma > 0.0 && z < 4.0,
                      label + ": luminosity with the window agrees with it off (z=" + std::to_string(z) + ")");
                // Ratio of two 29-dof standard deviations: 1.5 is exceeded by
                // chance in ~1.5% of runs when the errors are equal.
                check(b.crossingLuminosityStderr <= 1.5 * a.crossingLuminosityStderr,
                      label + ": generation error not worse with the window (" +
                      std::to_string(b.crossingLuminosityStderr / a.crossingLuminosityStderr) + " of off)");
            }
            metrics << "cap_extra_photons C MG " << C.forward.weightCapExtraPhotons
                    << " grey " << C.grey.weightCapExtraPhotons
                    << " D MG " << D.forward.weightCapExtraPhotons
                    << " grey " << D.grey.weightCapExtraPhotons << "\n";
            check(D.forward.weightCapExtraPhotons == 0 && D.grey.weightCapExtraPhotons == 0,
                  "learned cap off: no packets added");
            check(C.forward.weightCapExtraPhotons > 0 && C.grey.weightCapExtraPhotons > 0,
                  "learned cap on: packets added (MG " + std::to_string(C.forward.weightCapExtraPhotons) +
                  ", grey " + std::to_string(C.grey.weightCapExtraPhotons) + ")");
            metrics << "cap_cells C MG " << C.forward.weightCappedCells << " of " << C.forward.allocatedCells
                    << " grey " << C.grey.weightCappedCells << " of " << C.grey.allocatedCells << "\n";
            check(C.forward.weightCappedCells > 0 && C.forward.weightCappedCells < C.forward.allocatedCells &&
                  C.grey.weightCappedCells > 0 && C.grey.weightCappedCells < C.grey.allocatedCells,
                  "learned cap selective: raises some cells, not all (MG " +
                  std::to_string(C.forward.weightCappedCells) + "/" + std::to_string(C.forward.allocatedCells) +
                  ", grey " + std::to_string(C.grey.weightCappedCells) + "/" +
                  std::to_string(C.grey.allocatedCells) + " cell-generations)");
            check(RelDiff(C.forward.emittedEnergy, D.forward.emittedEnergy) < 1e-9 &&
                  RelDiff(C.grey.emittedEnergy, D.grey.emittedEnergy) < 1e-9,
                  "emitted energy unchanged by the learned cap");
            for (auto const& [label, on, off] : {std::make_tuple(std::string("grey"), C.grey, D.grey),
                                                std::make_tuple(std::string("MG"), C.forward, D.forward)}) {
                double const sigma = std::hypot(on.crossingLuminosityStderr, off.crossingLuminosityStderr);
                double const z = sigma > 0.0 ? std::abs(on.crossingLuminosity - off.crossingLuminosity) / sigma : 0.0;
                metrics << label << "_cap cap_on " << on.crossingLuminosity << " +- " << on.crossingLuminosityStderr
                        << " cap_off " << off.crossingLuminosity << " +- " << off.crossingLuminosityStderr
                        << " z " << z << "\n";
                check(sigma > 0.0 && z < 4.0,
                      label + ": luminosity with the learned cap agrees with it off (z=" + std::to_string(z) + ")");
                check(on.crossingLuminosityStderr <= 1.5 * off.crossingLuminosityStderr,
                      label + ": generation error not worse with the learned cap (" +
                      std::to_string(on.crossingLuminosityStderr / off.crossingLuminosityStderr) + " of off)");
            }
            // The load balance before the probe runs only with MPI; it must
            // predict the probe's packets, with and without the learned cap.
            if (gSize > 1) {
                for (auto const& [label, pass] : {std::make_tuple(std::string("A MG"), A.forward),
                                                  std::make_tuple(std::string("A grey"), A.grey),
                                                  std::make_tuple(std::string("C MG"), C.forward),
                                                  std::make_tuple(std::string("C grey"), C.grey),
                                                  std::make_tuple(std::string("D MG"), D.forward),
                                                  std::make_tuple(std::string("D grey"), D.grey)}) {
                    double const ratio = pass.probePackets > 0
                        ? static_cast<double>(pass.probePredictedPackets) / static_cast<double>(pass.probePackets)
                        : 0.0;
                    metrics << "probe_prediction " << label << " predicted " << pass.probePredictedPackets
                            << " created " << pass.probePackets << " ratio " << ratio << "\n";
                    check(pass.probePredictedPackets > 0 && std::abs(ratio - 1.0) < 0.15,
                          label + ": probe packets predicted for the load balance match the probe (" +
                          std::to_string(pass.probePredictedPackets) + " vs " +
                          std::to_string(pass.probePackets) + ")");
                }
            }
        }
    } catch (UniversalError const& eo) {
        reportError(eo, std::cerr);
        pass = false;
    } catch (std::exception const& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        pass = false;
    }

    int passFlag = pass ? 1 : 0;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &passFlag, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
    if (gRank == 0) {
        metrics << "pass " << passFlag << "\n";
        std::cout << (passFlag ? "imc_postprocess_weight_split PASSED" : "imc_postprocess_weight_split FAILED")
                  << std::endl;
    }
#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return passFlag ? 0 : 1;
}
