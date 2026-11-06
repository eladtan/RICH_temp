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
//
// Per-group window and diagnostics: Bd = B with provenance, generation records
// and band reports (science bit-identical to B); E = per-group window (paired
// with Bd: 12 observer-summed endpoints, max |t| < 4, and per quantity and
// group the mean over observers of the squared paired t < 2.5). V1B / V1E =
// Bd / E without scattering (Q = U = 0 exactly, finite records, ledger
// closes); V1Er = V1E with band reports (windows from the full snapshot equal
// those from the energy-only reduction, bit-identical).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
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

#include <H5Cpp.h>

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
constexpr double kScatteringKappaDefault = 0.2;   // cm^2/g
// Variant without scattering (runs V1*): emission is unpolarized and nothing
// polarizes it, so every Stokes Q and U must be exactly zero.
bool gNoScattering = false;
double ScatteringKappa() { return gNoScattering ? 0.0 : kScatteringKappaDefault; }
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
    { return ScatteringKappa() * cell.density; }
    double CalcScatteringOpacityAtTemperature(ComputationalCell3D const& cell, double) const override
    { return ScatteringKappa() * cell.density; }
    double CalcScatteringOpacity(ComputationalCell3D const& cell, double) const override
    { return ScatteringKappa() * cell.density; }
    double CalcScatteringOpacityAtTemperature(ComputationalCell3D const& cell, double, double) const override
    { return ScatteringKappa() * cell.density; }
    double CalcAbsorptionOpacity(ComputationalCell3D const& cell, double energy) const override
    { return CalcAbsorptionOpacityAtTemperature(cell, energy, cell.temperature); }
    double CalcAbsorptionOpacityAtTemperature(ComputationalCell3D const& cell, double energy, double) const override
    {
        double const factor = multigroup_ ? kGroupAbsorptionFactor[findGroup(energy)] : 1.0;
        return factor * kAbsorptionKappa * cell.density;
    }
    double CalcDiffusionCoefficient(ComputationalCell3D const& cell) const override
    { return CG::speed_of_light / (3.0 * (kAbsorptionKappa + ScatteringKappa()) * cell.density); }
    double CalcDiffusionCoefficient(ComputationalCell3D const& cell, double energy) const override
    {
        return CG::speed_of_light /
            (3.0 * (CalcAbsorptionOpacity(cell, energy) + ScatteringKappa() * cell.density));
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
    // Grey pass follows the MG values (the library defaults set its own).
    c.volumeEmission.explorationWeightFractionGrey = -1.0;
    c.volumeEmission.learnedPhotonsPerCellBudgetGrey = 0;
    // Cooling limit in temperature mode, as the checks were written for.
    c.volumeEmission.coolingLimitMode = CoolingLimitMode::Temperature;
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
                                      double explorationFraction = -1.0,
                                      std::function<void(PostProcessIMC::PostProcessConfig&)> const& tweak = {})
{
    PostProcessIMC::PostProcessScenario const scenario = MakeScenario(false);
    PostProcessIMC::PostProcessConfig config = scenario.defaults;
    if (tweak)
        tweak(config);
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

// Per-group window cases (band-SNR step 4):
//   Bd   B with packet provenance and generation Stokes records: the science
//        must be identical to B (the diagnostics change nothing)
//   E    B with the per-group window (split-group-scale kGroupScale): more
//        copies, also in the dimmest group, no budget refusal, emitted
//        energy unchanged, and per-group L, Q, U consistent with Bd
//   V1B, V1E  B and E without scattering: Q = U = 0 exactly in every bin of
//        both passes, and the box energy ledger closes
// Scale and share floor of the per-group window here: the scenario's scalar
// window is already 1e-4 of an observer share, so the plan's 0.2 / 1e-3
// would make the dim group's window ~1e-5 of a typical packet and cascade
// past the exact float split-factor range (the run stops by design).
constexpr double kGroupScale = 0.5;
constexpr double kGroupFloor = 1e-2;

std::vector<double> ReadDoubles(std::string const& file, std::string const& path)
{
    H5::H5File f(file, H5F_ACC_RDONLY);
    H5::DataSet d = f.openDataSet(path);
    hssize_t const n = d.getSpace().getSimpleExtentNpoints();
    std::vector<double> values(static_cast<size_t>(n), 0.0);
    if (n > 0)
        d.read(values.data(), H5::PredType::NATIVE_DOUBLE);
    return values;
}

bool Exists(std::string const& file, std::string const& path)
{
    H5::H5File f(file, H5F_ACC_RDONLY);
    return H5Lexists(f.getId(), path.c_str(), H5P_DEFAULT) > 0;
}

// Generation records of a pass: per generation, sums over observers of the
// raw energy, Stokes Q and U of every group ([gen][3 * G], G groups).
struct GenerationSeries
{
    size_t generations = 0;
    size_t groups = 0;
    std::vector<std::vector<double>> rows;
};

GenerationSeries ReadGenerationSeries(std::string const& file, std::string const& pass)
{
    std::string const p = pass + "/diagnostics/generation_stokes/";
    std::vector<double> const shape = ReadDoubles(file, p + "energy_shape");
    GenerationSeries series;
    series.generations = static_cast<size_t>(shape.at(0));
    size_t const observers = static_cast<size_t>(shape.at(1));
    series.groups = static_cast<size_t>(shape.at(2));
    std::vector<double> const e = ReadDoubles(file, p + "energy");
    std::vector<double> const q = ReadDoubles(file, p + "stokes_q");
    std::vector<double> const u = ReadDoubles(file, p + "stokes_u");
    series.rows.assign(series.generations, std::vector<double>(3 * series.groups, 0.0));
    for (size_t j = 0; j < series.generations; ++j)
        for (size_t o = 0; o < observers; ++o)
            for (size_t g = 0; g < series.groups; ++g) {
                size_t const k = (j * observers + o) * series.groups + g;
                series.rows[j][g] += e.at(k);
                series.rows[j][series.groups + g] += q.at(k);
                series.rows[j][2 * series.groups + g] += u.at(k);
            }
    return series;
}

// The same records, not summed over observers: values[quantity] at
// (j * observers + o) * groups + g, quantity 0 energy, 1 Stokes Q, 2 Stokes U.
struct GenerationCube
{
    size_t generations = 0;
    size_t observers = 0;
    size_t groups = 0;
    std::array<std::vector<double>, 3> values;
};

GenerationCube ReadGenerationCube(std::string const& file, std::string const& pass)
{
    std::string const p = pass + "/diagnostics/generation_stokes/";
    std::vector<double> const shape = ReadDoubles(file, p + "energy_shape");
    GenerationCube cube;
    cube.generations = static_cast<size_t>(shape.at(0));
    cube.observers = static_cast<size_t>(shape.at(1));
    cube.groups = static_cast<size_t>(shape.at(2));
    cube.values[0] = ReadDoubles(file, p + "energy");
    cube.values[1] = ReadDoubles(file, p + "stokes_q");
    cube.values[2] = ReadDoubles(file, p + "stokes_u");
    return cube;
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
        auto const diagnostics = [](PostProcessIMC::PostProcessConfig& c) {
            c.diagnostics.packetProvenance = true;
            c.diagnostics.generationStokes = true;
        };
        auto const perGroup = [&](PostProcessIMC::PostProcessConfig& c) {
            diagnostics(c);
            c.transport.splitPerGroup = true;
            c.transport.splitGroupScale = kGroupScale;
            c.transport.splitGroupFloor = kGroupFloor;
        };
        // Bd also prints the band reports: the science must not change.
        auto const Bd = Run("Bd", kSplitFraction, true, -1.0, [&](PostProcessIMC::PostProcessConfig& c) {
            diagnostics(c);
            c.diagnostics.bandReports = true;
        });
        // The scalar window of this scenario (1e-4 of an observer share)
        // already uses most of the default copy budget (1e6 per rank per
        // generation); the per-group window wants more copies, so E gets the
        // budget an accepted setting needs: no reduced or refused split.
        auto const E = Run("E", kSplitFraction, true, -1.0, [&](PostProcessIMC::PostProcessConfig& c) {
            perGroup(c);
            c.transport.splitCopyBudget = 20000000;
        });
        // The no-scattering variant is checked exactly, so a few final
        // generations suffice (its envelope splits heavily).
        auto const shortRun = [](std::function<void(PostProcessIMC::PostProcessConfig&)> const& base) {
            return [base](PostProcessIMC::PostProcessConfig& c) {
                base(c);
                c.transport.generations = 6;
            };
        };
        gNoScattering = true;
        auto const V1B = Run("V1B", kSplitFraction, true, -1.0, shortRun(diagnostics));
        auto const V1E = Run("V1E", kSplitFraction, true, -1.0, shortRun(perGroup));
        // V1E with the band reports: its group windows then come from the full
        // global snapshot instead of the energy-only reduction; the two must
        // agree exactly.
        auto const V1Er = Run("V1Er", kSplitFraction, true, -1.0, shortRun([&](PostProcessIMC::PostProcessConfig& c) {
            perGroup(c);
            c.diagnostics.bandReports = true;
        }));
        gNoScattering = false;
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
            // ---- per-group window and diagnostics (step 4) ----
            std::string const fB = kOutputDir + "/B.h5", fBd = kOutputDir + "/Bd.h5", fE = kOutputDir + "/E.h5";
            bool identical = B.forward.crossingLuminosity == Bd.forward.crossingLuminosity &&
                B.grey.crossingLuminosity == Bd.grey.crossingLuminosity &&
                B.forward.splitCopies == Bd.forward.splitCopies && B.grey.splitCopies == Bd.grey.splitCopies;
            std::string differing;
            for (char const* path : {"/passes/forward/statistics_state/group_energy/sum",
                                     "/passes/forward/statistics_state/group_energy/sum_sq",
                                     "/passes/forward/statistics_state/group_stokes_q/sum",
                                     "/passes/forward/statistics_state/group_stokes_u/sum",
                                     "/passes/forward/statistics_state/group_energy_weight_sq_sum",
                                     "/passes/forward/tally/multigroup/group_luminosity",
                                     "/passes/grey/tally/luminosity"}) {
                if (ReadDoubles(fB, path) != ReadDoubles(fBd, path)) {
                    identical = false;
                    differing += std::string(" ") + path;
                }
            }
            check(identical, "provenance, generation records and band reports leave the science unchanged" +
                  (differing.empty() ? std::string() : " (differs:" + differing + ")"));
            bool const recordsPresent =
                Exists(fBd, "/passes/forward/diagnostics/packet_provenance/class") &&
                Exists(fBd, "/passes/forward/diagnostics/generation_stokes/energy") &&
                Exists(fBd, "/passes/grey/diagnostics/generation_stokes/energy") &&
                !Exists(fB, "/passes/forward/diagnostics/generation_stokes/energy");
            check(recordsPresent, "provenance tables and generation records written only when asked");
            if (recordsPresent) {
                // Escapes whose history includes a source-face re-emission
                // (channel table [group][mask][x][quantity], quantity 0 = count,
                // face bit = mask 8): the guarded face handler ran, and its
                // RNG checks held for every re-emission (they throw).
                double faceEscapes = std::numeric_limits<double>::infinity();   // minimum over Bd, E
                for (std::string const& file : {fBd, fE}) {
                    double count = 0.0;
                    std::vector<double> const shape =
                        ReadDoubles(file, "/passes/forward/diagnostics/packet_provenance/channel_shape");
                    std::vector<double> const table =
                        ReadDoubles(file, "/passes/forward/diagnostics/packet_provenance/channel");
                    if (shape.size() == 4 && shape[1] == 16.0) {
                        size_t const nG = static_cast<size_t>(shape[0]), nX = static_cast<size_t>(shape[2]),
                                     nQ = static_cast<size_t>(shape[3]);
                        if (table.size() == nG * 16 * nX * nQ)
                            for (size_t g = 0; g < nG; ++g)
                                for (size_t mask = 8; mask < 16; ++mask)
                                    for (size_t x = 0; x < nX; ++x)
                                        count += table[((g * 16 + mask) * nX + x) * nQ];
                    }
                    metrics << "face_reemitted_escapes " << file << " " << count << "\n";
                    faceEscapes = std::min(faceEscapes, count);
                }
                check(faceEscapes > 0.0, "source-face re-emission exercised (escapes with a face channel: " +
                      std::to_string(faceEscapes) + ")");
            }
            if (recordsPresent) {
                GenerationSeries const series = ReadGenerationSeries(fBd, "/passes/forward");
                std::vector<double> const sum = ReadDoubles(fBd, "/passes/forward/statistics_state/group_energy/sum");
                double recordTotal = 0.0, statTotal = 0.0;
                for (auto const& row : series.rows)
                    for (size_t g = 0; g < series.groups; ++g)
                        recordTotal += row[g];
                for (double const x : sum)
                    statTotal += x;
                metrics << "records generations " << series.generations << " energy " << recordTotal
                        << " statistics " << statTotal << "\n";
                check(series.generations == 30 && RelDiff(recordTotal, statTotal) < 1e-12,
                      "generation records sum to the running statistics (" + std::to_string(series.generations) +
                      " generations)");
            }

            uint64_t eGroupCopies = 0;
            for (uint64_t const c : E.forward.groupSplitCopies)
                eGroupCopies += c;
            std::vector<double> const eLum = ReadDoubles(fE, "/passes/forward/tally/multigroup/group_luminosity");
            std::vector<double> groupShare(4, 0.0);
            for (size_t k = 0; k < eLum.size(); ++k)
                groupShare[k % 4] += eLum[k];
            size_t dim = 0;
            for (size_t g = 1; g < 4; ++g)
                if (groupShare[g] < groupShare[dim])
                    dim = g;
            metrics << "group_window copies E " << E.forward.splitCopies << " B " << B.forward.splitCopies
                    << " dim_group " << dim << " per_group";
            for (uint64_t const c : E.forward.groupSplitCopies)
                metrics << " " << c;
            metrics << " refusals " << E.forward.splitRefusals << " " << E.grey.splitRefusals << "\n";
            check(E.forward.groupSplitCopies.size() == 4 && eGroupCopies == E.forward.splitCopies &&
                  E.forward.splitCopies > B.forward.splitCopies &&
                  E.forward.groupSplitCopies[dim] > 0,
                  "per-group window: more copies than the scalar window, also in the dimmest group " +
                  std::to_string(dim));
            check(E.forward.splitRefusals == 0 && E.grey.splitRefusals == 0 && B.forward.splitRefusals == 0,
                  "no copy-budget refusal");
            check(RelDiff(A.forward.emittedEnergy, E.forward.emittedEnergy) < 1e-9,
                  "emitted energy unchanged by the per-group window");
            {
                // Predeclared family of 12 endpoints: per group the sums over
                // observers of the energy, Q and U per generation. Bd and E
                // share their seeds, so the comparison uses the per-generation
                // differences (paired); Bonferroni |t| < 4 over the 12
                // (29 degrees of freedom: family error about 0.5%). Records,
                // means and errors must be finite, and an endpoint without
                // generation-to-generation spread must agree exactly.
                GenerationSeries const sb = ReadGenerationSeries(fBd, "/passes/forward");
                GenerationSeries const se = ReadGenerationSeries(fE, "/passes/forward");
                bool valid = sb.rows.size() == 30 && se.rows.size() == 30 && sb.groups == 4 && se.groups == 4;
                double worst = 0.0;
                size_t const columns = valid ? sb.rows[0].size() : 0;
                for (size_t c = 0; c < columns; ++c) {
                    size_t const n = sb.rows.size();
                    double mean = 0.0, var = 0.0;
                    for (size_t j = 0; j < n; ++j)
                        mean += (se.rows[j][c] - sb.rows[j][c]) / static_cast<double>(n);
                    for (size_t j = 0; j < n; ++j) {
                        double const d = se.rows[j][c] - sb.rows[j][c] - mean;
                        var += d * d;
                    }
                    double const sem = std::sqrt(var / static_cast<double>(n - 1) / static_cast<double>(n));
                    double scale = 0.0;
                    for (size_t j = 0; j < n; ++j)
                        scale = std::max(scale, std::max(std::abs(sb.rows[j][c]), std::abs(se.rows[j][c])));
                    double t = 0.0;
                    if (!std::isfinite(mean) || !std::isfinite(sem)) {
                        valid = false;
                    } else if (sem > 0.0) {
                        t = std::abs(mean) / sem;
                    } else if (std::abs(mean) > 1e-12 * std::max(scale, 1e-300)) {
                        valid = false;   // no spread, yet different
                    }
                    worst = std::max(worst, t);
                    metrics << "group_window endpoint " << c << " mean_difference " << mean << " +- " << sem
                            << " t " << t << "\n";
                }
                check(valid && columns == 12 && worst < 4.0,
                      "per-group L, Q, U with the group window agree with the scalar window (paired, max |t| = " +
                      std::to_string(worst) + ")");
            }
            {
                // Observer-resolved family (predeclared): sums over observers
                // can cancel opposite changes in different viewing directions,
                // so per quantity (energy, Q, U) and group take the mean over
                // observers of the squared paired t of E - Bd. Pure noise gives
                // about 1.07 (29 degrees of freedom); fixed-code runs give
                // 0.8-1.8, the heaviest tails in the dimmest group. Limit 2.5
                // for each of the 12.
                GenerationCube const cb = ReadGenerationCube(fBd, "/passes/forward");
                GenerationCube const ce = ReadGenerationCube(fE, "/passes/forward");
                size_t const n = cb.generations, nObs = cb.observers, nG = cb.groups;
                bool valid = n == 30 && ce.generations == n && nObs > 0 && ce.observers == nObs &&
                    nG == 4 && ce.groups == nG;
                for (size_t qty = 0; qty < 3 && valid; ++qty)
                    valid = cb.values[qty].size() == n * nObs * nG && ce.values[qty].size() == n * nObs * nG;
                double worstMeanT2 = 0.0;
                for (size_t qty = 0; qty < 3 && valid; ++qty) {
                    for (size_t g = 0; g < nG && valid; ++g) {
                        double sumT2 = 0.0;
                        for (size_t o = 0; o < nObs && valid; ++o) {
                            double mean = 0.0, var = 0.0, scale = 0.0;
                            for (size_t j = 0; j < n; ++j) {
                                size_t const k = (j * nObs + o) * nG + g;
                                mean += (ce.values[qty][k] - cb.values[qty][k]) / static_cast<double>(n);
                                scale = std::max(scale, std::max(std::abs(cb.values[qty][k]), std::abs(ce.values[qty][k])));
                            }
                            for (size_t j = 0; j < n; ++j) {
                                size_t const k = (j * nObs + o) * nG + g;
                                double const d = ce.values[qty][k] - cb.values[qty][k] - mean;
                                var += d * d;
                            }
                            double const sem = std::sqrt(var / static_cast<double>(n - 1) / static_cast<double>(n));
                            if (!std::isfinite(mean) || !std::isfinite(sem))
                                valid = false;
                            else if (sem > 0.0)
                                sumT2 += (mean / sem) * (mean / sem);
                            else if (std::abs(mean) > 1e-12 * std::max(scale, 1e-300))
                                valid = false;   // no spread, yet different
                        }
                        double const meanT2 = sumT2 / static_cast<double>(nObs);
                        worstMeanT2 = std::max(worstMeanT2, meanT2);
                        metrics << "group_window observer_family quantity " << qty << " group " << g
                                << " mean_t2 " << meanT2 << "\n";
                    }
                }
                check(valid && worstMeanT2 < 2.5,
                      "per-observer L, Q, U with the group window agree with the scalar window (paired, worst mean t^2 "
                      "over observers = " + std::to_string(worstMeanT2) + ")");
            }
            for (auto const& [name, run] : {std::make_tuple(std::string("V1B"), V1B),
                                            std::make_tuple(std::string("V1E"), V1E)}) {
                std::string const f = kOutputDir + "/" + name + ".h5";
                double worstStokes = 0.0;
                // Records present, of equal length and finite: a NaN would
                // leave the std::max below unchanged and pass the check.
                bool recordsFinite = true;
                for (char const* pass : {"/passes/forward", "/passes/grey"}) {
                    std::string const p = std::string(pass) + "/diagnostics/generation_stokes/";
                    std::vector<double> const e = ReadDoubles(f, p + "energy");
                    std::vector<double> const q = ReadDoubles(f, p + "stokes_q");
                    std::vector<double> const u = ReadDoubles(f, p + "stokes_u");
                    if (e.empty() || q.size() != e.size() || u.size() != e.size()) {
                        recordsFinite = false;
                        continue;
                    }
                    for (size_t k = 0; k < e.size(); ++k) {
                        if (!std::isfinite(e[k]) || !std::isfinite(q[k]) || !std::isfinite(u[k]) || !(e[k] >= 0.0)) {
                            recordsFinite = false;
                            continue;
                        }
                        double const scale = std::max(e[k], 1e-300);
                        worstStokes = std::max(worstStokes, std::max(std::abs(q[k]), std::abs(u[k])) / scale);
                    }
                }
                double const emitted = ReadDoubles(f, "/passes/forward/diagnostics/emitted_energy").at(0);
                double const residual = ReadDoubles(f, "/passes/forward/diagnostics/transport_sink_residual").at(0);
                metrics << "no_scattering " << name << " max_abs_stokes_over_I " << worstStokes
                        << " ledger_residual_over_emitted " << residual / emitted << " copies " << run.forward.splitCopies << "\n";
                check(recordsFinite && worstStokes <= 1e-12,
                      name + ": without scattering Q = U = 0 in every bin of both passes (records finite: " +
                      std::to_string(recordsFinite) + ", max " + std::to_string(worstStokes) + ")");
                check(emitted > 0.0 && std::abs(residual) <= 1e-9 * emitted,
                      name + ": box energy ledger closes (residual/emitted " + std::to_string(residual / emitted) + ")");
            }
            check(V1E.forward.splitCopies > V1B.forward.splitCopies,
                  "no scattering: the per-group window splits more than the scalar window");
            {
                std::string const f1 = kOutputDir + "/V1E.h5", f2 = kOutputDir + "/V1Er.h5";
                bool same = V1E.forward.splitCopies == V1Er.forward.splitCopies &&
                    V1E.forward.groupSplitCopies == V1Er.forward.groupSplitCopies &&
                    V1E.forward.crossingLuminosity == V1Er.forward.crossingLuminosity;
                for (char const* path : {"/passes/forward/tally/multigroup/group_luminosity",
                                         "/passes/forward/statistics_state/group_energy/sum"})
                    same = same && ReadDoubles(f1, path) == ReadDoubles(f2, path);
                check(same, "group windows from the energy-only reduction equal those from the full snapshot "
                            "(V1E vs V1Er bit-identical)");
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
