// Regression checks for continuing a post-process result (restart.from).
//
// A small synthetic snapshot (an optically thick hot core inside a thin
// envelope) is post-processed with the paola configuration: CER flux source at
// the innermost per-group thermalization radius, volume emission, DDMC,
// polarization, adaptive source/observer/group allocation.
//
//   A   fresh, 3 final generations
//   R0  restart from A adding 0      -> identical to A (round trip)
//   X   restart from A adding 3      -> 6 generations
//   Y1  restart from A adding 1, Y restart from Y1 adding 2
//                                     -> Y matches X: a chain of restarts
//                                        reproduces one longer restart, so the
//                                        learned state is saved completely
//   C   fresh, 6 final generations   -> X agrees with C within the errors
//   a restart with a changed observer count, one that would overwrite its
//   source without restart.in-place, and one from a file that is not HDF5
//   must all be refused.
//
// The chain comparison needs one rank: there is no load balancing, so the
// decomposition and hence the random streams of X and Y are the same. Under
// MPI it is skipped and the rest (round trip, counts, statistics, refusals,
// plus the restart load balance) is checked.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <H5Cpp.h>

#include "source/3D/radiation/postprocess/IMCPostProcess.hpp"
#include "source/Radiation/OpacityCalculator.hpp"
#include "source/misc/universal_error.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/Radiation/conj_grad_solve.hpp"

#ifdef RICH_MPI
#include <mpi.h>
#endif

static_assert(ENERGY_GROUPS_NUM == 4,
              "imc_postprocess_restart is built with --energy_groups_num=4");

namespace
{

constexpr double kHalfBox = 1.0e14;          // cm
constexpr double kCoreRadius = 0.35 * kHalfBox;
constexpr std::array<double, 4> kGroupAbsorptionFactor = {0.5, 1.0, 2.0, 4.0};
constexpr double kAbsorptionKappa = 1.0;     // cm^2/g
constexpr double kScatteringKappa = 0.2;     // cm^2/g
constexpr double kGamma = 5.0 / 3.0;
constexpr double kMuMp = 0.6 * 1.67262192e-24;
// Ideal gas with e = f T, so the heat capacity is defined.
double SpecificHeatF() { return CG::boltzmann_constant / ((kGamma - 1.0) * kMuMp); }

// Constant mass opacities; the multigroup absorption depends on the group.
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
    {
        return CalcPlanckOpacityAtTemperature(cell, cell.temperature);
    }
    double CalcPlanckOpacityAtTemperature(ComputationalCell3D const& cell, double) const override
    {
        return kAbsorptionKappa * cell.density;
    }
    double CalcScatteringOpacity(ComputationalCell3D const& cell) const override
    {
        return CalcScatteringOpacityAtTemperature(cell, cell.temperature);
    }
    double CalcScatteringOpacityAtTemperature(ComputationalCell3D const& cell, double) const override
    {
        return kScatteringKappa * cell.density;
    }
    double CalcScatteringOpacity(ComputationalCell3D const& cell, double) const override
    {
        return kScatteringKappa * cell.density;
    }
    double CalcScatteringOpacityAtTemperature(ComputationalCell3D const& cell, double, double) const override
    {
        return kScatteringKappa * cell.density;
    }
    double CalcAbsorptionOpacity(ComputationalCell3D const& cell, double energy) const override
    {
        return CalcAbsorptionOpacityAtTemperature(cell, energy, cell.temperature);
    }
    double CalcAbsorptionOpacityAtTemperature(ComputationalCell3D const& cell, double energy, double) const override
    {
        double const factor = multigroup_ ? kGroupAbsorptionFactor[findGroup(energy)] : 1.0;
        return factor * kAbsorptionKappa * cell.density;
    }
    double CalcDiffusionCoefficient(ComputationalCell3D const& cell) const override
    {
        return CG::speed_of_light / (3.0 * (kAbsorptionKappa + kScatteringKappa) * cell.density);
    }
    double CalcDiffusionCoefficient(ComputationalCell3D const& cell, double energy) const override
    {
        return CG::speed_of_light /
            (3.0 * (CalcAbsorptionOpacity(cell, energy) + kScatteringKappa * cell.density));
    }

private:
    bool multigroup_;
};

Snapshot3D MakeSnapshot(int rank)
{
    size_t const n = 12;
    double const spacing = 2.0 * kHalfBox / static_cast<double>(n);
    std::mt19937_64 rng(20261008);
    std::uniform_real_distribution<double> jitter(-0.2, 0.2);
    double const radiation = CG::radiation_constant;

    Snapshot3D snapshot;
    snapshot.ll = Vector3D(-kHalfBox, -kHalfBox, -kHalfBox);
    snapshot.ur = Vector3D(kHalfBox, kHalfBox, kHalfBox);
    snapshot.time = 1.0;
    snapshot.cycle = 1;
    if (rank != 0)
        return snapshot;   // rank 0 holds every point, as a 1-rank snapshot file would
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
                cell.temperature = core ? 2e5 : 2e4;
                cell.internal_energy = SpecificHeatF() * cell.temperature;
                cell.pressure = (kGamma - 1.0) * cell.density * cell.internal_energy;
                cell.velocity = Vector3D(0.0, 0.0, 0.0);
                cell.Erad = radiation * std::pow(cell.temperature, 4) / cell.density;
                for (size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
                    cell.Eg[g] = cell.Erad / static_cast<double>(ENERGY_GROUPS_NUM);
                cell.ID = snapshot.cells.size();
                snapshot.mesh_points.push_back(point);
                snapshot.cells.push_back(cell);
            }
    return snapshot;
}

PostProcessIMC::PostProcessScenario MakeScenario()
{
    using namespace PostProcessIMC;
    PostProcessScenario scenario = MakeStaSnapshotScenario("imc-postprocess-restart");
    scenario.factories.loadSnapshot = [](PostProcessConfig::Input const&, ParallelContext const& parallel) {
        return MakeSnapshot(parallel.rank);
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
    c.volumeEmission.learnedPhotonsPerCellBudget = 20;
    c.volumeEmission.learnedMinPhotons = 1;
    c.volumeEmission.learnedMaxPhotons = 200;
    c.opacityScaling.mode = OpacityScaleMode::None;
    c.adaptive.source.enabled = true;
    c.adaptive.source.learnedPhotonsPerCellBudget = 20;
    c.adaptive.source.learnedMinPhotons = 2;
    c.adaptive.source.learnedMaxPhotons = 200;
    c.adaptive.source.scorePower = 0.5;
    c.adaptive.observer.equity = true;
    c.adaptive.observer.targetPolarizationSigma = 1e-2;
    c.adaptive.observer.normalizeDeficits = true;
    c.adaptive.group.quality = true;
    c.adaptive.group.sourceCells = true;
    c.adaptive.group.frequencySampling = true;
    c.loadBalance.measured = true;
    return scenario;
}

std::string const kOutputDir = "imc_postprocess_restart_output";

std::string Stem(std::string const& name) { return kOutputDir + "/" + name; }
std::string ResultFile(std::string const& name) { return Stem(name) + ".h5"; }

int gRank = 0;
int gSize = 1;

PostProcessIMC::PostProcessResult Run(
    PostProcessIMC::PostProcessScenario const& scenario,
    std::string const& name, size_t generations, std::string const& restartFrom)
{
    PostProcessIMC::PostProcessConfig config = scenario.defaults;
    config.output.stem = Stem(name);
    config.transport.generations = generations;
    config.restart.from = restartFrom;
    if (gRank == 0)
        std::cout << "\n##### run " << name << ": generations=" << generations
                  << " restart_from=" << (restartFrom.empty() ? "-" : restartFrom) << std::endl;
    return PostProcessIMC::RunPostProcess(config, scenario, PostProcessIMC::ParallelContext{gRank, gSize});
}

// ---------------------------------------------------------------------------
// HDF5 comparison
// ---------------------------------------------------------------------------

struct TreeComparison
{
    size_t datasets = 0;
    size_t structuralMismatches = 0;
    double maxRelativeDifference = 0.0;
    std::string worst;
};

void CompareGroup(H5::Group const& a, H5::Group const& b, std::string const& path,
                  TreeComparison& out)
{
    for (hsize_t i = 0; i < a.getNumObjs(); ++i) {
        std::string const name = a.getObjnameByIdx(i);
        std::string const child = path + "/" + name;
        if (!b.nameExists(name)) {
            ++out.structuralMismatches;
            std::cout << "  missing in second file: " << child << "\n";
            continue;
        }
        H5G_obj_t const type = a.getObjTypeByIdx(i);
        if (type == H5G_GROUP) {
            CompareGroup(a.openGroup(name), b.openGroup(name), child, out);
            continue;
        }
        if (type != H5G_DATASET)
            continue;
        H5::DataSet da = a.openDataSet(name);
        H5::DataSet db = b.openDataSet(name);
        if (da.getTypeClass() == H5T_STRING || da.getTypeClass() == H5T_VLEN)
            continue;
        hssize_t const na = da.getSpace().getSimpleExtentNpoints();
        hssize_t const nb = db.getSpace().getSimpleExtentNpoints();
        ++out.datasets;
        if (na != nb) {
            ++out.structuralMismatches;
            std::cout << "  size mismatch: " << child << " " << na << " vs " << nb << "\n";
            continue;
        }
        std::vector<double> va(static_cast<size_t>(na)), vb(static_cast<size_t>(nb));
        if (na > 0) {
            da.read(va.data(), H5::PredType::NATIVE_DOUBLE);
            db.read(vb.data(), H5::PredType::NATIVE_DOUBLE);
        }
        for (size_t k = 0; k < va.size(); ++k) {
            if (va[k] == vb[k] || (std::isnan(va[k]) && std::isnan(vb[k])))
                continue;
            double const scale = std::max(std::abs(va[k]), std::abs(vb[k]));
            double const rel = scale > 0.0 ? std::abs(va[k] - vb[k]) / scale : 0.0;
            if (rel > out.maxRelativeDifference) {
                out.maxRelativeDifference = rel;
                out.worst = child;
            }
        }
    }
}

TreeComparison CompareFiles(std::string const& fileA, std::string const& fileB,
                            std::vector<std::string> const& groups)
{
    TreeComparison out;
    H5::H5File a(fileA, H5F_ACC_RDONLY);
    H5::H5File b(fileB, H5F_ACC_RDONLY);
    for (std::string const& group : groups) {
        if (!a.nameExists(group) || !b.nameExists(group)) {
            ++out.structuralMismatches;
            std::cout << "  group missing: " << group << "\n";
            continue;
        }
        CompareGroup(a.openGroup(group), b.openGroup(group), group, out);
    }
    return out;
}

template <typename T>
T ReadScalar(std::string const& file, std::string const& path)
{
    H5::H5File f(file, H5F_ACC_RDONLY);
    T value{};
    H5::DataSet d = f.openDataSet(path);
    if constexpr (std::is_same_v<T, double>)
        d.read(&value, H5::PredType::NATIVE_DOUBLE);
    else
        d.read(&value, H5::PredType::NATIVE_ULLONG);
    return value;
}

std::string ReadString(std::string const& file, std::string const& path)
{
    H5::H5File f(file, H5F_ACC_RDONLY);
    H5::DataSet d = f.openDataSet(path);
    std::string value;
    d.read(value, d.getStrType());
    return value;
}

// Runs a post-process that must be refused. Its error report (with a stack
// trace) goes to a buffer, so the expected failure does not look like a crash
// to the regression framework's fatal-marker scan.
bool ThrowsContaining(std::function<void()> const& work, std::string const& needle,
                      std::string& message)
{
    std::ostringstream captured;
    std::streambuf* const original = std::cerr.rdbuf(captured.rdbuf());
    bool threw = false;
    try {
        work();
    } catch (UniversalError const& eo) {
        message = eo.getErrorMessage();
        threw = true;
    } catch (std::exception const& e) {
        message = e.what();
        threw = true;
    }
    std::cerr.rdbuf(original);
    if (!threw) {
        message = "no exception";
        return false;
    }
    return needle.empty() || message.find(needle) != std::string::npos;
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
    // Only rank 0 reads the results and reports.
    std::ofstream metrics;
    if (gRank == 0)
        metrics.open("imc_postprocess_restart_metrics.txt");
    metrics << std::setprecision(17);
    bool pass = true;
    auto check = [&](bool ok, std::string const& what) {
        if (gRank != 0)
            return;
        std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
        pass = pass && ok;
    };

    try {
        PostProcessIMC::PostProcessScenario const scenario = MakeScenario();
        std::vector<std::string> const passes = {"/passes/forward", "/passes/grey"};
        std::vector<std::string> stateGroups;
        for (std::string const& p : passes)
            for (char const* g : {"/tally", "/statistics_state", "/learned_state", "/load_balance"})
                stateGroups.push_back(p + g);

        Run(scenario, "A", 3, "");
        Run(scenario, "R0", 0, ResultFile("A"));
        Run(scenario, "X", 3, ResultFile("A"));
        Run(scenario, "Y1", 1, ResultFile("A"));
        Run(scenario, "Y", 2, ResultFile("Y1"));
        Run(scenario, "C", 6, "");
        metrics << "mpi_size " << gSize << "\n";

        if (gRank == 0) {
        // Round trip: adding nothing reproduces the result exactly.
        TreeComparison const roundTrip = CompareFiles(ResultFile("A"), ResultFile("R0"), stateGroups);
        metrics << "round_trip_datasets " << roundTrip.datasets << "\n"
                << "round_trip_structural_mismatches " << roundTrip.structuralMismatches << "\n"
                << "round_trip_max_rel " << roundTrip.maxRelativeDifference << "\n";
        check(roundTrip.datasets > 50 && roundTrip.structuralMismatches == 0 &&
              roundTrip.maxRelativeDifference == 0.0,
              "restart adding 0 generations reproduces A exactly (worst: " + roundTrip.worst + ")");

        // Chain: A+1+2 equals A+3 up to summation order of the learned maps.
        if (gSize == 1) {
            TreeComparison const chain = CompareFiles(ResultFile("X"), ResultFile("Y"), stateGroups);
            metrics << "chain_datasets " << chain.datasets << "\n"
                    << "chain_structural_mismatches " << chain.structuralMismatches << "\n"
                    << "chain_max_rel " << chain.maxRelativeDifference << "\n";
            check(chain.datasets > 50 && chain.structuralMismatches == 0 &&
                  chain.maxRelativeDifference < 1e-9,
                  "restart chain A+1+2 matches A+3 (worst: " + chain.worst + ")");
        } else {
            metrics << "chain_skipped_mpi 1\n";
        }

        // Generation counts and history.
        for (std::string const& p : passes) {
            size_t const a = ReadScalar<size_t>(ResultFile("A"), p + "/statistics_state/samples");
            size_t const x = ReadScalar<size_t>(ResultFile("X"), p + "/statistics_state/samples");
            size_t const y = ReadScalar<size_t>(ResultFile("Y"), p + "/statistics_state/samples");
            size_t const done = ReadScalar<size_t>(ResultFile("X"), p + "/learned_state/completed_final_generations");
            metrics << "samples" << (p == passes[0] ? "_forward" : "_grey")
                    << " " << a << " " << x << " " << y << "\n";
            check(a == 3 && x == 6 && y == 6 && done == 6, p + " generation counts 3/6/6");
            size_t const learned = ReadScalar<size_t>(ResultFile("X"), p + "/learned_state/adaptive/score_cell_ids_shape");
            check(learned > 0, p + " learned cells saved (" + std::to_string(learned) + ")");
        }
        size_t const segA = ReadScalar<size_t>(ResultFile("A"), "/restart/segment_count");
        size_t const segX = ReadScalar<size_t>(ResultFile("X"), "/restart/segment_count");
        size_t const segY = ReadScalar<size_t>(ResultFile("Y"), "/restart/segment_count");
        metrics << "segments " << segA << " " << segX << " " << segY << "\n";
        check(segA == 1 && segX == 2 && segY == 3, "run history segment counts 1/2/3");
        check(ReadString(ResultFile("Y"), "/restart/history/segment_0002/restart_from") == ResultFile("Y1"),
              "run history records the restart source");

        // Statistics: X (A continued) and C (fresh) estimate the same thing.
        for (std::string const& p : passes) {
            double const lx = ReadScalar<double>(ResultFile("X"), p + "/summary/crossing_luminosity");
            double const sx = ReadScalar<double>(ResultFile("X"), p + "/summary/crossing_luminosity_stderr");
            double const lc = ReadScalar<double>(ResultFile("C"), p + "/summary/crossing_luminosity");
            double const sc = ReadScalar<double>(ResultFile("C"), p + "/summary/crossing_luminosity_stderr");
            double const sigma = std::sqrt(sx * sx + sc * sc);
            double const z = sigma > 0.0 ? std::abs(lx - lc) / sigma : 0.0;
            metrics << "luminosity" << (p == passes[0] ? "_forward" : "_grey")
                    << " " << lx << " " << sx << " " << lc << " " << sc << " z=" << z << "\n";
            check(lx > 0.0 && lc > 0.0 && sx > 0.0 && sc > 0.0 && z < 5.0,
                  p + " restarted luminosity agrees with a fresh run (z=" + std::to_string(z) + ")");
        }

        }

        // Refusals (collective).
        std::string message;
        bool const mismatchRefused = ThrowsContaining([&]() {
            PostProcessIMC::PostProcessScenario changed = scenario;
            changed.defaults.observer.count = 64;
            Run(changed, "bad_observers", 1, ResultFile("A"));
        }, "observer.count", message);
        metrics << "mismatch_refused " << (mismatchRefused ? 1 : 0) << "\n";
        check(mismatchRefused, "restart with a different observer count is refused: " + message);
        bool const overwriteRefused = ThrowsContaining([&]() {
            Run(scenario, "A", 1, ResultFile("A"));
        }, "", message);
        metrics << "overwrite_refused " << (overwriteRefused ? 1 : 0) << "\n";
        // A file that is not HDF5 must stop every rank with a message, not
        // terminate rank 0 on an HDF5 C++ exception.
        if (gRank == 0) {
            std::ofstream junk(kOutputDir + "/not_hdf5.h5");
            junk << "not an HDF5 file\n";
        }
#ifdef RICH_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        bool const corruptRefused = ThrowsContaining([&]() {
            Run(scenario, "from_corrupt", 1, kOutputDir + "/not_hdf5.h5");
        }, "HDF5", message);
        metrics << "corrupt_refused " << (corruptRefused ? 1 : 0) << "\n";
        check(corruptRefused, "restart from a non-HDF5 file is refused on every rank: " + message);
        check(overwriteRefused &&
              (gRank != 0 || ReadScalar<size_t>(ResultFile("A"), "/restart/segment_count") == 1),
              "restart onto its own source without restart.in-place is refused");
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
        std::cout << (passFlag ? "imc_postprocess_restart PASSED" : "imc_postprocess_restart FAILED") << std::endl;
    }
#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return passFlag ? 0 : 1;
}
