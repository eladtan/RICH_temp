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
// Per-group weight window (transport.split-per-group, restart schema 3):
//   GA fresh with the window, GR0 = GA + 0, GX = GA + 3, GY1 = GA + 1,
//   GY = GY1 + 2. GR0 reproduces GA, GY matches GX (serial), and the saved
//   split state and spectrum generation follow the chain. Refused: the mode
//   on from a file written with it off, from a schema-2 file (accepted with
//   the mode off), and from files whose split state is missing, lists a cell
//   twice, has an inconsistent spectrum validity, or has NaN or infinite
//   group boundaries. The grey pass never stores split state.
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
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
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
    // Multigroup snapshot (E_g set) with the grey hydro face flux.
    c.fluxSource.faceFlux = FluxSourceFaceFlux::Hydro;
    c.volumeEmission.enabled = true;
    c.volumeEmission.cutoffFraction = 1e-10;
    c.volumeEmission.burninPacketsTarget = 20000;
    c.volumeEmission.gateGroups = false;
    c.volumeEmission.burninExact = true;
    c.volumeEmission.explorationWeightFraction = 0.05;
    c.volumeEmission.learnedPhotonsPerCellBudget = 20;
    c.volumeEmission.learnedMinPhotons = 1;
    c.volumeEmission.learnedMaxPhotons = 200;
    // Grey pass follows the MG values (the library defaults set its own).
    c.volumeEmission.explorationWeightFractionGrey = -1.0;
    c.volumeEmission.learnedPhotonsPerCellBudgetGrey = 0;
    // Cooling limit in temperature mode, as the checks were written for.
    c.volumeEmission.coolingLimitMode = CoolingLimitMode::Temperature;
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
// Rank 0: copies `from` to `to` and applies `edit` to the copy (HDF5).
void EditedCopy(std::string const& from, std::string const& to,
                std::function<void(H5::H5File&)> const& edit)
{
    std::filesystem::copy_file(from, to, std::filesystem::copy_options::overwrite_existing);
    H5::H5File f(to, H5F_ACC_RDWR);
    edit(f);
}

void DeleteLink(H5::H5File& f, std::string const& path)
{
    if (H5Lexists(f.getId(), path.c_str(), H5P_DEFAULT) > 0)
        H5Ldelete(f.getId(), path.c_str(), H5P_DEFAULT);
}

void WriteInt(H5::H5File& f, std::string const& path, int value)
{
    H5::DataSet d = f.openDataSet(path);
    d.write(&value, H5::PredType::NATIVE_INT);
}

// Rank 0: rewrites a 1-D double dataset (and its _shape) with new values.
void RewriteDoubles(H5::H5File& f, std::string const& path, std::vector<double> const& values)
{
    DeleteLink(f, path);
    DeleteLink(f, path + "_shape");
    hsize_t const dims[1] = {values.size()};
    H5::DataSet d = f.createDataSet(path, H5::PredType::NATIVE_DOUBLE, H5::DataSpace(1, dims));
    if (!values.empty())
        d.write(values.data(), H5::PredType::NATIVE_DOUBLE);
    unsigned long long const n = values.size();
    hsize_t const one[1] = {1};
    H5::DataSet shape = f.createDataSet(path + "_shape", H5::PredType::NATIVE_ULLONG, H5::DataSpace(1, one));
    shape.write(&n, H5::PredType::NATIVE_ULLONG);
}

std::vector<double> ReadDoublesAt(H5::H5File& f, std::string const& path)
{
    H5::DataSet d = f.openDataSet(path);
    hssize_t const n = d.getSpace().getSimpleExtentNpoints();
    std::vector<double> values(static_cast<size_t>(n));
    if (n > 0)
        d.read(values.data(), H5::PredType::NATIVE_DOUBLE);
    return values;
}

bool LinkExists(std::string const& file, std::string const& path)
{
    H5::H5File f(file, H5F_ACC_RDONLY);
    return H5Lexists(f.getId(), path.c_str(), H5P_DEFAULT) > 0;
}

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

        // ---- per-group weight window: restart schema 3 ----
        PostProcessIMC::PostProcessScenario group = scenario;
        group.defaults.transport.splitObserverFraction = 1e-4;
        group.defaults.transport.splitEscapeReference = 0.1;
        group.defaults.transport.splitPerGroup = true;
        group.defaults.transport.splitGroupScale = 0.2;
        Run(group, "GA", 3, "");
        Run(group, "GR0", 0, ResultFile("GA"));
        Run(group, "GX", 3, ResultFile("GA"));
        Run(group, "GY1", 1, ResultFile("GA"));
        Run(group, "GY", 2, ResultFile("GY1"));
        std::string const split = "/passes/forward/learned_state/weight_split";
        if (gRank == 0) {
            check(LinkExists(ResultFile("GA"), split + "/present") &&
                  ReadScalar<double>(ResultFile("GA"), split + "/present") == 1.0 &&
                  !LinkExists(ResultFile("GA"), "/passes/grey/learned_state/weight_split") &&
                  !LinkExists(ResultFile("A"), split),
                  "split state saved for the multigroup pass with the per-group window only");
            size_t const cellsSaved = ReadScalar<size_t>(ResultFile("GA"), split + "/count");
            metrics << "group_window_split_cells " << cellsSaved << "\n";
            check(cellsSaved > 0 && ReadScalar<double>(ResultFile("GA"), split + "/spectrum_valid") == 1.0,
                  "split state holds cells (" + std::to_string(cellsSaved) + ") and a valid spectrum");
            TreeComparison const groupRoundTrip = CompareFiles(ResultFile("GA"), ResultFile("GR0"), stateGroups);
            metrics << "group_round_trip_datasets " << groupRoundTrip.datasets
                    << " mismatches " << groupRoundTrip.structuralMismatches
                    << " max_rel " << groupRoundTrip.maxRelativeDifference << "\n";
            check(groupRoundTrip.datasets > 50 && groupRoundTrip.structuralMismatches == 0 &&
                  groupRoundTrip.maxRelativeDifference == 0.0,
                  "per-group window: restart adding 0 generations reproduces GA (worst: " + groupRoundTrip.worst + ")");
            double const genA = ReadScalar<double>(ResultFile("GA"), split + "/spectrum_generation");
            double const genX = ReadScalar<double>(ResultFile("GX"), split + "/spectrum_generation");
            double const genY = ReadScalar<double>(ResultFile("GY"), split + "/spectrum_generation");
            metrics << "group_spectrum_generation " << genA << " " << genX << " " << genY << "\n";
            check(genX == genA + 3.0 && genY == genX,
                  "spectrum generation is absolute along the restart chain");
            if (gSize == 1) {
                // The multigroup pass carries its split state, so the chain
                // reproduces one longer restart. The transport sink residual
                // is a cancellation (round-off) quantity: it is compared with
                // the emitted energy instead. The grey pass keeps the scalar
                // window, whose statistics restart from zero on every restart
                // (unchanged behaviour), so its chain is not reproduced.
                std::vector<std::string> forwardGroups;
                for (char const* g : {"/tally", "/learned_state", "/load_balance"})
                    forwardGroups.push_back(std::string("/passes/forward") + g);
                std::vector<std::string> forwardStatistics;
                {
                    H5::H5File f(ResultFile("GX"), H5F_ACC_RDONLY);
                    H5::Group stats = f.openGroup("/passes/forward/statistics_state");
                    for (hsize_t i = 0; i < stats.getNumObjs(); ++i) {
                        std::string const name = stats.getObjnameByIdx(i);
                        if (name != "transport_sink_residual" &&
                            stats.getObjTypeByIdx(i) == H5G_GROUP)
                            forwardStatistics.push_back("/passes/forward/statistics_state/" + name);
                    }
                }
                forwardGroups.insert(forwardGroups.end(), forwardStatistics.begin(), forwardStatistics.end());
                TreeComparison const groupChain = CompareFiles(ResultFile("GX"), ResultFile("GY"), forwardGroups);
                double const residualX = ReadScalar<double>(ResultFile("GX"),
                    "/passes/forward/statistics_state/transport_sink_residual/sum");
                double const residualY = ReadScalar<double>(ResultFile("GY"),
                    "/passes/forward/statistics_state/transport_sink_residual/sum");
                double const emitted = ReadScalar<double>(ResultFile("GX"),
                    "/passes/forward/statistics_state/emitted_energy/sum");
                metrics << "group_chain_datasets " << groupChain.datasets
                        << " mismatches " << groupChain.structuralMismatches
                        << " max_rel " << groupChain.maxRelativeDifference
                        << " residual_difference_over_emitted " << std::abs(residualX - residualY) / emitted << "\n";
                check(groupChain.datasets > 30 && groupChain.structuralMismatches == 0 &&
                      groupChain.maxRelativeDifference < 1e-9 &&
                      std::abs(residualX - residualY) <= 1e-12 * emitted,
                      "per-group window: multigroup restart chain GA+1+2 matches GA+3 (worst: " +
                      groupChain.worst + ")");
            }
            // Edited copies for the refusals.
            EditedCopy(ResultFile("A"), kOutputDir + "/A_schema2.h5", [](H5::H5File& f) {
                WriteInt(f, "/restart/schema_version", 2);
                for (char const* option : {"split-per-group", "split-group-scale", "split-group-floor",
                                           "split-importance-floor", "seed-offset"})
                    DeleteLink(f, std::string("/config/transport/") + option);
            });
            EditedCopy(ResultFile("GA"), kOutputDir + "/GA_nosplit.h5", [&](H5::H5File& f) {
                DeleteLink(f, split);
            });
            EditedCopy(ResultFile("GA"), kOutputDir + "/GA_duplicate.h5", [&](H5::H5File& f) {
                H5::DataSet d = f.openDataSet(split + "/cell_ids");
                hssize_t const n = d.getSpace().getSimpleExtentNpoints();
                std::vector<unsigned long long> ids(static_cast<size_t>(n));
                d.read(ids.data(), H5::PredType::NATIVE_ULLONG);
                if (ids.size() >= 2)
                    ids[1] = ids[0];
                d.write(ids.data(), H5::PredType::NATIVE_ULLONG);
            });
            EditedCopy(ResultFile("GA"), kOutputDir + "/GA_validity.h5", [&](H5::H5File& f) {
                WriteInt(f, split + "/spectrum_valid", 0);
            });
            // Schema-2 labels on files whose configuration matches: only the
            // explicit schema rule can refuse them.
            EditedCopy(ResultFile("GA"), kOutputDir + "/GA_schema2.h5", [](H5::H5File& f) {
                WriteInt(f, "/restart/schema_version", 2);
            });
            EditedCopy(ResultFile("A"), kOutputDir + "/A_schema2_config.h5", [](H5::H5File& f) {
                WriteInt(f, "/restart/schema_version", 2);
            });
            EditedCopy(ResultFile("GA"), kOutputDir + "/GA_unknown.h5", [&](H5::H5File& f) {
                H5::DataSet d = f.openDataSet(split + "/cell_ids");
                hssize_t const n = d.getSpace().getSimpleExtentNpoints();
                std::vector<unsigned long long> ids(static_cast<size_t>(n));
                d.read(ids.data(), H5::PredType::NATIVE_ULLONG);
                ids.back() = 999999999ULL;   // not a cell of the snapshot
                d.write(ids.data(), H5::PredType::NATIVE_ULLONG);
            });
            EditedCopy(ResultFile("GA"), kOutputDir + "/GA_length.h5", [&](H5::H5File& f) {
                std::vector<double> escaped = ReadDoublesAt(f, split + "/escaped");
                escaped.pop_back();
                RewriteDoubles(f, split + "/escaped", escaped);
            });
            EditedCopy(ResultFile("GA"), kOutputDir + "/GA_negative.h5", [&](H5::H5File& f) {
                std::vector<double> escaped = ReadDoublesAt(f, split + "/escaped");
                escaped[0] = -1.0;
                RewriteDoubles(f, split + "/escaped", escaped);
            });
            EditedCopy(ResultFile("GA"), kOutputDir + "/GA_nan.h5", [&](H5::H5File& f) {
                std::vector<double> emitted = ReadDoublesAt(f, split + "/emitted");
                emitted[0] = std::nan("");
                RewriteDoubles(f, split + "/emitted", emitted);
            });
            EditedCopy(ResultFile("GA"), kOutputDir + "/GA_flag.h5", [&](H5::H5File& f) {
                WriteInt(f, split + "/accumulated", 0);
            });
            EditedCopy(ResultFile("GA"), kOutputDir + "/GA_boolean.h5", [&](H5::H5File& f) {
                WriteInt(f, split + "/present", 2);
            });
            EditedCopy(ResultFile("GA"), kOutputDir + "/GA_bands.h5", [&](H5::H5File& f) {
                std::vector<double> bounds = ReadDoublesAt(f, split + "/group_boundaries");
                bounds[1] *= 1.01;
                RewriteDoubles(f, split + "/group_boundaries", bounds);
            });
            // Non-finite boundaries: the tolerance comparison alone accepted
            // both (NaN compares false; |inf - b| > tol * inf is inf > inf).
            EditedCopy(ResultFile("GA"), kOutputDir + "/GA_bands_nan.h5", [&](H5::H5File& f) {
                std::vector<double> bounds = ReadDoublesAt(f, split + "/group_boundaries");
                bounds[1] = std::nan("");
                RewriteDoubles(f, split + "/group_boundaries", bounds);
            });
            EditedCopy(ResultFile("GA"), kOutputDir + "/GA_bands_inf.h5", [&](H5::H5File& f) {
                std::vector<double> bounds = ReadDoublesAt(f, split + "/group_boundaries");
                bounds.back() = std::numeric_limits<double>::infinity();
                RewriteDoubles(f, split + "/group_boundaries", bounds);
            });
        }
#ifdef RICH_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        bool const fromOffRefused = ThrowsContaining([&]() {
            Run(group, "G_from_off", 1, ResultFile("A"));
        }, "transport.split-per-group", message);
        check(fromOffRefused, "per-group window from a file written without it is refused: " + message);
        bool legacyAccepted = true;
        try {
            Run(scenario, "L2", 1, kOutputDir + "/A_schema2.h5");
        } catch (...) {
            legacyAccepted = false;
        }
        check(legacyAccepted, "a schema-2 file is still continued with the per-group window off");
        bool const legacyRefused = ThrowsContaining([&]() {
            Run(group, "L2_on", 1, kOutputDir + "/A_schema2.h5");
        }, "transport.split-per-group", message);
        check(legacyRefused, "a schema-2 file is refused with the per-group window on: " + message);
        bool const missingRefused = ThrowsContaining([&]() {
            Run(group, "G_missing", 1, kOutputDir + "/GA_nosplit.h5");
        }, "weight-split state", message);
        check(missingRefused, "missing split state is refused: " + message);
        bool const duplicateRefused = ThrowsContaining([&]() {
            Run(group, "G_duplicate", 1, kOutputDir + "/GA_duplicate.h5");
        }, "twice", message);
        check(duplicateRefused, "split state listing a cell twice is refused: " + message);
        bool const validityRefused = ThrowsContaining([&]() {
            Run(group, "G_validity", 1, kOutputDir + "/GA_validity.h5");
        }, "marked invalid", message);
        check(validityRefused, "inconsistent spectrum validity is refused: " + message);
        metrics << "group_window_refusals " << fromOffRefused << legacyRefused << missingRefused
                << duplicateRefused << validityRefused << " legacy_accepted " << legacyAccepted << "\n";
        for (auto const& [file, needle, what] : {
                 std::make_tuple(std::string("GA_unknown"), std::string("not in this mesh"), std::string("unknown cell ID")),
                 std::make_tuple(std::string("GA_length"), std::string("size mismatch"), std::string("truncated escaped energies")),
                 std::make_tuple(std::string("GA_negative"), std::string("negative or non-finite"), std::string("negative energy")),
                 std::make_tuple(std::string("GA_nan"), std::string("negative or non-finite"), std::string("non-finite energy")),
                 std::make_tuple(std::string("GA_flag"), std::string("not accumulated"), std::string("accumulators without the accumulated flag")),
                 std::make_tuple(std::string("GA_boolean"), std::string("non-boolean"), std::string("non-boolean flag")),
                 std::make_tuple(std::string("GA_bands"), std::string("group boundaries"), std::string("other group boundaries")),
                 std::make_tuple(std::string("GA_bands_nan"), std::string("NaN, negative or non-increasing group boundaries"), std::string("a NaN group boundary")),
                 std::make_tuple(std::string("GA_bands_inf"), std::string("other group boundaries"), std::string("an infinite group boundary"))}) {
            bool const refused = ThrowsContaining([&]() {
                Run(group, "G_" + file, 1, kOutputDir + "/" + file + ".h5");
            }, needle, message);
            metrics << "split_state_refused " << file << " " << refused << "\n";
            check(refused, "split state with " + what + " is refused on every rank: " + message);
        }
        bool const schemaRuleGroup = ThrowsContaining([&]() {
            Run(group, "G_schema2", 1, kOutputDir + "/GA_schema2.h5");
        }, "(schema 2)", message);
        check(schemaRuleGroup, "schema 2 with the per-group window is refused by the schema rule: " + message);
        bool const schemaRuleSeed = ThrowsContaining([&]() {
            PostProcessIMC::PostProcessScenario seeded = scenario;
            seeded.defaults.transport.seedOffset = 1;
            Run(seeded, "S_schema2", 1, kOutputDir + "/A_schema2_config.h5");
        }, "(schema 2)", message);
        check(schemaRuleSeed, "schema 2 with a seed offset is refused by the schema rule: " + message);
        // A continuation with other ramp settings (not must-match) keeps the
        // absolute generation index of the file.
        PostProcessIMC::PostProcessScenario groupRamp = group;
        groupRamp.defaults.adaptive.source.learnedPhotonsPerCellBudget = 80;
        groupRamp.defaults.volumeEmission.learnedPhotonsPerCellBudget = 80;
        Run(groupRamp, "GZ", 2, ResultFile("GA"));
        if (gRank == 0) {
            double const nextA = ReadScalar<double>(ResultFile("GA"), "/passes/forward/learned_state/next_generation_id");
            double const nextX = ReadScalar<double>(ResultFile("GX"), "/passes/forward/learned_state/next_generation_id");
            double const nextZ = ReadScalar<double>(ResultFile("GZ"), "/passes/forward/learned_state/next_generation_id");
            double const genZ = ReadScalar<double>(ResultFile("GZ"), split + "/spectrum_generation");
            double const greyA = ReadScalar<double>(ResultFile("GA"), "/passes/grey/learned_state/next_generation_id");
            double const greyZ = ReadScalar<double>(ResultFile("GZ"), "/passes/grey/learned_state/next_generation_id");
            metrics << "generation_origin GA " << nextA << " GX " << nextX << " GZ " << nextZ
                    << " GZ_spectrum " << genZ << " grey GA " << greyA << " GZ " << greyZ << "\n";
            check(nextX == nextA + 3.0 && nextZ == nextA + 2.0 && genZ == nextA + 1.0 && greyZ == greyA + 2.0,
                  "absolute generation indices continue from the file whatever the continuation's ramp settings");
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
        std::cout << (passFlag ? "imc_postprocess_restart PASSED" : "imc_postprocess_restart FAILED") << std::endl;
    }
#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return passFlag ? 0 : 1;
}
