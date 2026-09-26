/*
 * TRE-3D: the canonical three-mode Thermal Radiative Eigenmode benchmark,
 * driven by RICH's Simulation with a periodic Cartesian tessellation.
 *
 * The physical domain is a periodic [0, 2*pi]^3 cube with L0 = 1 cm and
 * sigma_a = 1/cm.  The material and radiation fields are the exact
 * superposition from the benchmark proposal:
 *
 *   k1 = (1,0,0), k2 = (1,1,0), k3 = (1,1,1),
 *   A1 = 0.10, A2 = 0.07, A3 = 0.05.
 *
 * Each mode has its own full-transport thermal eigenvalue.  Packets are
 * initialized from the exact summed anisotropic angular eigenfunction; after
 * initialization the IMC step supplies thermal emission, transport,
 * absorption, and material feedback with no external source.
 *
 * The material equation of state is u = a T^4 / beta at unit density, which
 * is an IdealGas with f = a/beta, an energy exponent of four, and no density
 * dependence.
 */

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "source/3D/output/write3D.hpp"
#include "source/3D/output/write_vtu_3d.hpp"
#include "source/3D/radiation/RadiationIMC.hpp"
#include "source/3D/tessellation/Cartesian3D.hpp"
#include "source/Radiation/OpacityCalculator.hpp"
#include "source/monte/boundary/BoundaryCondition.hpp"
#include "source/monte/population/CombPopulationControl.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/ManualTimeStep.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"
#include "source/newtonian/three_dimensional/conserved_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"
#include "source/monte/deps/CMMC/src/units/units.hpp"

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double betaCoupling = 1.0;
constexpr double chi = 1.0;
constexpr double u0 = 1.0;
constexpr double referenceLengthCm = 1.0;
constexpr double cubeSide = 2.0 * pi;
constexpr double referenceDensity = 1.0;

struct Mode
{
    Vector3D waveVector;
    double amplitude;
    double magnitude;
    double decayRate;
    double radiationFactor;
};

struct Config
{
    std::size_t nx = 12;
    std::size_t ny = 12;
    std::size_t nz = 12;
    std::size_t initialParticlesPerCell = 300;
    std::size_t newPhotonsPerCell = 4;
    std::size_t populationPerCell = 500;
    double dtau = 0.02;
    double finalTau = 8.0;
    std::size_t snapshotEvery = 100;
    std::uint64_t seed = 20260828;
    std::string outputDirectory = "tre_3d_output";
};

void usage(const char *program)
{
    std::cerr
        << "Usage: " << program << " [options]\n"
        << "  --n N                     set nx=ny=nz (default 12)\n"
        << "  --nx N                   x cells (default 12)\n"
        << "  --ny N                   y cells (default 12)\n"
        << "  --nz N                   z cells (default 12)\n"
        << "  --initial-particles N    exact-mode packets/cell at tau=0 (default 300)\n"
        << "  --new-photons N          source packets/cell (default 4)\n"
        << "  --population N           population-control target/cell (default 500)\n"
        << "  --dtau X                 dimensionless timestep (default 0.02)\n"
        << "  --final-tau X            final dimensionless time (default 8)\n"
        << "  --snapshot-every N       cycles between VTK and restart snapshots, 0 for first and last only (default 100)\n"
        << "  --seed N                 random seed (default 20260828)\n"
        << "  --output DIR             output directory (default tre_3d_output)\n"
        << "  --help                   show this help\n";
}

std::string requireValue(int argc, char **argv, int &index)
{
    if(index + 1 >= argc)
        throw std::runtime_error(std::string(argv[index]) + " requires a value");
    return argv[++index];
}

Config parseConfig(int argc, char **argv)
{
    Config config;
    for(int i = 1; i < argc; ++i)
    {
        const std::string argument(argv[i]);
        if(argument == "--help" || argument == "-h")
        {
            usage(argv[0]);
            std::exit(0);
        }
        if(argument == "--n")
        {
            const std::size_t n = std::stoull(requireValue(argc, argv, i));
            config.nx = n;
            config.ny = n;
            config.nz = n;
        }
        else if(argument == "--nx")
            config.nx = std::stoull(requireValue(argc, argv, i));
        else if(argument == "--ny")
            config.ny = std::stoull(requireValue(argc, argv, i));
        else if(argument == "--nz")
            config.nz = std::stoull(requireValue(argc, argv, i));
        else if(argument == "--initial-particles")
            config.initialParticlesPerCell = std::stoull(requireValue(argc, argv, i));
        else if(argument == "--new-photons")
            config.newPhotonsPerCell = std::stoull(requireValue(argc, argv, i));
        else if(argument == "--population")
            config.populationPerCell = std::stoull(requireValue(argc, argv, i));
        else if(argument == "--dtau")
            config.dtau = std::stod(requireValue(argc, argv, i));
        else if(argument == "--final-tau")
            config.finalTau = std::stod(requireValue(argc, argv, i));
        else if(argument == "--snapshot-every")
            config.snapshotEvery = std::stoull(requireValue(argc, argv, i));
        else if(argument == "--seed")
            config.seed = std::stoull(requireValue(argc, argv, i));
        else if(argument == "--output")
            config.outputDirectory = requireValue(argc, argv, i);
        else
            throw std::runtime_error("unknown option: " + argument);
    }

    if(config.nx < 4 || config.ny < 4 || config.nz < 4 ||
       config.initialParticlesPerCell == 0 || config.newPhotonsPerCell == 0 ||
       config.populationPerCell == 0 || !(config.dtau > 0.0) ||
       !(config.finalTau > 0.0) || config.outputDirectory.empty())
        throw std::runtime_error("all mesh, packet, time-step, and final-time values must be positive");
    return config;
}

double dotProduct(const Vector3D &left, const Vector3D &right)
{
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

double norm(const Vector3D &value)
{
    return std::sqrt(dotProduct(value, value));
}

double principalDecayRate(double waveNumber)
{
    const double aMin = 1.0e-14;
    auto dispersion = [waveNumber](double s) {
        const double a = chi + s;
        const double G = (chi / waveNumber) * std::atan(waveNumber / a);
        return s - betaCoupling * chi * (G - 1.0);
    };

    double lo = -chi + aMin;
    double hi = -aMin;
    const double fLo = dispersion(lo);
    const double fHi = dispersion(hi);
    if(!(fLo < 0.0 && fHi > 0.0))
        throw std::runtime_error("failed to bracket TRE-3D eigenvalue");
    for(int iteration = 0; iteration < 160; ++iteration)
    {
        const double mid = 0.5 * (lo + hi);
        if(dispersion(mid) > 0.0)
            hi = mid;
        else
            lo = mid;
    }
    return 0.5 * (lo + hi);
}

double radiationEnergyModeFactor(double waveNumber, double decayRate)
{
    return (chi / waveNumber) *
           std::atan(waveNumber / (chi + decayRate));
}

std::vector<Mode> makeModes()
{
    std::vector<Mode> modes;
    modes.push_back({Vector3D(1.0, 0.0, 0.0), 0.10, 0.0, 0.0, 0.0});
    modes.push_back({Vector3D(1.0, 1.0, 0.0), 0.07, 0.0, 0.0, 0.0});
    modes.push_back({Vector3D(1.0, 1.0, 1.0), 0.05, 0.0, 0.0, 0.0});
    for(Mode &mode : modes)
    {
        mode.magnitude = norm(mode.waveVector);
        mode.decayRate = principalDecayRate(mode.magnitude);
        mode.radiationFactor = radiationEnergyModeFactor(
            mode.magnitude, mode.decayRate);
    }
    return modes;
}

double sinc(double argument)
{
    if(std::abs(argument) < 1.0e-14)
        return 1.0;
    return std::sin(argument) / argument;
}

double cellCosineIntegral(const Vector3D &center, const Vector3D &width,
                          const Vector3D &waveVector)
{
    // Integrate cos(k dot x), not a product of cosines.  Separation gives
    // the three sinc factors, but the phase remains cos(k dot x_center).
    const double xFactor = std::abs(waveVector.x) < 1.0e-14
        ? width.x
        : 2.0 * std::sin(0.5 * waveVector.x * width.x) / waveVector.x;
    const double yFactor = std::abs(waveVector.y) < 1.0e-14
        ? width.y
        : 2.0 * std::sin(0.5 * waveVector.y * width.y) / waveVector.y;
    const double zFactor = std::abs(waveVector.z) < 1.0e-14
        ? width.z
        : 2.0 * std::sin(0.5 * waveVector.z * width.z) / waveVector.z;
    return xFactor * yFactor * zFactor *
           std::cos(dotProduct(waveVector, center));
}

double cellModeFilter(const Cartesian3D &tess, const Mode &mode)
{
    return sinc(0.5 * mode.waveVector.x * tess.dx()) *
           sinc(0.5 * mode.waveVector.y * tess.dy()) *
           sinc(0.5 * mode.waveVector.z * tess.dz());
}

double materialCellAverage(const Cartesian3D &tess, std::size_t cellIndex,
                           const std::vector<Mode> &modes)
{
    const Vector3D center = tess.GetCellCM(cellIndex);
    const Vector3D width(tess.dx(), tess.dy(), tess.dz());
    const double volume = tess.GetVolume(cellIndex);
    double value = u0;
    for(const Mode &mode : modes)
    {
        value += mode.amplitude * cellCosineIntegral(
            center, width, mode.waveVector) / volume;
    }
    return value;
}

double radiationCellAverage(const Cartesian3D &tess, std::size_t cellIndex,
                            const std::vector<Mode> &modes)
{
    const Vector3D center = tess.GetCellCM(cellIndex);
    const Vector3D width(tess.dx(), tess.dy(), tess.dz());
    const double volume = tess.GetVolume(cellIndex);
    double value = u0;
    for(const Mode &mode : modes)
    {
        value += mode.amplitude * mode.radiationFactor *
                 cellCosineIntegral(center, width, mode.waveVector) / volume;
    }
    return value;
}

double evolvedMaterialCellAverage(const Cartesian3D &tess, std::size_t cellIndex,
                                  const std::vector<Mode> &modes, double tau)
{
    const Vector3D center = tess.GetCellCM(cellIndex);
    const Vector3D width(tess.dx(), tess.dy(), tess.dz());
    const double volume = tess.GetVolume(cellIndex);
    double value = u0;
    for(const Mode &mode : modes)
    {
        value += mode.amplitude * std::exp(mode.decayRate * tau) *
                 cellCosineIntegral(center, width, mode.waveVector) / volume;
    }
    return value;
}

double evolvedRadiationCellAverage(const Cartesian3D &tess, std::size_t cellIndex,
                                   const std::vector<Mode> &modes, double tau)
{
    const Vector3D center = tess.GetCellCM(cellIndex);
    const Vector3D width(tess.dx(), tess.dy(), tess.dz());
    const double volume = tess.GetVolume(cellIndex);
    double value = u0;
    for(const Mode &mode : modes)
    {
        value += mode.amplitude * mode.radiationFactor *
                 std::exp(mode.decayRate * tau) *
                 cellCosineIntegral(center, width, mode.waveVector) / volume;
    }
    return value;
}

double angularEigenfunction(const Vector3D &location, const Vector3D &direction,
                            const std::vector<Mode> &modes)
{
    double value = u0;
    for(const Mode &mode : modes)
    {
        const double a = chi + mode.decayRate;
        const double q = dotProduct(mode.waveVector, direction);
        const double phase = dotProduct(mode.waveVector, location);
        value += mode.amplitude * chi *
                 (a * std::cos(phase) + q * std::sin(phase)) /
                 (a * a + q * q);
    }
    return value;
}

std::vector<Particle3D> makeExactInitialParticles(
    const Cartesian3D &tess, const std::vector<ComputationalCell3D> &cells,
    std::size_t particlesPerCell, const std::vector<Mode> &modes,
    double energyDensityScale, std::uint64_t seed)
{
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<Particle3D> particles;
    particles.reserve(tess.GetPointNo() * particlesPerCell);

    double psiMax = u0;
    for(const Mode &mode : modes)
        psiMax += std::abs(mode.amplitude) * chi /
                  (chi + mode.decayRate);

    const Vector3D width(tess.dx(), tess.dy(), tess.dz());
    for(std::size_t cellIndex = 0; cellIndex < tess.GetPointNo(); ++cellIndex)
    {
        const Vector3D center = tess.GetCellCM(cellIndex);
        const double cellEnergy = tess.GetVolume(cellIndex) *
            radiationCellAverage(tess, cellIndex, modes);
        const double weight = energyDensityScale * cellEnergy /
                              static_cast<double>(particlesPerCell);

        for(std::size_t packetIndex = 0; packetIndex < particlesPerCell; ++packetIndex)
        {
            while(true)
            {
                const Vector3D location(
                    center.x + (uniform(generator) - 0.5) * width.x,
                    center.y + (uniform(generator) - 0.5) * width.y,
                    center.z + (uniform(generator) - 0.5) * width.z);
                const double mu = 2.0 * uniform(generator) - 1.0;
                const double phi = 2.0 * pi * uniform(generator);
                const double transverse = std::sqrt(std::max(0.0, 1.0 - mu * mu));
                const Vector3D direction(
                    mu, transverse * std::cos(phi), transverse * std::sin(phi));
                const double psi = angularEigenfunction(location, direction, modes);
                if(uniform(generator) * psiMax > psi)
                    continue;

                Particle3D particle;
                particle.location = location;
                particle.velocity = direction * units::clight;
                particle.frequency = 0.0;
                particle.weight = weight;
                particle.initialWeight = weight;
                particle.cellIndex = cellIndex;
                particle.cellID = cells[cellIndex].ID;
                particles.push_back(particle);
                break;
            }
        }
    }
    return particles;
}

/*
 * With a periodic tessellation every face of the cube leads to a real cell,
 * either local or an MPI halo, so transport never reaches a boundary face and
 * this condition is never applied.  It stays as a loud guard: silently
 * reflecting a packet here would corrupt the eigenmode decay.
 */
class TRE3DPeriodicBoundary final :
    public STORM::BoundaryCondition<Vector3D, Tessellation3D>
{
public:
    explicit TRE3DPeriodicBoundary(const Tessellation3D &tess)
        : STORM::BoundaryCondition<Vector3D, Tessellation3D>(tess)
    {}

    STORM::ParticleStatus apply(Particle3D &particle) override
    {
        this->grid.WrapPeriodicPoint(particle.location);
        const std::size_t destination = this->grid.GetContainingCell(particle.location);
        if(destination == std::numeric_limits<std::size_t>::max())
            throw std::runtime_error("TRE-3D periodic boundary reached a point owned by another rank");

        particle.cellIndex = destination;
        return STORM::ParticleStatus::REFLECT;
    }

    std::vector<Particle3D> generateNewBoundaryParticles(double) override
    {
        return {};
    }

    STORM::DDMCBoundaryFaceBehavior getDDMCBoundaryFaceBehavior(
        std::size_t, std::size_t, std::size_t) const override
    {
        return STORM::DDMCBoundaryFaceBehavior::Unsupported;
    }
};

class TREOpacity final : public OpacityCalculator
{
public:
    double CalcPlanckOpacity(const ComputationalCell3D &) const override
    {
        return chi / referenceLengthCm;
    }

    double CalcScatteringOpacity(const ComputationalCell3D &) const override
    {
        return 0.0;
    }
};

double projectMode(const std::vector<double> &field, const Cartesian3D &tess,
                  const Mode &mode, double background)
{
    double numerator = 0.0;
    double volume = 0.0;
    for(std::size_t i = 0; i < tess.GetPointNo(); ++i)
    {
        const double cellVolume = tess.GetVolume(i);
        numerator += cellVolume * (field[i] - background) *
                     std::cos(dotProduct(mode.waveVector, tess.GetCellCM(i)));
        volume += cellVolume;
    }
    double globalNumerator = 0.0;
    double globalVolume = 0.0;
    MPI_Allreduce(&numerator, &globalNumerator, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&volume, &globalVolume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // Fields are stored as cell averages.  Undo the exact Fourier response
    // of a cell average so the reported coefficient is the continuum mode.
    return 2.0 * globalNumerator / globalVolume / cellModeFilter(tess, mode);
}

double total(const std::vector<double> &values, const Cartesian3D &tess)
{
    double result = 0.0;
    for(std::size_t i = 0; i < tess.GetPointNo(); ++i)
        result += values[i] * tess.GetVolume(i);
    double globalResult = 0.0;
    MPI_Allreduce(&result, &globalResult, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return globalResult;
}

std::size_t particleCount(const std::vector<Particle3D> &particles)
{
    unsigned long long result = static_cast<unsigned long long>(particles.size());
    unsigned long long globalResult = 0;
    MPI_Allreduce(&result, &globalResult, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return static_cast<std::size_t>(globalResult);
}

std::string CycleStem(std::size_t cycle)
{
    std::ostringstream name;
    name << "tre_3d_cycle_" << std::setw(6) << std::setfill('0') << cycle;
    return name.str();
}

std::string SnapshotPath(const std::filesystem::path &directory, std::size_t cycle)
{
    return (directory / (CycleStem(cycle) + ".h5")).string();
}

std::string VtkPath(const std::filesystem::path &directory, std::size_t cycle)
{
    return (directory / (CycleStem(cycle) + ".pvtu")).string();
}

bool DueSnapshot(std::size_t cycle, std::size_t lastCycle, std::size_t snapshotEvery)
{
    if(cycle == 0 || cycle == lastCycle)
    {
        return true;
    }
    return snapshotEvery > 0 && cycle % snapshotEvery == 0;
}

void FillDiagnosticFields(const Cartesian3D &tess,
                          const std::vector<ComputationalCell3D> &cells,
                          const std::vector<Conserved3D> &extensives,
                          const std::vector<double> &radiationTimeAvg,
                          const std::vector<Mode> &modes, double tau,
                          double referenceTemperature, double referenceEnergyDensity,
                          std::vector<double> &material,
                          std::vector<double> &radiationCensus,
                          std::vector<double> &radiationAverage,
                          std::vector<double> &exactMaterial,
                          std::vector<double> &exactRadiation)
{
    const std::size_t cellCount = tess.GetPointNo();
    material.resize(cellCount);
    radiationCensus.resize(cellCount);
    radiationAverage.resize(cellCount);
    exactMaterial.resize(cellCount);
    exactRadiation.resize(cellCount);
    for(std::size_t i = 0; i < cellCount; ++i)
    {
        material[i] = std::pow(cells[i].temperature / referenceTemperature, 4);
        radiationCensus[i] = extensives[i].Erad /
            (referenceEnergyDensity * tess.GetVolume(i));
        if(i < radiationTimeAvg.size())
        {
            radiationAverage[i] = radiationTimeAvg[i] / referenceEnergyDensity;
        }
        else
        {
            radiationAverage[i] = radiationCensus[i];
        }
        exactMaterial[i] = evolvedMaterialCellAverage(tess, i, modes, tau);
        exactRadiation[i] = evolvedRadiationCellAverage(tess, i, modes, tau);
    }
}

void WriteTREVtk(const Cartesian3D &tess,
                 const std::vector<double> &material,
                 const std::vector<double> &radiationCensus,
                 const std::vector<double> &radiationAverage,
                 const std::vector<double> &exactMaterial,
                 const std::vector<double> &exactRadiation,
                 const std::vector<ComputationalCell3D> &cells,
                 double tau, std::size_t cycle, const std::string &filename)
{
    const std::size_t cellCount = tess.GetPointNo();
    std::vector<double> temperature(cellCount);
    for(std::size_t i = 0; i < cellCount; ++i)
    {
        temperature[i] = cells[i].temperature;
    }
    const std::vector<std::string> names = {
        "temperature", "material_U", "Erad_census", "Erad_time_avg",
        "U_exact", "E_exact"};
    const std::vector<std::vector<double>> data = {
        temperature, material, radiationCensus, radiationAverage,
        exactMaterial, exactRadiation};
    const std::vector<std::string> emptyNames;
    const std::vector<std::vector<std::string>> emptyStrings;
    const std::vector<std::string> emptyVectorNames;
    const std::vector<std::vector<Vector3D>> emptyVectors;
    const std::vector<std::pair<std::string, double>> scalars = {{"tau", tau}};
    // Cartesian3D is a Tessellation3D, not a Voronoi3D, so the VTK dump uses
    // write_vtu3d, the same backend WriteVoronoiVTKOnly uses for Voronoi meshes.
    write_vtu3d::write_vtu_3d(filename, names, data, emptyNames, emptyStrings,
                              emptyVectorNames, emptyVectors, scalars, tau, cycle,
                              tess);
}

void WriteFinalProfile(const Cartesian3D &tess,
                       const std::vector<double> &material,
                       const std::vector<double> &radiationCensus,
                       const std::vector<double> &radiationAverage,
                       const std::vector<double> &exactMaterial,
                       const std::vector<double> &exactRadiation,
                       double tau, const std::filesystem::path &path, int rank)
{
    const std::size_t localCells = tess.GetPointNo();
    constexpr int fieldCount = 8;
    std::vector<double> local(localCells * static_cast<std::size_t>(fieldCount));
    for(std::size_t i = 0; i < localCells; ++i)
    {
        const Vector3D center = tess.GetCellCM(i);
        const std::size_t offset = i * static_cast<std::size_t>(fieldCount);
        local[offset + 0] = center.x;
        local[offset + 1] = center.y;
        local[offset + 2] = center.z;
        local[offset + 3] = material[i];
        local[offset + 4] = radiationCensus[i];
        local[offset + 5] = radiationAverage[i];
        local[offset + 6] = exactMaterial[i];
        local[offset + 7] = exactRadiation[i];
    }

    int nlocal = static_cast<int>(local.size());
    int nprocs = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    std::vector<int> counts(static_cast<std::size_t>(nprocs), 0);
    std::vector<int> displs(static_cast<std::size_t>(nprocs), 0);
    MPI_Gather(&nlocal, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    int ntotal = 0;
    if(rank == 0)
    {
        for(int p = 0; p < nprocs; ++p)
        {
            displs[static_cast<std::size_t>(p)] = ntotal;
            ntotal += counts[static_cast<std::size_t>(p)];
        }
    }
    std::vector<double> gathered(static_cast<std::size_t>(std::max(ntotal, 0)));
    MPI_Gatherv(local.data(), nlocal, MPI_DOUBLE,
                gathered.data(), counts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    if(rank != 0)
    {
        return;
    }

    const std::size_t rowCount = static_cast<std::size_t>(ntotal) /
        static_cast<std::size_t>(fieldCount);
    std::vector<std::size_t> order(rowCount);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&gathered](std::size_t left, std::size_t right)
              {
                  const std::size_t leftOffset = left * static_cast<std::size_t>(fieldCount);
                  const std::size_t rightOffset = right * static_cast<std::size_t>(fieldCount);
                  if(gathered[leftOffset + 2] != gathered[rightOffset + 2])
                  {
                      return gathered[leftOffset + 2] < gathered[rightOffset + 2];
                  }
                  if(gathered[leftOffset + 1] != gathered[rightOffset + 1])
                  {
                      return gathered[leftOffset + 1] < gathered[rightOffset + 1];
                  }
                  return gathered[leftOffset] < gathered[rightOffset];
              });

    std::ofstream output(path);
    if(!output)
    {
        throw std::runtime_error("could not open spatial profile: " + path.string());
    }
    output << "# benchmark=TRE-3D\n"
           << "# tau=" << std::setprecision(17) << tau << "\n"
           << "x,y,z,U,E_census,E_time_avg,U_exact,E_exact\n"
           << std::scientific << std::setprecision(17);
    for(std::size_t rowIndex : order)
    {
        const std::size_t offset = rowIndex * static_cast<std::size_t>(fieldCount);
        output << gathered[offset] << ',' << gathered[offset + 1] << ','
               << gathered[offset + 2] << ',' << gathered[offset + 3] << ','
               << gathered[offset + 4] << ',' << gathered[offset + 5] << ','
               << gathered[offset + 6] << ',' << gathered[offset + 7] << '\n';
    }
}

void WriteRestartAndVtk(Simulation &simulation, const Cartesian3D &tess,
                        const std::vector<ComputationalCell3D> &cells,
                        const std::vector<Conserved3D> &extensives,
                        const std::vector<double> &radiationTimeAvg,
                        const std::vector<Mode> &modes, double tau,
                        double referenceTemperature, double referenceEnergyDensity,
                        std::size_t cycle, bool writeArchive,
                        const std::string &latestH5Path,
                        const std::filesystem::path &snapshotDirectory,
                        const std::filesystem::path &profilePath, int rank)
{
    std::vector<double> material;
    std::vector<double> radiationCensus;
    std::vector<double> radiationAverage;
    std::vector<double> exactMaterial;
    std::vector<double> exactRadiation;
    FillDiagnosticFields(tess, cells, extensives, radiationTimeAvg, modes, tau,
                         referenceTemperature, referenceEnergyDensity, material,
                         radiationCensus, radiationAverage, exactMaterial,
                         exactRadiation);

    WriteSimulation(simulation, latestH5Path);
    std::string archive;
    if(writeArchive)
    {
        archive = SnapshotPath(snapshotDirectory, cycle);
        WriteSimulation(simulation, archive);
    }
    const std::string vtk = VtkPath(snapshotDirectory, cycle);
    WriteTREVtk(tess, material, radiationCensus, radiationAverage, exactMaterial,
                exactRadiation, cells, tau, cycle, vtk);
    WriteFinalProfile(tess, material, radiationCensus, radiationAverage,
                      exactMaterial, exactRadiation, tau, profilePath, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0)
    {
        std::cout << "wrote latest checkpoint " << latestH5Path
                  << " and VTK " << vtk
                  << " and profile " << profilePath.string();
        if(writeArchive)
        {
            std::cout << " and archive " << archive;
        }
        std::cout << std::endl;
    }
}

} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    try
    {
        const Config config = parseConfig(argc, argv);
        const std::vector<Mode> modes = makeModes();
        const double referenceTemperature = units::kev_kelvin;
        const double referenceEnergyDensity = units::arad *
            std::pow(referenceTemperature, 4);

        Cartesian3D tess(Vector3D(0.0, 0.0, 0.0),
                         Vector3D(cubeSide, cubeSide, cubeSide),
                         config.nx, config.ny, config.nz);
        tess.SetPeriodicity(true, true, true);
        tess.BuildParallel(std::vector<double>(config.nx * config.ny * config.nz, 1.0));

        // u = a T^4 / beta at unit density, so cv = 4 a T^3 / beta.
        IdealGas eos(5.0 / 3.0, units::arad / betaCoupling, 4.0, 0.0);

        const std::size_t localCells = tess.GetPointNo();
        std::vector<ComputationalCell3D> initialCells(localCells);
        std::vector<double> initialMaterial(localCells);
        std::vector<double> initialRadiation(localCells);
        for(std::size_t i = 0; i < localCells; ++i)
        {
            const double material = materialCellAverage(tess, i, modes);
            const double radiation = radiationCellAverage(tess, i, modes);
            ComputationalCell3D &cell = initialCells[i];
            cell.density = referenceDensity;
            cell.velocity = Vector3D(0.0, 0.0, 0.0);
            cell.temperature = referenceTemperature * std::pow(material / u0, 0.25);
            cell.internal_energy = eos.dT2e(cell.density, cell.temperature,
                                            cell.tracers, ComputationalCell3D::tracerNames);
            cell.pressure = eos.de2p(cell.density, cell.internal_energy,
                                     cell.tracers, ComputationalCell3D::tracerNames);
            cell.Erad = referenceEnergyDensity * radiation / cell.density;
            cell.Erad_dt = 0.0;
            cell.Erad_dt_dt = 0.0;
            cell.Eg.resize(ENERGY_GROUPS_NUM);
            for(std::size_t group = 0; group < ENERGY_GROUPS_NUM; ++group)
            {
                cell.Eg[group] = cell.Erad / static_cast<double>(ENERGY_GROUPS_NUM);
            }
            initialMaterial[i] = material;
            initialRadiation[i] = radiation;
        }

        Simulation simulation(tess, initialCells, eos);
        simulation.SetTimeStepFunction(std::make_shared<ManualTimeStep>());
        std::vector<ComputationalCell3D> &cells = simulation.getCells();
        std::vector<Conserved3D> &extensives = simulation.getExtensives();
        extensives.resize(localCells);
        for(std::size_t i = 0; i < localCells; ++i)
        {
            extensives[i].Eg.resize(ENERGY_GROUPS_NUM);
            PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);
        }

        std::shared_ptr<IdealGas> eosPointer = std::make_shared<IdealGas>(eos);
        std::shared_ptr<TREOpacity> opacity = std::make_shared<TREOpacity>();
        std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> boundary =
            std::make_shared<TRE3DPeriodicBoundary>(tess);

        RadiationIMCParameters parameters;
        parameters.newPhotonsPerCell = config.newPhotonsPerCell;
        parameters.withHydro = false;
        parameters.withRandomWalk = false;
        parameters.withDDMC = false;
        parameters.withMultigroupOpacity = false;
        // A single group spanning every photon energy, which is gray transport.
        parameters.energyBoundaries[0] = 0.0;
        for(std::size_t group = 1; group <= ENERGY_GROUPS_NUM; ++group)
        {
            parameters.energyBoundaries[group] = 1.0e30 *
                static_cast<double>(group) / static_cast<double>(ENERGY_GROUPS_NUM);
        }
        parameters.energyBoundariesProvided = true;

        std::shared_ptr<RadiationIMC> physics = std::make_shared<RadiationIMC>(
            tess, boundary, cells, extensives, eosPointer, opacity, parameters);
        physics->reseedRNG(config.seed + 104729ULL * static_cast<std::uint64_t>(rank));
        std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> populationControl =
            std::make_shared<STORM::CombPopulationControl<Vector3D, Tessellation3D>>(
                tess, config.populationPerCell, 1.0);

        const std::vector<Particle3D> initialParticles = makeExactInitialParticles(
            tess, cells, config.initialParticlesPerCell, modes,
            referenceEnergyDensity, config.seed + 1);
        std::shared_ptr<RadiationMCStep> radiationStep =
            std::make_shared<RadiationMCStep>(tess, cells, extensives, physics,
                                              populationControl, boundary,
                                              initialParticles, 0, false,
                                              RadiationMCStep::ManagerType::RDMA);
        simulation.addPhysics(radiationStep);

        const std::filesystem::path outputDirectory(config.outputDirectory);
        const std::filesystem::path snapshotDirectory = outputDirectory / "snapshots";
        const std::filesystem::path historyPath =
            outputDirectory / "tre_3d_history.csv";
        const std::filesystem::path profilePath =
            outputDirectory / "tre_3d_final_profile.csv";
        const std::string latestH5Path = (outputDirectory / "latest.h5").string();

        const double initialMaterialEnergy = referenceEnergyDensity *
            total(initialMaterial, tess) / betaCoupling;
        const double initialRadiationEnergy = referenceEnergyDensity *
            total(initialRadiation, tess);
        std::vector<double> initialMaterialAmplitudes(modes.size());
        std::vector<double> initialRadiationAmplitudes(modes.size());
        for(std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex)
        {
            initialMaterialAmplitudes[modeIndex] = projectMode(initialMaterial, tess, modes[modeIndex], u0);
            initialRadiationAmplitudes[modeIndex] = projectMode(initialRadiation, tess, modes[modeIndex], u0);
        }
        const std::size_t initialParticleCount = particleCount(radiationStep->getParticles());

        std::unique_ptr<std::ofstream> output;
        if(rank == 0)
        {
            std::filesystem::create_directories(snapshotDirectory);
            output = std::make_unique<std::ofstream>(historyPath);
            if(!*output)
            {
                throw std::runtime_error("could not open history output file: " + historyPath.string());
            }
            *output << "# benchmark=TRE-3D\n"
                    << "# beta=" << std::setprecision(17) << betaCoupling << "\n"
                    << "# chi=" << chi << "\n"
                    << "# nx=" << config.nx << "\n"
                    << "# ny=" << config.ny << "\n"
                    << "# nz=" << config.nz << "\n"
                    << "# mpi_ranks=" << nprocs << "\n"
                    << "# dtau=" << config.dtau << "\n"
                    << "# final_tau=" << config.finalTau << "\n"
                    << "# mode1_k=1,0,0\n"
                    << "# mode2_k=1,1,0\n"
                    << "# mode3_k=1,1,1\n";
            for(std::size_t j = 0; j < modes.size(); ++j)
            {
                *output << "# mode" << (j + 1) << "_amplitude=" << modes[j].amplitude << "\n"
                        << "# mode" << (j + 1) << "_s=" << modes[j].decayRate << "\n"
                        << "# mode" << (j + 1) << "_G=" << modes[j].radiationFactor << "\n";
            }
            *output << "tau_end,tau_mid,A_U1,A_E1,A_U2,A_E2,A_U3,A_E3,"
                       "material_energy,radiation_energy,total_energy,particle_count\n"
                    << std::scientific << std::setprecision(17)
                    << 0.0 << ',' << 0.0;
            for(std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex)
            {
                *output << ',' << initialMaterialAmplitudes[modeIndex]
                        << ',' << initialRadiationAmplitudes[modeIndex];
            }
            *output << ',' << initialMaterialEnergy << ',' << initialRadiationEnergy
                    << ',' << initialMaterialEnergy + initialRadiationEnergy << ','
                    << initialParticleCount << '\n';
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const std::size_t steps = static_cast<std::size_t>(
            std::ceil(config.finalTau / config.dtau));
        WriteRestartAndVtk(simulation, tess, cells, extensives,
                           radiationStep->getEradTimeAvg(), modes, 0.0,
                           referenceTemperature, referenceEnergyDensity, 0, true,
                           latestH5Path, snapshotDirectory, profilePath, rank);
        double tau = 0.0;
        for(std::size_t step = 0; step < steps; ++step)
        {
            const double currentDtau = std::min(config.dtau, config.finalTau - tau);
            if(!(currentDtau > 0.0))
                break;
            const double currentDt = currentDtau * referenceLengthCm /
                                     units::clight;
            simulation.SetTimeStep(currentDt);
            simulation.step();
            tau += currentDtau;

            std::vector<double> material(tess.GetPointNo());
            for(std::size_t i = 0; i < tess.GetPointNo(); ++i)
                material[i] = std::pow(cells[i].temperature /
                                       referenceTemperature, 4);

            const std::vector<double> &radiation = radiationStep->getEradTimeAvg();
            std::vector<double> radiationDimensionless(tess.GetPointNo());
            std::vector<double> radiationCensusDimensionless(tess.GetPointNo());
            for(std::size_t i = 0; i < tess.GetPointNo(); ++i)
            {
                radiationDimensionless[i] = radiation[i] /
                                             referenceEnergyDensity;
                radiationCensusDimensionless[i] = extensives[i].Erad /
                    (referenceEnergyDensity * tess.GetVolume(i));
            }

            const double materialEnergy = referenceEnergyDensity *
                total(material, tess) / betaCoupling;
            const double radiationEnergy = referenceEnergyDensity *
                total(radiationCensusDimensionless, tess);
            std::vector<double> materialAmplitudes(modes.size());
            std::vector<double> radiationAmplitudes(modes.size());
            for(std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex)
            {
                materialAmplitudes[modeIndex] = projectMode(material, tess, modes[modeIndex], u0);
                radiationAmplitudes[modeIndex] = projectMode(radiationDimensionless, tess, modes[modeIndex], u0);
            }
            const std::size_t currentParticleCount = particleCount(radiationStep->getParticles());
            if(rank == 0)
            {
                *output << tau << ',' << tau - 0.5 * currentDtau;
                for(std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex)
                {
                    *output << ',' << materialAmplitudes[modeIndex]
                            << ',' << radiationAmplitudes[modeIndex];
                }
                *output << ',' << materialEnergy << ',' << radiationEnergy << ','
                        << materialEnergy + radiationEnergy << ','
                        << currentParticleCount << '\n';
            }

            // A restart snapshot carries the whole particle census, so it is
            // far heavier than the history row written every cycle.  VTK is
            // cheap by comparison and uses the same cadence.
            const bool writeSnapshot = DueSnapshot(step + 1, steps, config.snapshotEvery);
            if(writeSnapshot)
            {
                WriteRestartAndVtk(simulation, tess, cells, extensives,
                                   radiationStep->getEradTimeAvg(), modes, tau,
                                   referenceTemperature, referenceEnergyDensity,
                                   step + 1, true, latestH5Path,
                                   snapshotDirectory, profilePath, rank);
            }

            if(rank == 0 && ((step + 1) % 10 == 0 || writeSnapshot))
            {
                std::cout << "step " << (step + 1) << '/' << steps
                          << " tau=" << tau
                          << " A_U1=" << materialAmplitudes[0]
                          << " A_U2=" << materialAmplitudes[1]
                          << " A_U3=" << materialAmplitudes[2]
                          << " packets=" << currentParticleCount;
                if(writeSnapshot)
                {
                    std::cout << " snapshot=" << latestH5Path;
                }
                std::cout << std::endl;
            }
        }

        {
            std::vector<double> material;
            std::vector<double> radiationCensus;
            std::vector<double> radiationAverage;
            std::vector<double> exactMaterial;
            std::vector<double> exactRadiation;
            FillDiagnosticFields(tess, cells, extensives,
                                 radiationStep->getEradTimeAvg(), modes, tau,
                                 referenceTemperature, referenceEnergyDensity,
                                 material, radiationCensus, radiationAverage,
                                 exactMaterial, exactRadiation);
            WriteFinalProfile(tess, material, radiationCensus, radiationAverage,
                              exactMaterial, exactRadiation, tau, profilePath,
                              rank);
        }

        if(rank == 0)
        {
            output->close();
            std::cout << std::setprecision(17)
                      << "TRE-3D exact decay rates = "
                      << modes[0].decayRate << ", " << modes[1].decayRate << ", "
                      << modes[2].decayRate << '\n'
                      << "TRE-3D exact G = " << modes[0].radiationFactor << ", "
                      << modes[1].radiationFactor << ", "
                      << modes[2].radiationFactor << '\n'
                      << "wrote history " << historyPath.string() << '\n'
                      << "wrote latest restart " << latestH5Path << '\n'
                      << "wrote final profile " << profilePath.string() << '\n'
                      << "wrote snapshots in " << snapshotDirectory.string() << '\n';
        }
    }
    catch(const UniversalError &error)
    {
        std::cerr << "TRE-3D failed on rank " << rank << std::endl;
        reportError(error, std::cerr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    catch(const std::exception &error)
    {
        std::cerr << "TRE-3D failed on rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
