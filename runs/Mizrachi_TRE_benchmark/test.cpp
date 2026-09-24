/*
 * TRE-1: Thermal Radiative Eigenmode benchmark, using STORM's gray IMC.
 *
 * The physical coordinates use L0 = 1 cm, so the periodic domain is
 * [0, 2*pi] x [0, 1] x [0, 1] cm and sigma_a = 1/cm.  The y/z dimensions
 * are only a periodic extrusion of the one-dimensional problem.  The
 * material law is U_m = a*T^4/beta and Cv = 4*a*T^3/beta, with beta=1.
 *
 * Unlike STORM's normal initial-particle helper, TRE initializes packets
 * from the exact anisotropic angular eigenfunction.  The subsequent source,
 * transport, absorption, and material update are all performed by STORM.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "monte/deps/CMMC/src/units/units.hpp"

#include "monte/examples/Vector3D.hpp"
#include "MadCart/CartesianMesh3D.hpp"
#include "monte/boundary/BoundaryCondition.hpp"
#ifdef STORM_WITH_MPI
#include <mpi.h>
#include "monte/manager/MonteCarloManagerFactory.hpp"
#else
#include "monte/manager/MonteCarloManagerSerial.hpp"
#endif
#include "monte/particle/Particle.hpp"
#include "monte/population/CombPopulationControl.hpp"
#include "monte/radiation/RadiationCell.hpp"
#include "monte/radiation/RadiationIMC.hpp"
#include "monte/radiation/RadiationOpacityModel.hpp"
#include "monte/utils/RandomInCell.hpp"

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double beta = 1.0;
constexpr double chi = 1.0;
constexpr double u0 = 1.0;
constexpr double amplitude = 0.10;
constexpr double waveNumber = 1.0;
constexpr double referenceLengthCm = 1.0;

using Grid = MadCart::CartesianMesh3D<Vector3D>;
using Particle = STORM::Particle<Vector3D>;
using Cell = STORM::RadiationCell;
using Extensives = STORM::SimpleExtensives;

struct Config {
    std::size_t nx = 96;
    std::size_t initialParticlesPerCell = 300;
    std::size_t newPhotonsPerCell = 20;
    std::size_t populationPerCell = 350;
    double dtau = 0.02;
    double finalTau = 8.0;
    std::uint64_t seed = 20260827;
    std::string output = "tre_1d_history.csv";
};

struct HelpRequested : std::exception
{
    const char *what() const noexcept override
    {
        return "help requested";
    }
};

void usage(const char *program)
{
    std::cerr
        << "Usage: " << program << " [options]\n"
        << "  --nx N                    x cells (default 96)\n"
        << "  --initial-particles N    exact-mode packets/cell at tau=0 (default 300)\n"
        << "  --new-photons N          STORM source packets/cell (default 20)\n"
        << "  --population N            population-control target/cell (default 350)\n"
        << "  --dtau X                  dimensionless timestep (default 0.02)\n"
        << "  --final-tau X             final dimensionless time (default 8)\n"
        << "  --seed N                  random seed (default 20260827)\n"
        << "  --output FILE             history CSV (default tre_1d_history.csv)\n"
        << "  --help                    show this help\n";
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
            throw HelpRequested{};
        if(argument == "--nx")
            config.nx = std::stoull(requireValue(argc, argv, i));
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
        else if(argument == "--seed")
            config.seed = std::stoull(requireValue(argc, argv, i));
        else if(argument == "--output")
            config.output = requireValue(argc, argv, i);
        else
            throw std::runtime_error("unknown option: " + argument);
    }

    if(config.nx < 4 || config.initialParticlesPerCell == 0 ||
       config.newPhotonsPerCell == 0 || config.populationPerCell == 0 ||
       !(config.dtau > 0.0) || !(config.finalTau > 0.0))
        throw std::runtime_error("all mesh, packet, time-step, and final-time values must be positive");
    return config;
}

double principalDecayRate()
{
    const double aMin = 1.0e-14;
    auto dispersion = [](double s) {
        const double a = chi + s;
        const double G = (chi / waveNumber) * std::atan(waveNumber / a);
        return s - beta * chi * (G - 1.0);
    };

    double lo = -chi + aMin;
    double hi = -aMin;
    double fLo = dispersion(lo);
    double fHi = dispersion(hi);
    if(!(fLo < 0.0 && fHi > 0.0))
        throw std::runtime_error("failed to bracket TRE principal eigenvalue");
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

double radiationEnergyModeFactor(double decayRate)
{
    const double a = chi + decayRate;
    return (chi / waveNumber) * std::atan(waveNumber / a);
}

double angularEigenfunction(double x, double mu, double decayRate)
{
    const double a = chi + decayRate;
    const double q = waveNumber * mu;
    return u0 + amplitude * chi * (a * std::cos(waveNumber * x) +
                                   q * std::sin(waveNumber * x)) /
                         (a * a + q * q);
}

double cellIntegratedRadiationEnergy(double left, double right, double G)
{
    const double transverseArea = 1.0;
    return transverseArea * (u0 * (right - left) +
        amplitude * G * (std::sin(waveNumber * right) -
                         std::sin(waveNumber * left)) / waveNumber);
}

class TREPeriodicBoundary final :
    public STORM::BoundaryCondition<Vector3D, Grid>
{
public:
    explicit TREPeriodicBoundary(const Grid &grid)
        : STORM::BoundaryCondition<Vector3D, Grid>(grid)
    {}

    STORM::ParticleStatus apply(Particle &particle) override
    {
        const auto bounds = this->grid.GetBoxCoordinates();
        const Vector3D lower = bounds.first;
        const Vector3D upper = bounds.second;
        bool wrapped = false;

        auto wrap = [&wrapped](double &coordinate, double velocity,
                               double lo, double hi) {
            const double width = hi - lo;
            const double tolerance = 1.0e-10 * std::max(1.0, width);
            if(velocity > 0.0 && coordinate >= hi - tolerance)
            {
                // Keep the packet on the opposite face.  The manager then
                // nudges it into the destination cell.  In particular, do
                // not normalize hi back to lo for a lower-face crossing:
                // that would make a one-cell periodic dimension hit the
                // same face repeatedly at roundoff scale.
                coordinate = lo;
                wrapped = true;
            }
            else if(velocity < 0.0 && coordinate <= lo + tolerance)
            {
                coordinate = hi;
                wrapped = true;
            }
            while(coordinate < lo)
            {
                coordinate += width;
                wrapped = true;
            }
            while(coordinate > hi)
            {
                coordinate -= width;
                wrapped = true;
            }
        };

        wrap(particle.location.x, particle.velocity.x, lower.x, upper.x);
        wrap(particle.location.y, particle.velocity.y, lower.y, upper.y);
        wrap(particle.location.z, particle.velocity.z, lower.z, upper.z);

        if(!wrapped)
            throw std::runtime_error("TRE periodic boundary called away from a periodic face");

        // MonteCarloManagerSerial interprets REFLECT as "boundary handling
        // completed" and nudges toward particle.cellIndex.  Updating the
        // destination first makes that nudge enter the wrapped cell; no
        // velocity reflection takes place.
        particle.cellIndex = this->grid.GetContainingCell(particle.location);
        return STORM::ParticleStatus::REFLECT;
    }

    std::vector<Particle> generateNewBoundaryParticles(double) override
    {
        return {};
    }

    STORM::DDMCBoundaryFaceBehavior getDDMCBoundaryFaceBehavior(
        std::size_t, std::size_t, std::size_t) const override
    {
        return STORM::DDMCBoundaryFaceBehavior::Unsupported;
    }
};

class TREEOS
{
public:
    double dT2cv(double, double temperature,
                 const std::vector<double> &, const std::vector<std::string> &) const
    {
        return 4.0 * units::arad * std::pow(temperature, 3) / beta;
    }

    double de2T(double density, double specificEnergy,
                const std::vector<double> &, const std::vector<std::string> &) const
    {
        const double U = std::max(0.0, density * specificEnergy * beta);
        return std::pow(U / units::arad, 0.25);
    }
};

class TREOpacity final : public STORM::RadiationOpacityModel<
    Vector3D, Grid, Cell, 1>
{
public:
    double CalcPlanckOpacity(const Cell &) override { return chi / referenceLengthCm; }
    double CalcScatteringOpacity(const Cell &) override { return 0.0; }
};

std::vector<Particle> makeExactInitialParticles(
    const Grid &grid, std::size_t particlesPerCell, double decayRate,
    double G, double energyDensityScale, std::uint64_t seed)
{
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::vector<Particle> particles;
    particles.reserve(grid.GetPointNo() * particlesPerCell);

    const double psiMax = u0 + amplitude * chi / (chi + decayRate);
    for(std::size_t cellIndex = 0; cellIndex < grid.GetPointNo(); ++cellIndex)
    {
        const double left = grid.GetMeshPoint(cellIndex).x -
                            0.5 * (2.0 * pi / static_cast<double>(grid.nx()));
        const double right = left + 2.0 * pi / static_cast<double>(grid.nx());
        const double cellEnergy = cellIntegratedRadiationEnergy(left, right, G);
        const double weight = energyDensityScale * cellEnergy /
                              static_cast<double>(particlesPerCell);

        for(std::size_t packetIndex = 0; packetIndex < particlesPerCell; ++packetIndex)
        {
            while(true)
            {
                const double x = left + (right - left) * uniform(generator);
                const double mu = 2.0 * uniform(generator) - 1.0;
                const double phi = 2.0 * pi * uniform(generator);
                const double psi = angularEigenfunction(x, mu, decayRate);
                if(uniform(generator) * psiMax > psi)
                    continue;

                const double transverse = std::sqrt(std::max(0.0, 1.0 - mu * mu));
                Particle particle;
                particle.location = Vector3D(x, uniform(generator), uniform(generator));
                particle.velocity = Vector3D(
                    mu, transverse * std::cos(phi), transverse * std::sin(phi)) * units::clight;
                particle.frequency = 0.0;
                particle.weight = weight;
                particle.initialWeight = weight;
                particle.cellIndex = cellIndex;
                particles.push_back(particle);
                break;
            }
        }
    }
    return particles;
}

#ifdef STORM_WITH_MPI
struct GlobalMeshSamples
{
    std::vector<double> x;
    std::vector<double> volume;
};

GlobalMeshSamples gatherGlobalMeshSamples(const Grid &grid, int rank, int nprocs)
{
    const int localCount = static_cast<int>(grid.GetPointNo());
    std::vector<double> localX(localCount);
    std::vector<double> localVolume(localCount);
    for(int i = 0; i < localCount; ++i)
    {
        localX[static_cast<std::size_t>(i)] = grid.GetCellCM(static_cast<std::size_t>(i)).x;
        localVolume[static_cast<std::size_t>(i)] = grid.GetVolume(static_cast<std::size_t>(i));
    }

    std::vector<int> recvCounts(static_cast<std::size_t>(nprocs));
    std::vector<int> displacements(static_cast<std::size_t>(nprocs));
    MPI_Gather(&localCount, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    GlobalMeshSamples global;
    if(rank == 0)
    {
        displacements[0] = 0;
        for(int proc = 1; proc < nprocs; ++proc)
            displacements[static_cast<std::size_t>(proc)] =
                displacements[static_cast<std::size_t>(proc - 1)] +
                recvCounts[static_cast<std::size_t>(proc - 1)];
        const int totalCount = displacements.back() + recvCounts.back();
        global.x.resize(static_cast<std::size_t>(totalCount));
        global.volume.resize(static_cast<std::size_t>(totalCount));
    }

    MPI_Gatherv(localX.data(), localCount, MPI_DOUBLE,
                rank == 0 ? global.x.data() : nullptr,
                recvCounts.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(localVolume.data(), localCount, MPI_DOUBLE,
                rank == 0 ? global.volume.data() : nullptr,
                recvCounts.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    return global;
}

std::vector<double> gatherFieldToRoot(const std::vector<double> &local, int rank, int nprocs)
{
    const int localCount = static_cast<int>(local.size());
    std::vector<int> recvCounts(static_cast<std::size_t>(nprocs));
    std::vector<int> displacements(static_cast<std::size_t>(nprocs));
    MPI_Gather(&localCount, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<double> global;
    if(rank == 0)
    {
        displacements[0] = 0;
        for(int proc = 1; proc < nprocs; ++proc)
            displacements[static_cast<std::size_t>(proc)] =
                displacements[static_cast<std::size_t>(proc - 1)] +
                recvCounts[static_cast<std::size_t>(proc - 1)];
        global.resize(static_cast<std::size_t>(displacements.back() + recvCounts.back()));
    }

    MPI_Gatherv(const_cast<double *>(local.data()), localCount, MPI_DOUBLE,
                rank == 0 ? global.data() : nullptr,
                recvCounts.data(), displacements.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    return global;
}

std::size_t globalParticleCount(const std::vector<Particle> &particles)
{
    unsigned long long localValue = static_cast<unsigned long long>(particles.size());
    unsigned long long globalValue = 0;
    MPI_Allreduce(&localValue, &globalValue, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
                  MPI_COMM_WORLD);
    return static_cast<std::size_t>(globalValue);
}
#endif

double projectMode(const std::vector<double> &field,
                   const std::vector<double> &x,
                   const std::vector<double> &volume,
                   double background)
{
    double numerator = 0.0;
    double domainVolume = 0.0;
    for(std::size_t i = 0; i < field.size(); ++i)
    {
        numerator += volume[i] * (field[i] - background) * std::cos(x[i]);
        domainVolume += volume[i];
    }
    return 2.0 * numerator / domainVolume;
}

double totalField(const std::vector<double> &values, const std::vector<double> &volume)
{
    double result = 0.0;
    for(std::size_t i = 0; i < values.size(); ++i)
        result += values[i] * volume[i];
    return result;
}

#ifndef STORM_WITH_MPI
double projectMode(const std::vector<double> &field, const Grid &grid, double background)
{
    double numerator = 0.0;
    double volume = 0.0;
    for(std::size_t i = 0; i < grid.GetPointNo(); ++i)
    {
        const double cellVolume = grid.GetVolume(i);
        const double x = grid.GetCellCM(i).x;
        numerator += cellVolume * (field[i] - background) * std::cos(x);
        volume += cellVolume;
    }
    return 2.0 * numerator / volume;
}

double totalField(const std::vector<double> &values, const Grid &grid)
{
    double result = 0.0;
    for(std::size_t i = 0; i < grid.GetPointNo(); ++i)
        result += values[i] * grid.GetVolume(i);
    return result;
}
#endif

void ensureParentDirectory(const std::string &path)
{
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if(!parent.empty())
        std::filesystem::create_directories(parent);
}

} // namespace

int main(int argc, char **argv)
{
#ifdef STORM_WITH_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#else
    int rank = 0;
    int nprocs = 1;
#endif

    try
    {
#ifdef STORM_WITH_MPI
        { // MPI-dependent objects must be destroyed before MPI_Finalize
#endif
        const Config config = parseConfig(argc, argv);
        const double decayRate = principalDecayRate();
        const double G = radiationEnergyModeFactor(decayRate);
        const double referenceTemperature = units::kev_kelvin;
        const double referenceEnergyDensity = units::arad *
            std::pow(referenceTemperature, 4);

        Grid grid(Vector3D(0.0, 0.0, 0.0), Vector3D(2.0 * pi, 1.0, 1.0),
                  config.nx, 1, 1);
        grid.SetPeriodicity(true, true, true);
#ifdef STORM_WITH_MPI
        std::vector<double> uniformWeights(config.nx, 1.0);
        grid.BuildParallel(uniformWeights);
#endif
#ifdef STORM_WITH_MPI
        const GlobalMeshSamples globalMesh = gatherGlobalMeshSamples(grid, rank, nprocs);
#endif
        std::vector<Cell> cells(grid.GetPointNo());
        std::vector<Extensives> extensives(grid.GetPointNo());
        std::vector<double> initialMaterial(grid.GetPointNo());
        std::vector<double> initialRadiation(grid.GetPointNo());

        for(std::size_t i = 0; i < grid.GetPointNo(); ++i)
        {
            const double x = grid.GetCellCM(i).x;
            const double volume = grid.GetVolume(i);
            const double U = u0 + amplitude * std::cos(x);
            const double radiation = cellIntegratedRadiationEnergy(
                x - 0.5 * grid.dx(), x + 0.5 * grid.dx(), G) / volume;

            cells[i].temperature = referenceTemperature * std::pow(U / u0, 0.25);
            cells[i].internalEnergy = referenceEnergyDensity * (U / beta) * volume;
            cells[i].Erad = referenceEnergyDensity * radiation;
            extensives[i].mass = volume;
            extensives[i].internal_energy = cells[i].internalEnergy;
            extensives[i].Erad = referenceEnergyDensity * radiation * volume;
            initialMaterial[i] = U;
            initialRadiation[i] = radiation;
        }

        auto boundary = std::make_shared<TREPeriodicBoundary>(grid);
        auto eos = std::make_shared<TREEOS>();
        auto opacity = std::make_shared<TREOpacity>();

        STORM::RadiationIMCParameters<1> parameters;
        parameters.newPhotonsPerCell = config.newPhotonsPerCell;
        parameters.withHydro = false;
        parameters.withRandomWalk = false;
        parameters.withDDMC = false;
        parameters.withMultigroupOpacity = false;

        using IMC = STORM::RadiationIMC<
            Vector3D, Grid, Cell, Extensives, TREEOS, 1>;
        auto physics = std::make_shared<IMC>(
            grid, boundary, cells, extensives, eos, opacity, parameters,
            STORM::DirectRadiationIMCTraits<Vector3D, Cell, Extensives, 1>{},
            RandomInCellPositionSampler<Vector3D, Grid>{}, config.seed);
        auto populationControl = std::make_shared<
            STORM::CombPopulationControl<Vector3D, Grid>>(
                grid, config.populationPerCell, 1.0);

#ifdef STORM_WITH_MPI
        STORM::MonteCarloManager<Vector3D, Grid, IMC> manager =
            STORM::CreateMonteCarloManager<Vector3D, Grid>(
                grid, physics, populationControl, boundary);
#else
        STORM::MonteCarloManagerSerial<Vector3D, Grid> manager(
            grid, physics, populationControl, boundary);
#endif
        std::vector<Particle> initialParticles = makeExactInitialParticles(
            grid, config.initialParticlesPerCell, decayRate, G,
            referenceEnergyDensity, config.seed + 1);
#ifdef STORM_WITH_MPI
        manager.getParticles() = std::move(initialParticles);
        std::vector<Particle> &particles = manager.getParticles();
#else
        std::vector<Particle> particles = std::move(initialParticles);
#endif

#ifdef STORM_WITH_MPI
        const std::size_t initialParticleCount = globalParticleCount(particles);
        const std::vector<double> globalInitialMaterial =
            gatherFieldToRoot(initialMaterial, rank, nprocs);
        const std::vector<double> globalInitialRadiation =
            gatherFieldToRoot(initialRadiation, rank, nprocs);
#else
        const std::size_t initialParticleCount = particles.size();
#endif

        std::unique_ptr<std::ofstream> output;
        if(rank == 0)
        {
            ensureParentDirectory(config.output);
            output = std::make_unique<std::ofstream>(config.output);
            if(!*output)
                throw std::runtime_error("could not open output file: " + config.output);
            *output << "# benchmark=TRE-1\n"
                    << "# beta=" << std::setprecision(17) << beta << "\n"
                    << "# chi=" << chi << "\n"
                    << "# exact_decay_rate=" << decayRate << "\n"
                    << "# exact_G=" << G << "\n"
                    << "# amplitude=" << amplitude << "\n"
                    << "# nx=" << config.nx << "\n"
                    << "# mpi_ranks=" << nprocs << "\n"
                    << "# initial_particles_per_cell=" << config.initialParticlesPerCell << "\n"
                    << "# new_photons_per_cell=" << config.newPhotonsPerCell << "\n"
                    << "# population_per_cell=" << config.populationPerCell << "\n"
                    << "# dtau=" << config.dtau << "\n"
                    << "tau_end,tau_mid,A_U,A_E,material_energy,radiation_energy,total_energy,particle_count\n";

#ifdef STORM_WITH_MPI
            const double initialMaterialEnergy = referenceEnergyDensity *
                totalField(globalInitialMaterial, globalMesh.volume) / beta;
            const double initialRadiationEnergy = referenceEnergyDensity *
                totalField(globalInitialRadiation, globalMesh.volume);
            *output << std::scientific << std::setprecision(17)
                    << 0.0 << ',' << 0.0 << ','
                    << projectMode(globalInitialMaterial, globalMesh.x, globalMesh.volume, u0) << ','
                    << projectMode(globalInitialRadiation, globalMesh.x, globalMesh.volume, u0) << ','
                    << initialMaterialEnergy << ',' << initialRadiationEnergy << ','
                    << initialMaterialEnergy + initialRadiationEnergy << ','
                    << initialParticleCount << '\n';
#else
            const double initialMaterialEnergy = referenceEnergyDensity *
                totalField(initialMaterial, grid) / beta;
            const double initialRadiationEnergy = referenceEnergyDensity *
                totalField(initialRadiation, grid);
            *output << std::scientific << std::setprecision(17)
                    << 0.0 << ',' << 0.0 << ','
                    << projectMode(initialMaterial, grid, u0) << ','
                    << projectMode(initialRadiation, grid, u0) << ','
                    << initialMaterialEnergy << ',' << initialRadiationEnergy << ','
                    << initialMaterialEnergy + initialRadiationEnergy << ','
                    << initialParticleCount << '\n';
#endif
        }

        const std::size_t steps = static_cast<std::size_t>(
            std::ceil(config.finalTau / config.dtau));
        double tau = 0.0;
        for(std::size_t step = 0; step < steps; ++step)
        {
            const double currentDtau = std::min(config.dtau, config.finalTau - tau);
            if(!(currentDtau > 0.0))
                break;
            const double currentDt = currentDtau * referenceLengthCm / units::clight;
#ifdef STORM_WITH_MPI
            manager.step(currentDt);
#else
            particles = manager.step(std::move(particles), currentDt);
#endif
            tau += currentDtau;

            std::vector<double> material(grid.GetPointNo());
            for(std::size_t i = 0; i < grid.GetPointNo(); ++i)
                material[i] = std::pow(cells[i].temperature / referenceTemperature, 4);
            const std::vector<double> &radiation = physics->getEradTimeAvg();
            std::vector<double> radiationDimensionless(radiation.size());
            std::vector<double> radiationCensusDimensionless(radiation.size());
            for(std::size_t i = 0; i < radiation.size(); ++i)
            {
                radiationDimensionless[i] = radiation[i] / referenceEnergyDensity;
                radiationCensusDimensionless[i] = extensives[i].Erad /
                    (referenceEnergyDensity * grid.GetVolume(i));
            }

#ifdef STORM_WITH_MPI
            const std::size_t particleCount = globalParticleCount(particles);
            const std::vector<double> globalMaterial = gatherFieldToRoot(material, rank, nprocs);
            const std::vector<double> globalRadiation =
                gatherFieldToRoot(radiationDimensionless, rank, nprocs);
            const std::vector<double> globalRadiationCensus =
                gatherFieldToRoot(radiationCensusDimensionless, rank, nprocs);
            if(rank == 0)
            {
                const double materialEnergy = referenceEnergyDensity *
                    totalField(globalMaterial, globalMesh.volume) / beta;
                const double radiationEnergy = referenceEnergyDensity *
                    totalField(globalRadiationCensus, globalMesh.volume);
                *output << tau << ',' << tau - 0.5 * currentDtau << ','
                        << projectMode(globalMaterial, globalMesh.x, globalMesh.volume, u0) << ','
                        << projectMode(globalRadiation, globalMesh.x, globalMesh.volume, u0) << ','
                        << materialEnergy << ',' << radiationEnergy << ','
                        << materialEnergy + radiationEnergy << ','
                        << particleCount << '\n';
            }
#else
            if(rank == 0)
            {
                const double materialEnergy = referenceEnergyDensity *
                    totalField(material, grid) / beta;
                const double radiationEnergy = referenceEnergyDensity *
                    totalField(radiationCensusDimensionless, grid);
                *output << tau << ',' << tau - 0.5 * currentDtau << ','
                        << projectMode(material, grid, u0) << ','
                        << projectMode(radiationDimensionless, grid, u0) << ','
                        << materialEnergy << ',' << radiationEnergy << ','
                        << materialEnergy + radiationEnergy << ','
                        << particles.size() << '\n';
            }
#endif
        }

        if(rank == 0)
        {
            output->close();
        }
#ifdef STORM_WITH_MPI
        } // end MPI scope
#endif
    }
    catch(const HelpRequested &)
    {
        if(rank == 0)
            usage(argv[0]);
#ifdef STORM_WITH_MPI
        MPI_Finalize();
#endif
        return 0;
    }
    catch(const std::exception &error)
    {
        if(rank == 0)
            std::cerr << "TRE-1 failed: " << error.what() << '\n';
#ifdef STORM_WITH_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }
#ifdef STORM_WITH_MPI
    MPI_Finalize();
#endif
}
