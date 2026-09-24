/*
 * Jiang, Davis & Stone G3 radiative Rayleigh-Taylor IMC analogue.
 *
 * Frozen dimensionless physics: rho_- = 1, rho_+ = 4, g = 0.1, alpha = 1,
 * p_gas = 1, kappa_s = 1, kappa_P = 0, c = 1e4, and F_r,z = 0.1.
 * The ideal-gas gamma is configurable (default 5/3) because it controls only
 * the numerical compressible analogue. Vertical hydro boundaries are rigid
 * free-slip walls; radiation is injected at both vertical faces and escapes.
 *
 * A positive maximum-entropy angular distribution proportional to exp(beta*mu)
 * is used, with beta chosen so <mu> = F_r,z/E. It preserves both the stated
 * radiation energy and flux even near the upper boundary, where a linear P1
 * intensity would become negative. This IMC angular substitution is recorded
 * in the history metadata.
 */

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "source/3D/output/read3D.hpp"
#include "source/3D/output/write3D.hpp"
#include "source/3D/radiation/IMCCostCalculator.hpp"
#include "source/3D/radiation/RadiationIMC.hpp"
#include "source/3D/tessellation/Voronoi3D.hpp"
#include "source/Radiation/OpacityCalculator.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/monte/boundary/BoundaryCondition.hpp"
#include "source/monte/population/CombPopulationControl.hpp"
#include "source/monte/utils/RandomInCell.hpp"
#include "source/monte/utils/RandomOnFace.hpp"
#include "source/mpi/mpi_commands.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/ConservativeForce3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/Ghost3D.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"
#include "source/newtonian/three_dimensional/conserved_3d.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/eulerian_3d.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/HydroStep.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"
#include "source/utils/arguments/ArgumentParser.hpp"
#include "source/utils/debug/vtune.h"

namespace
{

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double rhoMinus = 1.0;
constexpr double rhoPlus = 4.0;
constexpr double gravity = 0.1;
constexpr double gasPressure = 1.0;
constexpr double massScatteringOpacity = 1.0;
constexpr double lightSpeed = 1.0e4;
constexpr double radiationFluxFraction = 0.1;
constexpr double radiationFlux = lightSpeed * radiationFluxFraction;
constexpr double targetTimeOne = 21.6;
constexpr double targetTimeTwo = 28.9;

struct Config
{
    std::size_t nx = 128;
    std::size_t ny = 1;
    std::size_t nz = 512;
    double finalTime = targetTimeTwo;
    double cfl = 0.25;
    std::size_t initialParticlesPerCell = 80;
    std::size_t boundaryParticlesPerFace = 8;
    std::size_t populationPerCell = 400;
    std::size_t historyEvery = 5;
    std::size_t logEvery = 500;
    std::size_t outputEvery = 50;
    std::size_t checkpointEvery = 500;
    std::size_t vtkEvery = 200;
    std::uint64_t seed = 20260831ULL;
    double gamma = 5.0 / 3.0;
    std::string outputDirectory = "jds_g3_output";
    bool writeDumps = false;
    std::string managerName = "p2p";
    std::string restartFile;
};

Config ParseConfig(ArgumentParser &arguments)
{
    Config config;
    config.nx = arguments.get<std::size_t>("nx");
    config.ny = arguments.get<std::size_t>("ny");
    config.nz = arguments.get<std::size_t>("nz");
    config.finalTime = arguments.get<double>("final-time");
    config.cfl = arguments.get<double>("cfl");
    config.initialParticlesPerCell = arguments.get<std::size_t>("initial-particles");
    config.boundaryParticlesPerFace = arguments.get<std::size_t>("boundary-particles");
    config.populationPerCell = arguments.get<std::size_t>("population");
    config.historyEvery = arguments.get<std::size_t>("history-every");
    config.logEvery = arguments.get<std::size_t>("log-every");
    config.outputEvery = arguments.get<std::size_t>("output-cycles");
    config.checkpointEvery = arguments.get<std::size_t>("checkpoint-cycles");
    config.vtkEvery = arguments.get<std::size_t>("vtk-cycles");
    config.seed = arguments.get<std::uint64_t>("seed");
    config.gamma = arguments.get<double>("gamma");
    config.managerName = arguments.get<std::string>("manager");
    if(arguments.wasSet("output"))
    {
        config.outputDirectory = arguments.get<std::string>("output");
        config.writeDumps = !config.outputDirectory.empty();
    }
    if(arguments.wasSet("restart"))
    {
        config.restartFile = arguments.get<std::string>("restart");
        if(!arguments.wasSet("output"))
        {
            const std::filesystem::path restartPath(config.restartFile);
            if(restartPath.parent_path().filename() == "snapshots")
            {
                config.outputDirectory = restartPath.parent_path().parent_path().string();
            }
            else
            {
                config.outputDirectory = restartPath.parent_path().string();
            }
        }
    }
    if(!config.writeDumps)
    {
        config.outputEvery = 0;
        config.checkpointEvery = 0;
        config.vtkEvery = 0;
    }
    if(config.nx == 0 || config.ny == 0 || config.nz == 0 ||
       !(config.finalTime > 0.0) || !(config.cfl > 0.0) || !(config.cfl < 1.0) ||
       !(config.gamma > 1.0) || config.initialParticlesPerCell == 0 ||
       config.boundaryParticlesPerFace == 0 || config.populationPerCell == 0 ||
       config.historyEvery == 0 || config.outputDirectory.empty())
    {
        throw std::runtime_error("invalid JDS G3 numerical configuration");
    }
    return config;
}

double BaseDensity(double z)
{
    return z < 0.0 ? rhoMinus : rhoPlus;
}

double RadiationEnergyDensity(double z)
{
    if(z < 0.0)
    {
        return 1.72 - 0.3 * (z + 1.0);
    }
    return 1.42 - 1.2 * z;
}

std::uint64_t MixSeed(std::uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::size_t CartesianCellID(const Vector3D &center, const Config &config)
{
    const std::size_t ix = std::min(config.nx - 1, static_cast<std::size_t>(
        std::max(0.0, std::floor((center.x + 0.5) * static_cast<double>(config.nx)))));
    const std::size_t iy = std::min(config.ny - 1, static_cast<std::size_t>(
        std::max(0.0, std::floor((center.y + 0.5) * static_cast<double>(config.ny)))));
    const std::size_t iz = std::min(config.nz - 1, static_cast<std::size_t>(
        std::max(0.0, std::floor(0.5 * (center.z + 1.0) * static_cast<double>(config.nz)))));
    return ix + config.nx * (iy + config.ny * iz);
}

double CellPerturbation(std::uint64_t seed, std::size_t cellID, double baseDensity)
{
    std::mt19937_64 generator(MixSeed(seed ^ static_cast<std::uint64_t>(cellID)));
    std::uniform_real_distribution<double> uniform(-0.5 * baseDensity, 0.5 * baseDensity);
    return uniform(generator);
}

double MeanMu(double beta)
{
    if(beta < 1.0e-6)
    {
        return beta / 3.0;
    }
    return 1.0 / std::tanh(beta) - 1.0 / beta;
}

double AngularBeta(double fluxFactor)
{
    if(!(fluxFactor >= 0.0) || !(fluxFactor < 1.0))
    {
        throw std::runtime_error("invalid JDS angular flux factor");
    }
    if(fluxFactor == 0.0)
    {
        return 0.0;
    }
    double lower = 0.0;
    double upper = 1.0;
    while(MeanMu(upper) < fluxFactor)
    {
        upper *= 2.0;
    }
    for(std::size_t iteration = 0; iteration < 100; ++iteration)
    {
        const double middle = 0.5 * (lower + upper);
        if(MeanMu(middle) < fluxFactor)
        {
            lower = middle;
        }
        else
        {
            upper = middle;
        }
    }
    return 0.5 * (lower + upper);
}

double SampleAngularMu(double beta, std::mt19937_64 &generator,
                       std::uniform_real_distribution<double> &uniform)
{
    if(beta < 1.0e-8)
    {
        return 2.0 * uniform(generator) - 1.0;
    }
    const double expMinusBeta = std::exp(-beta);
    const double expBeta = std::exp(beta);
    return std::log(expMinusBeta + uniform(generator) * (expBeta - expMinusBeta)) / beta;
}

double SampleIncomingMu(double beta, bool bottom, std::mt19937_64 &generator,
                        std::uniform_real_distribution<double> &uniform)
{
    const double minimumMu = bottom ? 0.0 : -1.0;
    const double maximumMu = bottom ? 1.0 : 0.0;
    double maximumDensity;
    if(bottom)
    {
        maximumDensity = std::exp(beta);
    }
    else
    {
        const double magnitudeAtMaximum = beta > 1.0 ? 1.0 / beta : 1.0;
        maximumDensity = magnitudeAtMaximum * std::exp(-beta * magnitudeAtMaximum);
    }
    for(;;)
    {
        const double mu = minimumMu + (maximumMu - minimumMu) * uniform(generator);
        const double density = std::abs(mu) * std::exp(beta * mu);
        if(maximumDensity * uniform(generator) <= density)
        {
            return mu;
        }
    }
}

double IncomingEnergyFlux(double energyDensity, bool bottom)
{
    const double beta = AngularBeta(radiationFluxFraction / energyDensity);
    if(beta < 1.0e-8)
    {
        return 0.25 * lightSpeed * energyDensity;
    }
    const double normalization = 2.0 * std::sinh(beta) / beta;
    const double angularIntegral = bottom
        ? (std::exp(beta) * (beta - 1.0) + 1.0) / (beta * beta)
        : (1.0 - (1.0 + beta) * std::exp(-beta)) / (beta * beta);
    return lightSpeed * energyDensity * angularIntegral / normalization;
}

class JDSHydroWall final : public Ghost3D
{
public:
    explicit JDSHydroWall(double gamma):
        gamma_(gamma)
    {}

    void operator()(const Tessellation3D &tess, const std::vector<ComputationalCell3D> &cells,
                    double, boost::container::flat_map<std::size_t, ComputationalCell3D> &result)
        const override
    {
        const std::vector<std::pair<std::size_t, std::size_t>> outerFaces =
            this->GetOuterFacesIndeces(tess);
        result.clear();
        for(const std::pair<std::size_t, std::size_t> &faceDescription : outerFaces)
        {
            const std::pair<std::size_t, std::size_t> neighbors =
                tess.GetFaceNeighbors(faceDescription.first);
            const std::size_t ghostIndex =
                faceDescription.second == 1 ? neighbors.first : neighbors.second;
            const std::size_t realIndex =
                faceDescription.second == 1 ? neighbors.second : neighbors.first;
            ComputationalCell3D ghost = cells[realIndex];
            const Vector3D normal = normalize(
                tess.GetMeshPoint(ghostIndex) - tess.GetMeshPoint(realIndex));
            ghost.velocity -= 2.0 * normal * ScalarProd(normal, ghost.velocity);
            ghost.pressure = gasPressure;
            ghost.internal_energy = ghost.pressure / (ghost.density * (gamma_ - 1.0));
            result.insert(std::make_pair(ghostIndex, ghost));
        }
    }

    Slope3D GetGhostGradient(const Tessellation3D &, const std::vector<ComputationalCell3D> &,
                             const std::vector<Slope3D> &, std::size_t, double,
                             std::size_t) const override
    {
        return Slope3D();
    }

private:
    double gamma_;
};

class JDSOpacity final : public OpacityCalculator
{
public:
    double CalcPlanckOpacity(const ComputationalCell3D &) const override
    {
        return 0.0;
    }

    double CalcScatteringOpacity(const ComputationalCell3D &cell) const override
    {
        return std::max(0.0, cell.density) * massScatteringOpacity;
    }
};

class JDSRadiationBoundary final :
    public STORM::BoundaryCondition<Vector3D, Tessellation3D>
{
public:
    JDSRadiationBoundary(const Tessellation3D &tess, std::size_t packetsPerFace,
                         std::uint64_t seed):
        STORM::BoundaryCondition<Vector3D, Tessellation3D>(tess),
        packetsPerFace_(packetsPerFace),
        seed_(seed),
        cycle_(0)
    {}

    void SetCycle(std::size_t cycle)
    {
        cycle_ = cycle;
    }

    STORM::ParticleStatus apply(Particle3D &) override
    {
        return STORM::ParticleStatus::REMOVE;
    }

    std::vector<Particle3D> generateNewBoundaryParticles(double fullDt) override
    {
        std::vector<Particle3D> particles;
        if(!(fullDt > 0.0) || packetsPerFace_ == 0)
        {
            return particles;
        }
        std::mt19937_64 generator(MixSeed(seed_ ^ (static_cast<std::uint64_t>(cycle_) << 32U)));
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        const std::pair<Vector3D, Vector3D> box = this->grid.GetBoxCoordinates();
        const double tolerance = 1.0e-8 * (box.second.z - box.first.z);
        for(std::size_t cellIndex = 0; cellIndex < this->grid.GetPointNo(); ++cellIndex)
        {
            const Vector3D &cellPoint = this->grid.GetMeshPoint(cellIndex);
            for(const std::size_t faceIndex : this->grid.GetCellFaces(cellIndex))
            {
                if(!this->grid.BoundaryFace(faceIndex))
                {
                    continue;
                }
                const Vector3D faceCenter = this->grid.FaceCM(faceIndex);
                const bool bottom = std::abs(faceCenter.z - box.first.z) < tolerance;
                const bool top = std::abs(faceCenter.z - box.second.z) < tolerance;
                if(!bottom && !top)
                {
                    continue;
                }
                const double boundaryZ = bottom ? box.first.z : box.second.z;
                const double energyDensity = RadiationEnergyDensity(boundaryZ);
                const double beta = AngularBeta(radiationFluxFraction / energyDensity);
                const double faceEnergy = IncomingEnergyFlux(energyDensity, bottom) *
                    this->grid.GetArea(faceIndex) * fullDt;
                const double packetWeight = faceEnergy / static_cast<double>(packetsPerFace_);
                for(std::size_t packetIndex = 0; packetIndex < packetsPerFace_; ++packetIndex)
                {
                    const double mu = SampleIncomingMu(beta, bottom, generator, uniform);
                    const double phi = 2.0 * pi * uniform(generator);
                    const double transverse = std::sqrt(std::max(0.0, 1.0 - mu * mu));
                    Particle3D particle;
                    particle.location = STORM::RandomPointOnFace<Vector3D, Tessellation3D>(
                        this->grid, faceIndex);
                    constexpr double nudge = 1.0e-8;
                    particle.location = particle.location * (1.0 - nudge) + cellPoint * nudge;
                    particle.velocity = Vector3D(transverse * std::cos(phi),
                                                 transverse * std::sin(phi), mu) * lightSpeed;
                    particle.frequency = 0.0;
                    particle.weight = packetIndex + 1 == packetsPerFace_
                        ? faceEnergy - packetWeight * static_cast<double>(packetsPerFace_ - 1)
                        : packetWeight;
                    particle.initialWeight = particle.weight;
                    particle.timeLeft = fullDt * uniform(generator);
                    particle.cellIndex = cellIndex;
                    particle.rngKey = MixSeed(seed_ ^
                        (static_cast<std::uint64_t>(cycle_) << 32U) ^
                        static_cast<std::uint64_t>(cellIndex * packetsPerFace_ + packetIndex));
                    particle.rngCounter = 0;
                    particles.push_back(particle);
                }
            }
        }
        return particles;
    }

    STORM::DDMCBoundaryFaceBehavior getDDMCBoundaryFaceBehavior(
        std::size_t, std::size_t, std::size_t) const override
    {
        return STORM::DDMCBoundaryFaceBehavior::Unsupported;
    }

private:
    std::size_t packetsPerFace_;
    std::uint64_t seed_;
    std::size_t cycle_;
};

std::vector<Particle3D> GenerateInitialParticles(
    const Tessellation3D &tess, const std::vector<ComputationalCell3D> &cells,
    std::size_t particlesPerCell, std::uint64_t seed, int rank)
{
    std::vector<Particle3D> particles;
    particles.reserve(tess.GetPointNo() * particlesPerCell);
    std::mt19937_64 generator(MixSeed(seed + static_cast<std::uint64_t>(rank)));
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    RandomInCellPositionSampler<Vector3D, Tessellation3D> positionSampler;
    for(std::size_t cellIndex = 0; cellIndex < tess.GetPointNo(); ++cellIndex)
    {
        const double z = tess.GetCellCM(cellIndex).z;
        const double energyDensity = RadiationEnergyDensity(z);
        const double beta = AngularBeta(radiationFluxFraction / energyDensity);
        const double totalEnergy = energyDensity * tess.GetVolume(cellIndex);
        const double packetWeight = totalEnergy / static_cast<double>(particlesPerCell);
        for(std::size_t packetIndex = 0; packetIndex < particlesPerCell; ++packetIndex)
        {
            const double mu = SampleAngularMu(beta, generator, uniform);
            const double phi = 2.0 * pi * uniform(generator);
            const double transverse = std::sqrt(std::max(0.0, 1.0 - mu * mu));
            Particle3D particle;
            particle.location = positionSampler(tess, cellIndex, generator, uniform);
            constexpr double nudge = 1.0e-10;
            particle.location = particle.location * (1.0 - nudge) +
                tess.GetMeshPoint(cellIndex) * nudge;
            particle.velocity = Vector3D(transverse * std::cos(phi),
                                         transverse * std::sin(phi), mu) * lightSpeed;
            particle.cellIndex = cellIndex;
            particle.cellID = cells[cellIndex].ID;
            particle.sourceCellID = cells[cellIndex].ID;
            particle.frequency = 0.0;
            particle.weight = packetIndex + 1 == particlesPerCell
                ? totalEnergy - packetWeight * static_cast<double>(particlesPerCell - 1)
                : packetWeight;
            particle.initialWeight = particle.weight;
            particle.timeLeft = 0.0;
            particle.rngKey = MixSeed(seed ^
                (static_cast<std::uint64_t>(rank) << 48U) ^
                static_cast<std::uint64_t>(cellIndex * particlesPerCell + packetIndex));
            particle.rngCounter = 0;
            particles.push_back(particle);
        }
    }
    return particles;
}

std::size_t GlobalParticleCount(const std::vector<Particle3D> &particles)
{
    unsigned long long localCount = static_cast<unsigned long long>(particles.size());
    unsigned long long globalCount = 0;
    MPI_Allreduce(&localCount, &globalCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return static_cast<std::size_t>(globalCount);
}

struct Diagnostics
{
    double zmax;
    double zmin;
    double mixingFraction;
    double radiationEnergy;
    double dimensionlessFrz;
};

Diagnostics CalculateDiagnostics(const Voronoi3D &tess,
                                 const std::vector<ComputationalCell3D> &cells,
                                 const std::vector<Particle3D> &particles)
{
    double localZmax = 0.0;
    double localZmin = 0.0;
    double localMixingVolume = 0.0;
    double localVolume = 0.0;
    double localRadiationEnergy = 0.0;
    double localFluxMoment = 0.0;
    for(std::size_t cellIndex = 0; cellIndex < tess.GetPointNo(); ++cellIndex)
    {
        const double z = tess.GetCellCM(cellIndex).z;
        const double density = cells[cellIndex].density;
        const double volume = tess.GetVolume(cellIndex);
        double initialDensity = cells[cellIndex].tracers[1];
        if(!(initialDensity > 0.0))
        {
            initialDensity = density;
        }
        if(std::abs(density - initialDensity) >= 0.1 * initialDensity)
        {
            if(z > 0.0)
            {
                localZmax = std::max(localZmax, z);
            }
            if(z < 0.0)
            {
                localZmin = std::min(localZmin, z);
            }
        }
        if(density >= 1.1 * rhoMinus && density <= 0.9 * rhoPlus)
        {
            localMixingVolume += volume;
        }
        localVolume += volume;
        localRadiationEnergy += density * cells[cellIndex].Erad * volume;
    }
    for(const Particle3D &particle : particles)
    {
        localFluxMoment += particle.weight * particle.velocity.z / lightSpeed;
    }
    Diagnostics result;
    MPI_Allreduce(&localZmax, &result.zmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&localZmin, &result.zmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    double globalMixingVolume = 0.0;
    double globalVolume = 0.0;
    double globalFluxMoment = 0.0;
    MPI_Allreduce(&localMixingVolume, &globalMixingVolume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localVolume, &globalVolume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&localRadiationEnergy, &result.radiationEnergy, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(&localFluxMoment, &globalFluxMoment, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    result.mixingFraction = globalMixingVolume / globalVolume;
    result.dimensionlessFrz = (globalVolume > 0.0) ? globalFluxMoment / globalVolume : 0.0;
    return result;
}

bool DueEvery(std::size_t cycle, std::size_t interval)
{
    return interval > 0 && cycle % interval == 0;
}

std::string CycleStem(std::size_t cycle)
{
    std::ostringstream stream;
    stream << "jds_g3_cycle_" << std::setw(7) << std::setfill('0') << cycle;
    return stream.str();
}

std::string TargetStem(double time)
{
    std::ostringstream stream;
    stream << "jds_g3_t" << std::fixed << std::setprecision(1) << time;
    return stream.str();
}

void WriteJDSVtk(const Voronoi3D &tess, const std::vector<ComputationalCell3D> &cells,
                 const std::shared_ptr<RadiationIMC> &physics, const std::string &filename)
{
    const std::size_t count = tess.GetPointNo();
    std::vector<double> density(count);
    std::vector<double> pressure(count);
    std::vector<double> erad(count);
    std::vector<double> eradTimeAverage(count);
    std::vector<double> velocityX(count);
    std::vector<double> velocityY(count);
    std::vector<double> velocityZ(count);
    std::vector<double> baseDensity(count);
    std::vector<double> initialDensity(count);
    const std::vector<double> &timeAverage = physics->getEradTimeAvg();
    for(std::size_t cellIndex = 0; cellIndex < count; ++cellIndex)
    {
        density[cellIndex] = cells[cellIndex].density;
        pressure[cellIndex] = cells[cellIndex].pressure;
        erad[cellIndex] = cells[cellIndex].density * cells[cellIndex].Erad;
        eradTimeAverage[cellIndex] = timeAverage[cellIndex];
        velocityX[cellIndex] = cells[cellIndex].velocity.x;
        velocityY[cellIndex] = cells[cellIndex].velocity.y;
        velocityZ[cellIndex] = cells[cellIndex].velocity.z;
        baseDensity[cellIndex] = cells[cellIndex].tracers[0];
        initialDensity[cellIndex] = cells[cellIndex].tracers[1];
    }
    WriteVoronoiVTKOnly(tess, filename,
                        {density, pressure, erad, eradTimeAverage, velocityX, velocityY,
                         velocityZ, baseDensity, initialDensity},
                        {"density", "pressure", "Er", "Er_time_avg", "velocity_x",
                         "velocity_y", "velocity_z", "base_density", "initial_density"});
}

} // namespace

int main(int argc, char **argv)
{
    vtune_stop();
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    int rank = 0;
    int processCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    try
    {
        ArgumentParser arguments("Jiang-Davis-Stone G3 IMC analogue");
        arguments.addOption<std::size_t>("nx", 128, "Cartesian cells in x");
        arguments.addOption<std::size_t>("ny", 1, "Cartesian cells in y");
        arguments.addOption<std::size_t>("nz", 512, "Cartesian cells in z");
        arguments.addOption<double>("final-time", targetTimeTwo, "simulation final time");
        arguments.addOption<double>("cfl", 0.25, "hydrodynamic CFL factor");
        arguments.addOption<std::size_t>("initial-particles", 80, "initial packets per cell");
        arguments.addOption<std::size_t>("boundary-particles", 8,
                                         "injected packets per vertical face per step");
        arguments.addOption<std::size_t>("population", 400,
                                         "population-control target per cell");
        arguments.addOption<std::size_t>("history-every", 5, "cycles between history rows");
        arguments.addOption<std::size_t>("log-every", 500, "cycles between progress messages");
        arguments.addOption<std::size_t>("output-cycles", 50,
                                         "cycles between latest HDF5 writes; zero disables");
        arguments.addOption<std::size_t>("checkpoint-cycles", 500,
                                         "cycles between archived HDF5 writes; zero disables");
        arguments.addOption<std::size_t>("vtk-cycles", 200,
                                         "cycles between VTK writes; zero disables");
        arguments.addOption<std::uint64_t>("seed", 20260831ULL,
                                           "deterministic perturbation and packet RNG seed");
        arguments.addOption<double>("gamma", 5.0 / 3.0,
                                    "ideal-gas gamma for the compressible analogue");
        arguments.addOption<std::string>("manager", "p2p", "Monte Carlo transport manager")
            .choices({"p2p", "rdma"});
        arguments.addOption<std::string>("output", "jds_g3_output",
                                         "output directory; VTK and HDF5 dumps require this flag");
        arguments.addOption<std::string>("restart", "",
                                         "HDF5 checkpoint from WriteSimulation");

        if(!arguments.parse(argc, argv))
        {
            if(rank == 0)
            {
                std::cout << arguments.help() << std::endl;
            }
            MPI_Finalize();
            return 0;
        }

        const Config config = ParseConfig(arguments);
        const bool restarting = !config.restartFile.empty();
        const Vector3D lowerLeft(-0.5, -0.5, -1.0);
        const Vector3D upperRight(0.5, 0.5, 1.0);
        Voronoi3D tess(lowerLeft, upperRight);
        tess.SetPeriodic(true, true, false);
        if(!restarting)
        {
            std::vector<Vector3D> points;
            if(rank == 0)
            {
                points = CartesianMesh(config.nx, config.ny, config.nz, lowerLeft, upperRight);
            }
            points = MPI_Spread(points, 0, MPI_COMM_WORLD);
            tess.BuildParallel(points);
        }

        IdealGas eos(config.gamma, 1.0 / (config.gamma - 1.0), 1.0, 0.0);
        ComputationalCell3D::tracerNames = {"BaseDensity", "InitialDensity"};
        std::vector<ComputationalCell3D> initialCells;
        if(!restarting)
        {
            initialCells.resize(tess.GetPointNo());
            for(std::size_t cellIndex = 0; cellIndex < tess.GetPointNo(); ++cellIndex)
            {
                const Vector3D center = tess.GetCellCM(cellIndex);
                const double z = center.z;
                const double baseDensity = BaseDensity(z);
                const std::size_t globalCellID = CartesianCellID(center, config);
                double density = baseDensity;
                if(std::abs(z) < 0.5)
                {
                    const double deltaDensity =
                        CellPerturbation(config.seed, globalCellID, baseDensity);
                    density += deltaDensity * (1.0 + std::cos(pi * z));
                }
                // The continuous formula reaches zero only at the endpoint random draw.
                density = std::max(density, std::numeric_limits<double>::min());
                ComputationalCell3D &cell = initialCells[cellIndex];
                cell.ID = globalCellID;
                cell.density = density;
                cell.velocity = Vector3D(0.0, 0.0, 0.0);
                cell.pressure = gasPressure;
                cell.internal_energy = cell.pressure / (cell.density * (config.gamma - 1.0));
                cell.tracers[0] = baseDensity;
                cell.tracers[1] = density;
                cell.temperature = eos.de2T(cell.density, cell.internal_energy, cell.tracers,
                                            ComputationalCell3D::tracerNames);
                cell.pressure = eos.de2p(cell.density, cell.internal_energy, cell.tracers,
                                         ComputationalCell3D::tracerNames);
                cell.Erad = RadiationEnergyDensity(z) / cell.density;
                cell.Erad_dt = 0.0;
                cell.Erad_dt_dt = 0.0;
                cell.Eg.resize(ENERGY_GROUPS_NUM);
                for(std::size_t group = 0; group < ENERGY_GROUPS_NUM; ++group)
                {
                    cell.Eg[group] = cell.Erad / static_cast<double>(ENERGY_GROUPS_NUM);
                }
            }
        }

        Simulation simulation(tess, initialCells, eos);
        ConstantAcceleration3D acceleration(Vector3D(0.0, 0.0, -gravity));
        ConservativeForce3D gravityForce(acceleration, false);
        std::shared_ptr<TimeStepFunction3D> timeStepFunction =
            std::make_shared<CourantFriedrichsLewy>(config.cfl, 1.0, gravityForce);
        simulation.SetTimeStepFunction(timeStepFunction);
        std::vector<ComputationalCell3D> &cells = simulation.getCells();
        std::vector<Conserved3D> &extensives = simulation.getExtensives();
        if(!restarting)
        {
            extensives.resize(tess.GetPointNo());
            for(std::size_t cellIndex = 0; cellIndex < tess.GetPointNo(); ++cellIndex)
            {
                extensives[cellIndex].Eg.resize(ENERGY_GROUPS_NUM);
                PrimitiveToConserved(cells[cellIndex], tess.GetVolume(cellIndex),
                                     extensives[cellIndex]);
            }
        }

        Hllc3D riemannSolver;
        JDSHydroWall ghostGenerator(config.gamma);
        LinearGauss3D interpolator(eos, ghostGenerator, true, 0.2, 0.5, 0.7, false,
                                   {"BaseDensity", "InitialDensity"});
        std::vector<std::pair<const ConditionActionFlux1::Condition3D *,
                              const ConditionActionFlux1::Action3D *>> fluxSequence;
        IsBulkFace3D bulkCondition;
        IsBoundaryFace3D boundaryCondition;
        RegularFlux3D regularFlux(riemannSolver);
        RigidWallFlux3D rigidFlux(riemannSolver);
        fluxSequence.push_back(std::make_pair(&boundaryCondition, &rigidFlux));
        fluxSequence.push_back(std::make_pair(&bulkCondition, &regularFlux));
        ConditionActionFlux1 fluxCalculator(fluxSequence, interpolator);
        DefaultCellUpdater cellUpdater;
        std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D *,
                              const ConditionExtensiveUpdater3D::Action3D *>> extensiveSequence;
        ConditionExtensiveUpdater3D extensiveUpdater(extensiveSequence);
        Eulerian3D pointMotion;
        HDSim3D hydroSimulation(tess, cells, extensives, eos, simulation.getTracker(),
                                pointMotion, *timeStepFunction, fluxCalculator, cellUpdater,
                                extensiveUpdater, gravityForce,
                                std::make_pair(ComputationalCell3D::tracerNames,
                                               ComputationalCell3D::stickerNames));
        std::shared_ptr<HydroStep> hydroStep =
            std::make_shared<HydroStep>(hydroSimulation, HydroStep::TIMEADVANCE_2);
        simulation.addPhysics(hydroStep);

        std::shared_ptr<IdealGas> eosPointer = std::make_shared<IdealGas>(eos);
        std::shared_ptr<JDSOpacity> opacity = std::make_shared<JDSOpacity>();
        std::shared_ptr<JDSRadiationBoundary> jdsBoundary =
            std::make_shared<JDSRadiationBoundary>(
                tess, config.boundaryParticlesPerFace, config.seed + 911ULL);
        std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> radiationBoundary =
            jdsBoundary;
        STORM::RadiationIMCParameters<ENERGY_GROUPS_NUM> radiationParameters;
        radiationParameters.newPhotonsPerCell = config.boundaryParticlesPerFace;
        radiationParameters.lightSpeed = lightSpeed;
        radiationParameters.withHydro = true;
        radiationParameters.withRandomWalk = false;
        radiationParameters.withDDMC = false;
        radiationParameters.withMultigroupOpacity = false;
        radiationParameters.diffusionPressureGradient = false;
        radiationParameters.noHydroFeedback = false;
        radiationParameters.staticScatterers = false;
        radiationParameters.energyBoundaries[0] = 0.0;
        for(std::size_t group = 1; group <= ENERGY_GROUPS_NUM; ++group)
        {
            radiationParameters.energyBoundaries[group] = 1.0e30;
        }
        radiationParameters.energyBoundariesProvided = true;
        std::shared_ptr<RadiationIMC> radiationPhysics = std::make_shared<RadiationIMC>(
            tess, radiationBoundary, cells, extensives, eosPointer, opacity, radiationParameters);
        if(!restarting)
        {
            radiationPhysics->reseedRNG(
                config.seed + 104729ULL * static_cast<std::uint64_t>(rank));
        }
        std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> populationControl =
            std::make_shared<STORM::CombPopulationControl<Vector3D, Tessellation3D>>(
                tess, config.populationPerCell, 1.0);
        std::vector<Particle3D> initialParticles;
        if(!restarting)
        {
            initialParticles = GenerateInitialParticles(
                tess, cells, config.initialParticlesPerCell, config.seed + 271ULL, rank);
        }
        std::shared_ptr<RadiationMCStep> radiationStep =
            std::make_shared<RadiationMCStep>(
                tess, cells, extensives, radiationPhysics, populationControl,
                radiationBoundary, initialParticles, config.initialParticlesPerCell, true,
                config.managerName == "rdma"
                    ? RadiationMCStep::ManagerType::RDMA
                    : RadiationMCStep::ManagerType::P2P);
        radiationStep->setCost(
            std::make_shared<IMCCostCalculator>(radiationStep->getManager()));
        simulation.addPhysics(radiationStep);

        if(restarting)
        {
            int exists = 0;
            if(rank == 0)
            {
                exists = std::filesystem::exists(config.restartFile) ? 1 : 0;
            }
            MPI_Bcast(&exists, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if(exists == 0)
            {
                throw std::runtime_error("restart file not found: " + config.restartFile);
            }
            ReadSimulation(config.restartFile, simulation);
        }
        extensives.resize(cells.size());
        for(std::size_t cellIndex = 0; cellIndex < tess.GetPointNo(); ++cellIndex)
        {
            extensives[cellIndex].Eg.resize(ENERGY_GROUPS_NUM);
            PrimitiveToConserved(cells[cellIndex], tess.GetVolume(cellIndex),
                                 extensives[cellIndex]);
        }

        const std::filesystem::path outputDirectory(config.outputDirectory);
        const std::filesystem::path snapshotDirectory = outputDirectory / "snapshots";
        const std::filesystem::path historyPath = outputDirectory / "jds_g3_history.csv";
        const std::string latestPath = (outputDirectory / "latest.h5").string();
        const std::string initH5Path = (outputDirectory / "init.h5").string();
        const std::string initVtkPath = (outputDirectory / "init.pvtu").string();
        const std::string finalH5Path = (outputDirectory / "final.h5").string();
        const std::string finalVtkPath = (outputDirectory / "final.pvtu").string();
        if(rank == 0)
        {
            std::filesystem::create_directories(outputDirectory);
            if(config.writeDumps)
            {
                std::filesystem::create_directories(snapshotDirectory);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        const auto writeBookend = [&](const std::string &h5Path, const std::string &vtkPath)
        {
            if(!config.writeDumps)
            {
                return;
            }
            WriteSimulation(simulation, h5Path);
            WriteJDSVtk(tess, cells, radiationPhysics, vtkPath);
            MPI_Barrier(MPI_COMM_WORLD);
            if(rank == 0)
            {
                std::cout << "Wrote " << h5Path << " and " << vtkPath << std::endl;
            }
        };
        const auto writeCycleOutputs = [&](std::size_t cycle)
        {
            if(!config.writeDumps)
            {
                return;
            }
            if(DueEvery(cycle, config.outputEvery))
            {
                WriteSimulation(simulation, latestPath);
            }
            if(DueEvery(cycle, config.checkpointEvery))
            {
                WriteSimulation(simulation,
                    (snapshotDirectory / (CycleStem(cycle) + ".h5")).string());
            }
            if(DueEvery(cycle, config.vtkEvery))
            {
                WriteJDSVtk(tess, cells, radiationPhysics,
                    (snapshotDirectory / (CycleStem(cycle) + ".pvtu")).string());
            }
        };
        const auto writeTargetOutputs = [&](double time)
        {
            if(!config.writeDumps)
            {
                return;
            }
            const std::string stem = TargetStem(time);
            WriteSimulation(simulation, (snapshotDirectory / (stem + ".h5")).string());
            WriteJDSVtk(tess, cells, radiationPhysics,
                        (snapshotDirectory / (stem + ".pvtu")).string());
        };

        if(rank == 0 && (!restarting || !std::filesystem::exists(historyPath)))
        {
            std::ofstream history(historyPath);
            history << std::setprecision(17)
                    << "# benchmark=Jiang_Davis_Stone_G3_IMC_analogue\n"
                    << "# domain=[-0.5,0.5]x[-0.5,0.5]x[-1,1]\n"
                    << "# periodic=x,y\n"
                    << "# vertical_hydro_boundary=rigid_free_slip\n"
                    << "# nx=" << config.nx << "\n"
                    << "# ny=" << config.ny << "\n"
                    << "# nz=" << config.nz << "\n"
                    << "# mpi_ranks=" << processCount << "\n"
                    << "# rho_minus=" << rhoMinus << "\n"
                    << "# rho_plus=" << rhoPlus << "\n"
                    << "# gravity=" << gravity << "\n"
                    << "# alpha=1\n"
                    << "# gas_pressure=" << gasPressure << "\n"
                    << "# gamma_gas=" << config.gamma << "\n"
                    << "# scattering_mass_opacity=" << massScatteringOpacity << "\n"
                    << "# planck_opacity=0\n"
                    << "# light_speed=" << lightSpeed << "\n"
                    << "# Frz_dimensionless=" << radiationFluxFraction << "\n"
                    << "# radiation_energy_flux=" << radiationFlux << "\n"
                    << "# static_scatterers=0\n"
                    << "# with_hydro=1\n"
                    << "# ddmc=0\n"
                    << "# random_walk=0\n"
                    << "# multigroup=0\n"
                    << "# perturbation=Eq15 deterministic cell-uniform\n"
                    << "# density_positive_guard=DBL_MIN\n"
                    << "# angular_distribution=positive maximum entropy exp(beta*mu)\n"
                    << "# angular_constraint=<mu>=Frz_dimensionless/Er; exact E and Frz moments\n"
                    << "# restart_source=" << (restarting ? config.restartFile : "none") << "\n"
                    << "# mixing_fraction=volume fraction of cells with 1.1*rho_minus<=rho<=0.9*rho_plus\n"
                    << "# zmax_zmin=max/min z of cells with |rho-rho_initial|>=0.1*rho_initial; rho_initial is the Eq15 perturbed state, not the two-layer base\n"
                    << "# Frz=census sum(weight*mu)/volume; mu=vz/c\n"
                    << "time,zmax,zmin,mixing_fraction,total_radiation_energy,Frz,particle_count\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if(!restarting)
        {
            const Diagnostics diagnostics = CalculateDiagnostics(
                tess, cells, radiationStep->getParticles());
            const std::size_t particles = GlobalParticleCount(radiationStep->getParticles());
            if(rank == 0)
            {
                std::ofstream history(historyPath, std::ios::app);
                history << std::scientific << std::setprecision(17)
                        << 0.0 << ',' << diagnostics.zmax << ',' << diagnostics.zmin << ','
                        << diagnostics.mixingFraction << ',' << diagnostics.radiationEnergy << ','
                        << diagnostics.dimensionlessFrz << ',' << particles << '\n';
            }
            writeBookend(initH5Path, initVtkPath);
            writeCycleOutputs(0);
        }

        double currentTime = simulation.GetTime();
        std::size_t cycle = simulation.GetCycle();
        const double cellSize = std::min(
            1.0 / static_cast<double>(config.nx),
            2.0 / static_cast<double>(config.nz));
        const double soundSpeed = std::sqrt(config.gamma * gasPressure / rhoMinus);
        const double cflDt = config.cfl * cellSize / soundSpeed;
        if(currentTime < config.finalTime)
        {
            double firstStop = config.finalTime;
            if(currentTime < targetTimeOne && targetTimeOne <= config.finalTime)
            {
                firstStop = std::min(firstStop, targetTimeOne);
            }
            if(currentTime < targetTimeTwo && targetTimeTwo <= config.finalTime)
            {
                firstStop = std::min(firstStop, targetTimeTwo);
            }
            simulation.SetTimeStep(std::min(cflDt, firstStop - currentTime));
        }
        while(currentTime < config.finalTime)
        {
            double nextStop = config.finalTime;
            if(currentTime < targetTimeOne && targetTimeOne <= config.finalTime)
            {
                nextStop = std::min(nextStop, targetTimeOne);
            }
            if(currentTime < targetTimeTwo && targetTimeTwo <= config.finalTime)
            {
                nextStop = std::min(nextStop, targetTimeTwo);
            }
            const double stepDt = std::min(
                std::min(cflDt, simulation.GetTimeStep()), nextStop - currentTime);
            if(!(stepDt > 0.0) || !std::isfinite(stepDt))
            {
                throw std::runtime_error("JDS G3 received an invalid timestep");
            }
            jdsBoundary->SetCycle(cycle);
            simulation.SetTimeStep(stepDt);
            simulation.step();
            currentTime = simulation.GetTime();
            cycle = simulation.GetCycle();
            const bool atTargetOne = std::abs(currentTime - targetTimeOne) <=
                16.0 * std::numeric_limits<double>::epsilon() * targetTimeOne;
            const bool atTargetTwo = std::abs(currentTime - targetTimeTwo) <=
                16.0 * std::numeric_limits<double>::epsilon() * targetTimeTwo;
            const bool writeHistory = DueEvery(cycle, config.historyEvery) ||
                atTargetOne || atTargetTwo;
            const bool writeLog = DueEvery(cycle, config.logEvery) ||
                atTargetOne || atTargetTwo;

            if(writeHistory || writeLog)
            {
                const Diagnostics diagnostics = CalculateDiagnostics(
                    tess, cells, radiationStep->getParticles());
                const std::size_t particles = GlobalParticleCount(radiationStep->getParticles());
                if(rank == 0)
                {
                    if(writeHistory)
                    {
                        std::ofstream history(historyPath, std::ios::app);
                        history << std::scientific << std::setprecision(17)
                                << currentTime << ',' << diagnostics.zmax << ','
                                << diagnostics.zmin << ',' << diagnostics.mixingFraction << ','
                                << diagnostics.radiationEnergy << ','
                                << diagnostics.dimensionlessFrz << ',' << particles << '\n';
                    }
                    if(writeLog)
                    {
                        std::cout << "cycle " << cycle << " time=" << currentTime
                                  << " zmax=" << diagnostics.zmax
                                  << " zmin=" << diagnostics.zmin
                                  << " mixing=" << diagnostics.mixingFraction
                                  << " Frz=" << diagnostics.dimensionlessFrz
                                  << " packets=" << particles << std::endl;
                    }
                }
            }
            writeCycleOutputs(cycle);
            if(atTargetOne)
            {
                writeTargetOutputs(targetTimeOne);
            }
            if(atTargetTwo)
            {
                writeTargetOutputs(targetTimeTwo);
            }
        }

        writeBookend(finalH5Path, finalVtkPath);
        if(rank == 0)
        {
            std::cout << "JDS G3 finished. History: " << historyPath.string() << std::endl;
        }
    }
    catch(const UniversalError &error)
    {
        std::cerr << "JDS G3 failed on rank " << rank << std::endl;
        reportError(error, std::cerr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    catch(const std::exception &error)
    {
        std::cerr << "JDS G3 failed on rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
