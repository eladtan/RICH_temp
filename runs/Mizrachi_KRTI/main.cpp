/*
 * Frozen KRTI-S-X CGS radiative Rayleigh-Taylor benchmark.
 *
 * Coupled hydro (HydroStep) + gray coherent isotropic scattering transport
 * (RadiationMCStep) on a periodic-x/y Voronoi slab with rigid hydro walls,
 * lower-boundary illumination, and escaping radiation at both vertical faces.
 *
 * The sharp material interface and the velocity, pressure, and anisotropic
 * radiation perturbations are initialized from the companion semi-analytic
 * eigenfunction package. RICH's ideal-gas hydro is the documented low-Mach
 * approximation to the frozen incompressible reference equations.
 */

#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <complex>
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
#include "source/monte/deps/CMMC/src/units/units.hpp"
#include "source/monte/population/CombPopulationControl.hpp"
#include "source/monte/utils/RandomInCell.hpp"
#include "source/monte/utils/RandomOnFace.hpp"
#include "source/mpi/mpi_commands.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/ConservativeForce3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/newtonian/three_dimensional/eulerian_3d.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"
#include "source/newtonian/three_dimensional/conserved_3d.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/Ghost3D.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/HydroStep.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"
#include "source/utils/arguments/ArgumentParser.hpp"
#include "source/utils/debug/vtune.h"

namespace
{

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double gammaGas = 5.0 / 3.0;
constexpr double atwoodNumber = 0.5;
constexpr double densityRatio = 3.0;
constexpr double radiationSupport = 0.5;
constexpr double defaultPerturbationKAmplitude = 1.0e-3;
constexpr double physicalHalfHeight = 1.0;
constexpr double physicalWaveNumber = 4.0;
constexpr double physicalGravity = 1.0e8;
constexpr double physicalLightSpeed = 2.99792458e10;
constexpr double physicalScatteringOpacity = 2.0;
constexpr double frozenBottomIntensity = 1.6856350e18;
constexpr double targetNetFlux = 7.4948114500e17;
constexpr double frozenInterfacePressure = 1.0e12;

struct PhysicalUnits
{
    double waveNumber;
    double gravity;
    double lightSpeed;
    double lengthScale;
    double timeScale;
    double densityScale;
    double pressureScale;
    double fluxScale;
};

struct Config
{
    std::size_t cellsPerWavelength = 32;
    std::size_t nx = 0;
    std::size_t ny = 0;
    std::size_t nz = 0;
    double interfaceThickness = 0.0;
    double perturbationKAmplitude = defaultPerturbationKAmplitude;
    bool twoModes = false;
    double referencePressure = frozenInterfacePressure;
    double finalTime = 0.0;
    double cfl = 0.25;
    std::size_t initialParticlesPerCell = 80;
    std::size_t newPhotonsPerCell = 8;
    std::size_t populationPerCell = 400;
    std::size_t historyEvery = 5;
    std::size_t logEvery = 500;
    std::size_t hdf5Every = 0;
    std::size_t hdf5ReplicaEvery = 0;
    std::size_t vtkEvery = 0;
    std::uint64_t seed = 20260828;
    bool periodicHorizontal = true;
    std::string outputDirectory = "krti_output";
    bool writeDumps = false;
    std::string managerName = "p2p";
    std::string restartFile;
    std::string referenceDirectory = "KRTI_S_X_reference_package/reference";
};

PhysicalUnits MakeUnits()
{
    PhysicalUnits units;
    units.waveNumber = physicalWaveNumber;
    units.gravity = physicalGravity;
    units.lightSpeed = physicalLightSpeed;
    units.lengthScale = 1.0 / units.waveNumber;
    units.timeScale = 1.0 / std::sqrt(atwoodNumber * units.gravity * units.waveNumber);
    units.densityScale = 1.0;
    units.pressureScale = 1.0;
    units.fluxScale = targetNetFlux;
    return units;
}

Config ParseConfig(ArgumentParser &arguments)
{
    Config config;
    (void)arguments.get<std::string>("case");
    config.cellsPerWavelength = arguments.get<std::size_t>("cells-per-lambda");
    config.nx = arguments.get<std::size_t>("nx");
    config.ny = arguments.get<std::size_t>("ny");
    config.nz = arguments.get<std::size_t>("nz");
    config.interfaceThickness = arguments.get<double>("interface-delta");
    config.perturbationKAmplitude = arguments.get<double>("k-eta0");
    if(arguments.wasSet("single-mode") && arguments.wasSet("two-modes"))
        throw std::runtime_error("--single-mode and --two-modes are mutually exclusive");
    config.twoModes = arguments.wasSet("two-modes");
    config.referencePressure = arguments.get<double>("reference-pressure");
    config.finalTime = arguments.get<double>("final-time");
    config.cfl = arguments.get<double>("cfl");
    config.initialParticlesPerCell = arguments.get<std::size_t>("initial-particles");
    config.newPhotonsPerCell = arguments.get<std::size_t>("new-photons");
    config.populationPerCell = arguments.get<std::size_t>("population");
    config.historyEvery = arguments.get<std::size_t>("history-every");
    config.logEvery = arguments.get<std::size_t>("log-every");
    config.seed = arguments.get<std::uint64_t>("seed");
    config.periodicHorizontal = !arguments.wasSet("no-periodic-horizontal");
    config.managerName = arguments.get<std::string>("manager");
    config.referenceDirectory = arguments.get<std::string>("reference-directory");
    if(arguments.wasSet("restart"))
        config.restartFile = arguments.get<std::string>("restart");
    if(arguments.wasSet("output"))
    {
        config.outputDirectory = arguments.get<std::string>("output");
        config.writeDumps = !config.outputDirectory.empty();
        config.hdf5Every = arguments.get<std::size_t>("output-cycles");
        config.hdf5ReplicaEvery = arguments.get<std::size_t>("archive-cycles");
        config.vtkEvery = arguments.get<std::size_t>("vtk-cycles");
    }
    if(config.outputDirectory.empty() && !config.restartFile.empty())
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
    if(!config.writeDumps)
    {
        config.hdf5Every = 0;
        config.hdf5ReplicaEvery = 0;
        config.vtkEvery = 0;
    }
    if(config.twoModes || config.interfaceThickness != 0.0 ||
       config.referencePressure != frozenInterfacePressure ||
       !config.periodicHorizontal || config.cellsPerWavelength < 8 ||
       config.perturbationKAmplitude <= 0.0 ||
       config.referencePressure <= 0.0 || config.cfl <= 0.0 ||
       config.outputDirectory.empty() || config.referenceDirectory.empty() ||
       config.initialParticlesPerCell == 0 || config.newPhotonsPerCell == 0 ||
       config.populationPerCell == 0 || config.historyEvery == 0)
    {
        throw std::runtime_error(
            "KRTI-S-X is frozen to case X, a sharp single-mode interface, p*=1e12, "
            "and periodic horizontal boundaries");
    }
    return config;
}

double interfacePerturbation(double x, double y, bool twoModes, double kAmplitude)
{
    const double mode1 = std::cos(physicalWaveNumber * x);
    if(!twoModes)
    {
        return kAmplitude * mode1 / physicalWaveNumber;
    }
    return kAmplitude *
        (mode1 + std::cos(physicalWaveNumber * y)) / physicalWaveNumber;
}

void GaussLegendreQuadrature(std::size_t count, std::vector<double> &directions,
                             std::vector<double> &weights)
{
    if(count < 2 || count % 2 != 0)
    {
        throw std::runtime_error("KRTI angular quadrature requires a positive even order");
    }
    directions.assign(count, 0.0);
    weights.assign(count, 0.0);
    const std::size_t half = count / 2;
    for(std::size_t rootIndex = 0; rootIndex < half; ++rootIndex)
    {
        double root = std::cos(pi * (static_cast<double>(rootIndex) + 0.75) /
                               (static_cast<double>(count) + 0.5));
        double derivative = 0.0;
        for(std::size_t iteration = 0; iteration < 100; ++iteration)
        {
            double previous = 1.0;
            double polynomial = root;
            for(std::size_t degree = 2; degree <= count; ++degree)
            {
                const double next =
                    ((2.0 * static_cast<double>(degree) - 1.0) * root * polynomial -
                     (static_cast<double>(degree) - 1.0) * previous) /
                    static_cast<double>(degree);
                previous = polynomial;
                polynomial = next;
            }
            derivative = static_cast<double>(count) * (root * polynomial - previous) /
                         (root * root - 1.0);
            const double correction = polynomial / derivative;
            root -= correction;
            if(std::abs(correction) < 1.0e-15)
            {
                break;
            }
        }
        const double weight = 2.0 / ((1.0 - root * root) * derivative * derivative);
        directions[rootIndex] = -root;
        directions[count - rootIndex - 1] = root;
        weights[rootIndex] = weight;
        weights[count - rootIndex - 1] = weight;
    }
}

double hydrostaticPressure(double z, double rhoLight, double rhoHeavy,
                           double effectiveGravity, double interfaceDelta,
                           double referencePressure)
{
    (void)interfaceDelta;
    const double density = z < 0.0 ? rhoLight : rhoHeavy;
    return referencePressure - density * effectiveGravity * z;
}

std::vector<std::string> SplitCSV(const std::string &line)
{
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while(std::getline(stream, field, ','))
    {
        fields.push_back(field);
    }
    return fields;
}

double InterpolateReal(const std::vector<double> &zNodes, const std::vector<double> &values,
                       double z)
{
    if(zNodes.empty() || zNodes.size() != values.size())
    {
        throw std::runtime_error("invalid KRTI reference interpolation table");
    }
    if(z <= zNodes.front())
    {
        return values.front();
    }
    if(z >= zNodes.back())
    {
        return values.back();
    }
    const std::vector<double>::const_iterator upper = std::upper_bound(zNodes.begin(), zNodes.end(), z);
    const std::size_t rightIndex = static_cast<std::size_t>(upper - zNodes.begin());
    const std::size_t leftIndex = rightIndex - 1;
    const double fraction = (z - zNodes[leftIndex]) / (zNodes[rightIndex] - zNodes[leftIndex]);
    return values[leftIndex] * (1.0 - fraction) + values[rightIndex] * fraction;
}

std::complex<double> InterpolateComplex(
    const std::vector<double> &zNodes, const std::vector<std::complex<double>> &values,
    double z)
{
    if(zNodes.empty() || zNodes.size() != values.size())
    {
        throw std::runtime_error("invalid KRTI complex reference interpolation table");
    }
    if(z <= zNodes.front())
    {
        return values.front();
    }
    if(z >= zNodes.back())
    {
        return values.back();
    }
    const std::vector<double>::const_iterator upper = std::upper_bound(zNodes.begin(), zNodes.end(), z);
    const std::size_t rightIndex = static_cast<std::size_t>(upper - zNodes.begin());
    const std::size_t leftIndex = rightIndex - 1;
    const double fraction = (z - zNodes[leftIndex]) / (zNodes[rightIndex] - zNodes[leftIndex]);
    return values[leftIndex] * (1.0 - fraction) + values[rightIndex] * fraction;
}

struct BackgroundDirection
{
    double mu = 0.0;
    std::vector<double> z;
    std::vector<double> intensity;
};

struct PerturbedDirection
{
    double mu = 0.0;
    double phi = 0.0;
    double xi = 0.0;
    double angularAverageWeight = 0.0;
    std::vector<double> z;
    std::vector<std::complex<double>> intensity;
};

class KRTIReferenceState
{
public:
    KRTIReferenceState(const std::filesystem::path &directory, double perturbationKAmplitude):
        perturbationKAmplitude_(perturbationKAmplitude)
    {
        this->LoadHydro(directory / "KRTI-S-X_hydro_eigenfunction.csv");
        this->LoadBackground(directory / "KRTI-S-X_background_intensity.csv");
        this->LoadPerturbedIntensity(directory / "KRTI-S-X_intensity_eigenfunction.csv");
    }

    std::complex<double> HydroU(double z) const
    {
        return InterpolateComplex(this->hydroZ_, this->hydroU_, z);
    }

    std::complex<double> HydroW(double z) const
    {
        return InterpolateComplex(this->hydroZ_, this->hydroW_, z);
    }

    std::complex<double> HydroPressure(double z) const
    {
        return InterpolateComplex(this->hydroZ_, this->hydroPressure_, z);
    }

    std::size_t DirectionCount() const
    {
        return this->directions_.size();
    }

    const PerturbedDirection &Direction(std::size_t index) const
    {
        return this->directions_.at(index);
    }

    double Intensity(std::size_t directionIndex, double x, double z) const
    {
        const PerturbedDirection &direction = this->directions_.at(directionIndex);
        const double background = this->BackgroundIntensity(direction.mu, z);
        const std::complex<double> perturbation = InterpolateComplex(
            direction.z, direction.intensity, z);
        const double phase = physicalWaveNumber * x;
        const double physicalPerturbation = perturbation.real() * std::cos(phase) -
                                            perturbation.imag() * std::sin(phase);
        const double result = background +
            (this->perturbationKAmplitude_ / physicalWaveNumber) * physicalPerturbation;
        if(!(result > 0.0) || !std::isfinite(result))
        {
            throw std::runtime_error("non-positive KRTI reference intensity");
        }
        return result;
    }

    double EnergyDensity(double x, double z) const
    {
        double angularMean = 0.0;
        for(std::size_t directionIndex = 0; directionIndex < this->directions_.size();
            ++directionIndex)
        {
            angularMean += this->directions_[directionIndex].angularAverageWeight *
                           this->Intensity(directionIndex, x, z);
        }
        return 4.0 * pi * angularMean / physicalLightSpeed;
    }

private:
    void LoadHydro(const std::filesystem::path &path)
    {
        std::ifstream input(path);
        if(!input)
        {
            throw std::runtime_error("cannot open KRTI hydro eigenfunction: " + path.string());
        }
        std::string line;
        std::getline(input, line);
        while(std::getline(input, line))
        {
            if(line.empty())
            {
                continue;
            }
            const std::vector<std::string> fields = SplitCSV(line);
            if(fields.size() != 7)
            {
                throw std::runtime_error("malformed KRTI hydro eigenfunction: " + path.string());
            }
            this->hydroZ_.push_back(std::stod(fields[0]));
            this->hydroW_.push_back({std::stod(fields[1]), std::stod(fields[2])});
            this->hydroU_.push_back({std::stod(fields[3]), std::stod(fields[4])});
            this->hydroPressure_.push_back({std::stod(fields[5]), std::stod(fields[6])});
        }
        if(this->hydroZ_.size() < 2)
        {
            throw std::runtime_error("empty KRTI hydro eigenfunction: " + path.string());
        }
    }

    void LoadBackground(const std::filesystem::path &path)
    {
        std::ifstream input(path);
        if(!input)
        {
            throw std::runtime_error("cannot open KRTI background intensity: " + path.string());
        }
        std::string line;
        std::getline(input, line);
        while(std::getline(input, line))
        {
            if(line.empty())
            {
                continue;
            }
            const std::vector<std::string> fields = SplitCSV(line);
            if(fields.size() != 3)
            {
                throw std::runtime_error("malformed KRTI background intensity: " + path.string());
            }
            const double mu = std::stod(fields[0]);
            if(this->background_.empty() || std::abs(mu - this->background_.back().mu) > 1.0e-13)
            {
                this->background_.push_back(BackgroundDirection());
                this->background_.back().mu = mu;
            }
            this->background_.back().z.push_back(std::stod(fields[1]));
            this->background_.back().intensity.push_back(std::stod(fields[2]));
        }
        if(this->background_.size() < 2)
        {
            throw std::runtime_error("empty KRTI background intensity: " + path.string());
        }
    }

    void LoadPerturbedIntensity(const std::filesystem::path &path)
    {
        std::ifstream input(path);
        if(!input)
        {
            throw std::runtime_error(
                "cannot open the direction-resolved KRTI eigenfunction " + path.string() +
                "; generate it with krti_s_reference.py --case X --nz 128 --nmu 24 --nphi 16");
        }
        std::string line;
        std::getline(input, line);
        std::size_t previousAngle = std::numeric_limits<std::size_t>::max();
        while(std::getline(input, line))
        {
            if(line.empty())
            {
                continue;
            }
            const std::vector<std::string> fields = SplitCSV(line);
            if(fields.size() != 7)
            {
                throw std::runtime_error("malformed KRTI intensity eigenfunction: " + path.string());
            }
            const std::size_t angleIndex = static_cast<std::size_t>(std::stoull(fields[0]));
            if(angleIndex != previousAngle)
            {
                if(angleIndex != this->directions_.size())
                {
                    throw std::runtime_error("non-sequential KRTI intensity angle index");
                }
                this->directions_.push_back(PerturbedDirection());
                this->directions_.back().mu = std::stod(fields[1]);
                this->directions_.back().phi = std::stod(fields[2]);
                this->directions_.back().xi = std::stod(fields[3]);
                previousAngle = angleIndex;
            }
            this->directions_.back().z.push_back(std::stod(fields[4]));
            this->directions_.back().intensity.push_back(
                {std::stod(fields[5]), std::stod(fields[6])});
        }
        if(this->directions_.empty())
        {
            throw std::runtime_error("empty KRTI intensity eigenfunction: " + path.string());
        }

        std::size_t phiCount = 0;
        while(phiCount < this->directions_.size() &&
              std::abs(this->directions_[phiCount].mu - this->directions_.front().mu) < 1.0e-13)
        {
            ++phiCount;
        }
        if(phiCount == 0 || this->directions_.size() % phiCount != 0)
        {
            throw std::runtime_error("invalid KRTI angular quadrature layout");
        }
        const std::size_t muCount = this->directions_.size() / phiCount;
        std::vector<double> mu;
        std::vector<double> muWeights;
        GaussLegendreQuadrature(muCount, mu, muWeights);
        for(std::size_t muIndex = 0; muIndex < muCount; ++muIndex)
        {
            for(std::size_t phiIndex = 0; phiIndex < phiCount; ++phiIndex)
            {
                const std::size_t directionIndex = muIndex * phiCount + phiIndex;
                if(std::abs(this->directions_[directionIndex].mu - mu[muIndex]) > 1.0e-12)
                {
                    throw std::runtime_error("KRTI angular quadrature does not match Gauss-Legendre");
                }
                this->directions_[directionIndex].angularAverageWeight =
                    muWeights[muIndex] / (2.0 * static_cast<double>(phiCount));
            }
        }
    }

    double BackgroundIntensity(double mu, double z) const
    {
        if(mu <= this->background_.front().mu)
        {
            return InterpolateReal(
                this->background_.front().z, this->background_.front().intensity, z);
        }
        if(mu >= this->background_.back().mu)
        {
            return InterpolateReal(
                this->background_.back().z, this->background_.back().intensity, z);
        }
        std::size_t rightIndex = 1;
        while(this->background_[rightIndex].mu < mu)
        {
            ++rightIndex;
        }
        const std::size_t leftIndex = rightIndex - 1;
        const double leftIntensity = InterpolateReal(
            this->background_[leftIndex].z, this->background_[leftIndex].intensity, z);
        const double rightIntensity = InterpolateReal(
            this->background_[rightIndex].z, this->background_[rightIndex].intensity, z);
        const double fraction = (mu - this->background_[leftIndex].mu) /
            (this->background_[rightIndex].mu - this->background_[leftIndex].mu);
        return leftIntensity * (1.0 - fraction) + rightIntensity * fraction;
    }

    std::vector<double> hydroZ_;
    std::vector<std::complex<double>> hydroU_;
    std::vector<std::complex<double>> hydroW_;
    std::vector<std::complex<double>> hydroPressure_;
    std::vector<BackgroundDirection> background_;
    std::vector<PerturbedDirection> directions_;
    double perturbationKAmplitude_;
};

double RealModeValue(const std::complex<double> &amplitude, double x)
{
    const double phase = physicalWaveNumber * x;
    return amplitude.real() * std::cos(phase) - amplitude.imag() * std::sin(phase);
}

class KRTIHydrostaticWall final : public Ghost3D
{
public:
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
            const double deltaZ =
                tess.GetMeshPoint(ghostIndex).z - tess.GetMeshPoint(realIndex).z;
            ghost.pressure -= ghost.density * physicalGravity *
                              (1.0 - radiationSupport) * deltaZ;
            ghost.internal_energy =
                ghost.pressure / (ghost.density * (gammaGas - 1.0));
            result.insert(std::make_pair(ghostIndex, ghost));
        }
    }

    Slope3D GetGhostGradient(const Tessellation3D &, const std::vector<ComputationalCell3D> &,
                             const std::vector<Slope3D> &, std::size_t, double,
                             std::size_t) const override
    {
        return Slope3D();
    }
};

class KRTIOpacity final : public OpacityCalculator
{
public:
    explicit KRTIOpacity(double massScatteringOpacity):
        massScatteringOpacity_(massScatteringOpacity)
    {}

    double CalcPlanckOpacity(const ComputationalCell3D &) const override
    {
        return 0.0;
    }

    double CalcScatteringOpacity(const ComputationalCell3D &cell) const override
    {
        return std::max(0.0, cell.density) * massScatteringOpacity_;
    }

private:
    double massScatteringOpacity_;
};

class KRTIRadiationBoundary final :
    public STORM::BoundaryCondition<Vector3D, Tessellation3D>
{
public:
    KRTIRadiationBoundary(const Tessellation3D &tess, double incidentFlux,
                          double lightSpeed, std::size_t packetsPerFace,
                          std::uint64_t seed)
        : STORM::BoundaryCondition<Vector3D, Tessellation3D>(tess),
          incidentFlux_(incidentFlux),
          lightSpeed_(lightSpeed),
          packetsPerFace_(packetsPerFace),
          seed_(seed)
    {}

    STORM::ParticleStatus apply(Particle3D &particle) override
    {
        const std::vector<typename Tessellation3D::Face_T> &faces = this->grid.GetBoxFaces();
        for(const typename Tessellation3D::Face_T &face : faces)
        {
            Vector3D normal;
            double faceScale = 0.0;
            if(!this->getInwardBoxFaceNormalIfClose(face, particle.location, normal, faceScale))
            {
                continue;
            }

            if(std::abs(normal.z) < 0.99)
            {
                continue;
            }
            return STORM::ParticleStatus::REMOVE;
        }
        return STORM::ParticleStatus::REMOVE;
    }

    std::vector<Particle3D> generateNewBoundaryParticles(double fullDt) override
    {
        std::vector<Particle3D> particles;
        if(fullDt <= 0.0 || packetsPerFace_ == 0)
            return particles;

        std::mt19937_64 generator(this->seed_++);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        const std::pair<Vector3D, Vector3D> box = this->grid.GetBoxCoordinates();
        const double slabHeight = box.second.z - box.first.z;
        const double bottomTolerance =
            1.0e-8 * std::max(1.0, slabHeight);
        const std::size_t localCells = this->grid.GetPointNo();
        for(std::size_t cellIndex = 0; cellIndex < localCells; ++cellIndex)
        {
            const Vector3D &cellPoint = this->grid.GetMeshPoint(cellIndex);
            for(const std::size_t faceIndex : this->grid.GetCellFaces(cellIndex))
            {
                if(!this->grid.BoundaryFace(faceIndex))
                {
                    continue;
                }
                const Vector3D faceCenter = this->grid.FaceCM(faceIndex);
                if(std::abs(faceCenter.z - box.first.z) > bottomTolerance)
                {
                    continue;
                }

                const double area = this->grid.GetArea(faceIndex);
                const double faceEnergy = incidentFlux_ * area * fullDt;
                const double packetWeight =
                    faceEnergy / static_cast<double>(packetsPerFace_);
                for(std::size_t packetIndex = 0; packetIndex < packetsPerFace_; ++packetIndex)
                {
                    Particle3D particle;
                    particle.location = STORM::RandomPointOnFace<Vector3D, Tessellation3D>(
                        this->grid, faceIndex);
                    constexpr double nudge = 1.0e-8;
                    particle.location = particle.location * (1.0 - nudge) + nudge * cellPoint;
                    const double mu = std::sqrt(uniform(generator));
                    const double phi = 2.0 * pi * uniform(generator);
                    const double transverse = std::sqrt(std::max(0.0, 1.0 - mu * mu));
                    particle.velocity = Vector3D(transverse * std::cos(phi),
                                               transverse * std::sin(phi),
                                               mu) * lightSpeed_;
                    particle.frequency = 0.0;
                    particle.weight = packetIndex + 1 == packetsPerFace_
                        ? faceEnergy - packetWeight * static_cast<double>(packetsPerFace_ - 1)
                        : packetWeight;
                    particle.initialWeight = particle.weight;
                    particle.timeLeft = fullDt * uniform(generator);
                    particle.cellIndex = cellIndex;
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
    double incidentFlux_;
    double lightSpeed_;
    std::size_t packetsPerFace_;
    mutable std::uint64_t seed_;
};

std::uint64_t MixParticleSeed(std::uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::vector<Particle3D> GenerateInitialRadiationParticles(
    const Tessellation3D &tess, const std::vector<ComputationalCell3D> &cells,
    const KRTIReferenceState &reference, std::size_t particlesPerCell,
    double lightSpeed, std::uint64_t seed, int rank)
{
    std::vector<Particle3D> particles;
    if(particlesPerCell == 0)
    {
        return particles;
    }
    const std::size_t localCells = tess.GetPointNo();
    particles.reserve(localCells * particlesPerCell);
    std::mt19937_64 generator(
        MixParticleSeed(seed + static_cast<std::uint64_t>(rank)));
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    RandomInCellPositionSampler<Vector3D, Tessellation3D> positionSampler;
    for(std::size_t cellIndex = 0; cellIndex < localCells; ++cellIndex)
    {
        const Vector3D cellCenter = tess.GetCellCM(cellIndex);
        std::vector<double> cumulative(reference.DirectionCount(), 0.0);
        double angularTotal = 0.0;
        for(std::size_t angleIndex = 0; angleIndex < reference.DirectionCount(); ++angleIndex)
        {
            angularTotal += reference.Direction(angleIndex).angularAverageWeight *
                            reference.Intensity(angleIndex, cellCenter.x, cellCenter.z);
            cumulative[angleIndex] = angularTotal;
        }
        const double radiationEnergy =
            reference.EnergyDensity(cellCenter.x, cellCenter.z) *
            tess.GetVolume(cellIndex);
        if(!(angularTotal > 0.0) || !(radiationEnergy > 0.0))
        {
            continue;
        }
        const double packetWeight =
            radiationEnergy / static_cast<double>(particlesPerCell);
        for(std::size_t packetIndex = 0; packetIndex < particlesPerCell; ++packetIndex)
        {
            const double angularSample = angularTotal * uniform(generator);
            const std::size_t angleIndex = static_cast<std::size_t>(
                std::lower_bound(cumulative.begin(), cumulative.end(), angularSample) -
                cumulative.begin());
            const std::size_t boundedAngle =
                std::min(angleIndex, reference.DirectionCount() - 1);
            const PerturbedDirection &direction = reference.Direction(boundedAngle);
            const double directionY = std::sqrt(std::max(0.0, 1.0 - direction.mu * direction.mu)) *
                                      std::sin(direction.phi);

            Particle3D particle;
            particle.location =
                positionSampler(tess, cellIndex, generator, uniform);
            constexpr double nudge = 1.0e-10;
            particle.location =
                particle.location * (1.0 - nudge) +
                tess.GetMeshPoint(cellIndex) * nudge;
            particle.velocity =
                Vector3D(direction.xi, directionY, direction.mu) * lightSpeed;
            particle.cellIndex = cellIndex;
            particle.cellID = cells[cellIndex].ID;
            particle.sourceCellID = cells[cellIndex].ID;
            particle.frequency = 0.0;
            particle.weight = packetWeight;
            particle.initialWeight = packetWeight;
            particle.timeLeft = 0.0;
            particle.rngKey = MixParticleSeed(
                seed ^ (static_cast<std::uint64_t>(rank) << 48U) ^
                static_cast<std::uint64_t>(cellIndex * particlesPerCell + packetIndex));
            particle.rngCounter = 0;
            particles.push_back(particle);
        }
    }
    return particles;
}

struct ModeDescriptor
{
    std::string label;
    double directionX;
    double directionY;
};

double projectTracerMode(const Voronoi3D &tess, const std::vector<ComputationalCell3D> &cells,
                         const ModeDescriptor &mode, double waveNumber)
{
    // Weighted least squares of the interface height onto {1, cos(phase)}. The constant term is
    // required: without it any mean offset of the interface leaks straight into the mode amplitude.
    double localSums[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    for(std::size_t faceIndex = 0; faceIndex < tess.GetTotalFacesNumber(); ++faceIndex)
    {
        if(tess.BoundaryFace(faceIndex))
        {
            continue;
        }
        const std::pair<std::size_t, std::size_t> neighbors =
            tess.GetFaceNeighbors(faceIndex);
        if(neighbors.first >= cells.size() || neighbors.second >= cells.size())
        {
            continue;
        }
        const double firstTracer = cells[neighbors.first].tracers[0];
        const double secondTracer = cells[neighbors.second].tracers[0];
        if((firstTracer - 0.5) * (secondTracer - 0.5) >= 0.0 ||
           std::abs(secondTracer - firstTracer) < 1.0e-14)
        {
            continue;
        }
        const Vector3D firstPoint = tess.GetMeshPoint(neighbors.first);
        const Vector3D secondPoint = tess.GetMeshPoint(neighbors.second);
        const double fraction = (0.5 - firstTracer) / (secondTracer - firstTracer);
        const Vector3D interfacePoint = firstPoint * (1.0 - fraction) + secondPoint * fraction;
        const Vector3D normal = normalize(secondPoint - firstPoint);
        const double projectedArea = tess.GetArea(faceIndex) * std::abs(normal.z);
        if(projectedArea <= 0.0)
        {
            continue;
        }
        const double phase = waveNumber *
            (mode.directionX * interfacePoint.x + mode.directionY * interfacePoint.y);
        const double cosine = std::cos(phase);
        localSums[0] += projectedArea;
        localSums[1] += projectedArea * cosine;
        localSums[2] += projectedArea * cosine * cosine;
        localSums[3] += projectedArea * interfacePoint.z;
        localSums[4] += projectedArea * interfacePoint.z * cosine;
    }
    double globalSums[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    MPI_Allreduce(localSums, globalSums, 5, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double weight = globalSums[0];
    const double weightedCosine = globalSums[1];
    const double weightedCosineSquared = globalSums[2];
    const double weightedHeight = globalSums[3];
    const double weightedHeightCosine = globalSums[4];
    const double determinant = weight * weightedCosineSquared - weightedCosine * weightedCosine;
    if(!(determinant > 0.0))
    {
        // No crossings, or every crossing landed on the same phase; fall back to the plain projection.
        if(weightedCosineSquared <= 0.0)
        {
            return 0.0;
        }
        return weightedHeightCosine / weightedCosineSquared;
    }
    return (weight * weightedHeightCosine - weightedCosine * weightedHeight) / determinant;
}

double projectConservedTracerMode(const Voronoi3D &tess,
                                  const std::vector<ComputationalCell3D> &cells,
                                  const ModeDescriptor &mode, double waveNumber,
                                  double heavyDensity)
{
    // The conserved heavy-material mass is rho * HeavyFraction * volume. For a sharp interface
    // z=eta*cos(kx), its Fourier coefficient is -rhoHeavy*eta*horizontalArea/2. Unlike locating
    // the tracer=0.5 contour between cell centres, this remains unbiased when the interface moves
    // through an Eulerian cell and the tracer becomes a cell average.
    double localHeavyMassCosine = 0.0;
    for(std::size_t cellIndex = 0; cellIndex < tess.GetPointNo(); ++cellIndex)
    {
        const Vector3D center = tess.GetCellCM(cellIndex);
        const double phase = waveNumber *
            (mode.directionX * center.x + mode.directionY * center.y);
        localHeavyMassCosine += tess.GetVolume(cellIndex) * cells[cellIndex].density *
                                cells[cellIndex].tracers[0] * std::cos(phase);
    }
    double globalHeavyMassCosine = 0.0;
    MPI_Allreduce(&localHeavyMassCosine, &globalHeavyMassCosine, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    const std::pair<Vector3D, Vector3D> box = tess.GetBoxCoordinates();
    const double horizontalArea = (box.second.x - box.first.x) *
                                  (box.second.y - box.first.y);
    return -2.0 * globalHeavyMassCosine / (heavyDensity * horizontalArea);
}

std::size_t particleCount(const std::vector<Particle3D> &particles)
{
    unsigned long long localCount = static_cast<unsigned long long>(particles.size());
    unsigned long long globalCount = 0;
    MPI_Allreduce(&localCount, &globalCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return static_cast<std::size_t>(globalCount);
}

std::string NumberedSnapshotStem(std::size_t cycle)
{
    std::ostringstream name;
    name << "krti_cycle_" << std::setw(6) << std::setfill('0') << cycle;
    return name.str();
}

std::string snapshotPath(const std::filesystem::path &directory, std::size_t cycle)
{
    return (directory / (NumberedSnapshotStem(cycle) + ".h5")).string();
}

std::string vtkSnapshotPath(const std::filesystem::path &directory, std::size_t cycle)
{
    return (directory / (NumberedSnapshotStem(cycle) + ".pvtu")).string();
}

bool DueEvery(std::size_t cycle, std::size_t interval)
{
    return interval > 0 && cycle % interval == 0;
}

void WriteKRTIVtu(const Voronoi3D &tess, const std::vector<ComputationalCell3D> &cells,
                  const std::shared_ptr<RadiationIMC> &physics, const std::string &filename)
{
    const std::size_t cellCount = tess.GetPointNo();
    std::vector<double> density(cellCount);
    std::vector<double> temperature(cellCount);
    std::vector<double> erad(cellCount);
    std::vector<double> eradTimeAvg(cellCount);
    std::vector<double> pressure(cellCount);
    std::vector<double> velocityX(cellCount);
    std::vector<double> velocityY(cellCount);
    std::vector<double> velocityZ(cellCount);
    std::vector<double> heavyFractionValues(cellCount);
    const std::vector<double> &timeAverage = physics->getEradTimeAvg();
    for(std::size_t cellIndex = 0; cellIndex < cellCount; ++cellIndex)
    {
        density[cellIndex] = cells[cellIndex].density;
        temperature[cellIndex] = cells[cellIndex].temperature;
        erad[cellIndex] = cells[cellIndex].Erad;
        eradTimeAvg[cellIndex] = timeAverage[cellIndex];
        pressure[cellIndex] = cells[cellIndex].pressure;
        velocityX[cellIndex] = cells[cellIndex].velocity.x;
        velocityY[cellIndex] = cells[cellIndex].velocity.y;
        velocityZ[cellIndex] = cells[cellIndex].velocity.z;
        heavyFractionValues[cellIndex] = cells[cellIndex].tracers[0];
    }
    WriteVoronoiVTKOnly(tess, filename,
                        {density, temperature, erad, eradTimeAvg, pressure, velocityX, velocityY,
                         velocityZ, heavyFractionValues},
                        {"density", "temperature", "Erad", "Erad_time_avg", "pressure",
                         "velocity_x", "velocity_y", "velocity_z", "heavy_fraction"});
}

} // namespace

int main(int argc, char **argv)
{
    vtune_stop();
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    try
    {
        ArgumentParser arguments("KRTI-S radiative Rayleigh-Taylor instability benchmark");
        arguments.addOption<std::string>("case", "X",
                                         "frozen benchmark case (only X is defined by the specification)")
            .choices({"X", "x"});
        arguments.addOption<std::size_t>("cells-per-lambda", 32, "cells per perturbation wavelength");
        arguments.addOption<std::size_t>("nx", 0, "override x cells (default 2*pi*k cells)");
        arguments.addOption<std::size_t>("ny", 0, "override y cells (default 1; canonical fields have no y dependence)");
        arguments.addOption<std::size_t>("nz", 0, "override z cells (default isotropic spacing)");
        arguments.addOption<double>("interface-delta", 0.0,
                                    "deprecated; the frozen interface is sharp and this must be zero");
        arguments.addOption<double>("k-eta0", defaultPerturbationKAmplitude,
                                    "dimensionless initial perturbation amplitude k*eta0");
        arguments.addFlag("single-mode", "use only cos(kx) perturbation (default)");
        arguments.addFlag("two-modes", "invalid for the frozen two-dimensional X case");
        arguments.addOption<double>(
            "reference-pressure", frozenInterfacePressure,
            "interface gas pressure in dyn/cm^2 (frozen value is 1e12)");
        arguments.addOption<double>("final-time", 0.0, "final time in RT units (default 12 growth times)");
        arguments.addOption<double>("cfl", 0.25, "hydro CFL factor");
        arguments.addOption<std::size_t>("initial-particles", 80, "packets per cell at t=0");
        arguments.addOption<std::size_t>(
            "new-photons", 8,
            "injected packets per lower-boundary face per step");
        arguments.addOption<std::size_t>("population", 400, "population-control target per cell");
        arguments.addOption<std::size_t>("history-every", 5, "cycles between history rows");
        arguments.addOption<std::size_t>("log-every", 500, "cycles between progress printouts");
        arguments.addOption<std::uint64_t>("seed", 20260828ULL, "RNG seed");
        arguments.addOption<std::string>("manager", "p2p", "Monte Carlo transport manager")
            .choices({"p2p", "rdma"});
        arguments.addFlag("no-periodic-horizontal",
                          "disable x/y periodic Voronoi (debug / local testing)");
        arguments.addOption<std::string>("output", "krti_output",
                                         "output directory; VTK and HDF5 dumps require this flag");
        arguments.addOption<std::size_t>("output-cycles", 50,
                                         "cycles between overwriting latest.h5");
        arguments.addOption<std::size_t>("archive-cycles", 500,
                                         "cycles between numbered HDF5 replica snapshots");
        arguments.addOption<std::size_t>("vtk-cycles", 200,
                                         "cycles between numbered VTK snapshots");
        arguments.addOption<std::string>("restart", "",
                                         "HDF5 checkpoint from WriteSimulation to resume from");
        arguments.addOption<std::string>(
            "reference-directory", "KRTI_S_X_reference_package/reference",
            "directory containing the frozen direction-resolved reference CSV files");

        if(!arguments.parse(argc, argv))
        {
            if(rank == 0)
                std::cout << arguments.help() << std::endl;
            MPI_Finalize();
            return 0;
        }

        const Config config = ParseConfig(arguments);
        const bool restarting = !config.restartFile.empty();
        const PhysicalUnits units = MakeUnits();
        const double rhoLight = units.densityScale;
        const double rhoHeavy = densityRatio * rhoLight;
        const double massScatteringOpacity = physicalScatteringOpacity;
        const double effectiveGravity = units.gravity * (1.0 - radiationSupport);
        const double slabHalfHeight = physicalHalfHeight;
        const double domainLength = 2.0 * pi / units.waveNumber;
        const double interfaceDelta = config.interfaceThickness / units.waveNumber;
        const double referencePressure =
            config.referencePressure * units.pressureScale;
        if(hydrostaticPressure(slabHalfHeight, rhoLight, rhoHeavy,
                               effectiveGravity, interfaceDelta,
                               referencePressure) <= 0.0)
        {
            throw std::runtime_error(
                "reference pressure is too small for a positive hydrostatic state");
        }
        const double finalTime = config.finalTime > 0.0
            ? config.finalTime * units.timeScale
            : 12.0 * units.timeScale;

        const std::size_t nx = config.nx > 0
            ? config.nx
            : std::max<std::size_t>(8, static_cast<std::size_t>(std::llround(
                config.cellsPerWavelength * domainLength * units.waveNumber / (2.0 * pi))));
        const std::size_t ny = config.ny > 0 ? config.ny : 1;
        const std::size_t requestedNz = config.nz > 0
            ? config.nz
            : std::max<std::size_t>(16, static_cast<std::size_t>(std::llround(
                config.cellsPerWavelength * 2.0 * slabHalfHeight / domainLength)));
        // An odd nz puts a row of cell centres exactly on the z=0 interface. The tracer test at
        // initialization then sits on a knife edge and resolves by centroid noise that correlates
        // with cos(kx), producing a one-cell staircase that swamps the sub-cell eta0 displacement.
        const std::size_t nz = requestedNz + (requestedNz % 2);
        if(rank == 0 && nz != requestedNz)
        {
            std::cout << "KRTI: nz " << requestedNz << " is odd; using " << nz
                      << " so the interface falls on a cell face" << std::endl;
        }

        Vector3D lowerLeft(0.0, 0.0, -slabHalfHeight);
        Vector3D upperRight(domainLength, domainLength, slabHalfHeight);
        Voronoi3D tess(lowerLeft, upperRight);
        tess.SetPeriodic(config.periodicHorizontal, config.periodicHorizontal, false);

        if(!restarting)
        {
            std::vector<Vector3D> points;
            if(rank == 0)
            {
                points = CartesianMesh(nx, ny, nz, lowerLeft, upperRight);
                for(Vector3D &point : points)
                {
                    point.z += interfacePerturbation(
                        point.x, point.y, false, config.perturbationKAmplitude);
                }
            }
            points = MPI_Spread(points, 0, MPI_COMM_WORLD);
            tess.BuildParallel(points);
        }

        const double chiLight = rhoLight * massScatteringOpacity;
        const double chiHeavy = rhoHeavy * massScatteringOpacity;
        std::unique_ptr<KRTIReferenceState> referenceState;
        if(!restarting)
        {
            referenceState = std::make_unique<KRTIReferenceState>(
                std::filesystem::path(config.referenceDirectory),
                config.perturbationKAmplitude);
        }

        IdealGas eos(gammaGas, 1.0 / (gammaGas - 1.0), 1.0, 0.0);
        ComputationalCell3D::tracerNames = {"HeavyFraction"};
        std::vector<ComputationalCell3D> initialCells;
        if(!restarting)
        {
            const std::size_t localCells = tess.GetPointNo();
            initialCells.resize(localCells);
            for(std::size_t cellIndex = 0; cellIndex < localCells; ++cellIndex)
            {
                const Vector3D center = tess.GetCellCM(cellIndex);
                ComputationalCell3D &cell = initialCells[cellIndex];
                const double interfaceHeight =
                    interfacePerturbation(
                        center.x, center.y, false, config.perturbationKAmplitude);
                const double materialFraction = center.z < interfaceHeight ? 0.0 : 1.0;
                cell.density = materialFraction < 0.5 ? rhoLight : rhoHeavy;
                cell.tracers[0] = materialFraction;
                const double eta0 = config.perturbationKAmplitude / units.waveNumber;
                cell.velocity = Vector3D(
                    eta0 * RealModeValue(referenceState->HydroU(center.z), center.x),
                    0.0,
                    eta0 * RealModeValue(referenceState->HydroW(center.z), center.x));
                cell.pressure = hydrostaticPressure(center.z, rhoLight, rhoHeavy,
                                                    effectiveGravity, interfaceDelta,
                                                    referencePressure) +
                                eta0 * RealModeValue(
                                    referenceState->HydroPressure(center.z), center.x);
                cell.internal_energy = cell.pressure / (cell.density * (gammaGas - 1.0));
                cell.temperature = eos.de2T(cell.density, cell.internal_energy, cell.tracers,
                                            ComputationalCell3D::tracerNames);
                cell.pressure = eos.de2p(cell.density, cell.internal_energy, cell.tracers,
                                         ComputationalCell3D::tracerNames);
                const double radiationEnergyDensity =
                    referenceState->EnergyDensity(center.x, center.z);
                cell.Erad = radiationEnergyDensity / std::max(cell.density, 1.0e-30);
                cell.Erad_dt = 0.0;
                cell.Erad_dt_dt = 0.0;
                cell.Eg.resize(ENERGY_GROUPS_NUM);
                for(std::size_t group = 0; group < ENERGY_GROUPS_NUM; ++group)
                    cell.Eg[group] = cell.Erad / static_cast<double>(ENERGY_GROUPS_NUM);
            }
            if(rank == 0)
            {
                std::cout << "KRTI frozen radiation boundary: F0=" << units.fluxScale
                          << " Ib=" << frozenBottomIntensity
                          << " Fin=" << pi * frozenBottomIntensity
                          << " chiL=" << chiLight << " chiH=" << chiHeavy
                          << std::endl;
            }
        }

        Simulation simulation(tess, initialCells, eos);
        ConstantAcceleration3D gravityAcceleration(Vector3D(0.0, 0.0, -units.gravity));
        ConservativeForce3D gravityForce(gravityAcceleration, false);
        std::shared_ptr<TimeStepFunction3D> timeStepFunction =
            std::make_shared<CourantFriedrichsLewy>(config.cfl, 1.0, gravityForce);
        simulation.SetTimeStepFunction(timeStepFunction);

        std::vector<ComputationalCell3D> &cells = simulation.getCells();
        std::vector<Conserved3D> &extensives = simulation.getExtensives();
        if(!restarting)
        {
            const std::size_t localCells = tess.GetPointNo();
            extensives.resize(localCells);
            for(std::size_t cellIndex = 0; cellIndex < localCells; ++cellIndex)
            {
                extensives[cellIndex].Eg.resize(ENERGY_GROUPS_NUM);
                PrimitiveToConserved(cells[cellIndex], tess.GetVolume(cellIndex), extensives[cellIndex]);
            }
        }
        Hllc3D riemannSolver;
        KRTIHydrostaticWall ghostGenerator;
        LinearGauss3D interpolator(eos, ghostGenerator, true, 0.2, 0.5, 0.7, false,
                                   {"HeavyFraction"});
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
        std::shared_ptr<KRTIOpacity> opacity =
            std::make_shared<KRTIOpacity>(massScatteringOpacity);
        std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> boundary =
            std::make_shared<KRTIRadiationBoundary>(
                tess, pi * frozenBottomIntensity, units.lightSpeed,
                config.newPhotonsPerCell, config.seed + 911ULL);
        STORM::RadiationIMCParameters<ENERGY_GROUPS_NUM> radiationParameters;
        radiationParameters.newPhotonsPerCell = config.newPhotonsPerCell;
        radiationParameters.lightSpeed = units.lightSpeed;
        radiationParameters.withHydro = true;
        radiationParameters.withRandomWalk = false;
        radiationParameters.withDDMC = false;
        radiationParameters.withMultigroupOpacity = false;
        radiationParameters.diffusionPressureGradient = false;
        radiationParameters.noHydroFeedback = false;
        radiationParameters.staticScatterers = true;
        radiationParameters.energyBoundaries[0] = 0.0;
        for(std::size_t group = 1; group <= ENERGY_GROUPS_NUM; ++group)
            radiationParameters.energyBoundaries[group] = 1.0e30;
        radiationParameters.energyBoundariesProvided = true;

        std::shared_ptr<RadiationIMC> radiationPhysics = std::make_shared<RadiationIMC>(
            tess, boundary, cells, extensives, eosPointer, opacity, radiationParameters);
        radiationPhysics->reseedRNG(config.seed + 104729ULL * static_cast<std::uint64_t>(rank));
        std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> populationControl =
            std::make_shared<STORM::CombPopulationControl<Vector3D, Tessellation3D>>(
                tess, config.populationPerCell, 1.0);
        std::vector<Particle3D> initialParticles;
        if(!restarting)
        {
            initialParticles = GenerateInitialRadiationParticles(
                tess, cells, *referenceState,
                config.initialParticlesPerCell, units.lightSpeed,
                config.seed + 271ULL, rank);
        }
        std::shared_ptr<RadiationMCStep> radiationStep =
            std::make_shared<RadiationMCStep>(tess, cells, extensives, radiationPhysics,
                                              populationControl, boundary, initialParticles,
                                              config.initialParticlesPerCell, true,
                                              config.managerName == "rdma"
                                                  ? RadiationMCStep::ManagerType::RDMA
                                                  : RadiationMCStep::ManagerType::P2P);
        radiationStep->setCost(
            std::make_shared<IMCCostCalculator>(radiationStep->getManager()));
        simulation.addPhysics(radiationStep);

        if(restarting)
        {
            int restartExists = 0;
            if(rank == 0)
                restartExists = std::filesystem::exists(config.restartFile) ? 1 : 0;
            MPI_Bcast(&restartExists, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if(restartExists == 0)
                throw std::runtime_error("restart file not found: " + config.restartFile);
            ReadSimulation(config.restartFile, simulation);
            if(rank == 0)
            {
                std::cout << "Restarted from " << config.restartFile
                          << ": cycle=" << simulation.GetCycle()
                          << ", tracker_time=" << simulation.GetTime()
                          << std::endl;
            }
        }

        // After HDSim3D's MPI ghost exchange, cells is sized to GetTotalPointNumber().
        // extensives must match (see Elad_paper_mach45/test.cpp).
        extensives.resize(cells.size());
        const std::size_t ownedCells = tess.GetPointNo();
        for(std::size_t cellIndex = 0; cellIndex < ownedCells; ++cellIndex)
        {
            extensives[cellIndex].Eg.resize(ENERGY_GROUPS_NUM);
            PrimitiveToConserved(cells[cellIndex], tess.GetVolume(cellIndex), extensives[cellIndex]);
        }

        const std::filesystem::path outputDirectory(config.outputDirectory);
        const std::filesystem::path historyPath = outputDirectory / "krti_history.csv";
        const std::filesystem::path snapshotDirectory = outputDirectory / "snapshots";
        const std::string latestH5Path = (outputDirectory / "latest.h5").string();
        const std::string initH5Path = (outputDirectory / "init.h5").string();
        const std::string initVtkPath = (outputDirectory / "init.pvtu").string();
        const std::string finalH5Path = (outputDirectory / "final.h5").string();
        const std::string finalVtkPath = (outputDirectory / "final.pvtu").string();

        const auto writeBookend = [&](const std::string &h5Path, const std::string &vtkPath)
        {
            if(!config.writeDumps)
            {
                return;
            }
            WriteSimulation(simulation, h5Path);
            WriteKRTIVtu(tess, cells, radiationPhysics, vtkPath);
            MPI_Barrier(MPI_COMM_WORLD);
            if(rank == 0)
            {
                std::cout << "Wrote " << h5Path << " and " << vtkPath << std::endl;
            }
        };

        const auto writeCheckpoints = [&](std::size_t cycle)
        {
            if(!config.writeDumps)
            {
                return;
            }
            const bool writeLatest = DueEvery(cycle, config.hdf5Every);
            const bool writeReplica = DueEvery(cycle, config.hdf5ReplicaEvery);
            const bool writeVtk = DueEvery(cycle, config.vtkEvery);
            if(!writeLatest && !writeReplica && !writeVtk)
            {
                return;
            }
            if(writeLatest)
            {
                WriteSimulation(simulation, latestH5Path);
            }
            if(writeReplica)
            {
                WriteSimulation(simulation, snapshotPath(snapshotDirectory, cycle));
            }
            if(writeVtk)
            {
                WriteKRTIVtu(tess, cells, radiationPhysics, vtkSnapshotPath(snapshotDirectory, cycle));
            }
            MPI_Barrier(MPI_COMM_WORLD);
            if(rank == 0)
            {
                bool wroteSomething = false;
                if(writeLatest)
                {
                    std::cout << "wrote latest checkpoint " << latestH5Path;
                    wroteSomething = true;
                }
                if(writeReplica)
                {
                    std::cout << (wroteSomething ? " and replica " : "wrote replica ")
                              << snapshotPath(snapshotDirectory, cycle);
                    wroteSomething = true;
                }
                if(writeVtk)
                {
                    std::cout << (wroteSomething ? " and VTK " : "wrote VTK ")
                              << vtkSnapshotPath(snapshotDirectory, cycle);
                }
                std::cout << std::endl;
            }
        };

        std::vector<ModeDescriptor> modes;
        modes.push_back({"mode1", 1.0, 0.0});
        if(config.twoModes)
            modes.push_back({"mode2", 0.0, 1.0});

        std::vector<double> initialAmplitudes(modes.size());
        std::vector<double> initialConservedAmplitudes(modes.size());
        for(std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex)
        {
            initialAmplitudes[modeIndex] = projectTracerMode(
                tess, cells, modes[modeIndex], units.waveNumber);
            initialConservedAmplitudes[modeIndex] = projectConservedTracerMode(
                tess, cells, modes[modeIndex], units.waveNumber, rhoHeavy);
        }

        const std::size_t initialParticleCount = particleCount(radiationStep->getParticles());
        if(rank == 0)
        {
            std::filesystem::create_directories(outputDirectory);
            if(config.writeDumps && (config.hdf5ReplicaEvery > 0 || config.vtkEvery > 0))
            {
                std::filesystem::create_directories(snapshotDirectory);
            }
        }
        if(!restarting)
        {
            if(rank == 0)
            {
                std::ofstream history(historyPath);
                history << "# benchmark=KRTI-S\n"
                        << "# case=X\n"
                        << "# theta=" << std::setprecision(17) << 1.0 << "\n"
                        << "# atwood=" << atwoodNumber << "\n"
                        << "# alpha=" << radiationSupport << "\n"
                        << "# nx=" << nx << "\n"
                        << "# ny=" << ny << "\n"
                        << "# nz=" << nz << "\n"
                        << "# mpi_ranks=" << nprocs << "\n"
                        << "# k_eta0=" << config.perturbationKAmplitude << "\n"
                        << "# H_cm=" << slabHalfHeight << "\n"
                        << "# k_cm_inv=" << units.waveNumber << "\n"
                        << "# rho_minus_g_cm3=" << rhoLight << "\n"
                        << "# rho_plus_g_cm3=" << rhoHeavy << "\n"
                        << "# gravity_cm_s2=" << units.gravity << "\n"
                        << "# gamma_gas=" << gammaGas << "\n"
                        << "# light_speed=" << units.lightSpeed << "\n"
                        << "# mass_scattering_opacity=" << massScatteringOpacity << "\n"
                        << "# net_flux=" << units.fluxScale << "\n"
                        << "# incident_flux=" << pi * frozenBottomIntensity << "\n"
                        << "# incident_intensity=" << frozenBottomIntensity << "\n"
                        << "# interface_delta=" << 0.0 << "\n"
                        << "# reference_pressure=" << referencePressure << "\n"
                        << "# two_modes=0\n"
                        << "# static_scatterers=1\n"
                        << "# initialization=full_reference_eigenmode\n";
                history << "time,time_rt";
                for(const ModeDescriptor &mode : modes)
                {
                    history << ",A_" << mode.label;
                    history << ",A_" << mode.label << "_conserved";
                }
                history << ",particle_count\n"
                        << std::scientific << std::setprecision(17)
                        << 0.0 << ',' << 0.0;
                for(std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex)
                {
                    history << ',' << initialAmplitudes[modeIndex]
                            << ',' << initialConservedAmplitudes[modeIndex];
                }
                history << ',' << initialParticleCount << '\n';
            }
            MPI_Barrier(MPI_COMM_WORLD);
            writeBookend(initH5Path, initVtkPath);
            writeCheckpoints(0);
        }
        else if(rank == 0)
        {
            std::cout << "Appending history to " << historyPath.string() << std::endl;
        }

        const double bottomPressure = hydrostaticPressure(
            -slabHalfHeight, rhoLight, rhoHeavy, effectiveGravity,
            interfaceDelta, referencePressure);
        const double topPressure = hydrostaticPressure(
            slabHalfHeight, rhoLight, rhoHeavy, effectiveGravity,
            interfaceDelta, referencePressure);
        const double soundSpeed = std::sqrt(gammaGas * std::max(
            bottomPressure / std::max(rhoLight, 1.0e-30),
            topPressure / std::max(rhoHeavy, 1.0e-30)));
        const double cflDt = config.cfl * std::min(domainLength / static_cast<double>(nx),
                                                    2.0 * slabHalfHeight / static_cast<double>(nz)) /
                              std::max(soundSpeed, 1.0e-12);
        double currentTime = simulation.GetTime();
        std::size_t cycle = simulation.GetCycle();
        if(restarting)
        {
            if(rank == 0)
            {
                std::cout << "Resuming main loop at cycle " << cycle
                          << ", currentTime=" << currentTime
                          << " (t/rt=" << currentTime / units.timeScale << ")"
                          << ", finalTime=" << finalTime
                          << std::endl;
            }
        }
        if(currentTime < finalTime)
        {
            simulation.SetTimeStep(std::min(cflDt, finalTime - currentTime));
        }
        while(currentTime < finalTime)
        {
            const double stepDt = std::min(
                std::min(cflDt, simulation.GetTimeStep()),
                finalTime - currentTime);
            if(!(stepDt > 0.0) || !std::isfinite(stepDt))
            {
                throw std::runtime_error("KRTI received an invalid timestep");
            }
            simulation.SetTimeStep(stepDt);
            simulation.step();
            currentTime = simulation.GetTime();
            cycle = simulation.GetCycle();

            const bool writeHistory = DueEvery(cycle, config.historyEvery);
            const bool writeLatest = DueEvery(cycle, config.hdf5Every);
            const bool writeReplica = DueEvery(cycle, config.hdf5ReplicaEvery);
            const bool writeVtk = DueEvery(cycle, config.vtkEvery);
            if(!writeHistory && !writeLatest && !writeReplica && !writeVtk)
            {
                continue;
            }

            if(writeHistory)
            {
                std::vector<double> amplitudes(modes.size());
                std::vector<double> conservedAmplitudes(modes.size());
                for(std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex)
                {
                    amplitudes[modeIndex] = projectTracerMode(
                        tess, cells, modes[modeIndex], units.waveNumber);
                    conservedAmplitudes[modeIndex] = projectConservedTracerMode(
                        tess, cells, modes[modeIndex], units.waveNumber, rhoHeavy);
                }
                const std::size_t globalParticleCount = particleCount(radiationStep->getParticles());
                if(rank == 0)
                {
                    std::ofstream history(historyPath, std::ios::app);
                    history << std::scientific << std::setprecision(17)
                            << currentTime << ',' << currentTime / units.timeScale;
                    for(std::size_t modeIndex = 0; modeIndex < modes.size(); ++modeIndex)
                    {
                        history << ',' << amplitudes[modeIndex]
                                << ',' << conservedAmplitudes[modeIndex];
                    }
                    history << ',' << globalParticleCount << '\n';
                }
                MPI_Barrier(MPI_COMM_WORLD);
                if(rank == 0 && config.logEvery > 0 && DueEvery(cycle, config.logEvery))
                {
                    std::cout << "cycle " << cycle
                              << " t/rt=" << currentTime / units.timeScale
                              << " A1=" << amplitudes[0];
                    if(amplitudes.size() > 1)
                    {
                        std::cout << " A2=" << amplitudes[1];
                    }
                    std::cout << " packets=" << globalParticleCount
                              << std::endl;
                }
            }
            writeCheckpoints(cycle);
        }

        writeBookend(finalH5Path, finalVtkPath);
        if(rank == 0)
        {
            std::cout << "KRTI finished. History: " << historyPath.string();
            if(config.hdf5Every > 0)
            {
                std::cout << "  latest: " << latestH5Path;
            }
            if(config.hdf5ReplicaEvery > 0 || config.vtkEvery > 0)
            {
                std::cout << "  archives: " << snapshotDirectory.string();
            }
            std::cout << std::endl;
        }
    }
    catch(const UniversalError &error)
    {
        std::cerr << "KRTI failed on rank " << rank << std::endl;
        reportError(error, std::cerr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    catch(const std::exception &error)
    {
        std::cerr << "KRTI failed on rank " << rank << ": " << error.what() << '\n';
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
