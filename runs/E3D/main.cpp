#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
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
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef RICH_MPI
#include <mpi.h>
#include "source/mpi/mpi_commands.hpp"
#endif

#include "source/3D/output/write_vtu_3d.hpp"
#include "source/3D/output/write3D.hpp"
#include "source/3D/radiation/IMCCostCalculator.hpp"
#include "source/3D/radiation/RadiationIMC.hpp"
#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/monte/population/CombPopulationControl.hpp"
#include "source/monte/particle/ParticleStatus.hpp"
#include "source/monte/utils/RandomOnFace.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/ManualTimeStep.hpp"
#include "source/newtonian/three_dimensional/conserved_3d.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"
#include "source/monte/deps/CMMC/src/units/units.hpp"
#include "source/utils/arguments/ArgumentParser.hpp"

namespace fs = std::filesystem;

namespace
{

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kBoundaryTolerance = 1.0e-10;

struct E3DConfig
{
    // Physical benchmark constants.
    double domainLength = 1.0;
    double sphereRadius = 0.0225;
    double sphereFraction = 0.1;
    std::size_t sphereCount = 2096;
    double sigmaSphere = 100.0;
    double sigmaBackground = 1.0;
    double rhoSphere = 1.0;
    double rhoBackground = 0.1;
    double CvKeV = 1.0e15;
    double sourceTemperatureKeV = 1.0;
    double initialTemperatureKeV = std::numeric_limits<double>::quiet_NaN();

    // Numerical controls.  The defaults are a developer-scale run; the
    // published reference scale is selected explicitly with the CLI.
    std::size_t meshPoints = 20000;
    std::size_t spherePointsPerSphere = 4;
    std::size_t interfacePointsPerSphere = 8;
    double interfaceOffset = 0.00225;
    std::size_t newPhotonsPerCell = 5;
    std::size_t maxPhotonsPerCell = 20;
    std::size_t boundaryPacketsPerFace = 20;
    std::size_t initialParticlesPerCell = 0;
    double dt0 = 1.0e-15;
    double dtMax = 1.0e-10;
    double dtGrowth = 1.1;
    double finalTime = 5.0e-9;
    std::size_t maxSteps = std::numeric_limits<std::size_t>::max();
    bool initialRadiationLTE = false;
    bool randomWalk = false;
    bool writeVtu = false;
    std::size_t vtuInterval = 10;

    std::uint64_t geometrySeed = 1000;
    std::uint64_t transportSeed = 9000;
    std::size_t realizationId = 0;
    std::string geometryFile;
    std::string outputPrefix;
    std::string outputDirectory;

#ifdef RICH_MPI
    RadiationMCStep::ManagerType manager = RadiationMCStep::ManagerType::AUTO_RDMA;
#endif
};

struct E3DSphere
{
    Vector3D center;
};

struct BinKey
{
    int x;
    int y;
    int z;

    bool operator==(const BinKey &other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct BinKeyHash
{
    std::size_t operator()(const BinKey &key) const
    {
        std::size_t value = static_cast<std::size_t>(key.x * 73856093);
        value ^= static_cast<std::size_t>(key.y * 19349663);
        value ^= static_cast<std::size_t>(key.z * 83492791);
        return value;
    }
};

class SphereRealization
{
public:
    SphereRealization(double domainLength, double radius)
        : domainLength_(domainLength), radius_(radius), binSize_(2.0 * radius)
    {}

    void setSpheres(std::vector<E3DSphere> spheres)
    {
        spheres_ = std::move(spheres);
        rebuildIndex();
    }

    const std::vector<E3DSphere> &spheres() const { return spheres_; }

    std::vector<E3DSphere> generate(std::size_t count, std::uint64_t seed) const
    {
        std::mt19937_64 generator(seed);
        std::uniform_real_distribution<double> uniform(radius_, domainLength_ - radius_);
        std::unordered_map<BinKey, std::vector<std::size_t>, BinKeyHash> bins;
        std::vector<E3DSphere> result;
        result.reserve(count);

        const std::uint64_t maxAttempts = std::max<std::uint64_t>(
            1000000ULL, static_cast<std::uint64_t>(count) * 100000ULL);
        std::uint64_t attempts = 0;
        while(result.size() < count)
        {
            if(++attempts > maxAttempts)
            {
                throw std::runtime_error(
                    "E3D sphere generator could not place the requested number of non-overlapping spheres");
            }

            Vector3D candidate(uniform(generator), uniform(generator), uniform(generator));
            BinKey bin = makeBin(candidate);
            bool accepted = true;
            for(int dx = -1; dx <= 1 && accepted; ++dx)
            {
                for(int dy = -1; dy <= 1 && accepted; ++dy)
                {
                    for(int dz = -1; dz <= 1 && accepted; ++dz)
                    {
                        auto it = bins.find(BinKey{bin.x + dx, bin.y + dy, bin.z + dz});
                        if(it == bins.end())
                        {
                            continue;
                        }
                        for(std::size_t index : it->second)
                        {
                            Vector3D delta = candidate - result[index].center;
                            if(ScalarProd(delta, delta) < binSize_ * binSize_)
                            {
                                accepted = false;
                                break;
                            }
                        }
                    }
                }
            }
            if(accepted)
            {
                std::size_t index = result.size();
                result.push_back(E3DSphere{candidate});
                bins[bin].push_back(index);
            }
        }
        return result;
    }

    void validate(std::size_t expectedCount) const
    {
        if(spheres_.size() != expectedCount)
        {
            throw std::runtime_error("E3D geometry has an unexpected sphere count");
        }

        std::unordered_map<BinKey, std::vector<std::size_t>, BinKeyHash> bins;
        for(std::size_t i = 0; i < spheres_.size(); ++i)
        {
            const Vector3D &center = spheres_[i].center;
            if(!std::isfinite(center.x) || !std::isfinite(center.y) || !std::isfinite(center.z))
            {
                throw std::runtime_error("E3D geometry contains a non-finite sphere center");
            }
            if(center.x < radius_ - kBoundaryTolerance || center.x > domainLength_ - radius_ + kBoundaryTolerance ||
               center.y < radius_ - kBoundaryTolerance || center.y > domainLength_ - radius_ + kBoundaryTolerance ||
               center.z < radius_ - kBoundaryTolerance || center.z > domainLength_ - radius_ + kBoundaryTolerance)
            {
                throw std::runtime_error("E3D sphere crosses the domain boundary");
            }

            BinKey bin = makeBin(center);
            for(int dx = -1; dx <= 1; ++dx)
            {
                for(int dy = -1; dy <= 1; ++dy)
                {
                    for(int dz = -1; dz <= 1; ++dz)
                    {
                        auto it = bins.find(BinKey{bin.x + dx, bin.y + dy, bin.z + dz});
                        if(it == bins.end())
                        {
                            continue;
                        }
                        for(std::size_t previous : it->second)
                        {
                            Vector3D delta = center - spheres_[previous].center;
                            if(ScalarProd(delta, delta) < binSize_ * binSize_ - 1.0e-14)
                            {
                                throw std::runtime_error("E3D geometry contains overlapping spheres");
                            }
                        }
                    }
                }
            }
            bins[bin].push_back(i);
        }
    }

    bool isSphereMaterial(const Vector3D &point) const
    {
        BinKey bin = makeBin(point);
        for(int dx = -1; dx <= 1; ++dx)
        {
            for(int dy = -1; dy <= 1; ++dy)
            {
                for(int dz = -1; dz <= 1; ++dz)
                {
                    auto it = bins_.find(BinKey{bin.x + dx, bin.y + dy, bin.z + dz});
                    if(it == bins_.end())
                    {
                        continue;
                    }
                    for(std::size_t index : it->second)
                    {
                        Vector3D delta = point - spheres_[index].center;
                        if(ScalarProd(delta, delta) <= radius_ * radius_)
                        {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

    double analyticSphereVolume() const
    {
        return static_cast<double>(spheres_.size()) * (4.0 * kPi / 3.0) * radius_ * radius_ * radius_;
    }

private:
    BinKey makeBin(const Vector3D &point) const
    {
        return BinKey{
            static_cast<int>(std::floor(point.x / binSize_)),
            static_cast<int>(std::floor(point.y / binSize_)),
            static_cast<int>(std::floor(point.z / binSize_))};
    }

    void rebuildIndex()
    {
        bins_.clear();
        for(std::size_t i = 0; i < spheres_.size(); ++i)
        {
            bins_[makeBin(spheres_[i].center)].push_back(i);
        }
    }

    double domainLength_;
    double radius_;
    double binSize_;
    std::vector<E3DSphere> spheres_;
    std::unordered_map<BinKey, std::vector<std::size_t>, BinKeyHash> bins_;
};

struct E3DTallies
{
    double inputStep = 0.0;
    double transmittedStep = 0.0;
    double reflectedStep = 0.0;
    double inputCumulative = 0.0;
    double transmittedCumulative = 0.0;
    double reflectedCumulative = 0.0;

    void beginStep()
    {
        inputStep = 0.0;
        transmittedStep = 0.0;
        reflectedStep = 0.0;
    }

    void addInput(double energy) { inputStep += energy; }
    void addTransmission(double energy) { transmittedStep += energy; }
    void addReflection(double energy) { reflectedStep += energy; }

    void reduce()
    {
        double values[3] = {inputStep, transmittedStep, reflectedStep};
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, values, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
        inputStep = values[0];
        transmittedStep = values[1];
        reflectedStep = values[2];
        inputCumulative += inputStep;
        transmittedCumulative += transmittedStep;
        reflectedCumulative += reflectedStep;
    }
};

class E3DOpacity final : public OpacityCalculator
{
public:
    E3DOpacity(double sphereOpacity, double backgroundOpacity)
        : sphereOpacity_(sphereOpacity), backgroundOpacity_(backgroundOpacity)
    {}

    double CalcPlanckOpacity(const ComputationalCell3D &cell) const override
    {
        return cell.tracers[0] > 0.5 ? sphereOpacity_ : backgroundOpacity_;
    }

    double CalcAbsorptionOpacity(const ComputationalCell3D &cell, double /*energy*/) const override
    {
        return CalcPlanckOpacity(cell);
    }

    double CalcScatteringOpacity(const ComputationalCell3D & /*cell*/) const override
    {
        return 0.0;
    }

    double CalcScatteringOpacity(const ComputationalCell3D & /*cell*/, double /*energy*/) const override
    {
        return 0.0;
    }

private:
    double sphereOpacity_;
    double backgroundOpacity_;
};

// Step from the sampled point toward the cell's generator, doubling the
// fraction, until it is strictly inside the box and inside the declared cell.
// A fixed relative nudge cannot do this for sliver wall cells whose generator
// sits ~1e-7 from a wall: the point stays outside the box or in a neighbour.
static void NudgeIntoCell(const Tessellation3D &grid, Vector3D &location, std::size_t cellIndex)
{
    const Vector3D original = location;
    const Vector3D target = grid.GetMeshPoint(cellIndex);
    double t = 1.0e-8;
    while((grid.IsPointOutsideBox(location) || !grid.IsPointInCell(location, cellIndex)) && t < 1.0)
    {
        location = original + t * (target - original);
        t *= 2.0;
    }
}

class E3DBoundary final : public STORM::BoundaryCondition<Vector3D, Tessellation3D>
{
public:
    E3DBoundary(const Tessellation3D &grid,
                double sourceTemperature,
                std::size_t packetsPerFace,
                std::uint64_t transportSeed,
                std::shared_ptr<E3DTallies> tallies)
        : STORM::BoundaryCondition<Vector3D, Tessellation3D>(grid),
          sourceTemperature_(sourceTemperature),
          packetsPerFace_(packetsPerFace),
          tallies_(std::move(tallies)),
          uniform_(0.0, 1.0)
    {
        int rank = 0;
#ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
        generator_.seed(transportSeed + 0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(rank + 1));
    }

    STORM::ParticleStatus apply(Particle3D &particle) override
    {
        std::vector<BoxFacePlane> planes = CollectBoxFacePlanes(particle.location);
        if(planes.empty())
        {
            std::ostringstream message;
            message << "E3DBoundary could not identify a valid box face for " << particle;
            throw std::runtime_error(message.str());
        }

        // The wall with the most negative inward distance is the one the packet
        // actually left through.  Choosing the *nearest* wall instead lets a
        // packet that sits on one wall while being far outside another one be
        // projected back onto the wrong wall forever, which strands it outside
        // the domain with no time progress.
        const BoxFacePlane *worst = &planes[0];
        for(const BoxFacePlane &plane : planes)
        {
            if(plane.signedDistance < worst->signedDistance)
            {
                worst = &plane;
            }
        }

        bool leftThroughZ = false;
        bool transmitted = false;
        bool anyViolated = false;
        for(const BoxFacePlane &plane : planes)
        {
            if(plane.signedDistance >= 0.0)
            {
                continue;
            }
            anyViolated = true;
            MoveInsidePlane(particle, plane);
            if(std::abs(plane.inwardNormal.z) > 0.99)
            {
                leftThroughZ = true;
                transmitted = plane.inwardNormal.z < 0.0;
            }
        }
        if(!anyViolated)
        {
            MoveInsidePlane(particle, *worst);
        }

        const double inwardVelocity = ScalarProd(particle.velocity, worst->inwardNormal);
        if(leftThroughZ || (std::abs(worst->inwardNormal.z) > 0.99 && inwardVelocity < 0.0))
        {
            if(leftThroughZ ? transmitted : worst->inwardNormal.z < 0.0)
            {
                tallies_->addTransmission(particle.weight);
            }
            else
            {
                tallies_->addReflection(particle.weight);
            }
            return STORM::ParticleStatus::REMOVE;
        }

        if(inwardVelocity < 0.0)
        {
            particle.velocity -= 2.0 * inwardVelocity * worst->inwardNormal;
        }
        // Keep the reflected packet inside its declared cell: the fixed push-in of
        // MoveInsidePlane exceeds the height of sliver wall cells, and nothing
        // downstream re-checks the cell.
        NudgeIntoCell(this->grid, particle.location, particle.cellIndex);
        return STORM::ParticleStatus::REFLECT;
    }

    bool isEscape(STORM::ParticleStatus status) const override
    {
        return status == STORM::ParticleStatus::REMOVE;
    }

    std::vector<Particle3D> generateNewBoundaryParticles(double fullDt) override
    {
        std::vector<Particle3D> particles;
        if(packetsPerFace_ == 0)
        {
            return particles;
        }

        const double sourceTemperatureFourth = std::pow(sourceTemperature_, 4);
        const std::size_t cellCount = this->grid.GetPointNo();
        for(std::size_t cellIndex = 0; cellIndex < cellCount; ++cellIndex)
        {
            const Vector3D &cellPoint = this->grid.GetMeshPoint(cellIndex);
            for(std::size_t faceIndex : this->grid.GetCellFaces(cellIndex))
            {
                const auto &neighbors = this->grid.GetFaceNeighbors(faceIndex);
                std::size_t outsideIndex = neighbors.first == cellIndex ? neighbors.second : neighbors.first;
                if(outsideIndex < cellCount || !this->grid.IsPointOutsideBox(outsideIndex))
                {
                    continue;
                }

                Vector3D outwardNormal = this->grid.GetMeshPoint(outsideIndex) - cellPoint;
                double normalLength = abs(outwardNormal);
                if(!(normalLength > 0.0))
                {
                    continue;
                }
                outwardNormal *= 1.0 / normalLength;
                if(outwardNormal.z > -0.99)
                {
                    continue;
                }

                double faceEnergy = units::sigma_sb * sourceTemperatureFourth *
                                    this->grid.GetArea(faceIndex) * fullDt;
                tallies_->addInput(faceEnergy);
                double packetEnergy = faceEnergy / static_cast<double>(packetsPerFace_);
                particles.reserve(particles.size() + packetsPerFace_);
                for(std::size_t packetIndex = 0; packetIndex < packetsPerFace_; ++packetIndex)
                {
                    Particle3D particle;
                    particle.location = STORM::RandomPointOnFace<Vector3D, Tessellation3D>(this->grid, faceIndex);
                    NudgeIntoCell(this->grid, particle.location, cellIndex);

                    double mu = std::sqrt(uniform_(generator_));
                    double transverse = std::sqrt(std::max(0.0, 1.0 - mu * mu));
                    double phi = 2.0 * kPi * uniform_(generator_);
                    particle.velocity = Vector3D(transverse * std::cos(phi),
                                                 transverse * std::sin(phi), mu);
                    particle.velocity *= units::clight;
                    particle.frequency = 0.0;
                    particle.weight = packetIndex + 1 == packetsPerFace_
                        ? faceEnergy - packetEnergy * static_cast<double>(packetsPerFace_ - 1)
                        : packetEnergy;
                    particle.initialWeight = particle.weight;
                    particle.timeLeft = fullDt * uniform_(generator_);
                    particle.cellIndex = cellIndex;
                    particles.push_back(particle);
                }
            }
        }
        return particles;
    }

    STORM::DDMCBoundaryFaceBehavior getDDMCBoundaryFaceBehavior(
        std::size_t /*faceIndex*/, std::size_t /*insideCellIndex*/, std::size_t /*outsidePointIndex*/) const override
    {
        return STORM::DDMCBoundaryFaceBehavior::Unsupported;
    }

private:
    struct BoxFacePlane
    {
        Vector3D inwardNormal;
        double faceScale;
        // Distance from the packet to the plane along the inward normal.
        // Negative means the packet is outside this wall.
        double signedDistance;
    };

    std::vector<BoxFacePlane> CollectBoxFacePlanes(const Vector3D &location) const
    {
        const std::pair<Vector3D, Vector3D> box = this->grid.GetBoxCoordinates();
        const Vector3D boxCenter = 0.5 * (box.first + box.second);
        std::vector<BoxFacePlane> planes;
        planes.reserve(6);
        for(const Tessellation3D::Face_T &face : this->grid.GetBoxFaces())
        {
            if(face.vertices.size() < 3)
            {
                continue;
            }
            const Vector3D u = face.vertices[1] - face.vertices[0];
            const Vector3D v = face.vertices[2] - face.vertices[0];
            Vector3D inwardNormal = CrossProduct(u, v);
            const double normalNorm = abs(inwardNormal);
            const double faceScale = std::min(abs(u), abs(v));
            if(!(normalNorm > 0.0) || !std::isfinite(normalNorm) ||
               !(faceScale > 0.0) || !std::isfinite(faceScale))
            {
                continue;
            }
            inwardNormal *= 1.0 / normalNorm;
            const Vector3D &onFace = face.vertices[0];
            if(ScalarProd(inwardNormal, boxCenter - onFace) < 0.0)
            {
                inwardNormal *= -1.0;
            }
            BoxFacePlane plane;
            plane.inwardNormal = inwardNormal;
            plane.faceScale = faceScale;
            plane.signedDistance = ScalarProd(location - onFace, inwardNormal);
            planes.push_back(plane);
        }
        return planes;
    }

    static void MoveInsidePlane(Particle3D &particle, const BoxFacePlane &plane)
    {
        particle.location -= plane.signedDistance * plane.inwardNormal;
        particle.location += 1e-6 * plane.faceScale * plane.inwardNormal;
    }

    double sourceTemperature_;
    std::size_t packetsPerFace_;
    std::shared_ptr<E3DTallies> tallies_;
    std::mt19937_64 generator_;
    std::uniform_real_distribution<double> uniform_;
};

// The parallel transport managers normally apply the boundary condition at
// the instant a packet crosses a box face.  With distributed Voronoi ghosts,
// a packet can occasionally reach the end-of-step census with a small
// overshoot instead (for example z=1.0004 at the top face).  Comb population
// control intentionally rejects such input packets, so normalize and process
// these escaped packets here before handing the census to Comb.
class E3DPopulationControl final
    : public STORM::PopulationControl<Vector3D, Tessellation3D>
{
public:
    E3DPopulationControl(
        const Tessellation3D &grid,
        std::shared_ptr<E3DBoundary> boundary,
        std::size_t minimumParticlesPerCell)
        : STORM::PopulationControl<Vector3D, Tessellation3D>(grid),
          comb_(grid, minimumParticlesPerCell, 1.0),
          boundary_(std::move(boundary))
    {}

    std::vector<Particle3D> activate(
        const std::vector<Particle3D> &particles) override
    {
        std::vector<Particle3D> sanitized;
        sanitized.reserve(particles.size());
        const auto [lower, upper] = this->grid.GetBoxCoordinates();

        for(const Particle3D &input : particles)
        {
            if(!this->grid.IsPointOutsideBox(input.location))
            {
                sanitized.push_back(input);
                continue;
            }

            Particle3D particle = input;
            particle.location.x = std::max(lower.x, std::min(upper.x, particle.location.x));
            particle.location.y = std::max(lower.y, std::min(upper.y, particle.location.y));
            particle.location.z = std::max(lower.z, std::min(upper.z, particle.location.z));

            STORM::ParticleStatus status = boundary_->apply(particle);
            if(status == STORM::ParticleStatus::REMOVE)
            {
                continue;
            }
            if(status != STORM::ParticleStatus::REFLECT)
            {
                throw std::runtime_error(
                    "E3D boundary repair returned an unsupported particle status");
            }

            // Keep a reflected packet strictly inside its declared cell for
            // the subsequent population-control pass.
            if(particle.cellIndex >= this->grid.GetPointNo())
            {
                throw std::runtime_error(
                    "E3D boundary repair encountered an invalid cell index");
            }
            constexpr double nudge = 1.0e-8;
            particle.location = (1.0 - nudge) * particle.location +
                                nudge * this->grid.GetMeshPoint(particle.cellIndex);
            sanitized.push_back(particle);
        }

        return comb_.activate(sanitized);
    }

private:
    STORM::CombPopulationControl<Vector3D, Tessellation3D> comb_;
    std::shared_ptr<E3DBoundary> boundary_;
};

struct ParsedArguments
{
    E3DConfig config;
    bool showHelp = false;
    std::string helpText;
};

#ifdef RICH_MPI
RadiationMCStep::ManagerType parseManager(const std::string &value)
{
    if(value == "auto")
    {
        return RadiationMCStep::ManagerType::AUTO_RDMA;
    }
    if(value == "p2p")
    {
        return RadiationMCStep::ManagerType::P2P;
    }
    if(value == "rdma")
    {
        return RadiationMCStep::ManagerType::RDMA;
    }
    if(value == "rdma-ibv")
    {
        return RadiationMCStep::ManagerType::RDMA_IBV;
    }
    if(value == "legacy")
    {
        return RadiationMCStep::ManagerType::LEGACY_AUTO_RDMA;
    }
    throw std::runtime_error("Unknown --manager value: " + value);
}
#endif

void RequirePositive(std::size_t value, const std::string &name)
{
    if(value == 0)
    {
        throw std::runtime_error(name + " must be a positive integer");
    }
}

void RequirePositive(double value, const std::string &name)
{
    if(!std::isfinite(value) || !(value > 0.0))
    {
        throw std::runtime_error(name + " must be a positive finite number");
    }
}

void AddE3DArguments(ArgumentParser &arguments)
{
    const E3DConfig defaults;
    arguments.addPositional<std::size_t>("mesh_points", defaults.meshPoints, "Background Voronoi points");
    arguments.addPositional<std::size_t>("new_photons_per_cell", defaults.newPhotonsPerCell, "New IMC packets per cell per step");
    arguments.addPositional<std::size_t>("max_photons_per_cell", defaults.maxPhotonsPerCell, "Target retained packets per cell");
    arguments.addOption<std::size_t>("realization", defaults.realizationId, "Geometry realization id");
    arguments.addOption<std::uint64_t>("geometry-seed", defaults.geometrySeed, "Seed used only to generate spheres");
    arguments.addOption<std::uint64_t>("transport-seed", defaults.transportSeed, "Seed used only by radiation/source RNGs");
    arguments.addOption<std::string>("geometry-file", "", "Load/save the deterministic sphere CSV");
    arguments.addOption<std::string>("output", "", "Write tallies, HDF5, and VTK to this directory; omitted disables dumps");
    arguments.addOption<std::size_t>("mesh-points", defaults.meshPoints, "Background Voronoi points");
    arguments.addOption<std::size_t>("sphere-count", defaults.sphereCount, "Number of spheres");
    arguments.addOption<std::size_t>("sphere-points-per-sphere", defaults.spherePointsPerSphere, "Interior points per sphere");
    arguments.addOption<std::size_t>("interface-points-per-sphere", defaults.interfacePointsPerSphere, "Shell directions per sphere");
    arguments.addOption<double>("interface-offset", defaults.interfaceOffset, "Shell offset around R, in cm");
    arguments.addOption<std::size_t>("source-packets-per-face", defaults.boundaryPacketsPerFace, "Boundary packets per z=0 face");
    arguments.addOption<std::size_t>("initial-particles-per-cell", defaults.initialParticlesPerCell, "LTE initial field packet count");
    arguments.addOption<double>("T0-keV", "Required initial material temperature in keV")
        .required()
        .alias("initial-material-temperature-keV");
    arguments.addOption<std::string>("initial-radiation", std::string("empty"), "Initial radiation convention")
        .choices({"empty", "lte"});
    arguments.addOption<double>("dt0", defaults.dt0, "Initial time step in seconds");
    arguments.addOption<double>("dt-max", defaults.dtMax, "Maximum time step in seconds");
    arguments.addOption<double>("dt-growth", defaults.dtGrowth, "Time-step growth factor");
    arguments.addOption<double>("t-final", defaults.finalTime, "Final time in seconds");
    arguments.addOption<std::size_t>("max-steps", "Optional development run limit");
    arguments.addFlag("random-walk", "Enable IMC random-walk acceleration");
    arguments.addFlag("no-vtu", "Disable VTK even if --output is given");
#ifdef RICH_MPI
    arguments.addOption<std::string>("manager", std::string("auto"), "Monte Carlo communication manager")
        .choices({"auto", "p2p", "rdma", "rdma-ibv", "legacy"});
#endif
}

std::size_t ChooseCount(const ArgumentParser &arguments, const std::string &positionalName, const std::string &optionName)
{
    if(arguments.wasSet(optionName))
    {
        return arguments.get<std::size_t>(optionName);
    }
    return arguments.get<std::size_t>(positionalName);
}

void FinishE3DConfig(E3DConfig &config)
{
    RequirePositive(config.meshPoints, "mesh_points");
    RequirePositive(config.newPhotonsPerCell, "new_photons_per_cell");
    RequirePositive(config.maxPhotonsPerCell, "max_photons_per_cell");
    RequirePositive(config.sphereCount, "sphere-count");
    RequirePositive(config.spherePointsPerSphere, "sphere-points-per-sphere");
    RequirePositive(config.interfacePointsPerSphere, "interface-points-per-sphere");
    RequirePositive(config.boundaryPacketsPerFace, "source-packets-per-face");
    RequirePositive(config.interfaceOffset, "interface-offset");
    RequirePositive(config.initialTemperatureKeV, "T0-keV");
    RequirePositive(config.dt0, "dt0");
    RequirePositive(config.dtMax, "dt-max");
    RequirePositive(config.finalTime, "t-final");
    if(config.initialRadiationLTE && config.initialParticlesPerCell == 0)
    {
        throw std::runtime_error("--initial-radiation lte requires --initial-particles-per-cell > 0");
    }
    if(config.dtMax < config.dt0)
    {
        throw std::runtime_error("--dt-max must be at least --dt0");
    }
    if(config.dtGrowth < 1.0)
    {
        throw std::runtime_error("--dt-growth must be at least 1");
    }
    if(config.interfaceOffset >= config.sphereRadius)
    {
        throw std::runtime_error("--interface-offset must be smaller than the sphere radius");
    }
    if(config.geometryFile.empty())
    {
        config.geometryFile = "geometry_realization_" + std::to_string(config.realizationId) + ".csv";
    }
    std::ostringstream defaultPrefix;
    defaultPrefix << "e3d_r" << std::setw(2) << std::setfill('0') << config.realizationId;
    if(config.outputDirectory.empty())
    {
        config.writeVtu = false;
        if(config.outputPrefix.empty())
        {
            config.outputPrefix = defaultPrefix.str();
        }
    }
    else if(config.outputPrefix.empty())
    {
        config.outputPrefix = (fs::path(config.outputDirectory) / defaultPrefix.str()).string();
    }
}

ParsedArguments parseArguments(int argc, char *argv[])
{
    ArgumentParser arguments("Brantley-Novellino explicit-sphere E3D IMC benchmark");
    AddE3DArguments(arguments);
    ParsedArguments parsed;
    if(!arguments.parse(argc, argv))
    {
        parsed.showHelp = true;
        parsed.helpText = arguments.help();
        return parsed;
    }

    E3DConfig &config = parsed.config;
    config.meshPoints = ChooseCount(arguments, "mesh_points", "mesh-points");
    config.newPhotonsPerCell = arguments.get<std::size_t>("new_photons_per_cell");
    config.maxPhotonsPerCell = arguments.get<std::size_t>("max_photons_per_cell");
    config.realizationId = arguments.get<std::size_t>("realization");
    config.geometrySeed = arguments.get<std::uint64_t>("geometry-seed");
    config.transportSeed = arguments.get<std::uint64_t>("transport-seed");
    config.geometryFile = arguments.get<std::string>("geometry-file");
    config.outputDirectory = arguments.get<std::string>("output");
    config.writeVtu = arguments.wasSet("output") && !arguments.get<bool>("no-vtu");
    config.sphereCount = arguments.get<std::size_t>("sphere-count");
    config.spherePointsPerSphere = arguments.get<std::size_t>("sphere-points-per-sphere");
    config.interfacePointsPerSphere = arguments.get<std::size_t>("interface-points-per-sphere");
    config.interfaceOffset = arguments.get<double>("interface-offset");
    config.boundaryPacketsPerFace = arguments.get<std::size_t>("source-packets-per-face");
    config.initialParticlesPerCell = arguments.get<std::size_t>("initial-particles-per-cell");
    config.initialTemperatureKeV = arguments.get<double>("T0-keV");
    config.initialRadiationLTE = arguments.get<std::string>("initial-radiation") == "lte";
    config.dt0 = arguments.get<double>("dt0");
    config.dtMax = arguments.get<double>("dt-max");
    config.dtGrowth = arguments.get<double>("dt-growth");
    config.finalTime = arguments.get<double>("t-final");
    if(arguments.wasSet("max-steps"))
    {
        config.maxSteps = arguments.get<std::size_t>("max-steps");
        RequirePositive(config.maxSteps, "max-steps");
    }
    config.randomWalk = arguments.get<bool>("random-walk");
#ifdef RICH_MPI
    config.manager = parseManager(arguments.get<std::string>("manager"));
#endif
    FinishE3DConfig(config);
    return parsed;
}

void printUsage(const char *program)
{
    ArgumentParser arguments("Brantley-Novellino explicit-sphere E3D IMC benchmark");
    AddE3DArguments(arguments);
    char *helpArgv[] = {const_cast<char*>(program), const_cast<char*>("--help"), nullptr};
    arguments.parse(2, helpArgv);
    std::cerr << arguments.help();
}

std::vector<E3DSphere> loadGeometry(const fs::path &path)
{
    std::ifstream input(path);
    if(!input)
    {
        throw std::runtime_error("Could not open E3D geometry file for reading: " + path.string());
    }

    std::vector<E3DSphere> spheres;
    std::string line;
    while(std::getline(input, line))
    {
        if(line.empty() || line[0] == '#' || line.rfind("index", 0) == 0)
        {
            continue;
        }
        std::stringstream row(line);
        std::size_t index = 0;
        char comma = 0;
        E3DSphere sphere;
        if(!(row >> index >> comma >> sphere.center.x >> comma >> sphere.center.y >> comma >> sphere.center.z))
        {
            throw std::runtime_error("Malformed E3D geometry row in " + path.string());
        }
        if(index != spheres.size())
        {
            throw std::runtime_error("E3D geometry indices are not contiguous in " + path.string());
        }
        spheres.push_back(sphere);
    }
    return spheres;
}

void saveGeometry(const fs::path &path, const std::vector<E3DSphere> &spheres,
                  double radius, std::uint64_t geometrySeed, double analyticFraction)
{
    if(!path.parent_path().empty())
    {
        fs::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::trunc);
    if(!output)
    {
        throw std::runtime_error("Could not open E3D geometry file for writing: " + path.string());
    }
    output << std::setprecision(17);
    output << "# E3D explicit non-overlapping sphere realization\n";
    output << "# radius_cm=" << radius << "\n";
    output << "# geometry_seed=" << geometrySeed << "\n";
    output << "# analytic_sphere_volume_fraction=" << analyticFraction << "\n";
    output << "index,x_cm,y_cm,z_cm\n";
    for(std::size_t i = 0; i < spheres.size(); ++i)
    {
        output << i << ',' << spheres[i].center.x << ',' << spheres[i].center.y << ',' << spheres[i].center.z << '\n';
    }
}

std::vector<E3DSphere> loadOrGenerateGeometry(const E3DConfig &config, int rank)
{
    SphereRealization realization(config.domainLength, config.sphereRadius);
    std::vector<E3DSphere> spheres;
    fs::path path(config.geometryFile);
    if(rank == 0)
    {
        if(fs::exists(path))
        {
            spheres = loadGeometry(path);
            std::cout << "Loading E3D geometry from " << path << std::endl;
        }
        else
        {
            std::cout << "Generating " << config.sphereCount << " E3D spheres with seed "
                      << config.geometrySeed << std::endl;
            spheres = realization.generate(config.sphereCount, config.geometrySeed);
            realization.setSpheres(spheres);
            realization.validate(config.sphereCount);
            saveGeometry(path, spheres, config.sphereRadius,
                         config.geometrySeed, realization.analyticSphereVolume() /
                         (config.domainLength * config.domainLength * config.domainLength));
        }
    }

#ifdef RICH_MPI
    unsigned long long count = static_cast<unsigned long long>(spheres.size());
    MPI_Bcast(&count, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    if(rank != 0)
    {
        spheres.resize(static_cast<std::size_t>(count));
    }
    std::vector<double> packed(static_cast<std::size_t>(count) * 3);
    if(rank == 0)
    {
        for(std::size_t i = 0; i < spheres.size(); ++i)
        {
            packed[3 * i] = spheres[i].center.x;
            packed[3 * i + 1] = spheres[i].center.y;
            packed[3 * i + 2] = spheres[i].center.z;
        }
    }
    MPI_Bcast(packed.data(), static_cast<int>(packed.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(rank != 0)
    {
        for(std::size_t i = 0; i < spheres.size(); ++i)
        {
            spheres[i].center = Vector3D(packed[3 * i], packed[3 * i + 1], packed[3 * i + 2]);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    realization.setSpheres(spheres);
    realization.validate(config.sphereCount);
    return spheres;
}

void appendRandomPoint(std::vector<Vector3D> &points, std::mt19937_64 &generator,
                       double length)
{
    constexpr double endpointNudge = 1.0e-9;
    std::uniform_real_distribution<double> uniform(endpointNudge, length - endpointNudge);
    points.emplace_back(uniform(generator), uniform(generator), uniform(generator));
}

Vector3D randomPointInSphere(const Vector3D &center, double radius, std::mt19937_64 &generator)
{
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    double mu = 2.0 * uniform(generator) - 1.0;
    double phi = 2.0 * kPi * uniform(generator);
    double radial = radius * std::cbrt(uniform(generator));
    double transverse = std::sqrt(std::max(0.0, 1.0 - mu * mu));
    return center + Vector3D(radial * transverse * std::cos(phi),
                             radial * transverse * std::sin(phi), radial * mu);
}

std::vector<Vector3D> buildMeshPoints(const E3DConfig &config, const std::vector<E3DSphere> &spheres)
{
    std::mt19937_64 generator(config.geometrySeed ^ 0xd1b54a32d192ed03ULL);
    std::vector<Vector3D> points;
    points.reserve(config.meshPoints + spheres.size() *
                   (config.spherePointsPerSphere + 2 * config.interfacePointsPerSphere));
    for(std::size_t i = 0; i < config.meshPoints; ++i)
    {
        appendRandomPoint(points, generator, config.domainLength);
    }

    for(const E3DSphere &sphere : spheres)
    {
        for(std::size_t i = 0; i < config.spherePointsPerSphere; ++i)
        {
            points.push_back(randomPointInSphere(sphere.center, config.sphereRadius, generator));
        }

        // Fibonacci directions give deterministic, approximately isotropic
        // shell points without requiring a second stochastic stream.
        for(std::size_t i = 0; i < config.interfacePointsPerSphere; ++i)
        {
            double z = 1.0 - 2.0 * (static_cast<double>(i) + 0.5) /
                                  static_cast<double>(config.interfacePointsPerSphere);
            double radial = std::sqrt(std::max(0.0, 1.0 - z * z));
            double phi = kPi * (3.0 - std::sqrt(5.0)) * static_cast<double>(i);
            Vector3D direction(radial * std::cos(phi), radial * std::sin(phi), z);
            Vector3D inside = sphere.center + (config.sphereRadius - config.interfaceOffset) * direction;
            Vector3D outside = sphere.center + (config.sphereRadius + config.interfaceOffset) * direction;
            auto inDomain = [&config](const Vector3D &point)
            {
                return point.x > 0.0 && point.x < config.domainLength &&
                       point.y > 0.0 && point.y < config.domainLength &&
                       point.z > 0.0 && point.z < config.domainLength;
            };
            if(inDomain(inside))
            {
                points.push_back(inside);
            }
            if(inDomain(outside))
            {
                points.push_back(outside);
            }
        }
    }
    return points;
}

void writeManifest(const E3DConfig &config, const SphereRealization &geometry,
                   std::size_t globalCells, int mpiRanks, std::size_t steps,
                   const std::string &initialRadiationMode)
{
    fs::path path(config.outputPrefix + "_manifest.json");
    if(!path.parent_path().empty())
    {
        fs::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::trunc);
    output << std::setprecision(17);
    output << "{\n";
    output << "  \"benchmark\": \"Brantley-Novellino explicit-sphere E3D\",\n";
    output << "  \"domain_length_cm\": " << config.domainLength << ",\n";
    output << "  \"transport_axis\": \"z\",\n";
    output << "  \"sphere_radius_cm\": " << config.sphereRadius << ",\n";
    output << "  \"sphere_count\": " << geometry.spheres().size() << ",\n";
    output << "  \"target_sphere_volume_fraction\": " << config.sphereFraction << ",\n";
    output << "  \"analytic_sphere_volume_fraction\": "
            << geometry.analyticSphereVolume() /
               std::pow(config.domainLength, 3) << ",\n";
    output << "  \"sigma_absorption_sphere_cm_inv\": " << config.sigmaSphere << ",\n";
    output << "  \"sigma_absorption_background_cm_inv\": " << config.sigmaBackground << ",\n";
    output << "  \"rho_sphere_g_cm3\": " << config.rhoSphere << ",\n";
    output << "  \"rho_background_g_cm3\": " << config.rhoBackground << ",\n";
    output << "  \"Cv_erg_g_keV\": " << config.CvKeV << ",\n";
    output << "  \"source_temperature_keV\": " << config.sourceTemperatureKeV << ",\n";
    output << "  \"initial_temperature_keV\": " << config.initialTemperatureKeV << ",\n";
    output << "  \"initial_radiation_mode\": \"" << initialRadiationMode << "\",\n";
    output << "  \"geometry_seed\": " << config.geometrySeed << ",\n";
    output << "  \"transport_seed\": " << config.transportSeed << ",\n";
    output << "  \"realization_id\": " << config.realizationId << ",\n";
    output << "  \"geometry_file\": \"" << config.geometryFile << "\",\n";
    output << "  \"mesh_points_requested\": " << config.meshPoints << ",\n";
    output << "  \"mesh_points_actual\": " << globalCells << ",\n";
    output << "  \"sphere_points_per_sphere\": " << config.spherePointsPerSphere << ",\n";
    output << "  \"interface_points_per_sphere\": " << config.interfacePointsPerSphere << ",\n";
    output << "  \"interface_offset_cm\": " << config.interfaceOffset << ",\n";
    output << "  \"new_photons_per_cell\": " << config.newPhotonsPerCell << ",\n";
    output << "  \"population_target_per_cell\": " << config.maxPhotonsPerCell << ",\n";
    output << "  \"boundary_packets_per_face\": " << config.boundaryPacketsPerFace << ",\n";
    output << "  \"initial_particles_per_cell\": " << config.initialParticlesPerCell << ",\n";
    output << "  \"dt0_s\": " << config.dt0 << ",\n";
    output << "  \"dt_max_s\": " << config.dtMax << ",\n";
    output << "  \"dt_growth\": " << config.dtGrowth << ",\n";
    output << "  \"t_final_s\": " << config.finalTime << ",\n";
    output << "  \"steps_completed\": " << steps << ",\n";
    output << "  \"mpi_ranks\": " << mpiRanks << ",\n";
    output << "  \"hydrodynamics\": false,\n";
    output << "  \"physical_scattering\": false,\n";
    output << "  \"random_walk\": " << (config.randomWalk ? "true" : "false") << "\n";
    output << "}\n";
}

struct EnergyState
{
    double material = 0.0;
    double radiation = 0.0;
};

EnergyState computeEnergyState(const Tessellation3D &tess,
                               const std::vector<Conserved3D> &extensives)
{
    EnergyState local;
    for(std::size_t i = 0; i < tess.GetPointNo(); ++i)
    {
        local.material += extensives[i].internal_energy;
        local.radiation += extensives[i].Erad;
    }
#ifdef RICH_MPI
    double values[2] = {local.material, local.radiation};
    MPI_Allreduce(MPI_IN_PLACE, values, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    local.material = values[0];
    local.radiation = values[1];
#endif
    return local;
}

void writeTalliesHeader(const fs::path &path)
{
    if(!path.parent_path().empty())
    {
        fs::create_directories(path.parent_path());
    }
    std::ofstream output(path, std::ios::trunc);
    output << "# E3D gray IMC tallies; energies are erg\n";
    output << "cycle,time_s,dt_s,E_in_step,E_T_step,E_R_step,T_step,R_step,E_in_cumulative,E_T_cumulative,E_R_cumulative,material_delta,radiation_delta,energy_residual\n";
}

void appendTally(const fs::path &path, std::size_t cycle, double time, double dt,
                 const E3DTallies &tallies, const EnergyState &initial,
                 const EnergyState &current)
{
    double transmission = tallies.inputStep > 0.0 ? tallies.transmittedStep / tallies.inputStep : 0.0;
    double reflection = tallies.inputStep > 0.0 ? tallies.reflectedStep / tallies.inputStep : 0.0;
    double materialDelta = current.material - initial.material;
    double radiationDelta = current.radiation - initial.radiation;
    double residual = tallies.inputCumulative - tallies.transmittedCumulative -
                      tallies.reflectedCumulative - materialDelta - radiationDelta;
    std::ofstream output(path, std::ios::app);
    output << std::setprecision(17)
           << cycle << ',' << time << ',' << dt << ','
           << tallies.inputStep << ',' << tallies.transmittedStep << ',' << tallies.reflectedStep << ','
           << transmission << ',' << reflection << ','
           << tallies.inputCumulative << ',' << tallies.transmittedCumulative << ','
           << tallies.reflectedCumulative << ',' << materialDelta << ',' << radiationDelta << ','
           << residual << '\n';
}

void writeSnapshot(const E3DConfig &config, Simulation &simulation, const Tessellation3D &tess,
                   const std::vector<ComputationalCell3D> &cells,
                   double time, std::size_t cycle)
{
    if(!config.writeVtu || config.outputDirectory.empty())
    {
        return;
    }
    WriteSimulation(simulation, (fs::path(config.outputDirectory) / "latest.h5").string());
    std::vector<double> material(cells.size());
    std::vector<double> density(cells.size());
    std::vector<double> temperatureKeV(cells.size());
    std::vector<double> radiation(cells.size());
    for(std::size_t i = 0; i < cells.size(); ++i)
    {
        material[i] = cells[i].tracers[0] > 0.5 ? 1.0 : 2.0;
        density[i] = cells[i].density;
        temperatureKeV[i] = cells[i].temperature / units::kev_kelvin;
        radiation[i] = cells[i].Erad;
    }
    std::vector<std::string> names = {"material_id", "density_g_cm3", "temperature_keV", "Erad_specific"};
    std::vector<std::vector<double>> data = {material, density, temperatureKeV, radiation};
    std::vector<std::string> emptyNames;
    std::vector<std::vector<std::string>> emptyStringData;
    std::vector<std::string> emptyVectorNames;
    std::vector<std::vector<Vector3D>> emptyVectors;
    std::vector<std::pair<std::string, double>> scalars = {
        {"sphere_volume_fraction_analytic",
         static_cast<double>(config.sphereCount) * (4.0 * kPi / 3.0) *
             std::pow(config.sphereRadius, 3) / std::pow(config.domainLength, 3)}};
    std::ostringstream fileName;
    fileName << "cycle_" << std::setw(6) << std::setfill('0') << cycle << ".vtu";
    write_vtu3d::write_vtu_3d(fs::path(config.outputDirectory) / fileName.str(),
                              names, data, emptyNames, emptyStringData,
                              emptyVectorNames, emptyVectors, scalars, time, cycle, tess);
}

void writeBookend(const E3DConfig &config, Simulation &simulation,
                  const Voronoi3D &tess,
                  const std::vector<ComputationalCell3D> &cells,
                  const std::string &stem)
{
    if(config.outputDirectory.empty())
    {
        return;
    }
    const fs::path h5Path = fs::path(config.outputDirectory) / (stem + ".h5");
    const fs::path vtkPath = fs::path(config.outputDirectory) / (stem + ".pvtu");
    WriteSimulation(simulation, h5Path.string());
    if(!config.writeVtu)
    {
        return;
    }
    const std::size_t cellCount = tess.GetPointNo();
    std::vector<double> material(cellCount);
    std::vector<double> density(cellCount);
    std::vector<double> temperatureKeV(cellCount);
    std::vector<double> radiation(cellCount);
    for(std::size_t i = 0; i < cellCount; ++i)
    {
        material[i] = cells[i].tracers[0] > 0.5 ? 1.0 : 2.0;
        density[i] = cells[i].density;
        temperatureKeV[i] = cells[i].temperature / units::kev_kelvin;
        radiation[i] = cells[i].Erad;
    }
    WriteVoronoiVTKOnly(tess, vtkPath.string(),
                        {material, density, temperatureKeV, radiation},
                        {"material_id", "density_g_cm3", "temperature_keV", "Erad_specific"});
}

bool ShouldWriteVtu(const E3DConfig &config, std::size_t cycle, double time)
{
    if(!config.writeVtu || config.outputDirectory.empty() || config.vtuInterval == 0)
    {
        return false;
    }
    if(time >= config.finalTime)
    {
        return true;
    }
    return cycle % config.vtuInterval == 0;
}

std::vector<Vector3D> broadcastOrBuildMeshPoints(const E3DConfig &config,
                                                 const std::vector<E3DSphere> &spheres,
                                                 int rank)
{
    std::vector<Vector3D> points;
    if(rank == 0)
    {
        points = buildMeshPoints(config, spheres);
        std::cout << "Generated " << points.size() << " Voronoi generator points" << std::endl;
    }
#ifdef RICH_MPI
    return MPI_Spread(points, 0, MPI_COMM_WORLD);
#else
    return points;
#endif
}

} // namespace

int main(int argc, char *argv[])
{
    int rank = 0;
    int mpiRanks = 1;
#ifdef RICH_MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiRanks);
#else
    (void)argc;
    (void)argv;
#endif

    try
    {
        ParsedArguments parsed = parseArguments(argc, argv);
        if(parsed.showHelp)
        {
            if(rank == 0)
            {
                std::cout << parsed.helpText << std::endl;
            }
#ifdef RICH_MPI
            MPI_Finalize();
#endif
            return 0;
        }
        E3DConfig const &config = parsed.config;

        std::vector<E3DSphere> spheres = loadOrGenerateGeometry(config, rank);
        SphereRealization geometry(config.domainLength, config.sphereRadius);
        geometry.setSpheres(spheres);
        geometry.validate(config.sphereCount);

        Vector3D lower(0.0, 0.0, 0.0);
        Vector3D upper(config.domainLength, config.domainLength, config.domainLength);
        std::vector<Vector3D> localPoints = broadcastOrBuildMeshPoints(config, spheres, rank);

        Voronoi3D tess(lower, upper);
#ifdef RICH_MPI
        if(mpiRanks == 1)
        {
            tess.Build(localPoints);
        }
        else
        {
            tess.BuildParallel(localPoints);
        }
#else
        tess.Build(localPoints);
#endif

        std::size_t localCellCount = tess.GetPointNo();
        std::size_t globalCellCount = localCellCount;
#ifdef RICH_MPI
        unsigned long long globalCount = static_cast<unsigned long long>(localCellCount);
        MPI_Allreduce(MPI_IN_PLACE, &globalCount, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        globalCellCount = static_cast<std::size_t>(globalCount);
#endif

        ComputationalCell3D::tracerNames.clear();
        ComputationalCell3D::tracerNames.push_back("E3D_Sphere");
        ComputationalCell3D::tracerNames.push_back("E3D_Background");

        double const initialTemperature = config.initialTemperatureKeV * units::kev_kelvin;
        double const sourceTemperature = config.sourceTemperatureKeV * units::kev_kelvin;
        double const cvPerMass = config.CvKeV / units::kev_kelvin;
        IdealGas eos(1.4, cvPerMass, 1.0, 0.0);

        std::vector<ComputationalCell3D> initialCells(localCellCount);
        double localSphereVolume = 0.0;
        std::size_t localSphereCells = 0;
        for(std::size_t i = 0; i < localCellCount; ++i)
        {
            bool const inSphere = geometry.isSphereMaterial(tess.GetMeshPoint(i));
            double const density = inSphere ? config.rhoSphere : config.rhoBackground;
            ComputationalCell3D &cell = initialCells[i];
            cell.density = density;
            cell.temperature = initialTemperature;
            cell.velocity = Vector3D(0.0, 0.0, 0.0);
            cell.tracers.fill(0.0);
            cell.tracers[0] = inSphere ? 1.0 : 0.0;
            cell.tracers[1] = inSphere ? 0.0 : 1.0;
            cell.internal_energy = eos.dT2e(cell.density, cell.temperature,
                                             cell.tracers, ComputationalCell3D::tracerNames);
            cell.pressure = eos.de2p(cell.density, cell.internal_energy,
                                     cell.tracers, ComputationalCell3D::tracerNames);
            cell.Erad = config.initialRadiationLTE
                ? units::arad * std::pow(initialTemperature, 4) / density
                : 0.0;
            if(inSphere)
            {
                localSphereVolume += tess.GetVolume(i);
                ++localSphereCells;
            }
        }

        double globalSphereVolume = localSphereVolume;
        unsigned long long globalSphereCells = static_cast<unsigned long long>(localSphereCells);
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &globalSphereVolume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &globalSphereCells, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
        if(rank == 0)
        {
            std::cout << "E3D mesh: " << globalCellCount << " cells on " << mpiRanks << " MPI ranks\n"
                      << "  sphere cells=" << globalSphereCells
                      << ", mesh sphere volume fraction=" << globalSphereVolume /
                         std::pow(config.domainLength, 3)
                      << ", analytic fraction=" << geometry.analyticSphereVolume() /
                         std::pow(config.domainLength, 3) << std::endl;
        }

        Simulation simulation(tess, initialCells, eos);
        simulation.SetTimeStepFunction(std::make_shared<ManualTimeStep>());
        std::vector<ComputationalCell3D> &cells = simulation.getCells();
        std::vector<Conserved3D> &extensives = simulation.getExtensives();

        auto tallies = std::make_shared<E3DTallies>();
        auto boundary = std::make_shared<E3DBoundary>(
            tess, sourceTemperature, config.boundaryPacketsPerFace,
            config.transportSeed, tallies);
        auto opacity = std::make_shared<E3DOpacity>(config.sigmaSphere, config.sigmaBackground);
        auto eosPtr = std::make_shared<IdealGas>(eos);

        RadiationIMCParameters parameters;
        parameters.newPhotonsPerCell = config.newPhotonsPerCell;
        parameters.withHydro = false;
        parameters.diffusionPressureGradient = false;
        parameters.MMC = false;
        parameters.withMultigroupOpacity = false;
        parameters.withRandomWalk = config.randomWalk;
        parameters.noHydroFeedback = false;
        parameters.energyBoundaries[0] = 0.0;
        for(std::size_t g = 1; g <= ENERGY_GROUPS_NUM; ++g)
        {
            parameters.energyBoundaries[g] = 1.0e30 * static_cast<double>(g) /
                                             static_cast<double>(ENERGY_GROUPS_NUM);
        }
        parameters.energyBoundariesProvided = true;

        auto physics = std::make_shared<::RadiationIMC>(
            tess, boundary, cells, extensives, eosPtr, opacity, parameters);
        physics->reseedRNG(config.transportSeed);

        auto population = std::make_shared<E3DPopulationControl>(
            tess, boundary, config.maxPhotonsPerCell);
        std::vector<Particle3D> initialParticles;
        auto mcStep = std::make_shared<RadiationMCStep>(
            tess, cells, extensives, physics, population, boundary,
            initialParticles, config.initialParticlesPerCell, false
#ifdef RICH_MPI
            , config.manager
#endif
        );
#ifdef RICH_MPI
        mcStep->setCost(std::make_shared<IMCCostCalculator>(mcStep->getManager()));
        simulation.addMigrationBuffer(mcStep->getManager()->GetCellsStepsCounters());
        simulation.addMigrationBuffer(mcStep->getManager()->GetBeginningParticleCount());
#endif
        simulation.addPhysics(mcStep);

        EnergyState initialEnergy = computeEnergyState(tess, extensives);
        if(rank == 0 && !config.outputDirectory.empty())
        {
            fs::create_directories(config.outputDirectory);
        }
#ifdef RICH_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        fs::path tallyPath(config.outputPrefix + "_tallies.csv");
        if(rank == 0)
        {
            if(!tallyPath.parent_path().empty())
            {
                fs::create_directories(tallyPath.parent_path());
            }
            writeTalliesHeader(tallyPath);
        }
#ifdef RICH_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
        if(!config.outputDirectory.empty())
        {
            writeBookend(config, simulation, tess, cells, "init");
#ifdef RICH_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            if(rank == 0)
            {
                std::cout << "Wrote "
                          << (fs::path(config.outputDirectory) / "init.h5").string()
                          << " and "
                          << (fs::path(config.outputDirectory) / "init.pvtu").string()
                          << std::endl;
            }
        }
        if(ShouldWriteVtu(config, 0, 0.0))
        {
            writeSnapshot(config, simulation, tess, cells, 0.0, 0);
        }

        double time = 0.0;
        double dt = config.dt0;
        std::size_t steps = 0;
        auto wallStart = std::chrono::high_resolution_clock::now();
        while(time < config.finalTime && steps < config.maxSteps)
        {
            double stepDt = std::min(dt, config.finalTime - time);
            tallies->beginStep();
            simulation.SetTimeStep(stepDt);
            simulation.step();
            time = simulation.GetTime();
            ++steps;
            tallies->reduce();
            EnergyState currentEnergy = computeEnergyState(tess, extensives);
            if(rank == 0)
            {
                appendTally(tallyPath, steps, time, stepDt, *tallies, initialEnergy, currentEnergy);
                if(steps == 1 || steps % 10 == 0 || time >= config.finalTime)
                {
                    double Tstep = tallies->inputStep > 0.0
                        ? tallies->transmittedStep / tallies->inputStep : 0.0;
                    double Rstep = tallies->inputStep > 0.0
                        ? tallies->reflectedStep / tallies->inputStep : 0.0;
                    std::cout << "Cycle " << steps << "/" << config.maxSteps
                              << ", t=" << time << " s, dt=" << stepDt
                              << " s, T=" << Tstep << ", R=" << Rstep << std::endl;
                }
            }
            if(ShouldWriteVtu(config, steps, time))
            {
                writeSnapshot(config, simulation, tess, cells, time, steps);
            }
            dt = std::min(config.dtMax, stepDt * config.dtGrowth);
        }

        auto wallEnd = std::chrono::high_resolution_clock::now();
        double wallSeconds = std::chrono::duration<double>(wallEnd - wallStart).count();
        if(!config.outputDirectory.empty())
        {
            writeBookend(config, simulation, tess, cells, "final");
#ifdef RICH_MPI
            MPI_Barrier(MPI_COMM_WORLD);
#endif
            if(rank == 0)
            {
                std::cout << "Wrote "
                          << (fs::path(config.outputDirectory) / "final.h5").string()
                          << " and "
                          << (fs::path(config.outputDirectory) / "final.pvtu").string()
                          << std::endl;
            }
        }
        if(rank == 0)
        {
            writeManifest(config, geometry, globalCellCount, mpiRanks, steps,
                          config.initialRadiationLTE ? "LTE" : "EMPTY");
            std::cout << "Finished E3D realization " << config.realizationId
                      << " at t=" << time << " s after " << steps
                      << " steps; wall time=" << wallSeconds << " s" << std::endl;
        }
#ifdef RICH_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif

#ifdef RICH_MPI
        MPI_Finalize();
#endif
        return 0;
    }
    catch(const std::exception &error)
    {
        std::cerr << "E3D failure on rank " << rank << ": " << error.what() << std::endl;
        if(rank == 0)
        {
            printUsage(argv[0]);
        }
#ifdef RICH_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }
}
