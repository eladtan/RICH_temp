#include <mpi.h>
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
#include <vector>

#include "source/mpi/mpi_commands.hpp"
#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#ifdef RICH_MPI
#include <MeshDecomposer3D/load_balancing/OneDimensionalLoadBalancer.hpp>
#endif
#include "source/3D/radiation/IMCCostCalculator.hpp"
#include "source/3D/radiation/RadiationIMC.hpp"
#include "source/Radiation/OpacityCalculator.hpp"
#include "source/monte/boundary/RigidBoundary.hpp"
#include "source/monte/population/CombPopulationControl.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/ManualTimeStep.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"
#include "source/newtonian/three_dimensional/conserved_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"
#include "source/3D/output/write3D.hpp"
#include "source/3D/output/read3D.hpp"
#include "source/monte/deps/CMMC/src/units/units.hpp"
#include "source/utils/arguments/ArgumentParser.hpp"
#include "source/utils/debug/SmartTimer.hpp"
#include "source/utils/debug/vtune.h"

namespace
{

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr double ns = 1.0e-9;

struct Benchmark
{
    int number;
    double radius;
    double alpha;
    double lambda;
    double opacityCoefficient;
    double eosCoefficient;
    double beta;
    double mu;
    double omega;
    double densityCoefficient;
    double delta;
    double temperatureScale;
    double profileCoefficient;
    double profileTimeExponent;
    double bathCorrection;
    double bathTimeExponent;
    double lambdaRadiusExponent;
    double lambdaWExponent;
    std::array<double, 3> snapshotPaperTimesNs;
};

Benchmark GetBenchmark(int number)
{
    const double hevKelvin = 0.1 * units::kev_kelvin;

    if(number == 1)
    {
        return Benchmark{1, 1.0e-3, 1.5, 0.2, 7200.0,
                        3.4e13, 1.6, 0.14, 0.0, 19.3,
                        0.679501, hevKelvin, 1.34503465,
                        0.0920519, 0.103502, -0.541423,
                        1.0, -1.5,
                         {-22.122309, -9.4484244, -1.0}};
    }
    if(number == 2)
    {
        return Benchmark{2, 5.0e-2, 3.0, 0.4, 1.5e4, 3.0e13, 2.0, 0.6, -0.5, 1.0, 0.51765,
                         hevKelvin, 0.809892, 0.100238, 0.385372, -0.579294, 1.2, -1.0,
                         {-58.251607, -19.068532, -1.0}};
    }
    if(number == 3)
    {
        return Benchmark{3, 1.0e-3, 3.5, 0.4, 1.0e3, 1.0e13, 2.0, 0.25, 0.45, 1.0, 1.1157536,
                         hevKelvin, 1.1982, 0.0276392, 0.075821, -0.316092, 0.6625, -1.0,
                         {-6.5918976, -3.926451, -1.0}};
    }
    if(number == 4)
    {
        const double materialCoefficient = 0.25 * units::arad * std::pow(units::kev_kelvin, 4.0);
        return Benchmark{4, 10.0, 3.5, 1.0, 1.0, materialCoefficient, 4.0, 1.0, -1.0, 1.0, 0.462367,
                         units::kev_kelvin, 0.552154, 0.242705, 0.083391, -0.537633, 1.0, 0.0,
                         {-94.706889, -27.126998, -1.0}};
    }
    throw std::runtime_error("The benchmark number must be between 1 and 4");
}

double InitialPaperTimeNs(const Benchmark &benchmark)
{
    return -std::pow(10.0, 1.0 / benchmark.delta);
}

// Paper Eq. (58): the heat front reaches the system boundary at the initial
// time and 0.1R at t = -1 ns.
double FrontRadius(const Benchmark &benchmark, double paperTimeNs)
{
    return 0.1 * benchmark.radius * std::pow(std::max(-paperTimeNs, 0.0), benchmark.delta);
}

double PaperTimeFromFrontRadius(const Benchmark &benchmark, double frontRadius)
{
    return -std::pow(frontRadius / (0.1 * benchmark.radius), 1.0 / benchmark.delta);
}

double Density(const Benchmark &benchmark, double radius)
{
    return benchmark.densityCoefficient * std::pow(std::max(radius, benchmark.radius * 1.0e-12), -benchmark.omega);
}

double FittedW(const Benchmark &benchmark, double xi)
{
    if(xi <= 1.0)
    {
        return 0.0;
    }
    if(benchmark.number == 1)
    {
        if(xi <= 2.0)
        {
            return std::pow(xi - 1.0, 0.4057) * (1.521 - 0.3762 * xi + 0.06558 * xi * xi);
        }
        return std::pow(xi - 1.0, 0.2955) * (1.082 - 0.02718 * xi + 0.001055 * xi * xi);
    }
    if(benchmark.number == 2)
    {
        if(xi <= 2.0)
        {
            return std::pow(xi - 1.0, 0.3977) * (1.244 - 0.1757 * xi + 0.03186 * xi * xi);
        }
        return std::pow(xi - 1.0, 0.3401) * (1.021 - 0.0007123 * xi + 0.0001726 * xi * xi);
    }
    if(benchmark.number == 3)
    {
        if(xi <= 2.0)
        {
            return std::pow(xi - 1.0, 0.3575) * (1.979 - 0.6195 * xi + 0.1106 * xi * xi);
        }
        return std::pow(xi - 1.0, 0.2101) * (1.27 - 0.04707 * xi + 0.001797 * xi * xi);
    }
    if(xi <= 2.0)
    {
        return std::pow(xi - 1.0, 1.141) * (0.2251 + 0.127 * xi + 0.001626 * xi * xi);
    }
    return std::pow(xi - 1.0, 1.102) * (0.1846 + 0.1505 * xi + 0.00004394 * xi * xi);
}

double FittedV(const Benchmark &benchmark, double xi)
{
    if(benchmark.number == 1)
    {
        return 0.4345 * std::pow(xi, -2.752) + 0.2451 * std::pow(xi, -1.454);
    }
    if(benchmark.number == 2)
    {
        return 0.262 * std::pow(xi, -3.24) + 0.2558 * std::pow(xi, -1.88);
    }
    if(benchmark.number == 3)
    {
        return 0.8879 * std::pow(xi, -2.233) + 0.2278 * std::pow(xi, -1.037);
    }
    return 0.06247 * std::pow(xi, -3.836) + 0.3999 * std::pow(xi, -2.157);
}

double BathTemperature(const Benchmark &benchmark, double paperTimeNs, double temperatureFloor)
{
    const double timeMagnitude = std::max(-paperTimeNs, 1.0e-14);
    const double xi = 10.0 / std::pow(timeMagnitude, benchmark.delta);
    const double W = FittedW(benchmark, xi);
    if(W <= 0.0)
    {
        return temperatureFloor;
    }
    const double V = FittedV(benchmark, xi);
    const double surfaceScaled = benchmark.profileCoefficient * std::pow(timeMagnitude, benchmark.profileTimeExponent) * std::pow(W, 1.0 / benchmark.beta);
    const double lambdaProfile = std::pow(xi, benchmark.lambdaRadiusExponent) * V * std::pow(W, benchmark.lambdaWExponent);
    const double correction = 1.0 + benchmark.bathCorrection * std::pow(timeMagnitude, benchmark.bathTimeExponent) * lambdaProfile;
    return std::max(surfaceScaled * std::pow(std::max(correction, 0.0), 0.25) * benchmark.temperatureScale, temperatureFloor);
}

class BenchmarkOpacity : public OpacityCalculator
{
public:
    explicit BenchmarkOpacity(const Benchmark &benchmark): benchmark_(benchmark)
    {}

    double CalcPlanckOpacity(const ComputationalCell3D &cell) const override
    {
        return PlanckOpacity(cell.density, cell.temperature);
    }

    // Face-averaged DDMC evaluates the diffusion coefficient at a temperature
    // interpolated across the face rather than at either cell's own state, so
    // the opacity law has to be callable away from cell.temperature.  Without
    // these the base class throws and every rank aborts on the first DDMC face.
    double CalcPlanckOpacityAtTemperature(const ComputationalCell3D &cell, double temperature) const override
    {
        return PlanckOpacity(cell.density, temperature);
    }

    double CalcScatteringOpacity(const ComputationalCell3D &) const override
    {
        return 0.0;
    }

    double CalcAbsorptionOpacity(const ComputationalCell3D &cell, double) const override
    {
        return CalcPlanckOpacity(cell);
    }

    double CalcAbsorptionOpacityAtTemperature(const ComputationalCell3D &cell, double, double temperature) const override
    {
        return PlanckOpacity(cell.density, temperature);
    }

    double CalcDiffusionCoefficient(const ComputationalCell3D &cell) const override
    {
        return units::clight / (3.0 * CalcPlanckOpacity(cell));
    }

    double CalcDiffusionCoefficient(const ComputationalCell3D &cell, double) const override
    {
        return CalcDiffusionCoefficient(cell);
    }

private:
    double PlanckOpacity(double density, double temperature) const
    {
        const double scaledTemperature = std::max(temperature / benchmark_.temperatureScale, 1.0e-12);
        return benchmark_.opacityCoefficient * std::pow(scaledTemperature, -benchmark_.alpha) * std::pow(density, benchmark_.lambda + 1.0);
    }

    Benchmark benchmark_;
};

std::vector<Vector3D> FibonacciShell(double radius, std::size_t count, double phase)
{
    std::vector<Vector3D> points;
    points.reserve(count);
    const double goldenAngle = pi * (3.0 - std::sqrt(5.0));
    for(std::size_t index = 0; index < count; ++index)
    {
        const double z = 1.0 - 2.0 * (static_cast<double>(index) + 0.5) / static_cast<double>(count);
        const double cylindricalRadius = std::sqrt(std::max(0.0, 1.0 - z * z));
        const double phi = goldenAngle * static_cast<double>(index) + phase;
        points.emplace_back(radius * cylindricalRadius * std::cos(phi), radius * cylindricalRadius * std::sin(phi), radius * z);
    }
    return points;
}

// The bath sits between the sphere surface and the outer boundary; bathZone is
// its thickness in units of the sphere radius.  A thicker bath carries more
// shells, so the thermostat layer that actually seeds packets is resolved by
// more cells rather than by a handful next to the surface.
double HalfBox(const Benchmark &benchmark, double bathZone)
{
    return (1.0 + bathZone) * benchmark.radius;
}

// The hottest state the material reaches is the surface temperature at the
// final time, so the mean free path evaluated there is the shortest photon
// mean free path that the heated region ever has.
double CharacteristicTemperature(const Benchmark &benchmark)
{
    return BathTemperature(benchmark, -1.0, 0.0);
}

double MeanFreePath(const Benchmark &benchmark, double radius, double temperature)
{
    const double density = Density(benchmark, std::min(radius, benchmark.radius));
    const double opacity = benchmark.opacityCoefficient * std::pow(temperature / benchmark.temperatureScale, -benchmark.alpha) *
                           std::pow(density, benchmark.lambda + 1.0);
    return 1.0 / opacity;
}

struct RadialPlan
{
    std::vector<double> centers;
    std::vector<double> edges;

    double Width(std::size_t shell) const
    {
        return edges[shell + 1] - edges[shell];
    }

    // Width of the cell that contains the given radius, used to size the time
    // step from the analytic front speed.
    double WidthAt(double radius) const
    {
        const std::vector<double>::const_iterator upper = std::upper_bound(edges.begin(), edges.end(), radius);
        const std::size_t shell = std::min<std::size_t>(centers.size() - 1,
                                                       static_cast<std::size_t>(std::max<std::ptrdiff_t>(0, upper - edges.begin() - 1)));
        return Width(shell);
    }
};

// A converging Marshak wave is a front whose width is a few mean free paths of
// the material just behind it.  Cells wider than that cannot represent the
// front: implicit Monte Carlo then teleports energy a full cell per emission
// and runs the wave too fast, while discrete diffusion sees a neighbour that is
// many decades more opaque and stalls it.  The radial spacing therefore tracks
// the local mean free path, which for these benchmarks follows the density
// power law, so the mesh refines wherever the material is optically dense.
RadialPlan PlanRadialShells(const Benchmark &benchmark, double cellsPerMeanFreePath, std::size_t minimumShells, std::size_t maximumShells)
{
    const double temperature = CharacteristicTemperature(benchmark);
    const double widestCell = benchmark.radius / static_cast<double>(minimumShells);
    const double narrowestCell = benchmark.radius / static_cast<double>(maximumShells);
    RadialPlan plan;
    plan.edges.push_back(0.0);
    double edge = 0.0;
    while(edge < benchmark.radius * (1.0 - 1.0e-12) && plan.centers.size() < maximumShells)
    {
        double width = std::clamp(MeanFreePath(benchmark, edge, temperature) / cellsPerMeanFreePath, narrowestCell, widestCell);
        for(std::size_t iteration = 0; iteration < 3; ++iteration)
        {
            const double candidate = std::min(MeanFreePath(benchmark, edge, temperature),
                                              MeanFreePath(benchmark, edge + width, temperature)) / cellsPerMeanFreePath;
            width = std::clamp(candidate, narrowestCell, widestCell);
        }
        width = std::min(width, benchmark.radius - edge);
        plan.centers.push_back(edge + 0.5 * width);
        edge += width;
        plan.edges.push_back(edge);
    }
    return plan;
}

// The bath region between the sphere and the cube is filled with a jittered
// Cartesian lattice whose sites sit at cell centres.  That keeps every
// generator half a lattice spacing away from the box faces, which the Monte
// Carlo transport requires: a generator flush against a box face produces
// sliver cells whose box-clipping faces cannot be resolved reliably.
std::vector<Vector3D> GenerateMeshPoints(const Benchmark &benchmark, const RadialPlan &plan, std::size_t angularPoints,
                                         std::size_t driverShells, std::size_t exteriorPoints, double bathGrading, double bathZone,
                                         std::uint64_t seed)
{
    std::vector<Vector3D> points;
    // The angular pattern is identical on every shell so that the generators
    // form radial columns; the benchmark is one dimensional in radius and this
    // keeps the Voronoi faces perpendicular to the radial direction.
    for(std::size_t shell = 0; shell < plan.centers.size(); ++shell)
    {
        std::vector<Vector3D> shellPoints = FibonacciShell(plan.centers[shell], angularPoints, 0.0);
        points.insert(points.end(), shellPoints.begin(), shellPoints.end());
    }

    const double halfBox = HalfBox(benchmark, bathZone);
    const double sphereFraction = (4.0 / 3.0) * pi * std::pow(benchmark.radius, 3.0) / std::pow(2.0 * halfBox, 3.0);
    const double retainedFraction = std::max(0.05, 1.0 - sphereFraction);
    const std::size_t sitesPerAxis = std::max<std::size_t>(8, static_cast<std::size_t>(std::ceil(std::cbrt(static_cast<double>(exteriorPoints) / retainedFraction))));
    const double latticeSpacing = 2.0 * halfBox / static_cast<double>(sitesPerAxis);

    // The thermostat shells and the innermost bath lattice sites all have to fit
    // in the gap between the sphere and the box, so the shell spacing cannot
    // simply follow the outermost radial cell: a coarse mesh would otherwise
    // push generators past the box faces and out of the tessellation.
    const double driverWidth = std::min(plan.Width(plan.centers.size() - 1),
                                        (halfBox - benchmark.radius) / static_cast<double>(driverShells + 2));
    double driverRadius = benchmark.radius;
    for(std::size_t shell = 0; shell < driverShells; ++shell)
    {
        driverRadius = benchmark.radius + (static_cast<double>(shell) + 0.5) * driverWidth;
        std::vector<Vector3D> shellPoints = FibonacciShell(driverRadius, angularPoints, 0.0);
        points.insert(points.end(), shellPoints.begin(), shellPoints.end());
    }

    // The thermostat shells are as thin as the outermost material cell, so on a
    // finely resolved mesh they leave the whole gap out to the first bath lattice
    // site unmeshed.  A bath cell spanning that gap borders a very large number of
    // surface cells, and the Voronoi face it shares with the next bath cell gets a
    // comparable number of vertices.  A grading factor above one bridges the gap
    // with shells whose width grows by that factor until it reaches the lattice
    // spacing, keeping the jump between adjacent cells bounded.
    double bathReferenceWidth = driverWidth;
    if(bathGrading > 1.0)
    {
        const double outermostShell = halfBox - 0.5 * latticeSpacing;
        double edge = benchmark.radius + static_cast<double>(driverShells) * driverWidth;
        while(bathReferenceWidth < latticeSpacing)
        {
            const double width = bathReferenceWidth * bathGrading;
            if(edge + width > outermostShell)
            {
                break;
            }
            std::vector<Vector3D> shellPoints = FibonacciShell(edge + 0.5 * width, angularPoints, 0.0);
            points.insert(points.end(), shellPoints.begin(), shellPoints.end());
            driverRadius = edge + 0.5 * width;
            edge += width;
            bathReferenceWidth = width;
        }
    }

    const double minimumBathRadius = std::max(driverRadius + 2.0 * bathReferenceWidth, driverRadius + 0.5 * latticeSpacing);
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<double> jitter(-0.05 * latticeSpacing, 0.05 * latticeSpacing);
    for(std::size_t i = 0; i < sitesPerAxis; ++i)
    {
        for(std::size_t j = 0; j < sitesPerAxis; ++j)
        {
            for(std::size_t k = 0; k < sitesPerAxis; ++k)
            {
                const Vector3D site(-halfBox + (static_cast<double>(i) + 0.5) * latticeSpacing + jitter(generator),
                                    -halfBox + (static_cast<double>(j) + 0.5) * latticeSpacing + jitter(generator),
                                    -halfBox + (static_cast<double>(k) + 0.5) * latticeSpacing + jitter(generator));
                if(abs(site) > minimumBathRadius)
                {
                    points.push_back(site);
                }
            }
        }
    }
    return points;
}

// RigidBoundary declares its box faces reflecting for DDMC but leaves the
// device behaviour at HostOnly, so the portable grey kernel hands a box-face
// crossing back to the host, loses the packet's track, and reports
// TransportError::NoIntersection.  The sphere never exercised that path: its
// box faces are the cube at r >= 1.25 R, far outside the radius where packets
// are born, so nothing ever reached them.  Every wedge cell has four lateral
// walls, so the walls have to reflect inside the kernel instead.
template<typename T, typename Grid>
class FrustumBoundary : public RigidBoundaryCondition<T, Grid>
{
public:
    explicit FrustumBoundary(const Grid &grid): RigidBoundaryCondition<T, Grid>(grid)
    {}

    STORM::DeviceBoundaryFaceBehavior getDeviceBoundaryFaceBehavior(std::size_t, std::size_t, std::size_t) const override
    {
        return STORM::DeviceBoundaryFaceBehavior::ReflectingRigid;
    }
};

// A one-dimensional debug geometry.  Instead of the full sphere the benchmark
// is solved inside a narrow square pyramid whose apex sits at the origin and
// whose axis is +z, so every radial shell carries exactly one cell.  The four
// lateral walls are registered as box faces, and MadVoro mirrors generators
// across box faces, so the walls act as symmetry planes: the wedge sees the
// same solution as the full sphere while costing one cell per shell instead of
// angular-points of them.  The mirrors are also what keeps the Delaunay
// tessellation well posed, since the generators themselves are collinear.
//
// A square pyramid whose half-angle about both transverse axes is alpha
// subtends Omega = 4 asin(sin^2 alpha); this inverts that for tan(alpha).
double ConeHalfTangent(double solidAngle)
{
    const double sinAlphaSquared = std::sin(0.25 * solidAngle);
    const double sinAlpha = std::sqrt(sinAlphaSquared);
    return sinAlpha / std::sqrt(1.0 - sinAlphaSquared);
}

// Cell volumes in the wedge follow (Omega/3) r^3 only up to the difference
// between the flat Voronoi cut perpendicular to the axis and the spherical cap
// it stands in for.  That error is O(alpha^2), which this reports so the run
// header can state how close the wedge is to a true spherical segment.
double ConeVolumeError(double tangent)
{
    return tangent * tangent;
}

// Vertex order follows the convention GetBoxNormals relies on: the normal is
// cross(v2 - v0, v1 - v0) and has to point out of the domain.  A positive
// innerRadius truncates the apex into a genuine frustum, which trades the
// needle-shaped cells nearest the origin for a reflecting inner cap.
std::vector<Face> BuildConeBoxFaces(double tangent, double innerRadius, double height)
{
    const double inner = tangent * innerRadius;
    const double outer = tangent * height;
    const std::array<double, 4> cornerX = {-1.0, 1.0, 1.0, -1.0};
    const std::array<double, 4> cornerY = {-1.0, -1.0, 1.0, 1.0};
    std::array<Vector3D, 4> top;
    std::array<Vector3D, 4> bottom;
    for(std::size_t corner = 0; corner < 4; ++corner)
    {
        top[corner] = Vector3D(cornerX[corner] * outer, cornerY[corner] * outer, height);
        bottom[corner] = Vector3D(cornerX[corner] * inner, cornerY[corner] * inner, innerRadius);
    }

    std::vector<Face> faces;
    Face outerCap;
    outerCap.vertices.push_back(top[0]);
    outerCap.vertices.push_back(top[3]);
    outerCap.vertices.push_back(top[2]);
    outerCap.vertices.push_back(top[1]);
    faces.push_back(outerCap);

    for(std::size_t corner = 0; corner < 4; ++corner)
    {
        const std::size_t next = (corner + 1) % 4;
        Face wall;
        if(innerRadius > 0.0)
        {
            wall.vertices.push_back(bottom[corner]);
            wall.vertices.push_back(top[corner]);
            wall.vertices.push_back(top[next]);
            wall.vertices.push_back(bottom[next]);
        }
        else
        {
            wall.vertices.push_back(Vector3D(0.0, 0.0, 0.0));
            wall.vertices.push_back(top[corner]);
            wall.vertices.push_back(top[next]);
        }
        faces.push_back(wall);
    }

    if(innerRadius > 0.0)
    {
        Face innerCap;
        innerCap.vertices.push_back(bottom[0]);
        innerCap.vertices.push_back(bottom[1]);
        innerCap.vertices.push_back(bottom[2]);
        innerCap.vertices.push_back(bottom[3]);
        faces.push_back(innerCap);
    }
    return faces;
}

// The wedge counterpart of GenerateMeshPoints.  Every generator sits on the
// axis, so the Voronoi faces between them are planes perpendicular to the
// radius and each cell is a frustum of the pyramid.  There is no bath lattice:
// the thermostat and bath are the same on-axis shells continued out to the cap.
std::vector<Vector3D> GenerateConeMeshPoints(const Benchmark &benchmark, const RadialPlan &plan, std::size_t driverShells,
                                             double bathGrading, double innerRadius, double height)
{
    std::vector<Vector3D> points;
    for(std::size_t shell = 0; shell < plan.centers.size(); ++shell)
    {
        if(plan.centers[shell] > innerRadius)
        {
            points.emplace_back(0.0, 0.0, plan.centers[shell]);
        }
    }

    const double driverWidth = std::min(plan.Width(plan.centers.size() - 1),
                                        (height - benchmark.radius) / static_cast<double>(driverShells + 2));
    double edge = benchmark.radius;
    for(std::size_t shell = 0; shell < driverShells; ++shell)
    {
        points.emplace_back(0.0, 0.0, edge + 0.5 * driverWidth);
        edge += driverWidth;
    }

    // Only the bath cells within ten mean free paths of the surface emit, so
    // the shells past the thermostat exist to carry the boundary, not to
    // resolve it; a grading factor keeps their count down on fine meshes.
    const double growth = bathGrading > 1.0 ? bathGrading : 1.0;
    double width = driverWidth;
    while(edge + width < height)
    {
        points.emplace_back(0.0, 0.0, edge + 0.5 * width);
        edge += width;
        width = std::min(width * growth, height - benchmark.radius);
    }
    return points;
}

struct ProfilePoint : public Serializable
{
    double radius;
    double volume;
    double density;
    double materialTemperature;
    double radiationTemperature;
    double materialEnergyDensity;
    double radiationEnergyDensity;

    ProfilePoint(): radius(0.0), volume(0.0), density(0.0), materialTemperature(0.0), radiationTemperature(0.0),
                    materialEnergyDensity(0.0), radiationEnergyDensity(0.0)
    {}

    std::size_t dump(Serializer *serializer) const override
    {
        std::size_t offset = 0;
        offset += serializer->insert(radius);
        offset += serializer->insert(volume);
        offset += serializer->insert(density);
        offset += serializer->insert(materialTemperature);
        offset += serializer->insert(radiationTemperature);
        offset += serializer->insert(materialEnergyDensity);
        offset += serializer->insert(radiationEnergyDensity);
        return offset;
    }

    std::size_t load(const Serializer *serializer, std::size_t offset) override
    {
        std::size_t read = 0;
        read += serializer->extract(radius, offset + read);
        read += serializer->extract(volume, offset + read);
        read += serializer->extract(density, offset + read);
        read += serializer->extract(materialTemperature, offset + read);
        read += serializer->extract(radiationTemperature, offset + read);
        read += serializer->extract(materialEnergyDensity, offset + read);
        read += serializer->extract(radiationEnergyDensity, offset + read);
        return read;
    }

    bool operator<(const ProfilePoint &other) const
    {
        return radius < other.radius;
    }
};

void SetThermalState(ComputationalCell3D &cell, Conserved3D &conserved, const IdealGas &eos, double temperature, double volume)
{
    cell.temperature = temperature;
    cell.internal_energy = eos.dT2e(cell.density, temperature, cell.tracers, ComputationalCell3D::tracerNames);
    cell.Erad = units::arad * std::pow(temperature, 4.0) / cell.density;
    cell.Erad_dt = 0.0;
    cell.Erad_dt_dt = 0.0;
    cell.Eg.resize(ENERGY_GROUPS_NUM);
    conserved.Eg.resize(ENERGY_GROUPS_NUM);
    for(std::size_t group = 0; group < ENERGY_GROUPS_NUM; ++group)
    {
        cell.Eg[group] = cell.Erad / static_cast<double>(ENERGY_GROUPS_NUM);
    }
    cell.velocity = Vector3D(0.0, 0.0, 0.0);
    cell.pressure = eos.de2p(cell.density, cell.internal_energy, cell.tracers, ComputationalCell3D::tracerNames);
    PrimitiveToConserved(cell, volume, conserved);
}

void SetBathThermalStateAndRadiation(const Tessellation3D &tess, std::vector<ComputationalCell3D> &cells,
                                     std::vector<Conserved3D> &extensives, RadiationIMC &physics,
                                     std::vector<Particle3D> &particles, const IdealGas &eos,
                                     const Benchmark &benchmark, double bathRadius, double temperature,
                                     std::size_t particlesPerCell)
{
    if(particlesPerCell == 0)
    {
        throw std::runtime_error("The bath radiation field requires at least one particle per cell");
    }

    // Bath density is pinned to the surface value, so the gray mean free path
    // is the same in every thermostat cell at the current bath temperature.
    // Packets are spawned only in the layer that can actually reach r = R:
    // cells farther than 10 mean free paths cannot contribute incoming flux.
    const double bathMeanFreePath = MeanFreePath(benchmark, bathRadius, temperature);
    const double particleRadius = bathRadius + 10.0 * bathMeanFreePath;

    // Remove the old census first: changing cell.Erad does not reweight the
    // Monte Carlo packets that already represent the radiation field.
    particles.erase(std::remove_if(particles.begin(), particles.end(),
                                   [&tess, bathRadius](const Particle3D &particle)
                                   {
                                       return particle.cellIndex < tess.GetPointNo() &&
                                              abs(tess.GetMeshPoint(particle.cellIndex)) >= bathRadius;
                                   }),
                    particles.end());

    const std::size_t localCells = tess.GetPointNo();
    for(std::size_t index = 0; index < localCells; ++index)
    {
        const double radius = abs(tess.GetMeshPoint(index));
        if(radius < bathRadius)
        {
            continue;
        }

        const double volume = tess.GetVolume(index);
        SetThermalState(cells[index], extensives[index], eos, temperature, volume);
        if(radius > particleRadius)
        {
            continue;
        }
        const double packetWeight = units::arad * std::pow(temperature, 4.0) * volume /
                                    static_cast<double>(particlesPerCell);
        for(std::size_t packet = 0; packet < particlesPerCell; ++packet)
        {
            // generateSingleParticle samples uniformly in the Voronoi cell and
            // samples an isotropic direction. This benchmark is gray, so the
            // Planck field is fully specified by its integrated a*T^4 energy.
            Particle3D photon = physics.generateSingleParticle(index, cells[index]);
            photon.weight = packetWeight;
            photon.initialWeight = packetWeight;
            particles.push_back(photon);
        }
    }
}

void WriteProfile(const std::string &path, double paperTimeNs, const Benchmark &benchmark, const Tessellation3D &tess,
                  const std::vector<ComputationalCell3D> &cells, const RadiationIMC &physics, rank_t rank)
{
    const std::vector<double> &radiationEnergy = physics.getEradTimeAvg();
    std::vector<ProfilePoint> local;
    const std::size_t localCells = tess.GetPointNo();
    if(cells.size() < localCells || radiationEnergy.size() < localCells)
    {
        throw std::runtime_error("Local profile arrays do not match the tessellation");
    }
    for(std::size_t index = 0; index < localCells; ++index)
    {
        const double radius = abs(tess.GetMeshPoint(index));
        if(radius >= benchmark.radius)
        {
            continue;
        }
        ProfilePoint point;
        point.radius = radius;
        point.volume = tess.GetVolume(index);
        point.density = cells[index].density;
        point.materialTemperature = cells[index].temperature;
        point.radiationEnergyDensity = std::max(radiationEnergy[index], 0.0);
        point.radiationTemperature = std::pow(point.radiationEnergyDensity / units::arad, 0.25);
        point.materialEnergyDensity = cells[index].density * cells[index].internal_energy;
        local.push_back(point);
    }

    std::vector<ProfilePoint> gathered = MPI_Gatherv_serializable(local, 0, MPI_COMM_WORLD);
    if(rank != 0)
    {
        return;
    }
    std::sort(gathered.begin(), gathered.end());
    std::ofstream output(path);
    if(!output)
    {
        throw std::runtime_error("Cannot open profile output " + path);
    }
    output << std::scientific << std::setprecision(12);
    output << "# benchmark=" << benchmark.number << " paper_time_ns=" << paperTimeNs << "\n";
    output << "radius_cm,volume_cm3,density_g_cm3,Tmat_K,Trad_K,umat_erg_cm3,erad_erg_cm3\n";
    for(const ProfilePoint &point : gathered)
    {
        output << point.radius << ',' << point.volume << ',' << point.density << ',' << point.materialTemperature << ','
               << point.radiationTemperature << ',' << point.materialEnergyDensity << ',' << point.radiationEnergyDensity << '\n';
    }
    std::cout << "Wrote " << path << " with " << gathered.size() << " interior cells" << std::endl;
}

std::string SnapshotPath(const std::string &outputDirectory, int benchmark, std::size_t snapshot)
{
    std::ostringstream stream;
    stream << outputDirectory << "/test" << benchmark << "_snapshot" << snapshot + 1 << ".csv";
    return stream.str();
}

bool DueEvery(std::size_t cycle, std::size_t interval)
{
    return interval > 0 && cycle > 0 && (cycle % interval) == 0;
}

void ReduceTransportDiagnostics(const RadiationIMC &physics,
                                const std::shared_ptr<MonteCarloManager3D> &manager,
                                unsigned long long *globalCounts)
{
    unsigned long long localCounts[6];
    localCounts[0] = physics.getDDMCStepCount();
    localCounts[1] = physics.getDDMCLeakCount();
    localCounts[2] = physics.getDDMCCensusCount();
    localCounts[3] = physics.getDDMCFallbackCount();
    localCounts[4] = physics.getRandomWalkStepCount();
    unsigned long long managerSteps = 0;
    const std::vector<std::size_t> &stepCounters = manager->GetCellsStepsCounters();
    for(std::size_t i = 0; i < stepCounters.size(); ++i)
    {
        managerSteps += stepCounters[i];
    }
    localCounts[5] = managerSteps;
    MPI_Reduce(localCounts, globalCounts, 6, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
}

std::string NumberedDumpPath(const std::string &directory, std::size_t cycle, const std::string &extension)
{
    std::ostringstream stream;
    stream << directory << "/cycle_" << std::setw(6) << std::setfill('0') << cycle << extension;
    return stream.str();
}

void WriteMarshakVTK(const Voronoi3D &tess, const std::vector<ComputationalCell3D> &cells, const RadiationIMC &physics, const std::string &path)
{
    const std::size_t localCells = tess.GetPointNo();
    std::vector<double> temperature(localCells);
    std::vector<double> density(localCells);
    std::vector<double> pressure(localCells);
    std::vector<double> radiationEnergyDensity(localCells);
    const std::vector<double> &radiationEnergy = physics.getEradTimeAvg();
    for(std::size_t index = 0; index < localCells; ++index)
    {
        temperature[index] = cells[index].temperature;
        density[index] = cells[index].density;
        pressure[index] = cells[index].pressure;
        radiationEnergyDensity[index] = (index < radiationEnergy.size()) ? radiationEnergy[index] : 0.0;
    }
    WriteVoronoiVTKOnly(tess, path, {temperature, density, pressure, radiationEnergyDensity},
                        {"temperature", "density", "pressure", "Erad_time_avg"});
}

}

int main(int argc, char *argv[])
{
    vtune_stop();
    DISABLE_TIMERS();
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);

    rank_t rank = 0;
    rank_t worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    try
    {
        ArgumentParser arguments("Converging Marshak waves in a three-dimensional sphere");
        arguments.addPositional<int>("test", "paper benchmark number (1-4)").required();
        arguments.addOption<double>("cells-per-mfp", 1.0, "radial cells per mean free path of the heated material");
        arguments.addOption<std::size_t>("min-shells", 64, "lower bound on the number of radial shells");
        arguments.addOption<std::size_t>("max-shells", 200, "upper bound on the number of radial shells");
        arguments.addOption<std::size_t>("angular-points", 2000, "generators per radial shell");
        arguments.addOption<std::size_t>("driver-shells", 10, "bath thermostat shells immediately outside the sphere");
        arguments.addOption<std::size_t>("exterior-points", 20000, "bath-region background generators");
        arguments.addOption<double>("bath-grading", 0.0, "growth factor of the shells bridging the sphere surface and the bath lattice; 0 disables the bridge");
        arguments.addOption<double>("bath-zone", 0.25, "thickness of the bath region outside the sphere, in units of the sphere radius");
        arguments.addOption<std::size_t>("new-photons", 15, "new IMC packets per cell per step");
        arguments.addOption<std::size_t>("population", 50, "target retained packets per cell");
        arguments.addOption<std::size_t>("boundary-packets", 1, "unused compatibility option");
        arguments.addOption<std::size_t>("steps", 100000, "hard cap on the number of time steps");
        arguments.addOption<double>("front-cells", 1.0, "maximum analytic front advance per step, in radial cells");
        arguments.addOption<double>("max-relative-step", 0.1, "maximum fractional decrease of the remaining time per step");
        arguments.addOption<double>("max-dt", 1.0e-11, "hard cap on the time step, in seconds");
        arguments.addOption<bool>("random-walk", true, "accelerate optically thick cells with random walk");
        // The Fleck factor in the heated material is of order 1e-5, so a packet
        // that fails the random walk test pays about 1e5 effective scatters one
        // collision at a time.  The stock thresholds (25 and 5) leave far too
        // many packets on that path for these opacities.
        arguments.addOption<double>("rw-cell-depth", 10.0, "minimum cell optical depth for a random walk step");
        arguments.addOption<double>("rw-particle-depth", 4.5, "minimum packet-to-face optical depth for a random walk step");
        // Discrete diffusion stalls these fronts: the leakage rate into an
        // unheated cell, whose optical depth is many orders of magnitude larger
        // than the heated cell behind the front, is diffusion limited and cannot
        // absorb the incident bath flux.  Plain IMC with random walk does.
        arguments.addOption<bool>("ddmc", false, "accelerate optically thick cells with discrete diffusion");
        arguments.addOption<std::uint64_t>("seed", 1847, "mesh and transport seed");
        arguments.addOption<std::string>("output-dir", "results", "profile output directory");
        arguments.addOption<std::string>("output", "", "directory for VTK and HDF5 dumps; omitted disables dumps");
        arguments.addOption<std::size_t>("hdf5-cycles", 500, "cycles between numbered WriteSimulation HDF5 dumps");
        arguments.addOption<std::size_t>("vtk-cycles", 200, "cycles between numbered VTK dumps");
        arguments.addOption<std::size_t>("latest-cycles", 200, "cycles between rolling latest.h5 checkpoints");
        arguments.addOption<std::string>("resume", "", "HDF5 checkpoint from WriteSimulation to resume from (same MPI rank count)");
        arguments.addOption<std::string>("manager", "new-rdma-auto", "Monte Carlo communication manager")
            .choices({"new-rdma-auto", "new-rdma-ibv", "legacy-rdma-auto", "p2p"});
        arguments.addFlag("frustum", "solve a narrow cone with one cell per radial shell instead of the full sphere")
            .flagAlias("cone", true);
        arguments.addOption<double>("cone-solid-angle", 1.0e-4, "solid angle of the frustum, in steradians");
        arguments.addOption<double>("cone-inner-radius", 0.0, "truncate the frustum apex at this radius; 0 keeps the apex");

        if(!arguments.parse(argc, argv))
        {
            if(rank == 0)
            {
                std::cout << arguments.help() << std::endl;
            }
            MPI_Finalize();
            return 0;
        }

        const Benchmark benchmark = GetBenchmark(arguments.get<int>("test"));
        const double cellsPerMeanFreePath = arguments.get<double>("cells-per-mfp");
        const std::size_t minimumShells = arguments.get<std::size_t>("min-shells");
        const std::size_t maximumShells = arguments.get<std::size_t>("max-shells");
        const std::size_t angularPoints = arguments.get<std::size_t>("angular-points");
        const std::size_t driverShells = arguments.get<std::size_t>("driver-shells");
        const std::size_t exteriorPoints = arguments.get<std::size_t>("exterior-points");
        const double bathGrading = arguments.get<double>("bath-grading");
        const double bathZone = arguments.get<double>("bath-zone");
        const std::size_t newPhotons = arguments.get<std::size_t>("new-photons");
        const std::size_t population = arguments.get<std::size_t>("population");
        const std::size_t targetSteps = arguments.get<std::size_t>("steps");
        const double frontCellsPerStep = arguments.get<double>("front-cells");
        const double maxRelativeStep = arguments.get<double>("max-relative-step");
        const double maxTimeStep = arguments.get<double>("max-dt");
        const bool withRandomWalk = arguments.get<bool>("random-walk");
        const double rwCellDepth = arguments.get<double>("rw-cell-depth");
        const double rwParticleDepth = arguments.get<double>("rw-particle-depth");
        const bool withDDMC = arguments.get<bool>("ddmc");
        const std::uint64_t seed = arguments.get<std::uint64_t>("seed");
        const std::string outputDirectory = arguments.get<std::string>("output-dir");
        const std::string checkpointDirectory = arguments.get<std::string>("output");
        const std::size_t hdf5Cycles = arguments.get<std::size_t>("hdf5-cycles");
        const std::size_t vtkCycles = arguments.get<std::size_t>("vtk-cycles");
        const std::size_t latestHdf5Cycles = arguments.get<std::size_t>("latest-cycles");
        const bool writeCheckpoints = arguments.wasSet("output") && !checkpointDirectory.empty();
        const std::string resumeFile = arguments.get<std::string>("resume");
        const bool resuming = arguments.wasSet("resume") && !resumeFile.empty();
        const std::string managerName = arguments.get<std::string>("manager");
        const bool coneMode = arguments.get<bool>("frustum");
        const double coneSolidAngle = arguments.get<double>("cone-solid-angle");
        const double coneInnerRadius = arguments.get<double>("cone-inner-radius");
        if(minimumShells < 4 || maximumShells < minimumShells || targetSteps < 10)
        {
            throw std::runtime_error("Use at least 4 radial shells, a consistent shell range, and 10 time steps");
        }
        if((angularPoints < 12 && !coneMode) || driverShells < 1 || !(cellsPerMeanFreePath > 0.0))
        {
            throw std::runtime_error("Use at least 12 angular points, one driver shell, and a positive cells per mean free path");
        }
        if(coneMode && !(coneSolidAngle > 0.0 && coneSolidAngle < 4.0 * pi))
        {
            throw std::runtime_error("Use a frustum solid angle in (0, 4 pi) steradians");
        }
        if(coneMode && !(coneInnerRadius >= 0.0 && coneInnerRadius < benchmark.radius))
        {
            throw std::runtime_error("Use a frustum inner radius between zero and the sphere radius");
        }
        if(!(frontCellsPerStep > 0.0) || !(maxRelativeStep > 0.0 && maxRelativeStep < 1.0))
        {
            throw std::runtime_error("Use a positive front advance per step and a relative step in (0, 1)");
        }
        if(!(maxTimeStep > 0.0))
        {
            throw std::runtime_error("Use a positive time step cap");
        }
        if(bathGrading != 0.0 && !(bathGrading > 1.0 && bathGrading <= 4.0))
        {
            throw std::runtime_error("Use a bath grading of 0 to disable the bridge shells, or a growth factor in (1, 4]");
        }
        if(!(bathZone > 0.0))
        {
            throw std::runtime_error("Use a positive bath zone thickness");
        }
        const RadialPlan radialPlan = PlanRadialShells(benchmark, cellsPerMeanFreePath, minimumShells, maximumShells);
        if(rank == 0)
        {
            if(!outputDirectory.empty())
            {
                std::filesystem::create_directories(outputDirectory);
            }
            if(writeCheckpoints)
            {
                std::filesystem::create_directories(checkpointDirectory);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        // RadiationMCStep::AUTO_RDMA is an alias for LEGACY_AUTO_RDMA, so naming it
        // selects MonteCarloManagerLegacy, whose progress counter is commented out.
        // That manager prints neither the ten second [Progress] line nor the
        // [StuckParticle] warning, which is exactly what one needs to tell a slow
        // transport sweep from a hung one.  NEW_RDMA is the manager that prints both.
        RadiationMCStep::ManagerType managerType = RadiationMCStep::ManagerType::NEW_RDMA;
        if(managerName == "new-rdma-ibv")
        {
            managerType = RadiationMCStep::ManagerType::NEW_IBV_RDMA;
        }
        else if(managerName == "legacy-rdma-auto")
        {
            managerType = RadiationMCStep::ManagerType::LEGACY_AUTO_RDMA;
        }
        else if(managerName == "p2p")
        {
            managerType = RadiationMCStep::ManagerType::P2P;
        }

        // The wedge keeps the same radial extent as the sphere run: its cap sits
        // where the cube face used to, so the thermostat and bath span the same
        // radii and the material region is resolved by the same radial plan.
        const double halfBox = HalfBox(benchmark, bathZone);
        const double coneTangent = coneMode ? ConeHalfTangent(coneSolidAngle) : 0.0;
        const double coneHalfWidth = coneTangent * halfBox;
        const Vector3D lowerLeft = coneMode ? Vector3D(-coneHalfWidth, -coneHalfWidth, coneInnerRadius)
                                            : Vector3D(-halfBox, -halfBox, -halfBox);
        const Vector3D upperRight = coneMode ? Vector3D(coneHalfWidth, coneHalfWidth, halfBox)
                                             : Vector3D(halfBox, halfBox, halfBox);
        std::vector<Vector3D> points;
        if(rank == 0)
        {
            points = coneMode ? GenerateConeMeshPoints(benchmark, radialPlan, driverShells, bathGrading, coneInnerRadius, halfBox)
                              : GenerateMeshPoints(benchmark, radialPlan, angularPoints, driverShells, exteriorPoints, bathGrading, bathZone, seed);
            std::cout << "Generated " << points.size() << " three-dimensional Voronoi generators on "
                      << radialPlan.centers.size() << " radial shells, cell width from "
                      << radialPlan.Width(0) << " to " << radialPlan.Width(radialPlan.centers.size() - 1)
                      << " cm, bath grading " << bathGrading << ", bath zone " << bathZone << " R, heated mean free path at the surface "
                      << MeanFreePath(benchmark, benchmark.radius, CharacteristicTemperature(benchmark)) << " cm" << std::endl;
            if(coneMode)
            {
                std::cout << "Frustum mode: solid angle " << coneSolidAngle << " sr, half-angle "
                          << std::atan(coneTangent) << " rad, one cell per radial shell, apex "
                          << (coneInnerRadius > 0.0 ? "truncated at " : "at the origin")
                          << (coneInnerRadius > 0.0 ? std::to_string(coneInnerRadius) + " cm" : std::string())
                          << ", spherical-segment volume error " << ConeVolumeError(coneTangent) << std::endl;
            }
        }
        points = MPI_Spread(points, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        // Voronoi3D has a constructor that takes box faces directly, but it
        // leaves the MPI Hilbert points manager bound to a degenerate zero-size
        // box.  Constructing from the bounding box and then replacing the faces
        // keeps that manager correct; BuildPartiallyParallel pushes the new
        // faces into the engine before it builds.
        Voronoi3D tess(lowerLeft, upperRight);
        if(coneMode)
        {
            tess.ModifyBoxFaces() = BuildConeBoxFaces(coneTangent, coneInnerRadius, halfBox);
#ifdef RICH_MPI
            // The wedge is one cell wide, so its generators differ only in z.
            // A Hilbert curve over a 200:1 needle bounding box wastes most of
            // its resolution on the two transverse axes; splitting on z alone
            // gives contiguous radial slabs, which is also what the radial
            // transport wants for locality.
            tess.PresetLoadBalancer(std::make_shared<OneDimensionalLoadBalancer<Vector3D>>(lowerLeft, upperRight, Axis::Z));
#endif
        }
        tess.BuildParallel(points);

        const double eosCoefficientKelvin = benchmark.eosCoefficient / std::pow(benchmark.temperatureScale, benchmark.beta);
        IdealGas eos(5.0 / 3.0, eosCoefficientKelvin, benchmark.beta, benchmark.mu);
        const double temperatureFloor = 1.0e-3 * benchmark.temperatureScale;
        const std::size_t localCells = tess.GetPointNo();
        std::vector<ComputationalCell3D> initialCells(localCells);
        for(std::size_t index = 0; index < localCells; ++index)
        {
            ComputationalCell3D &cell = initialCells[index];
            const double radius = abs(tess.GetMeshPoint(index));
            cell.density = Density(benchmark, std::min(radius, benchmark.radius));
            cell.temperature = temperatureFloor;
            cell.velocity = Vector3D(0.0, 0.0, 0.0);
            cell.internal_energy = eos.dT2e(cell.density, cell.temperature, cell.tracers, ComputationalCell3D::tracerNames);
            cell.pressure = eos.de2p(cell.density, cell.internal_energy, cell.tracers, ComputationalCell3D::tracerNames);
            cell.Erad = units::arad * std::pow(temperatureFloor, 4.0) / cell.density;
            cell.Erad_dt = 0.0;
            cell.Erad_dt_dt = 0.0;
            cell.Eg.resize(ENERGY_GROUPS_NUM);
            for(std::size_t group = 0; group < ENERGY_GROUPS_NUM; ++group)
            {
                cell.Eg[group] = cell.Erad / static_cast<double>(ENERGY_GROUPS_NUM);
            }
        }

        Simulation simulation(tess, initialCells, eos);
        std::shared_ptr<TimeStepFunction3D> timeStepFunction = std::make_shared<ManualTimeStep>();
        simulation.SetTimeStepFunction(timeStepFunction);
        std::vector<ComputationalCell3D> &cells = simulation.getCells();
        std::vector<Conserved3D> &extensives = simulation.getExtensives();
        extensives.resize(tess.GetPointNo());
        for(std::size_t index = 0; index < tess.GetPointNo(); ++index)
        {
            extensives[index].Eg.resize(ENERGY_GROUPS_NUM);
            PrimitiveToConserved(cells[index], tess.GetVolume(index), extensives[index]);
        }

        std::shared_ptr<IdealGas> eosPointer = std::make_shared<IdealGas>(eos);
        std::shared_ptr<BenchmarkOpacity> opacity = std::make_shared<BenchmarkOpacity>(benchmark);
        std::shared_ptr<BoundaryCondition<Vector3D, Tessellation3D>> boundary;
        if(coneMode)
        {
            boundary = std::make_shared<FrustumBoundary<Vector3D, Tessellation3D>>(tess);
        }
        else
        {
            boundary = std::make_shared<RigidBoundaryCondition<Vector3D, Tessellation3D>>(tess);
        }

        RadiationIMCParameters parameters;
        parameters.newPhotonsPerCell = newPhotons;
        parameters.withHydro = false;
        parameters.withRandomWalk = withRandomWalk;
        parameters.rwMinCellOpticalDepth = rwCellDepth;
        parameters.rwMinParticleOpticalDepth = rwParticleDepth;
        parameters.withDDMC = withDDMC;
        parameters.energyBoundaries[0] = 0.0;
        for(std::size_t group = 1; group <= ENERGY_GROUPS_NUM; ++group)
        {
            parameters.energyBoundaries[group] = 1.0e30 * static_cast<double>(group) / static_cast<double>(ENERGY_GROUPS_NUM);
        }
        parameters.energyBoundariesProvided = true;
        std::shared_ptr<RadiationIMC> physics = std::make_shared<RadiationIMC>(tess, boundary, cells, extensives, eosPointer, opacity, parameters);
        physics->reseedRNG(seed + 104729ULL * static_cast<std::uint64_t>(rank));
        std::shared_ptr<PopulationControl<Vector3D, Tessellation3D>> populationControl = std::make_shared<STORM::CombPopulationControl<Vector3D, Tessellation3D>>(tess, population);
        std::vector<MonteCarloParticle<Vector3D>> initialParticles;
        std::shared_ptr<RadiationMCStep> radiationStep = std::make_shared<RadiationMCStep>(tess, cells, extensives, physics, populationControl, boundary, initialParticles, 0, false, managerType);
#ifdef RICH_MPI
        radiationStep->setCost(std::make_shared<IMCCostCalculator>(radiationStep->getManager()));
#endif
        simulation.addPhysics(radiationStep);

        const double initialPaperNs = InitialPaperTimeNs(benchmark);
        const double endPaperNs = -1.0;
        const double totalPaperSpan = endPaperNs - initialPaperNs;
        double paperTime = initialPaperNs;
        std::size_t nextSnapshot = 0;
        std::size_t cycle = 0;
        if(resuming)
        {
            int resumeExists = 0;
            if(rank == 0)
            {
                resumeExists = std::filesystem::exists(resumeFile) ? 1 : 0;
            }
            MPI_Bcast(&resumeExists, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if(resumeExists == 0)
            {
                throw std::runtime_error("resume file not found: " + resumeFile);
            }
            ReadSimulation(resumeFile, simulation);
            cycle = simulation.GetCycle();
            paperTime = initialPaperNs + simulation.GetTime() / ns;
            while(nextSnapshot < benchmark.snapshotPaperTimesNs.size() &&
                  paperTime >= benchmark.snapshotPaperTimesNs[nextSnapshot] * (1.0 + 1.0e-12))
            {
                ++nextSnapshot;
            }
            if(rank == 0)
            {
                std::cout << "Resumed from " << resumeFile << ": cycle=" << cycle
                          << ", paper t=" << paperTime << " ns, tracker_time=" << simulation.GetTime() << " s";
                if(nextSnapshot < benchmark.snapshotPaperTimesNs.size())
                {
                    std::cout << ", next paper snapshot=" << nextSnapshot + 1;
                }
                else
                {
                    std::cout << ", all paper snapshots already passed";
                }
                std::cout << std::endl;
            }
        }

        // Step size follows the analytic front advance (front-cells) and the
        // remaining paper time.  RadiationMCStep still prints a suggested dt
        // each cycle for diagnostics, but the driver does not follow it: at
        // low packet counts that suggestion collapses from Monte Carlo noise.
        const std::chrono::high_resolution_clock::time_point wallStart = std::chrono::high_resolution_clock::now();
        const double etaPaperStart = paperTime;

        if(rank == 0)
        {
            std::cout << std::setprecision(10)
                      << "Converging Marshak test " << benchmark.number << ": R=" << benchmark.radius << " cm, paper t=["
                      << initialPaperNs << ", " << endPaperNs << "] ns, ranks=" << worldSize
                      << ", radial shells=" << radialPlan.centers.size()
                      << (coneMode ? ", frustum " : ", angular points=")
                      << (coneMode ? std::to_string(coneSolidAngle) + " sr (1 angular cell)" : std::to_string(angularPoints))
                      << ", local cells(rank 0)=" << localCells << ", new photons/cell=" << newPhotons
                      << ", front cells per step=" << frontCellsPerStep << ", max relative step=" << maxRelativeStep
                      << ", max dt=" << maxTimeStep << " s"
                      << ", step cap=" << targetSteps << std::endl;
        }

        const std::string latestH5Path = writeCheckpoints ? (std::filesystem::path(checkpointDirectory) / "latest.h5").string() : "";
        const std::string initH5Path = writeCheckpoints ? (std::filesystem::path(checkpointDirectory) / "init.h5").string() : "";
        const std::string initVtkPath = writeCheckpoints ? (std::filesystem::path(checkpointDirectory) / "init.pvtu").string() : "";
        const std::string finalH5Path = writeCheckpoints ? (std::filesystem::path(checkpointDirectory) / "final.h5").string() : "";
        const std::string finalVtkPath = writeCheckpoints ? (std::filesystem::path(checkpointDirectory) / "final.pvtu").string() : "";
        const auto writeBookend = [&](const std::string &h5Path, const std::string &vtkPath)
        {
            if(!writeCheckpoints)
            {
                return;
            }
            WriteSimulation(simulation, h5Path);
            WriteMarshakVTK(tess, cells, *physics, vtkPath);
            MPI_Barrier(MPI_COMM_WORLD);
            if(rank == 0)
            {
                std::cout << "Wrote " << h5Path << " and " << vtkPath << std::endl;
            }
        };
        const auto dumpCheckpoints = [&](std::size_t dumpCycle, bool forceLatest)
        {
            if(!writeCheckpoints)
            {
                return;
            }
            const bool writeLatest = forceLatest || DueEvery(dumpCycle, latestHdf5Cycles);
            const bool writeHdf5 = DueEvery(dumpCycle, hdf5Cycles);
            const bool writeVtk = DueEvery(dumpCycle, vtkCycles);
            if(!writeLatest && !writeHdf5 && !writeVtk)
            {
                return;
            }
            if(writeLatest)
            {
                WriteSimulation(simulation, latestH5Path);
            }
            if(writeHdf5)
            {
                WriteSimulation(simulation, NumberedDumpPath(checkpointDirectory, dumpCycle, ".h5"));
            }
            if(writeVtk)
            {
                WriteMarshakVTK(tess, cells, *physics, NumberedDumpPath(checkpointDirectory, dumpCycle, ".pvtu"));
            }
            MPI_Barrier(MPI_COMM_WORLD);
            if(rank == 0)
            {
                bool wroteSomething = false;
                if(writeLatest)
                {
                    std::cout << "Wrote " << latestH5Path;
                    wroteSomething = true;
                }
                if(writeHdf5)
                {
                    std::cout << (wroteSomething ? " and HDF5 " : "Wrote HDF5 ") << NumberedDumpPath(checkpointDirectory, dumpCycle, ".h5");
                    wroteSomething = true;
                }
                if(writeVtk)
                {
                    std::cout << (wroteSomething ? " and VTK " : "Wrote VTK ") << NumberedDumpPath(checkpointDirectory, dumpCycle, ".pvtu");
                }
                std::cout << std::endl;
            }
        };

        if(!resuming)
        {
            writeBookend(initH5Path, initVtkPath);
        }

        while(paperTime < endPaperNs * (1.0 + 1.0e-13) && cycle < targetSteps)
        {
            const double frontRadius = FrontRadius(benchmark, paperTime);
            const double frontLimitedTime = PaperTimeFromFrontRadius(benchmark, std::max(frontRadius - frontCellsPerStep * radialPlan.WidthAt(frontRadius), 0.1 * benchmark.radius));
            const double dtLimit = std::min(maxTimeStep, std::min((frontLimitedTime - paperTime) * ns, -maxRelativeStep * paperTime * ns));
            double targetPaperTime = endPaperNs;
            if(nextSnapshot < benchmark.snapshotPaperTimesNs.size())
            {
                targetPaperTime = std::min(targetPaperTime, benchmark.snapshotPaperTimesNs[nextSnapshot]);
            }
            const double dt = std::min(dtLimit, (targetPaperTime - paperTime) * ns);
            const double nextPaperTime = paperTime + dt / ns;
            if(!(dt > 0.0))
            {
                throw std::runtime_error("The similarity time step stopped advancing");
            }

            const double bathTemperature = BathTemperature(benchmark, 0.5 * (paperTime + nextPaperTime), temperatureFloor);
            if(cells.size() < tess.GetPointNo() or extensives.size() < tess.GetPointNo())
            {
                throw std::runtime_error("Local state arrays do not match the tessellation after load balancing");
            }
            SetBathThermalStateAndRadiation(tess, cells, extensives, *physics, radiationStep->getParticles(), eos,
                                            benchmark, benchmark.radius, bathTemperature, population);

            simulation.SetTimeStep(dt);
            simulation.step();
            paperTime = nextPaperTime;
            ++cycle;
            dumpCheckpoints(cycle, false);

            if(nextSnapshot < benchmark.snapshotPaperTimesNs.size() &&
               paperTime >= benchmark.snapshotPaperTimesNs[nextSnapshot] * (1.0 + 1.0e-12))
            {
                WriteProfile(SnapshotPath(outputDirectory, benchmark.number, nextSnapshot), benchmark.snapshotPaperTimesNs[nextSnapshot],
                             benchmark, tess, cells, *physics, rank);
                ++nextSnapshot;
            }

            if(cycle % 10 == 0 or paperTime >= endPaperNs * (1.0 + 1.0e-13))
            {
                unsigned long long transportCounts[6];
                ReduceTransportDiagnostics(*physics, radiationStep->getManager(), transportCounts);
                if(rank == 0)
                {
                    const double fraction = (paperTime - initialPaperNs) / totalPaperSpan;
                    const double progressed = paperTime - etaPaperStart;
                    const double remaining = endPaperNs - paperTime;
                    const double wallSeconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - wallStart).count();
                    const double eta = progressed > 0.0 ? wallSeconds * remaining / progressed : 0.0;
                    const unsigned long long ddmcSteps = transportCounts[0];
                    const unsigned long long managerSteps = transportCounts[5];
                    const double ddmcFraction = managerSteps > 0 ? static_cast<double>(ddmcSteps) / static_cast<double>(managerSteps) : 0.0;
                    std::cout << "Cycle " << cycle << "  " << static_cast<int>(100.0 * fraction) << "%  paper t=" << paperTime
                              << " ns  dt=" << dt << " s  front=" << frontRadius << " cm  bath="
                              << bathTemperature / benchmark.temperatureScale
                              << (benchmark.number == 4 ? " keV" : " HeV")
                              << "  bath_mfp=" << MeanFreePath(benchmark, benchmark.radius, bathTemperature)
                              << " cm  ETA=" << eta / 60.0 << " min"
                              << "  ddmc_steps=" << ddmcSteps
                              << "  ddmc_leaks=" << transportCounts[1]
                              << "  ddmc_census=" << transportCounts[2]
                              << "  ddmc_fallback=" << transportCounts[3]
                              << "  rw_steps=" << transportCounts[4]
                              << "  manager_steps=" << managerSteps
                              << "  ddmc_frac=" << ddmcFraction << std::endl;
                }
            }
        }
        if(nextSnapshot < benchmark.snapshotPaperTimesNs.size())
        {
            throw std::runtime_error("The run ended before every paper profile time was reached; raise the step cap");
        }
        dumpCheckpoints(cycle, true);
        writeBookend(finalH5Path, finalVtkPath);

        if(rank == 0)
        {
            const double wallSeconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - wallStart).count();
            std::cout << "Completed test " << benchmark.number << " in " << wallSeconds << " s and " << cycle << " steps" << std::endl;
        }
    }
    catch(const UniversalError &error)
    {
        std::cerr << "UniversalError on rank " << rank << std::endl;
        reportError(error, std::cerr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    // MadVoroException does not derive from std::exception, so without this the
    // tessellation errors reach terminate() and their diagnostics are lost.
    catch(const MadVoro::Exception::MadVoroException &error)
    {
        std::cerr << "MadVoroException on rank " << rank << std::endl;
        MadVoro::Exception::reportError(error, std::cerr);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    catch(const std::exception &error)
    {
        std::cerr << "Error on rank " << rank << ": " << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Finalize();
    return 0;
}
