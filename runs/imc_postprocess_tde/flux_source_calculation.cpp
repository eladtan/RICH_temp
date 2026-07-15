#include "flux_source_calculation.hpp"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <tuple>
#include <utility>
#include <vector>

#include "source/3D/monte/MonteCarloManager3D.hpp"
#include "source/Radiation/Diffusion.hpp"
#include "source/misc/universal_error.hpp"
#include "source/misc/utils.hpp"
#include "source/monte/MonteCarloFunctionality.hpp"
#include "source/monte/MonteCarloParticleStatus.hpp"
#include "source/monte/boundary/Vacuum.hpp"
#include "source/monte/population/NoControl.hpp"
#include "source/monte/physics/MonteCarloPhysics.hpp"

#ifdef RICH_MPI
#include <mpi.h>
#include "source/mpi/mpi_commands.hpp"
#endif

namespace imc_postprocess_tde {
namespace {

double EffectiveThermalizationOpacity(
    OpacityCalculator const& opacity,
    ComputationalCell3D const& cell)
{
    double const absorption = opacity.CalcPlanckOpacity(cell);
    double const scattering = opacity.CalcScatteringOpacity(cell);
    if(!std::isfinite(absorption) || !std::isfinite(scattering) ||
       absorption < 0.0 || scattering < 0.0)
        throw UniversalError(
            "Flux-source thermalization probe encountered invalid opacity");
    return std::sqrt(std::max(
        0.0, 3.0 * absorption * (absorption + scattering)));
}

class GreyThermalizationProbePhysics
    : public MonteCarloPhysics<Vector3D, Tessellation3D>
{
public:
    using Particle = MonteCarloParticle<Vector3D, Tessellation3D>;
    using Functionality = MonteCarloFunctionality<Vector3D, Tessellation3D>;
    using BoundaryCond = BoundaryCondition<Vector3D, Tessellation3D>;

    GreyThermalizationProbePhysics(
        Tessellation3D const& grid,
        std::shared_ptr<BoundaryCond> const& boundary,
        std::vector<ComputationalCell3D> const& cells,
        OpacityCalculator const& opacity,
        Vector3D center,
        double targetTau,
        size_t observerCount)
        : MonteCarloPhysics<Vector3D, Tessellation3D>(grid, boundary),
          cells_(cells), opacity_(opacity), center_(center),
          targetTau_(targetTau), radius_(observerCount, -1.0),
          valid_(observerCount, 0)
    {}

    std::vector<Particle> preStep(double) override { return {}; }
    void postStep(std::vector<Particle> const&, double) override {}

    Functionality step(Particle& particle, std::vector<Particle>&) override
    {
        Functionality result;
        result.change = MonteCarloParticleStatus::REMOVE;
        if(particle.id >= radius_.size() || particle.cellIndex >= cells_.size())
            return result;

        double const directionNorm = abs(particle.velocity);
        if(!(directionNorm > 0.0) || !std::isfinite(directionNorm))
            return result;
        Vector3D const direction = particle.velocity * (1.0 / directionNorm);
        particle.velocity = direction;

        auto const intersection = this->getIntersectionDetails(particle);
        double const ds = std::get<1>(intersection);
        size_t const nextCell = std::get<2>(intersection);
        if(!(ds >= 0.0) || !std::isfinite(ds))
            return result;

        double const sigma = EffectiveThermalizationOpacity(
            opacity_, cells_[particle.cellIndex]);
        double const oldTau = particle.weight;
        double const newTau = oldTau + sigma * ds;
        if(oldTau < targetTau_ && newTau >= targetTau_)
        {
            double fraction = (sigma > 0.0 && ds > 0.0)
                ? (targetTau_ - oldTau) / (sigma * ds) : 0.0;
            fraction = std::clamp(fraction, 0.0, 1.0);
            Vector3D const crossing = particle.location
                + direction * (fraction * ds);
            radius_[particle.id] = fastabs(crossing - center_);
            valid_[particle.id] = 1;
            return result;
        }

        particle.weight = newTau;
        particle.location += direction * ds;
        particle.timeLeft -= ds;
        if(this->grid.IsPointOutsideBox(nextCell))
            return result;
        result.change = MonteCarloParticleStatus::CELL_MOVE;
        result.nextCellIndex = nextCell;
        return result;
    }

    std::vector<double> const& radius() const { return radius_; }
    std::vector<int> const& valid() const { return valid_; }

private:
    std::vector<ComputationalCell3D> const& cells_;
    OpacityCalculator const& opacity_;
    Vector3D center_;
    double targetTau_;
    std::vector<double> radius_;
    std::vector<int> valid_;
};

size_t NearestObserverDirection(
    Vector3D const& point,
    Vector3D const& center,
    std::vector<Vector3D> const& directions)
{
    Vector3D radial = point - center;
    double const radius = abs(radial);
    if(!(radius > 0.0) || directions.empty())
        return 0;
    radial *= 1.0 / radius;
    size_t best = 0;
    double bestDot = -std::numeric_limits<double>::infinity();
    for(size_t i = 0; i < directions.size(); ++i)
    {
        double const dot = ScalarProd(radial, directions[i]);
        if(dot > bestDot)
        {
            bestDot = dot;
            best = i;
        }
    }
    return best;
}

std::vector<unsigned char> BuildOutsideSurfaceMask(
    Config const& cfg,
    PostprocessRuntime const& runtime)
{
    size_t const pointCount = runtime.tess.getMeshPoints().size();
    std::vector<unsigned char> outside(pointCount, 0);
    std::vector<Vector3D> const& directions = runtime.fluxSourceDirections;
    if(directions.empty() || directions.size() != runtime.fluxSourceRadius.size())
        throw UniversalError("Flux-source directions are unavailable");
    for(size_t pointIndex = 0; pointIndex < pointCount; ++pointIndex)
    {
        Vector3D const point = runtime.tess.GetMeshPoint(pointIndex);
        double const radius = fastabs(point - cfg.center);
        if(!(radius > 0.0))
            continue;
        size_t const direction = NearestObserverDirection(
            point, cfg.center, directions);
        if(direction >= runtime.fluxSourceRadius.size() ||
           direction >= runtime.fluxSourceRadiusDirectlyResolved.size())
            throw UniversalError("Flux-source angular surface index is out of range");
        // A direction that never reaches the requested tau_eff has no CER.
        // Treat that angular channel as open instead of inventing a surface
        // radius from a neighboring ray.
        if(runtime.fluxSourceRadiusDirectlyResolved[direction] == 0)
            outside[pointIndex] = 1;
        else
            outside[pointIndex] =
                radius >= runtime.fluxSourceRadius[direction] ? 1 : 0;
    }
    return outside;
}

std::vector<Vector3D> ComputeGreyFldFlux(PostprocessRuntime& runtime)
{
    size_t const nCells = runtime.tess.GetPointNo();
    std::vector<double> radiationEnergy(nCells, 0.0);
    std::vector<double> diffusionCoefficient(nCells, 0.0);
    double localRadiationEnergy = 0.0;
    for(size_t i = 0; i < nCells; ++i)
    {
        radiationEnergy[i] = runtime.cells[i].density * runtime.cells[i].Erad;
        diffusionCoefficient[i] =
            runtime.greyOpacity->CalcDiffusionCoefficient(runtime.cells[i]);
        if(!std::isfinite(radiationEnergy[i]) || radiationEnergy[i] < 0.0 ||
           !std::isfinite(diffusionCoefficient[i]) ||
           diffusionCoefficient[i] < 0.0)
        {
            UniversalError eo("Flux-source FLD reconstruction encountered invalid cell data");
            eo.addEntry("Cell index", i);
            eo.addEntry("Radiation energy density", radiationEnergy[i]);
            eo.addEntry("Diffusion coefficient", diffusionCoefficient[i]);
            throw eo;
        }
        localRadiationEnergy += radiationEnergy[i] * runtime.tess.GetVolume(i);
    }

    double globalRadiationEnergy = localRadiationEnergy;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &globalRadiationEnergy, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_exchange_data(runtime.tess, radiationEnergy, true);
#endif
    if(!(globalRadiationEnergy > 0.0) || !std::isfinite(globalRadiationEnergy))
        throw UniversalError(
            "Flux-source comparison requires positive snapshot radiation energy (Erad)");

    std::vector<Vector3D> gradient(nCells, Vector3D(0.0, 0.0, 0.0));
    std::vector<size_t> neighbors;
    for(size_t i = 0; i < nCells; ++i)
    {
        auto const& faces = runtime.tess.GetCellFaces(i);
        runtime.tess.GetNeighbors(i, neighbors);
        if(faces.size() != neighbors.size())
            throw UniversalError("Flux-source FLD face/neighbor count mismatch");
        Vector3D grad(0.0, 0.0, 0.0);
        for(size_t j = 0; j < neighbors.size(); ++j)
        {
            size_t const neighbor = neighbors[j];
            if(runtime.tess.IsPointOutsideBox(neighbor) ||
               neighbor >= radiationEnergy.size())
                continue;
            Vector3D separation = runtime.tess.GetMeshPoint(neighbor)
                - runtime.tess.GetMeshPoint(i);
            double const distance = fastabs(separation);
            if(!(distance > 0.0) || !std::isfinite(distance))
                continue;
            Vector3D const normal = separation * (1.0 / distance);
            double const faceEnergy =
                0.5 * (radiationEnergy[i] + radiationEnergy[neighbor]);
            double const area = runtime.tess.GetArea(faces[j]);
            if(!(area > 0.0) || !std::isfinite(area))
                continue;
            grad += normal * (area * faceEnergy);
        }
        double const volume = runtime.tess.GetVolume(i);
        if(volume > 0.0 && std::isfinite(volume))
            grad *= 1.0 / volume;
        gradient[i] = grad;
    }

    std::vector<Vector3D> flux(nCells, Vector3D(0.0, 0.0, 0.0));
    for(size_t i = 0; i < nCells; ++i)
    {
        double const limiter = CG::CalcSingleFluxLimiter(
            gradient[i], diffusionCoefficient[i], radiationEnergy[i]);
        flux[i] = gradient[i] * (-limiter * diffusionCoefficient[i]);
        if(!std::isfinite(flux[i].x) || !std::isfinite(flux[i].y) ||
           !std::isfinite(flux[i].z))
            throw UniversalError("Flux-source FLD reconstruction produced non-finite flux");
    }
#ifdef RICH_MPI
    MPI_exchange_data(runtime.tess, flux, true);
#endif
    return flux;
}

} // namespace

void InitializeFluxSourceSurface(
    Config const& cfg,
    PostprocessRuntime& runtime)
{
    size_t const nObservers = runtime.observer->getNumObservers();
    if(nObservers == 0 || nObservers > static_cast<size_t>(INT_MAX))
        throw UniversalError("Flux-source surface requires a valid observer count");

    std::vector<Vector3D> const& directions = runtime.observer->getDirections();
    if(directions.size() != nObservers)
        throw UniversalError("Flux-source observer direction count mismatch");
    runtime.fluxSourceDirections = directions;
    std::vector<GreyThermalizationProbePhysics::Particle> particles;
    particles.reserve(nObservers / std::max(1, runtime.mpiSize) + 1);

    for(size_t observerIndex = 0; observerIndex < nObservers; ++observerIndex)
    {
        Vector3D direction = directions[observerIndex];
        double const directionNorm = abs(direction);
        if(!(directionNorm > 0.0))
            continue;
        direction *= 1.0 / directionNorm;
        Vector3D const spherePoint = cfg.center + cfg.radius * direction;

        bool local = false;
#ifdef RICH_MPI
        if(!runtime.tess.IsPointOutsideBox(spherePoint))
            local = runtime.tess.GetOwner(spherePoint) == runtime.rank;
#else
        local = !runtime.tess.IsPointOutsideBox(spherePoint);
#endif
        if(!local)
            continue;

        size_t const cellIndex = runtime.tess.GetContainingCell(spherePoint);
        if(cellIndex >= runtime.tess.GetPointNo())
            continue;
        GreyThermalizationProbePhysics::Particle particle;
        particle.id = observerIndex;
        particle.location = spherePoint;
        particle.velocity = -1.0 * direction;
        particle.cellIndex = cellIndex;
        particle.cellID = runtime.cells[cellIndex].ID;
        particle.sourceCellID = particle.cellID;
        particle.weight = 0.0;
        particle.initialWeight = 0.0;
        particle.frequency = 0.0;
        particles.push_back(particle);
    }

    auto boundary = std::make_shared<
        VacuumBoundaryCondition<Vector3D, Tessellation3D>>(runtime.tess);
    auto physics = std::make_shared<GreyThermalizationProbePhysics>(
        runtime.tess, boundary, runtime.cells, *runtime.greyOpacity,
        cfg.center, cfg.fluxSourceThermalizationTau, nObservers);
    auto population = std::make_shared<
        NoPopulationControl<Vector3D, Tessellation3D>>(runtime.tess);
    std::shared_ptr<MonteCarloManager3D> manager;
#ifdef RICH_MPI
    manager = std::make_shared<RDMAMonteCarloManager3D>(
        runtime.tess, physics, population, boundary);
#else
    manager = std::make_shared<MonteCarloManagerSerial3D>(
        runtime.tess, physics, population, boundary);
#endif
    (void)manager->step(
        std::move(particles), runtime.cells, 2.01 * cfg.radius);

    runtime.fluxSourceRadius = physics->radius();
    runtime.fluxSourceRadiusDirectlyResolved = physics->valid();
#ifdef RICH_MPI
    if(!runtime.fluxSourceRadius.empty())
    {
        MPI_Allreduce(MPI_IN_PLACE, runtime.fluxSourceRadius.data(),
                      static_cast<int>(runtime.fluxSourceRadius.size()),
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE,
                      runtime.fluxSourceRadiusDirectlyResolved.data(),
                      static_cast<int>(runtime.fluxSourceRadiusDirectlyResolved.size()),
                      MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    }
#endif

    size_t const directCount = static_cast<size_t>(std::count(
        runtime.fluxSourceRadiusDirectlyResolved.begin(),
        runtime.fluxSourceRadiusDirectlyResolved.end(), 1));
    if(directCount == 0)
        throw UniversalError(
            "No inward grey rays reached the requested flux-source optical depth");

    for(size_t i = 0; i < nObservers; ++i)
    {
        if(runtime.fluxSourceRadiusDirectlyResolved[i] == 0)
            continue;
        double const radius = runtime.fluxSourceRadius[i];
        if(!(radius > 0.0) || !(radius < cfg.radius) ||
           !std::isfinite(radius))
            throw UniversalError("Flux-source ray returned an invalid radius");
    }

    runtime.fluxSourceEnabled = true;
    runtime.fluxSourceTau = cfg.fluxSourceThermalizationTau;
    runtime.fluxSourceDirectlyResolvedFraction =
        static_cast<double>(directCount) / static_cast<double>(nObservers);
    if(runtime.rank == 0)
        std::cout << "FLUX_SOURCE_SURFACE tau_eff="
                  << runtime.fluxSourceTau
                  << " directly_resolved=" << directCount << "/"
                  << nObservers
                  << " fraction="
                  << runtime.fluxSourceDirectlyResolvedFraction
                  << std::endl;
}

void ConfigureFluxSourceForCurrentDecomposition(
    Config const& cfg,
    PostprocessRuntime& runtime,
    RadiationIMC& physics)
{
    if(!cfg.fluxSourceCompare)
    {
        physics.clearPostProcessExternalSources();
        return;
    }
    if(!runtime.fluxSourceEnabled || runtime.fluxSourceRadius.empty())
        throw UniversalError("Flux-source surface was not initialized");

    std::vector<Vector3D> const fldFlux = ComputeGreyFldFlux(runtime);
    std::vector<unsigned char> const outside =
        BuildOutsideSurfaceMask(cfg, runtime);
    size_t const nCells = runtime.tess.GetPointNo();
    std::vector<RadiationIMC::PostProcessExternalSource> sources;
    std::vector<size_t> neighbors;
    double localLuminosity = 0.0;
    double localNetLuminosity = 0.0;
    double localInwardLuminosity = 0.0;
    uint64_t localBoundaryFaces = 0;
    uint64_t localEmittingFaces = 0;

    for(size_t outerCell = 0; outerCell < nCells; ++outerCell)
    {
        if(outerCell >= outside.size() || outside[outerCell] == 0)
            continue;

        Vector3D const outerPoint = runtime.tess.GetMeshPoint(outerCell);
        auto const& faces = runtime.tess.GetCellFaces(outerCell);
        runtime.tess.GetNeighbors(outerCell, neighbors);
        if(faces.size() != neighbors.size())
            throw UniversalError("Flux-source face/neighbor count mismatch");
        for(size_t j = 0; j < neighbors.size(); ++j)
        {
            size_t const innerCell = neighbors[j];
            if(runtime.tess.IsPointOutsideBox(innerCell) ||
               innerCell >= fldFlux.size() || innerCell >= outside.size() ||
               outside[innerCell] != 0)
                continue;

            Vector3D const innerPoint = runtime.tess.GetMeshPoint(innerCell);
            Vector3D normal = outerPoint - innerPoint;
            double const distance = abs(normal);
            double const area = runtime.tess.GetArea(faces[j]);
            if(!(distance > 0.0) || !std::isfinite(distance) ||
               !(area > 0.0) || !std::isfinite(area))
                throw UniversalError("Flux-source surface has invalid face geometry");
            normal *= 1.0 / distance;

            Vector3D const faceFlux = 0.5 *
                (fldFlux[outerCell] + fldFlux[innerCell]);
            double const signedOutwardFlux = ScalarProd(faceFlux, normal);
            if(!std::isfinite(signedOutwardFlux))
                throw UniversalError("Flux-source surface has non-finite face flux");
            double const signedLuminosity = signedOutwardFlux * area;
            double const luminosity = std::max(0.0, signedLuminosity);
            localNetLuminosity += signedLuminosity;
            localInwardLuminosity += std::max(0.0, -signedLuminosity);

            RadiationIMC::PostProcessExternalSource source;
            source.faceIndex = faces[j];
            source.cellID = runtime.cells[outerCell].ID;
            source.location = runtime.tess.FaceCM(faces[j]);
            source.outwardNormal = normal;
            source.luminosity = luminosity;
            sources.push_back(source);
            ++localBoundaryFaces;
            if(luminosity > 0.0)
            {
                localLuminosity += luminosity;
                ++localEmittingFaces;
            }
        }
    }

    double rawGlobalLuminosity = localLuminosity;
    double globalNetLuminosity = localNetLuminosity;
    double globalInwardLuminosity = localInwardLuminosity;
    uint64_t globalBoundaryFaces = localBoundaryFaces;
    uint64_t globalEmittingFaces = localEmittingFaces;
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &rawGlobalLuminosity, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &globalNetLuminosity, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &globalInwardLuminosity, 1,
                  MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &globalBoundaryFaces, 1,
                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &globalEmittingFaces, 1,
                  MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
    if(!(rawGlobalLuminosity > 0.0) || globalBoundaryFaces == 0 ||
       globalEmittingFaces == 0)
        throw UniversalError(
            "Flux-source CER produced no outward positive-flux faces");

    double luminosityScale = 1.0;
    if(runtime.fluxSourceInjectedLuminosity > 0.0)
    {
        double const reference = runtime.fluxSourceInjectedLuminosity;
        double const relDifference = std::abs(rawGlobalLuminosity - reference) /
            std::max({rawGlobalLuminosity, reference,
                      std::numeric_limits<double>::min()});
        if(relDifference > 1.0e-8)
        {
            UniversalError eo(
                "Flux-source luminosity changed after MPI repartition/rebuild");
            eo.addEntry("Reference luminosity", reference);
            eo.addEntry("Rebuilt luminosity", rawGlobalLuminosity);
            eo.addEntry("Relative difference", relDifference);
            throw eo;
        }
        if(runtime.fluxSourceBoundaryFaceCount != globalBoundaryFaces ||
           runtime.fluxSourceEmittingFaceCount != globalEmittingFaces)
            throw UniversalError(
                "Flux-source face topology changed after MPI repartition/rebuild");
        // Signed fluxes can be near zero after cancellation.  Normalize their
        // rebuild tolerance to the positive source luminosity as well, rather
        // than amplifying harmless partition-order roundoff by dividing by a
        // nearly zero signed diagnostic.
        double const netScale = std::max({
            reference,
            std::abs(runtime.fluxSourceNetLuminosity),
            std::abs(globalNetLuminosity),
            std::numeric_limits<double>::min()});
        double const inwardScale = std::max({
            reference,
            runtime.fluxSourceInwardLuminosity,
            globalInwardLuminosity,
            std::numeric_limits<double>::min()});
        if(std::abs(runtime.fluxSourceNetLuminosity - globalNetLuminosity) /
               netScale > 1.0e-8 ||
           std::abs(runtime.fluxSourceInwardLuminosity -
                    globalInwardLuminosity) / inwardScale > 1.0e-8)
            throw UniversalError(
                "Flux-source signed flux changed after MPI repartition/rebuild");
        luminosityScale = reference / rawGlobalLuminosity;
    }
    else
    {
        runtime.fluxSourceInjectedLuminosity = rawGlobalLuminosity;
        runtime.fluxSourceNetLuminosity = globalNetLuminosity;
        runtime.fluxSourceInwardLuminosity = globalInwardLuminosity;
        runtime.fluxSourceBoundaryFaceCount = globalBoundaryFaces;
        runtime.fluxSourceEmittingFaceCount = globalEmittingFaces;
    }

    if(luminosityScale != 1.0)
    {
        for(auto& source : sources)
            source.luminosity *= luminosityScale;
    }
    physics.setPostProcessExternalSources(std::move(sources));

    if(runtime.rank == 0)
        std::cout << "FLUX_SOURCE_CONFIG boundary_faces="
                  << globalBoundaryFaces
                  << " emitting_faces=" << globalEmittingFaces
                  << " raw_luminosity=" << rawGlobalLuminosity
                  << " normalized_luminosity="
                  << runtime.fluxSourceInjectedLuminosity
                  << " net_luminosity=" << globalNetLuminosity
                  << " clipped_inward_luminosity="
                  << globalInwardLuminosity
                  << " inward_fraction="
                  << (rawGlobalLuminosity > 0.0
                      ? globalInwardLuminosity / rawGlobalLuminosity : 0.0)
                  << " erg/s" << std::endl;
}

FluxSourcePolarizationSummary ComputeFluxSourcePolarizationSummary(
    SphericalObserver::ObserverQualitySnapshot const& snapshot)
{
    FluxSourcePolarizationSummary result;
    if(!snapshot.polarizationEnabled)
        return result;

    size_t const count = std::min(
        snapshot.energy.size(),
        std::min(snapshot.stokesQ.size(), snapshot.stokesU.size()));
    double totalEnergy = 0.0;
    double polarizedEnergy = 0.0;
    for(size_t i = 0; i < count; ++i)
    {
        double const energy = snapshot.energy[i];
        double const q = snapshot.stokesQ[i];
        double const u = snapshot.stokesU[i];
        if(!(energy > 0.0) || !std::isfinite(energy) ||
           !std::isfinite(q) || !std::isfinite(u))
            continue;
        totalEnergy += energy;
        polarizedEnergy += std::sqrt(q * q + u * u);
        ++result.observerCount;
    }
    if(totalEnergy > 0.0)
        result.luminosityWeightedDegree = std::clamp(
            polarizedEnergy / totalEnergy, 0.0, 1.0);
    return result;
}

} // namespace imc_postprocess_tde
