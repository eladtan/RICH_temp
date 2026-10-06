// Regression checks for three transport properties:
//  A. The path-length estimator integrates the laboratory packet weight,
//     which decays at the Doppler-scaled rate f kappa_a c D, through both the
//     shared grey kernel and the host multigroup event loop.
//  B. Multigroup DDMC applies the Densmore admission test to thermal-boundary
//     packets inside the diffusive band; for an isotropic incident intensity
//     the kept energy fraction equals the conversion coefficient C.  (DDMC with
//     multigroup opacity always uses the multigroup mode; the grey-DDMC plus
//     multigroup-opacity combination is rejected at construction.)
//  C. The DDMC flux force raises only the kinetic energy: in a pure scatterer
//     the internal energy is unchanged and E = E_int + p^2/2m holds.
#ifdef RICH_MPI
#include <mpi.h>
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "source/3D/radiation/RadiationIMC.hpp"
#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/Radiation/OpacityCalculator.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/misc/universal_error.hpp"
#include "source/monte/boundary/RigidBoundary.hpp"
#include "source/monte/population/NoPopulationControl.hpp"
#include "source/monte/radiation/ddmc/DDMCWollaegerInterface.hpp"
#include "source/monte/utils/RandomOnFace.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"
#include "source/newtonian/three_dimensional/conserved_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"
#include "CMMC/src/units/units.hpp"

static_assert(ENERGY_GROUPS_NUM == 4,
              "imc_ddmc_review_fixes is built with --energy_groups_num=4");

namespace
{
class ConstantOpacity final : public OpacityCalculator
{
public:
    ConstantOpacity(double absorption, double scattering):
        absorption_(absorption), scattering_(scattering) {}

    double CalcPlanckOpacity(const ComputationalCell3D &cell) const override
    {
        return CalcPlanckOpacityAtTemperature(cell, cell.temperature);
    }
    double CalcPlanckOpacityAtTemperature(const ComputationalCell3D &, double) const override
    {
        return absorption_;
    }
    double CalcScatteringOpacity(const ComputationalCell3D &cell) const override
    {
        return CalcScatteringOpacityAtTemperature(cell, cell.temperature);
    }
    double CalcScatteringOpacityAtTemperature(const ComputationalCell3D &, double) const override
    {
        return scattering_;
    }
    double CalcScatteringOpacity(const ComputationalCell3D &, double) const override
    {
        return scattering_;
    }
    double CalcScatteringOpacityAtTemperature(const ComputationalCell3D &, double, double) const override
    {
        return scattering_;
    }
    double CalcAbsorptionOpacity(const ComputationalCell3D &cell, double energy) const override
    {
        return CalcAbsorptionOpacityAtTemperature(cell, energy, cell.temperature);
    }
    double CalcAbsorptionOpacityAtTemperature(const ComputationalCell3D &, double, double) const override
    {
        return absorption_;
    }

private:
    double absorption_;
    double scattering_;
};

// Thermal source on the x = ll.x face, reflecting elsewhere.  Packets enter
// with an isotropic intensity (mu = sqrt(xi)) and a fixed frequency.
class SourceFaceBoundary final : public STORM::BoundaryCondition<Vector3D, Tessellation3D>
{
public:
    SourceFaceBoundary(const Tessellation3D &grid, double temperature,
                       std::size_t packetsPerFace, double frequency):
        STORM::BoundaryCondition<Vector3D, Tessellation3D>(grid),
        temperature_(temperature), packetsPerFace_(packetsPerFace),
        frequency_(frequency), rng_(20261006) {}

    STORM::ParticleStatus apply(STORM::Particle<Vector3D> &particle) override
    {
        const auto &[ll, ur] = this->grid.GetBoxCoordinates();
        STORM::ParticleStatus status = STORM::ParticleStatus::DONE;
        for(const auto &face : this->grid.GetBoxFaces())
        {
            Vector3D normal;
            double faceScale = 0.0;
            if(this->getInwardBoxFaceNormalIfClose(face, particle.location, normal, faceScale))
            {
                if(std::abs(normal.x) > 0.99 &&
                   std::abs(particle.location.x - ll.x) < std::abs(ur.x - particle.location.x))
                {
                    return STORM::ParticleStatus::REMOVE;
                }
                if(this->reflectParticleOnBoxFace(particle, face))
                {
                    status = STORM::ParticleStatus::REFLECT;
                }
            }
        }
        if(status != STORM::ParticleStatus::REFLECT)
        {
            throw UniversalError("SourceFaceBoundary: particle not on any boundary");
        }
        return status;
    }

    std::vector<STORM::Particle<Vector3D>> generateNewBoundaryParticles(double fullDt) override
    {
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        std::vector<STORM::Particle<Vector3D>> packets;
        const double flux = units::sigma_sb * std::pow(temperature_, 4);
        const std::size_t N = this->grid.GetPointNo();
        for(std::size_t i = 0; i < N; ++i)
        {
            for(const std::size_t face : this->grid.GetCellFaces(i))
            {
                if(!IsSourceFace(face, i))
                {
                    continue;
                }
                const double energy = flux * this->grid.GetArea(face) * fullDt;
                generatedEnergy_ += energy;
                for(std::size_t j = 0; j < packetsPerFace_; ++j)
                {
                    STORM::Particle<Vector3D> packet;
                    packet.location = STORM::RandomPointOnFace<Vector3D, Tessellation3D>(this->grid, face);
                    const double mu = std::sqrt(unif(rng_));
                    const double sinTheta = std::sqrt(std::max(0.0, 1.0 - mu * mu));
                    const double phi = 2.0 * M_PI * unif(rng_);
                    packet.velocity = Vector3D(mu, sinTheta * std::cos(phi), sinTheta * std::sin(phi)) * units::clight;
                    packet.frequency = frequency_;
                    packet.weight = energy / static_cast<double>(packetsPerFace_);
                    packet.initialWeight = packet.weight;
                    packet.timeLeft = fullDt * unif(rng_);
                    packet.cellIndex = i;
                    packets.push_back(packet);
                }
            }
        }
        return packets;
    }

    STORM::DDMCBoundaryFaceBehavior getDDMCBoundaryFaceBehavior(
        std::size_t faceIdx, std::size_t insideCellIndex, std::size_t) const override
    {
        return IsSourceFace(faceIdx, insideCellIndex)
            ? STORM::DDMCBoundaryFaceBehavior::ThermalSource
            : STORM::DDMCBoundaryFaceBehavior::ReflectingRigid;
    }

    STORM::DeviceBoundaryFaceBehavior getDeviceBoundaryFaceBehavior(
        std::size_t faceIdx, std::size_t insideCellIndex, std::size_t) const override
    {
        return IsSourceFace(faceIdx, insideCellIndex)
            ? STORM::DeviceBoundaryFaceBehavior::HostOnly
            : STORM::DeviceBoundaryFaceBehavior::ReflectingRigid;
    }

    double generatedEnergy() const { return generatedEnergy_; }

private:
    bool IsSourceFace(std::size_t face, std::size_t cell) const
    {
        const auto &neighbors = this->grid.GetFaceNeighbors(face);
        const std::size_t other = neighbors.first == cell ? neighbors.second : neighbors.first;
        if(other < this->grid.GetPointNo() || !this->grid.IsPointOutsideBox(other))
        {
            return false;
        }
        const Vector3D outward = this->grid.GetMeshPoint(other) - this->grid.GetMeshPoint(cell);
        return outward.x < -0.99 * std::sqrt(ScalarProd(outward, outward));
    }

    double temperature_;
    std::size_t packetsPerFace_;
    double frequency_;
    std::mt19937_64 rng_;
    double generatedEnergy_ = 0.0;
};

std::vector<ComputationalCell3D> MakeCells(const Tessellation3D &tess, const IdealGas &eos,
                                           double temperature, const Vector3D &velocity)
{
    std::vector<ComputationalCell3D> cells(tess.GetPointNo());
    for(ComputationalCell3D &cell : cells)
    {
        cell.density = 1.0;
        cell.temperature = temperature;
        cell.velocity = velocity;
        cell.internal_energy = eos.dT2e(cell.density, cell.temperature, cell.tracers,
                                        ComputationalCell3D::tracerNames);
        cell.pressure = eos.de2p(cell.density, cell.internal_energy, cell.tracers,
                                 ComputationalCell3D::tracerNames);
        cell.Erad = 0.0;
    }
    return cells;
}

std::vector<Conserved3D> MakeExtensives(const Tessellation3D &tess,
                                        const std::vector<ComputationalCell3D> &cells)
{
    std::vector<Conserved3D> extensives(cells.size());
    for(std::size_t i = 0; i < cells.size(); ++i)
    {
        PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);
    }
    return extensives;
}

void SetGroups(RadiationIMCParameters &parameters, double kT)
{
    const std::array<double, ENERGY_GROUPS_NUM + 1> edges{0.01, 0.5, 1.5, 4.0, 30.0};
    for(std::size_t g = 0; g < edges.size(); ++g)
    {
        parameters.energyBoundaries[g] = edges[g] * kT;
    }
    parameters.energyBoundariesProvided = true;
}

double SumWeights(const std::vector<Particle3D> &particles)
{
    double sum = 0.0;
    for(const Particle3D &particle : particles)
    {
        sum += particle.weight;
    }
    return sum;
}

// A. One absorbing packet in a single cell moving at 0.1c along x.
double EstimatorRelativeError(bool multigroup, double mu)
{
    const double c = units::clight;
    const double dt = 1.0 / c;
    const Vector3D lower(0.0, 0.0, 0.0);
    const Vector3D upper(4.0, 4.0, 4.0);
    const Vector3D centre(2.0, 2.0, 2.0);
    Voronoi3D tess(lower, upper);
    tess.Build(std::vector<Vector3D>{centre});

    const double beta = 0.1;
    const double temperature = 0.01;
    const double kappa = 1.0;
    IdealGas eos(5.0 / 3.0, 1.0, 1.0, 0.0);
    std::vector<ComputationalCell3D> cells = MakeCells(tess, eos, temperature, Vector3D(beta * c, 0.0, 0.0));
    std::vector<Conserved3D> extensives = MakeExtensives(tess, cells);
    auto boundary = std::make_shared<RigidBoundaryCondition<Vector3D, Tessellation3D>>(tess);

    RadiationIMCParameters parameters;
    parameters.newPhotonsPerCell = 1;
    parameters.withHydro = true;
    parameters.diffusionPressureGradient = false;
    parameters.MMC = false;
    parameters.withDDMC = false;
    parameters.noHydroFeedback = true;
    parameters.withRandomWalk = false;
    parameters.withMultigroupOpacity = multigroup;
    const double kT = units::k_boltz * temperature;
    SetGroups(parameters, kT);

    auto physics = std::make_shared<RadiationIMC>(
        tess, boundary, cells, extensives, std::make_shared<IdealGas>(eos),
        std::make_shared<ConstantOpacity>(kappa, 0.0), parameters);
    if(multigroup)
    {
        // Skip the shared kernel so the host event loop is exercised.
        physics->implementation().imcDiffForceLegacy_ = true;
    }

    Particle3D packet;
    packet.location = centre;
    packet.velocity = Vector3D(mu * c, 0.0, 0.0);
    packet.weight = 1.0;
    packet.initialWeight = 1.0;
    packet.frequency = 2.0 * kT;
    packet.timeLeft = dt;
    packet.cellIndex = 0;
    auto population = std::make_shared<STORM::NoPopulationControl<Vector3D, Tessellation3D>>(tess);
    RadiationMCStep step(tess, cells, extensives, physics, population, boundary,
                         std::vector<Particle3D>{packet}, 0, true);
    step.step(dt);

    const double fleck = physics->getFactorFleck()[0];
    const double gamma = 1.0 / std::sqrt(1.0 - beta * beta);
    const double doppler = gamma * (1.0 - beta * mu);
    const double rate = fleck * kappa * c * doppler;
    const double expected = -std::expm1(-rate * dt) / rate / (dt * tess.GetVolume(0));
    const double measured = physics->getEradTimeAvg()[0];
    return std::abs(measured - expected) / expected;
}

// B. Thermal-boundary admission into multigroup DDMC; every group is diffusive.
struct AdmissionResult
{
    double keptFraction = 0.0;
    double expected = 0.0;
    double sigma = 0.0;
    std::size_t ddmcSteps = 0;
};

AdmissionResult ThermalAdmission()
{
    const double c = units::clight;
    const Vector3D lower(0.0, 0.0, 0.0);
    const Vector3D upper(4.0, 1.0, 1.0);
    Voronoi3D tess(lower, upper);
    tess.Build(CartesianMesh(4, 1, 1, lower, upper));

    const double temperature = 1.0e6;
    const double kappaA = 1.0e-8;
    const double kappaS = 20.0;
    const std::size_t packets = 400000;
    IdealGas eos(5.0 / 3.0, 1.0, 1.0, 0.0);
    std::vector<ComputationalCell3D> cells = MakeCells(tess, eos, temperature, Vector3D(0.0, 0.0, 0.0));
    std::vector<Conserved3D> extensives = MakeExtensives(tess, cells);
    const double kT = units::k_boltz * temperature;
    auto boundary = std::make_shared<SourceFaceBoundary>(tess, temperature, packets, 2.0 * kT);

    RadiationIMCParameters parameters;
    parameters.newPhotonsPerCell = 1;
    parameters.withHydro = false;
    parameters.diffusionPressureGradient = false;
    parameters.MMC = false;
    parameters.withDDMC = true;
    parameters.withMultigroupOpacity = true;
    parameters.withMultigroupDDMC = true;
    parameters.ddmcMinCellOpticalDepth = 5.0;
    parameters.noHydroFeedback = true;
    parameters.withRandomWalk = false;
    SetGroups(parameters, kT);

    auto physics = std::make_shared<RadiationIMC>(
        tess, boundary, cells, extensives, std::make_shared<IdealGas>(eos),
        std::make_shared<ConstantOpacity>(kappaA, kappaS), parameters);
    auto population = std::make_shared<STORM::NoPopulationControl<Vector3D, Tessellation3D>>(tess);
    RadiationMCStep step(tess, cells, extensives, physics, population, boundary,
                         std::vector<Particle3D>(), 0, false);
    // Short enough that admitted energy has no time to leak back out.
    step.step(1.0e-3 / c);

    AdmissionResult result;
    result.keptFraction = SumWeights(step.getParticles()) / boundary->generatedEnergy();
    const double sigmaT = kappaA + kappaS;
    const double albedo = STORM::ddmc::Densmore2006SingleScatterAlbedo(
        sigmaT, physics->getFactorFleck()[0] * kappaA);
    result.expected = STORM::ddmc::Densmore2006CellCoefficient(sigmaT, albedo, 0.5);
    result.sigma = std::sqrt(result.expected * (1.0 - result.expected) / static_cast<double>(packets));
    result.ddmcSteps = physics->getDDMCStepCount();
    return result;
}

// C. DDMC force in a pure scatterer with a radiation-energy gradient.
struct ForceResult
{
    double internalRelChange = 0.0;
    double invariantRelError = 0.0;
    double maxMomentum = 0.0;
    std::size_t feedbackCount = 0;
};

ForceResult DDMCForceInvariant()
{
    const double c = units::clight;
    const Vector3D lower(0.0, 0.0, 0.0);
    const Vector3D upper(8.0, 1.0, 1.0);
    Voronoi3D tess(lower, upper);
    tess.Build(CartesianMesh(8, 1, 1, lower, upper));

    IdealGas eos(5.0 / 3.0, 1.0, 1.0, 0.0);
    std::vector<ComputationalCell3D> cells = MakeCells(tess, eos, 1.0, Vector3D(0.0, 0.0, 0.0));
    for(std::size_t i = 0; i < cells.size(); ++i)
    {
        cells[i].Erad = 1.0 + tess.GetMeshPoint(i).x;
    }
    std::vector<Conserved3D> extensives = MakeExtensives(tess, cells);
    auto boundary = std::make_shared<RigidBoundaryCondition<Vector3D, Tessellation3D>>(tess);

    RadiationIMCParameters parameters;
    parameters.newPhotonsPerCell = 1;
    parameters.withHydro = true;
    parameters.diffusionPressureGradient = false;
    parameters.MMC = false;
    parameters.withDDMC = true;
    parameters.ddmcMinCellOpticalDepth = 15.0;
    parameters.noHydroFeedback = false;
    parameters.withMultigroupOpacity = false;
    parameters.withRandomWalk = false;
    SetGroups(parameters, units::k_boltz);

    auto physics = std::make_shared<RadiationIMC>(
        tess, boundary, cells, extensives, std::make_shared<IdealGas>(eos),
        std::make_shared<ConstantOpacity>(0.0, 100.0), parameters);
    auto population = std::make_shared<STORM::NoPopulationControl<Vector3D, Tessellation3D>>(tess);
    RadiationMCStep step(tess, cells, extensives, physics, population, boundary,
                         std::vector<Particle3D>(), 256, true);

    std::vector<double> internalBefore(extensives.size());
    for(std::size_t i = 0; i < extensives.size(); ++i)
    {
        internalBefore[i] = extensives[i].internal_energy;
    }
    step.step(200.0 / c);

    ForceResult result;
    for(std::size_t i = 0; i < extensives.size(); ++i)
    {
        const Conserved3D &ext = extensives[i];
        const double kinetic = 0.5 * ScalarProd(ext.momentum, ext.momentum) / ext.mass;
        result.internalRelChange = std::max(result.internalRelChange,
            std::abs(ext.internal_energy - internalBefore[i]) / std::abs(internalBefore[i]));
        result.invariantRelError = std::max(result.invariantRelError,
            std::abs(ext.energy - ext.internal_energy - kinetic) / std::abs(ext.energy));
        result.maxMomentum = std::max(result.maxMomentum, std::sqrt(ScalarProd(ext.momentum, ext.momentum)));
    }
    result.feedbackCount = physics->implementation().getDDMCMomentumFeedbackCount();
    return result;
}
}

int main(int argc, char **argv)
{
#ifdef RICH_MPI
    MPI_Init(&argc, &argv);
#else
    (void)argc;
    (void)argv;
#endif
    int exitCode = 1;
    try
    {
        const std::array<double, 4> estimatorErrors{
            EstimatorRelativeError(false, 1.0), EstimatorRelativeError(false, -1.0),
            EstimatorRelativeError(true, 1.0), EstimatorRelativeError(true, -1.0)};
        const double estimatorMax = *std::max_element(estimatorErrors.begin(), estimatorErrors.end());
        const AdmissionResult admission = ThermalAdmission();
        const ForceResult force = DDMCForceInvariant();

        const double admissionDeviation =
            std::abs(admission.keptFraction - admission.expected) / admission.sigma;
        const bool estimatorPass = std::isfinite(estimatorMax) && estimatorMax < 1.0e-9;
        const bool admissionPass = std::isfinite(admissionDeviation) &&
            admissionDeviation < 6.0 && admission.ddmcSteps > 0;
        const bool forcePass = force.feedbackCount > 0 && force.maxMomentum > 0.0 &&
            force.internalRelChange < 1.0e-12 && force.invariantRelError < 1.0e-12;
        const bool pass = estimatorPass && admissionPass && forcePass;

        std::ofstream out("imc_ddmc_review_fixes_metrics.txt");
        out << std::scientific << std::setprecision(16);
        out << "estimator_grey_plus_rel " << estimatorErrors[0] << std::endl;
        out << "estimator_grey_minus_rel " << estimatorErrors[1] << std::endl;
        out << "estimator_host_mg_plus_rel " << estimatorErrors[2] << std::endl;
        out << "estimator_host_mg_minus_rel " << estimatorErrors[3] << std::endl;
        out << "estimator_max_rel " << estimatorMax << std::endl;
        out << "admission_kept_fraction " << admission.keptFraction << std::endl;
        out << "admission_expected " << admission.expected << std::endl;
        out << "admission_sigma " << admission.sigma << std::endl;
        out << "admission_deviation_sigma " << admissionDeviation << std::endl;
        out << "admission_ddmc_steps " << admission.ddmcSteps << std::endl;
        out << "force_internal_rel_change " << force.internalRelChange << std::endl;
        out << "force_invariant_rel_error " << force.invariantRelError << std::endl;
        out << "force_max_momentum " << force.maxMomentum << std::endl;
        out << "force_feedback_count " << force.feedbackCount << std::endl;
        out << "pass " << (pass ? 1 : 0) << std::endl;

        std::cout << "estimator max rel error " << estimatorMax
                  << ", admission kept " << admission.keptFraction
                  << " expected " << admission.expected
                  << " (" << admissionDeviation << " sigma)"
                  << ", force internal change " << force.internalRelChange
                  << " invariant " << force.invariantRelError
                  << " feedback " << force.feedbackCount << std::endl;
        std::cout << (pass ? "PASS" : "FAIL") << std::endl;
        exitCode = pass ? 0 : 1;
    }
    catch(const UniversalError &error)
    {
        reportError(error);
    }
    catch(const std::exception &error)
    {
        std::cerr << "std::exception: " << error.what() << std::endl;
    }
#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return exitCode;
}
