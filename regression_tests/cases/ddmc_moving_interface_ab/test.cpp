#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "source/mpi/mpi_commands.hpp"
#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/Radiation/CMMC/src/units/units.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"
#include "source/newtonian/three_dimensional/conserved_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/3D/radiation/DDMCWollaegerInterface.hpp"
#include "source/3D/radiation/LorentzTransformation.hpp"
#include "source/3D/radiation/RadiationIMC.hpp"
#include "source/monte/boundary/Rigid.hpp"

namespace fs = std::filesystem;

namespace
{
class InterfaceOpacity final : public OpacityCalculator
{
public:
    double CalcPlanckOpacity(const ComputationalCell3D &) const override
    {
        return 0.0;
    }

    double CalcScatteringOpacity(const ComputationalCell3D &cell) const override
    {
        return cell.density > 1.5 ? 100.0 : 0.0;
    }

    double CalcAbsorptionOpacity(const ComputationalCell3D &, double) const override
    {
        return 0.0;
    }
};

struct ModeResult
{
    size_t incidents = 0;
    size_t admitted = 0;
    size_t reflected = 0;
    size_t unexpected = 0;
    size_t diagnosticIncidents = 0;
    size_t diagnosticAdmitted = 0;
    size_t diagnosticReflected = 0;
    size_t guApplied = 0;
    size_t guFallback = 0;
    size_t bypass = 0;
    size_t splitPackets = 0;
    double admittedWeight = 0.0;
};

double ExtractDebugValue(const std::string &text, const std::string &name)
{
    const std::string needle = name + "=";
    const size_t begin = text.find(needle);
    if(begin == std::string::npos)
    {
        UniversalError error("Missing DDMC debug field");
        error.addEntry("Field", name);
        error.addEntry("Debug text", text);
        throw error;
    }

    const size_t valueBegin = begin + needle.size();
    const size_t valueEnd = text.find(' ', valueBegin);
    try
    {
        const double value = std::stod(text.substr(valueBegin, valueEnd - valueBegin));
        if(!std::isfinite(value))
            throw std::runtime_error("non-finite debug value");
        return value;
    }
    catch(const std::exception &)
    {
        UniversalError error("Invalid DDMC debug field");
        error.addEntry("Field", name);
        error.addEntry("Debug text", text);
        throw error;
    }
}

size_t ExtractDebugCount(const std::string &text, const std::string &name)
{
    const double value = ExtractDebugValue(text, name);
    const double rounded = std::round(value);
    if(value < 0.0 || std::abs(value - rounded) > 1e-9 ||
       rounded > static_cast<double>(std::numeric_limits<size_t>::max()))
    {
        UniversalError error("Invalid DDMC debug counter");
        error.addEntry("Field", name);
        error.addEntry("Value", value);
        throw error;
    }
    return static_cast<size_t>(rounded);
}

ModeResult RunMode(bool useMovingCorrection,
                   Tessellation3D &tess,
                   const std::vector<ComputationalCell3D> &baseCells,
                   const std::vector<Conserved3D> &baseExtensives,
                   const IdealGas &eos,
                   size_t sourceCell,
                   size_t targetCell,
                   const Vector3D &materialVelocity,
                   const Vector3D &incomingDirectionCo,
                   size_t incidentCount)
{
    std::vector<ComputationalCell3D> cells = baseCells;
    std::vector<Conserved3D> extensives = baseExtensives;

    auto eosPtr = std::make_shared<IdealGas>(eos);
    auto opacity = std::make_shared<InterfaceOpacity>();
    auto boundary = std::make_shared<RigidBoundaryCondition<Vector3D, Tessellation3D>>(tess);

    RadiationIMCParameters parameters;
    parameters.newPhotonsPerCell = 0;
    parameters.withHydro = true;
    parameters.diffusionPressureGradient = false;
    parameters.MMC = false;
    parameters.withMultigroupOpacity = false;
    parameters.withRandomWalk = false;
    parameters.withDDMC = true;
    parameters.ddmcMinCellOpticalDepth = 15.0;
    parameters.ddmcUseMovingInterfaceCorrection = useMovingCorrection;
    parameters.ddmcMaxInterfaceVelocityOverC = 0.1;
    parameters.ddmcInterfaceTargetWeightRatio = 100.0;
    parameters.ddmcMaxInterfaceSplits = 4;
    parameters.noHydroFeedback = true;

    auto physics = std::make_shared<RadiationIMC>(
        tess, boundary, cells, extensives, eosPtr, opacity, parameters);

    const double transportTime = 4.0 / units::clight;
    const std::vector<Particle3D> generated = physics->preStep(transportTime);
    if(!generated.empty())
        throw UniversalError("DDMC moving-interface A/B test unexpectedly generated source particles");

    Particle3D templateParticle;
    templateParticle.cellIndex = sourceCell;
    templateParticle.cellID = cells[sourceCell].ID;
    templateParticle.sourceCellID = cells[sourceCell].ID;
    templateParticle.location = tess.GetMeshPoint(sourceCell);
    templateParticle.velocity = units::clight * incomingDirectionCo;
    templateParticle.frequency = 1.0;
    templateParticle.weight = 1.0;
    templateParticle.initialWeight = 1.0;
    templateParticle.timeLeft = transportTime;
#ifdef RICH_MPI
    templateParticle.rank = 0;
#endif

    ComovingToLabPacket(templateParticle, materialVelocity);
    templateParticle.initialWeight = std::abs(templateParticle.weight);

    ModeResult result;
    result.incidents = incidentCount;
    for(size_t n = 0; n < incidentCount; ++n)
    {
        Particle3D particle = templateParticle;
        std::vector<Particle3D> particlesToAdd;
        const auto functionality = physics->step(particle, particlesToAdd);

        if(functionality.change == MonteCarloParticleStatus::CELL_MOVE &&
           functionality.nextCellIndex == targetCell && particle.ddmcMode)
        {
            ++result.admitted;
            result.admittedWeight += particle.weight;
            for(const Particle3D &extra : particlesToAdd)
                result.admittedWeight += extra.weight;
        }
        else if(functionality.change == MonteCarloParticleStatus::NO_CELL_MOVE &&
                !particle.ddmcMode)
        {
            ++result.reflected;
        }
        else
        {
            ++result.unexpected;
        }
    }

    const std::string debug = physics->getAccelerationDebugInfo(targetCell, 1.0);
    result.diagnosticIncidents = ExtractDebugCount(
        debug, "ddmc_interface_incident");
    result.diagnosticAdmitted = ExtractDebugCount(
        debug, "ddmc_interface_admitted");
    result.diagnosticReflected = ExtractDebugCount(
        debug, "ddmc_interface_reflected");
    result.guApplied = ExtractDebugCount(
        debug, "ddmc_interface_gu_applied");
    result.guFallback = ExtractDebugCount(
        debug, "ddmc_interface_gu_fallback");
    result.bypass = ExtractDebugCount(
        debug, "ddmc_interface_bypass");
    result.splitPackets = ExtractDebugCount(
        debug, "ddmc_interface_split_packets");

    return result;
}
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int exitCode = 0;
    try
    {
        if(worldSize != 1)
            throw UniversalError("DDMC moving-interface A/B test must run with one MPI rank");

        const Vector3D ll(0.0, -10.0, -10.0);
        const Vector3D ur(2.0, 10.0, 10.0);
        std::vector<Vector3D> points;
        if(rank == 0)
            points = CartesianMesh(2, 1, 1, ll, ur);
        points = MPI_Spread(points, 0, MPI_COMM_WORLD);

        Voronoi3D tess(ll, ur);
        tess.BuildParallel(points);

        IdealGas eos(1.4, 1.0, 1, 0);
        ComputationalCell3D initialCell;
        initialCell.density = 1.0;
        initialCell.temperature = 1.0;
        initialCell.velocity = Vector3D(0.0, 0.0, 0.0);
        initialCell.internal_energy = eos.dT2e(
            initialCell.density, initialCell.temperature,
            initialCell.tracers, ComputationalCell3D::tracerNames);
        initialCell.pressure = eos.de2p(
            initialCell.density, initialCell.internal_energy,
            initialCell.tracers, ComputationalCell3D::tracerNames);
        initialCell.Erad = 0.0;

        std::vector<ComputationalCell3D> initialCells(tess.GetPointNo(), initialCell);
        size_t sourceCell = std::numeric_limits<size_t>::max();
        size_t targetCell = std::numeric_limits<size_t>::max();
        for(size_t i = 0; i < initialCells.size(); ++i)
        {
            if(tess.GetMeshPoint(i).x < 1.0)
            {
                sourceCell = i;
                initialCells[i].density = 1.0;
            }
            else
            {
                targetCell = i;
                initialCells[i].density = 2.0;
            }
        }
        if(sourceCell == std::numeric_limits<size_t>::max() ||
           targetCell == std::numeric_limits<size_t>::max())
        {
            throw UniversalError("Could not identify source and target cells");
        }

        size_t interfaceFace = std::numeric_limits<size_t>::max();
        for(size_t face : tess.GetCellFaces(sourceCell))
        {
            const auto neighbors = tess.GetFaceNeighbors(face);
            const size_t other = neighbors.first == sourceCell
                ? neighbors.second : neighbors.first;
            if(other == targetCell)
            {
                interfaceFace = face;
                break;
            }
        }
        if(interfaceFace == std::numeric_limits<size_t>::max())
            throw UniversalError("Could not identify the IMC-DDMC interface face");

        Vector3D normalOutOfDDMC = normalize(tess.Normal(interfaceFace));
        const Vector3D sourceCenter = tess.GetMeshPoint(sourceCell);
        const Vector3D targetCenter = tess.GetMeshPoint(targetCell);
        if(ScalarProd(normalOutOfDDMC, sourceCenter - targetCenter) < 0.0)
            normalOutOfDDMC = -1.0 * normalOutOfDDMC;

        constexpr double betaNormal = 0.04;
        constexpr double mu = 0.35;
        const Vector3D materialVelocity =
            betaNormal * units::clight * normalOutOfDDMC;

        Vector3D helper(0.0, 1.0, 0.0);
        if(std::abs(ScalarProd(helper, normalOutOfDDMC)) > 0.9)
            helper = Vector3D(0.0, 0.0, 1.0);
        const Vector3D tangent = normalize(
            helper - ScalarProd(helper, normalOutOfDDMC) * normalOutOfDDMC);
        const Vector3D incomingDirectionCo = normalize(
            -mu * normalOutOfDDMC + std::sqrt(1.0 - mu * mu) * tangent);

        for(ComputationalCell3D &cell : initialCells)
        {
            cell.velocity = materialVelocity;
            cell.internal_energy = eos.dT2e(
                cell.density, cell.temperature,
                cell.tracers, ComputationalCell3D::tracerNames);
            cell.pressure = eos.de2p(
                cell.density, cell.internal_energy,
                cell.tracers, ComputationalCell3D::tracerNames);
        }

        Simulation sim(tess, initialCells, eos);
        std::vector<ComputationalCell3D> baseCells = sim.getCells();
        std::vector<Conserved3D> baseExtensives(baseCells.size());
        for(size_t i = 0; i < baseCells.size(); ++i)
            PrimitiveToConserved(baseCells[i], tess.GetVolume(i), baseExtensives[i]);

        constexpr size_t incidentCount = 60000;
        const ModeResult staticResult = RunMode(
            false, tess, baseCells, baseExtensives, eos,
            sourceCell, targetCell, materialVelocity,
            incomingDirectionCo, incidentCount);
        const ModeResult correctedResult = RunMode(
            true, tess, baseCells, baseExtensives, eos,
            sourceCell, targetCell, materialVelocity,
            incomingDirectionCo, incidentCount);

        const double expectedFactor =
            DDMCWollaeger::MovingFactor(mu, betaNormal);
        const double staticMeanWeight = staticResult.admitted > 0
            ? staticResult.admittedWeight / static_cast<double>(staticResult.admitted)
            : std::numeric_limits<double>::quiet_NaN();
        const double correctedMeanWeight = correctedResult.admitted > 0
            ? correctedResult.admittedWeight / static_cast<double>(correctedResult.admitted)
            : std::numeric_limits<double>::quiet_NaN();
        const double measuredFactor = correctedResult.admittedWeight /
            std::max(staticResult.admittedWeight,
                     std::numeric_limits<double>::min());
        const double factorRelError = std::abs(measuredFactor - expectedFactor) /
            std::max(std::abs(expectedFactor), 1.0);
        const double staticWeightError = std::abs(staticMeanWeight - 1.0);

        const bool pass =
            staticResult.admitted == correctedResult.admitted &&
            staticResult.admitted > 500 &&
            staticResult.reflected + staticResult.admitted == incidentCount &&
            correctedResult.reflected + correctedResult.admitted == incidentCount &&
            staticResult.unexpected == 0 && correctedResult.unexpected == 0 &&
            staticResult.diagnosticIncidents == incidentCount &&
            correctedResult.diagnosticIncidents == incidentCount &&
            staticResult.diagnosticAdmitted == staticResult.admitted &&
            correctedResult.diagnosticAdmitted == correctedResult.admitted &&
            staticResult.diagnosticReflected == staticResult.reflected &&
            correctedResult.diagnosticReflected == correctedResult.reflected &&
            staticResult.guApplied == 0 &&
            correctedResult.guApplied == incidentCount &&
            staticResult.guFallback == 0 && correctedResult.guFallback == 0 &&
            staticResult.bypass == 0 && correctedResult.bypass == 0 &&
            staticResult.splitPackets == 0 && correctedResult.splitPackets == 0 &&
            std::isfinite(expectedFactor) &&
            std::abs(expectedFactor - 1.0) > 0.02 &&
            staticWeightError < 5e-12 &&
            factorRelError < 5e-11;

        if(rank == 0)
        {
            const std::string caseDir = fs::path(__FILE__).parent_path().string();
            std::ofstream out(caseDir + "/ddmc_moving_interface_ab_metrics.txt");
            out << std::setprecision(17);
            out << "incident_count " << incidentCount << "\n";
            out << "static_admitted " << staticResult.admitted << "\n";
            out << "corrected_admitted " << correctedResult.admitted << "\n";
            out << "static_reflected " << staticResult.reflected << "\n";
            out << "corrected_reflected " << correctedResult.reflected << "\n";
            out << "static_mean_admitted_weight " << staticMeanWeight << "\n";
            out << "corrected_mean_admitted_weight " << correctedMeanWeight << "\n";
            out << "expected_moving_factor " << expectedFactor << "\n";
            out << "measured_moving_factor " << measuredFactor << "\n";
            out << "moving_factor_rel_error " << factorRelError << "\n";
            out << "static_weight_error " << staticWeightError << "\n";
            out << "static_diagnostic_incidents " << staticResult.diagnosticIncidents << "\n";
            out << "corrected_diagnostic_incidents " << correctedResult.diagnosticIncidents << "\n";
            out << "static_gu_applied " << staticResult.guApplied << "\n";
            out << "corrected_gu_applied " << correctedResult.guApplied << "\n";
            out << "static_gu_fallback " << staticResult.guFallback << "\n";
            out << "corrected_gu_fallback " << correctedResult.guFallback << "\n";
            out << "static_bypass " << staticResult.bypass << "\n";
            out << "corrected_bypass " << correctedResult.bypass << "\n";
            out << "pass " << (pass ? 1 : 0) << "\n";

            std::cout << "DDMC moving-interface A/B: admitted="
                      << staticResult.admitted
                      << " expected_G=" << expectedFactor
                      << " measured_G=" << measuredFactor
                      << " rel_error=" << factorRelError
                      << " pass=" << pass << std::endl;
        }

        exitCode = pass ? 0 : 1;
    }
    catch(const UniversalError &error)
    {
        std::cerr << "=== UniversalError on rank " << rank << " ===" << std::endl;
        reportError(error);
        exitCode = 1;
    }
    catch(const std::exception &error)
    {
        std::cerr << "=== std::exception on rank " << rank
                  << ": " << error.what() << " ===" << std::endl;
        exitCode = 1;
    }

    int globalExitCode = 0;
    MPI_Allreduce(&exitCode, &globalExitCode, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return globalExitCode;
}
