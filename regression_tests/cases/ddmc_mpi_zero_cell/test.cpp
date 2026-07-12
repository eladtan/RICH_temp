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
#include "source/3D/tessellation/loadBalancing/HilbertLoadBalancer.hpp"
#include "source/3D/hilbert/rectangular/HilbertRectangularConvertor3D.hpp"
#include "source/Radiation/CMMC/src/units/units.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/computational_cell.hpp"
#include "source/newtonian/three_dimensional/conserved_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/3D/radiation/RadiationIMC.hpp"
#include "source/monte/boundary/Rigid.hpp"
#include "source/monte/population/Comb.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"

namespace fs = std::filesystem;

namespace
{
class ScatteringOpacity final : public OpacityCalculator
{
public:
    double CalcPlanckOpacity(const ComputationalCell3D &) const override
    {
        return 0.0;
    }

    double CalcScatteringOpacity(const ComputationalCell3D &) const override
    {
        return 100.0;
    }

    double CalcAbsorptionOpacity(const ComputationalCell3D &, double) const override
    {
        return 0.0;
    }
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

unsigned long long ExtractDebugCount(const std::string &text,
                                     const std::string &name)
{
    const double value = ExtractDebugValue(text, name);
    const double rounded = std::round(value);
    if(value < 0.0 || std::abs(value - rounded) > 1e-9)
    {
        UniversalError error("Invalid DDMC debug counter");
        error.addEntry("Field", name);
        error.addEntry("Value", value);
        throw error;
    }
    return static_cast<unsigned long long>(rounded);
}

double SumParticleWeights(const std::vector<Particle3D> &particles)
{
    double sum = 0.0;
    for(const Particle3D &particle : particles)
        sum += particle.weight;
    return sum;
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
        constexpr size_t physicalCellCount = 2;
        const Vector3D ll(0.0, -0.5, -0.5);
        const Vector3D ur(2.0, 0.5, 0.5);

        if(worldSize <= static_cast<int>(physicalCellCount))
            throw UniversalError("DDMC zero-cell MPI test requires more ranks than physical cells");

        // MPI_Spread assigns all points to the final rank when points.size() is
        // smaller than the communicator. That would test empty ranks, but not a
        // cross-rank DDMC face. Build a deterministic Hilbert partition instead:
        // one cell belongs to rank 0, one to the final rank, and every rank in
        // between owns zero cells.
        const std::vector<Vector3D> globalPoints =
            CartesianMesh(physicalCellCount, 1, 1, ll, ur);
        if(globalPoints.size() != physicalCellCount)
            throw UniversalError("DDMC zero-cell MPI test generated the wrong number of points");

        auto indexing = std::make_shared<const Kernelization3D::Identity>();
        auto convertor = std::make_shared<HilbertRectangularConvertor3D>(
            ll, ur, 8);
        const curve_index_t d0 = convertor->xyz2d(globalPoints[0]);
        const curve_index_t d1 = convertor->xyz2d(globalPoints[1]);
        const curve_index_t lowIndex = std::min(d0, d1);
        const curve_index_t highIndex = std::max(d0, d1);
        if(highIndex - lowIndex < 2)
            throw UniversalError("DDMC zero-cell MPI test Hilbert points are not separable");

        const size_t lowPointIndex = d0 < d1 ? 0 : 1;
        const size_t highPointIndex = 1 - lowPointIndex;
        const curve_index_t partitionCut =
            lowIndex + (highIndex - lowIndex) / 2;
        std::vector<curve_index_t> forcedBoundaries(
            static_cast<size_t>(worldSize), partitionCut);
        auto forcedLoadBalancer = std::make_shared<HilbertLoadBalancer>(
            convertor, indexing, forcedBoundaries);

        if(forcedLoadBalancer->getOwner(globalPoints[lowPointIndex]) != 0 ||
           forcedLoadBalancer->getOwner(globalPoints[highPointIndex]) != worldSize - 1)
        {
            throw UniversalError("DDMC zero-cell MPI test failed to construct the forced partition");
        }

        std::vector<Vector3D> points;
        if(rank == 0)
            points.push_back(globalPoints[lowPointIndex]);
        else if(rank == worldSize - 1)
            points.push_back(globalPoints[highPointIndex]);

        Voronoi3D tess(ll, ur);
        tess.PresetLoadBalancer(forcedLoadBalancer);
        tess.BuildParallel(points, std::vector<double>(points.size(), 1.0),
                           true, true);

        const size_t localCellCount = tess.GetPointNo();
        int localZeroRank = localCellCount == 0 ? 1 : 0;
        int zeroRankCount = 0;
        MPI_Allreduce(&localZeroRank, &zeroRankCount, 1, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD);

        unsigned long long localCrossRankFaces = 0;
        for(size_t i = 0; i < localCellCount; ++i)
        {
            for(size_t face : tess.GetCellFaces(i))
            {
                const auto neighbors = tess.GetFaceNeighbors(face);
                const size_t other = neighbors.first == i
                    ? neighbors.second : neighbors.first;
                if(other >= localCellCount &&
                   other < tess.getMeshPoints().size() &&
                   !tess.IsPointOutsideBox(other))
                {
                    ++localCrossRankFaces;
                }
            }
        }
        unsigned long long crossRankFaces = 0;
        MPI_Allreduce(&localCrossRankFaces, &crossRankFaces, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

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
        initialCell.Erad = 1.0;

        std::vector<ComputationalCell3D> initialCells(localCellCount, initialCell);
        Simulation sim(tess, initialCells, eos);
        std::vector<ComputationalCell3D> &cells = sim.getCells();
        std::vector<Conserved3D> &extensives = sim.getExtensives();
        extensives.resize(cells.size());
        for(size_t i = 0; i < cells.size(); ++i)
            PrimitiveToConserved(cells[i], tess.GetVolume(i), extensives[i]);

        auto eosPtr = std::make_shared<IdealGas>(eos);
        auto opacity = std::make_shared<ScatteringOpacity>();
        auto boundary = std::make_shared<RigidBoundaryCondition<Vector3D, Tessellation3D>>(tess);

        RadiationIMCParameters parameters;
        parameters.newPhotonsPerCell = 0;
        parameters.withHydro = false;
        parameters.diffusionPressureGradient = false;
        parameters.MMC = false;
        parameters.withMultigroupOpacity = false;
        parameters.withRandomWalk = false;
        parameters.withDDMC = true;
        parameters.ddmcMinCellOpticalDepth = 15.0;
        parameters.ddmcUseMovingInterfaceCorrection = false;
        parameters.noHydroFeedback = true;

        auto physics = std::make_shared<RadiationIMC>(
            tess, boundary, cells, extensives, eosPtr, opacity, parameters);
        auto population = std::make_shared<CombPopulationControl<Vector3D, Tessellation3D>>(
            tess, 512, 2.0);

        std::vector<Particle3D> initialParticles;
        constexpr size_t initialParticlesPerCell = 512;
        auto mcStep = std::make_shared<RadiationMCStep>(
            tess, cells, extensives, physics, population, boundary,
            initialParticles, initialParticlesPerCell, false,
            RadiationMCStep::ManagerType::P2P);

        double localInitialWeight = SumParticleWeights(mcStep->getParticles());
        double initialWeight = 0.0;
        MPI_Allreduce(&localInitialWeight, &initialWeight, 1, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);

        const double dt = 200.0 / units::clight;
        mcStep->step(dt);

        double localFinalWeight = SumParticleWeights(mcStep->getParticles());
        double finalWeight = 0.0;
        MPI_Allreduce(&localFinalWeight, &finalWeight, 1, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);

        unsigned long long localDDMCSteps =
            static_cast<unsigned long long>(physics->getDDMCStepCount());
        unsigned long long localLeaks =
            static_cast<unsigned long long>(physics->getDDMCLeakCount());
        unsigned long long totalDDMCSteps = 0;
        unsigned long long totalLeaks = 0;
        MPI_Allreduce(&localDDMCSteps, &totalDDMCSteps, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&localLeaks, &totalLeaks, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

        unsigned long long localRemoteLeaks = 0;
        unsigned long long localFluxReductions = 0;
        unsigned long long localInvalidGeometry = 0;
        double localReciprocityMax = 0.0;
        double localConductanceMin = std::numeric_limits<double>::infinity();
        double localConductanceMax = 0.0;
        double localRateConductanceConsistencyMax = 0.0;
        unsigned long long localConductanceCells = 0;
        if(localCellCount > 0)
        {
            const std::string rankDebug =
                physics->getAccelerationDebugInfo(0, 1.0);
            localRemoteLeaks = ExtractDebugCount(
                rankDebug, "ddmc_remote_resident_leaks");
            localFluxReductions = ExtractDebugCount(
                rankDebug, "ddmc_mpi_face_flux_reductions");
            localInvalidGeometry = ExtractDebugCount(
                rankDebug, "ddmc_leak_invalid_geometry");
            localReciprocityMax = ExtractDebugValue(
                rankDebug, "ddmc_leak_reciprocity_max");

            for(size_t i = 0; i < localCellCount; ++i)
            {
                const std::string debug =
                    physics->getAccelerationDebugInfo(i, 1.0);
                const double internalRate = ExtractDebugValue(
                    debug, "ddmc_internal_leak_rate_sum");
                const double directConductance = ExtractDebugValue(
                    debug, "ddmc_internal_conductance_sum");
                const double rateConductance = tess.GetVolume(i) * internalRate;
                const double scale = std::max({std::abs(rateConductance),
                                               std::abs(directConductance),
                                               std::numeric_limits<double>::min()});
                const double consistency =
                    std::abs(rateConductance - directConductance) / scale;

                if(directConductance > 0.0 && std::isfinite(directConductance))
                {
                    localConductanceMin = std::min(
                        localConductanceMin, rateConductance);
                    localConductanceMax = std::max(
                        localConductanceMax, rateConductance);
                    localRateConductanceConsistencyMax = std::max(
                        localRateConductanceConsistencyMax, consistency);
                    ++localConductanceCells;
                }
            }
        }

        unsigned long long remoteLeaks = 0;
        unsigned long long fluxReductions = 0;
        unsigned long long invalidGeometry = 0;
        double reciprocityMax = 0.0;
        double conductanceMin = 0.0;
        double conductanceMax = 0.0;
        double rateConductanceConsistencyMax = 0.0;
        unsigned long long conductanceCells = 0;
        MPI_Allreduce(&localRemoteLeaks, &remoteLeaks, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&localFluxReductions, &fluxReductions, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&localInvalidGeometry, &invalidGeometry, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&localReciprocityMax, &reciprocityMax, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&localConductanceMin, &conductanceMin, 1,
                      MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&localConductanceMax, &conductanceMax, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&localRateConductanceConsistencyMax,
                      &rateConductanceConsistencyMax, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        MPI_Allreduce(&localConductanceCells, &conductanceCells, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

        const double crossRankReciprocityError =
            std::abs(conductanceMax - conductanceMin) /
            std::max({std::abs(conductanceMax), std::abs(conductanceMin),
                      std::numeric_limits<double>::min()});
        const double weightRelError = std::abs(finalWeight - initialWeight) /
            std::max(std::abs(initialWeight),
                     std::numeric_limits<double>::min());

        const bool pass =
            worldSize > static_cast<int>(physicalCellCount) &&
            zeroRankCount > 0 &&
            zeroRankCount < worldSize &&
            crossRankFaces > 0 &&
            initialWeight > 0.0 && finalWeight > 0.0 &&
            totalDDMCSteps > 0 && totalLeaks > 0 && remoteLeaks > 0 &&
            fluxReductions > 0 &&
            invalidGeometry == 0 &&
            conductanceCells == physicalCellCount &&
            std::isfinite(reciprocityMax) && reciprocityMax < 1e-10 &&
            std::isfinite(crossRankReciprocityError) &&
            crossRankReciprocityError < 1e-12 &&
            std::isfinite(rateConductanceConsistencyMax) &&
            rateConductanceConsistencyMax < 1e-12 &&
            std::isfinite(weightRelError) && weightRelError < 1e-10;

        if(rank == 0)
        {
            const std::string caseDir = fs::path(__FILE__).parent_path().string();
            std::ofstream out(caseDir + "/ddmc_mpi_zero_cell_metrics.txt");
            out << std::setprecision(17);
            out << "world_size " << worldSize << "\n";
            out << "physical_cells " << physicalCellCount << "\n";
            out << "zero_rank_count " << zeroRankCount << "\n";
            out << "cross_rank_faces " << crossRankFaces << "\n";
            out << "ddmc_steps " << totalDDMCSteps << "\n";
            out << "ddmc_leaks " << totalLeaks << "\n";
            out << "remote_resident_leaks " << remoteLeaks << "\n";
            out << "mpi_face_flux_reductions " << fluxReductions << "\n";
            out << "invalid_geometry " << invalidGeometry << "\n";
            out << "reciprocity_max " << reciprocityMax << "\n";
            out << "conductance_cells " << conductanceCells << "\n";
            out << "conductance_min " << conductanceMin << "\n";
            out << "conductance_max " << conductanceMax << "\n";
            out << "cross_rank_reciprocity_rel_error "
                << crossRankReciprocityError << "\n";
            out << "rate_conductance_consistency_max "
                << rateConductanceConsistencyMax << "\n";
            out << "initial_weight " << initialWeight << "\n";
            out << "final_weight " << finalWeight << "\n";
            out << "weight_rel_error " << weightRelError << "\n";
            out << "pass " << (pass ? 1 : 0) << "\n";

            std::cout << "DDMC zero-cell/cross-rank MPI: zero_ranks="
                      << zeroRankCount
                      << " cross_rank_faces=" << crossRankFaces
                      << " remote_leaks=" << remoteLeaks
                      << " flux_reductions=" << fluxReductions
                      << " cross_rank_reciprocity="
                      << crossRankReciprocityError
                      << " weight_rel_error=" << weightRelError
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
