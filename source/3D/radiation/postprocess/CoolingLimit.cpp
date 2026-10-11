#include "CoolingLimit.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "FluxSourceCalculation.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/SourceTerm3D.hpp"
#include "source/Radiation/conj_grad_solve.hpp"
#include "source/misc/universal_error.hpp"

#ifdef RICH_MPI
#include <mpi.h>
#include "source/mpi/mpi_commands_3d.hpp"
#endif

namespace imc_postprocess_tde {
namespace {

// Smallest s in (0, sMax] at which rate(s) reaches target, by bisection;
// rate(0) = 0 < target < rate(sMax). rate need not be monotonic (the opacity
// depends on temperature); any crossing is a valid cap.
double SolveForRate(std::function<double(double)> const& rate, double sMax, double target)
{
    double lo = 0.0;
    double hi = sMax;
    for(int iteration = 0; iteration < 4000; ++iteration)
    {
        if(hi - lo <= 1e-14 * hi)
            return 0.5 * (lo + hi);
        double const mid = 0.5 * (lo + hi);
        double const value = rate(mid);
        if(!std::isfinite(value))
        {
            UniversalError eo("Cooling limit: non-finite exchange rate during the temperature solve");
            eo.addEntry("s", mid);
            throw eo;
        }
        if(value > target)
            hi = mid;
        else
            lo = mid;
    }
    throw UniversalError("Cooling limit: temperature solve did not converge");
}

UniversalError CellError(std::string const& what, double T, double E, double V)
{
    UniversalError eo(what);
    eo.addEntry("Temperature", T);
    eo.addEntry("Radiation energy density", E);
    eo.addEntry("Volume", V);
    return eo;
}

struct CourantCell
{
    double time = std::numeric_limits<double>::max();  // without the CFL factor
    size_t id = 0;
    double radius = 0.0;
    double density = 0.0;
    double temperature = 0.0;
};

// Hydro time step from the RICH Courant condition. The snapshot has no mesh
// velocities; on RICH's Lagrangian mesh the faces move with the gas, so each
// face gets the mean velocity of its two cells (the one cell's velocity at
// the box boundary).
double CourantHydroTime(Config const& cfg, PostprocessRuntime& runtime, CourantCell& limiting)
{
    Voronoi3D& tess = runtime.tess;
    size_t const nCells = tess.GetPointNo();
    std::vector<double> velocity(3 * nCells);
    for(size_t i = 0; i < nCells; ++i)
    {
        velocity[3 * i] = runtime.cells[i].velocity.x;
        velocity[3 * i + 1] = runtime.cells[i].velocity.y;
        velocity[3 * i + 2] = runtime.cells[i].velocity.z;
    }
#ifdef RICH_MPI
    MPI_exchange_data(tess, velocity, true, 3);
#endif
    size_t const nPoints = velocity.size() / 3;
    auto hasVelocity = [&](size_t point) {
        return point < nPoints && !tess.IsPointOutsideBox(point);
    };
    auto pointVelocity = [&](size_t point) {
        return Vector3D(velocity[3 * point], velocity[3 * point + 1], velocity[3 * point + 2]);
    };
    size_t const nFaces = tess.GetTotalFacesNumber();
    std::vector<Vector3D> faceVelocity(nFaces, Vector3D(0.0, 0.0, 0.0));
    for(size_t face = 0; face < nFaces; ++face)
    {
        std::pair<size_t, size_t> const& neighbors = tess.GetFaceNeighbors(face);
        bool const first = hasVelocity(neighbors.first);
        bool const second = hasVelocity(neighbors.second);
        if(first && second)
            faceVelocity[face] = 0.5 * (pointVelocity(neighbors.first) + pointVelocity(neighbors.second));
        else if(first)
            faceVelocity[face] = pointVelocity(neighbors.first);
        else if(second)
            faceVelocity[face] = pointVelocity(neighbors.second);
    }

    ZeroForce3D noSource;
    CourantFriedrichsLewy courant(cfg.volumeEmissionCoolingLimitCfl, 1.0, noSource);
    double const hydroTime = courant(tess, runtime.cells, *runtime.eos, faceVelocity, runtime.snapshotTime);

    // Report only: the cell with the smallest Courant time, same expression
    // as CourantFriedrichsLewy::operator().
    CourantCell local;
    for(size_t i = 0; i < nCells; ++i)
    {
        ComputationalCell3D const& cell = runtime.cells[i];
        double const soundSpeed = runtime.eos->de2c(
            cell.density, cell.internal_energy, cell.tracers, ComputationalCell3D::tracerNames);
        face_vec const& faces = tess.GetCellFaces(i);
        double signal = 0.0;
        double maxFaceArea = 0.0;
        for(size_t j = 0; j < faces.size(); ++j)
        {
            Vector3D n = tess.Normal(faces[j]);
            n *= 1.0 / fastabs(n);
            signal = std::max(signal, soundSpeed + std::abs(ScalarProd(n, cell.velocity - faceVelocity[faces[j]])));
            maxFaceArea = std::max(maxFaceArea, tess.GetArea(faces[j]));
        }
        double const time = std::min(tess.GetWidth(i), tess.GetVolume(i) / maxFaceArea) / signal;
        if(time < local.time)
        {
            local.time = time;
            local.id = cell.ID;
            local.radius = fastabs(tess.GetMeshPoint(i) - cfg.center);
            local.density = cell.density;
            local.temperature = cell.temperature;
        }
    }
    limiting = local;
#ifdef RICH_MPI
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    struct { double value; int rank; } mine{local.time, rank}, owner{};
    MPI_Allreduce(&mine, &owner, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
    double report[5] = {local.time, static_cast<double>(local.id), local.radius, local.density, local.temperature};
    MPI_Bcast(report, 5, MPI_DOUBLE, owner.rank, MPI_COMM_WORLD);
    limiting.time = report[0];
    limiting.id = static_cast<size_t>(report[1]);
    limiting.radius = report[2];
    limiting.density = report[3];
    limiting.temperature = report[4];
#endif
    if(!(hydroTime > 0.0) || !std::isfinite(hydroTime))
    {
        UniversalError eo("Cooling limit: the Courant hydro time step is not positive and finite");
        eo.addEntry("Hydro time", hydroTime);
        throw eo;
    }
    return hydroTime;
}

} // namespace

CoolingLimitOutcome LimitCellCooling(
    double const T,
    double const E,
    double const V,
    double const hydroTime,
    CoolingLimitMaterial const& material)
{
    if(!(T > 0.0) || !std::isfinite(T) || !(E >= 0.0) || !std::isfinite(E) ||
       !(V > 0.0) || !std::isfinite(V))
        throw CellError("Cooling limit received an invalid cell state", T, E, V);
    if(!(hydroTime > 0.0) || !std::isfinite(hydroTime))
        throw CellError("Cooling limit requires a positive, finite hydro time", T, E, V);

    double const a = CG::radiation_constant;
    double const c = CG::speed_of_light;
    double const aT4 = a * T * T * T * T;
    double const kappa = material.planckOpacity(T);
    if(!(kappa >= 0.0) || !std::isfinite(kappa))
        throw CellError("Cooling limit: invalid Planck opacity", T, E, V);

    CoolingLimitOutcome outcome;
    outcome.temperature = T;
    outcome.netBefore = c * kappa * (aT4 - E) * V;
    outcome.netAfter = outcome.netBefore;

    // Radiative equilibrium to round-off (snapshot E = rho * (aT^4 / rho)).
    if(std::abs(aT4 - E) <= 1e-12 * std::max(aT4, E))
        return outcome;
    if(aT4 > E)
    {
        double const cv = material.heatCapacity;
        double const u = material.snapshotInternalEnergyDensity;
        if(!(cv > 0.0) || !std::isfinite(cv) || !(u > 0.0) || !std::isfinite(u))
        {
            UniversalError eo = CellError("Cooling limit: invalid heat capacity or internal energy", T, E, V);
            eo.addEntry("Heat capacity", cv);
            eo.addEntry("Internal energy density", u);
            throw eo;
        }
        // Energy the hydro's Fleck step removes per step when
        // beta c kappa dt >> 1 (Diffusion.cpp), per hydro time.
        double const beta = std::max(1.0, 0.5 * cv * T / u) * 4.0 * a * T * T * T / cv;
        double const cap = (aT4 - E) * V / (beta * hydroTime);
        if(!(outcome.netBefore > cap))
            return outcome;
        // Net emission at T' with s = aT'^4 - E in (0, aT^4 - E].
        auto net = [&](double s) {
            return c * material.planckOpacity(std::pow((E + s) / a, 0.25)) * s * V;
        };
        double const s = SolveForRate(net, aT4 - E, cap);
        outcome.kind = CoolingLimitKind::Hot;
        outcome.temperature = std::pow((E + s) / a, 0.25);
        outcome.netAfter = net(s);
    }
    else if(aT4 < E)
    {
        double const radiationTemperature = std::pow(E / a, 0.25);
        double const deficit =
            (material.internalEnergyDensity(radiationTemperature) - material.internalEnergyDensity(T)) * V;
        if(!(deficit > 0.0) || !std::isfinite(deficit))
        {
            UniversalError eo = CellError(
                "Cooling limit: internal energy does not increase with temperature", T, E, V);
            eo.addEntry("Energy to reach the radiation temperature", deficit);
            throw eo;
        }
        double const cap = deficit / hydroTime;
        if(!(-outcome.netBefore > cap))
            return outcome;
        // Net absorption at T' with s = E - aT'^4 in (0, E - aT^4].
        auto absorption = [&](double s) {
            return c * material.planckOpacity(std::pow((E - s) / a, 0.25)) * s * V;
        };
        double const s = SolveForRate(absorption, E - aT4, cap);
        outcome.kind = CoolingLimitKind::Cold;
        outcome.temperature = std::pow((E - s) / a, 0.25);
        outcome.netAfter = -absorption(s);
    }
    return outcome;
}

void ApplyVolumeEmissionCoolingLimit(
    Config const& cfg,
    PostprocessRuntime& runtime)
{
    if(!cfg.fluxSourceCompare || !cfg.volumeEmissionEnabled || !cfg.volumeEmissionCoolingLimit)
        return;
    if(runtime.coolingLimit.applied)
        throw UniversalError("Cooling limit applied twice to the same snapshot");
    if(!runtime.eos || !runtime.greyOpacity)
        throw UniversalError("Cooling limit needs the EOS and the grey opacity");
    size_t const nCells = runtime.tess.GetPointNo();
    if(runtime.cells.size() < nCells)
        throw UniversalError("Cooling limit: fewer cells than tessellation points");

    PostprocessRuntime::CoolingLimitSummary summary;
    CourantCell courantCell;
    summary.hydroTimeFromCourant = !(cfg.volumeEmissionCoolingLimitDt > 0.0);
    summary.hydroTime = summary.hydroTimeFromCourant
        ? CourantHydroTime(cfg, runtime, courantCell)
        : cfg.volumeEmissionCoolingLimitDt;
    summary.courantCellID = courantCell.id;
    summary.courantCellRadius = courantCell.radius;
    summary.courantCellDensity = courantCell.density;
    summary.courantCellTemperature = courantCell.temperature;

    std::vector<unsigned char> const outside = BuildOutsideFluxSourceMask(cfg, runtime);
    uint64_t counts[3] = {0, 0, 0};  // outside, hot limited, cold limited
    double sums[2] = {0.0, 0.0};      // net before, net after
    double maxChange = 0.0;
    for(size_t i = 0; i < nCells; ++i)
    {
        if(i >= outside.size() || outside[i] == 0)
            continue;
        ComputationalCell3D& cell = runtime.cells[i];
        double const rho = cell.density;
        CoolingLimitMaterial material;
        material.planckOpacity = [&](double temperature) {
            return runtime.greyOpacity->CalcPlanckOpacityAtTemperature(cell, temperature);
        };
        material.internalEnergyDensity = [&](double temperature) {
            return rho * runtime.eos->dT2e(rho, temperature, cell.tracers, ComputationalCell3D::tracerNames);
        };
        material.heatCapacity = runtime.eos->dT2cv(rho, cell.temperature, cell.tracers, ComputationalCell3D::tracerNames);
        material.snapshotInternalEnergyDensity = rho * cell.internal_energy;
        CoolingLimitOutcome outcome;
        try
        {
            outcome = LimitCellCooling(
                cell.temperature, rho * cell.Erad, runtime.tess.GetVolume(i), summary.hydroTime, material);
        }
        catch(UniversalError& eo)
        {
            eo.addEntry("Cell ID", cell.ID);
            eo.addEntry("Density", rho);
            throw;
        }
        ++counts[0];
        sums[0] += outcome.netBefore;
        sums[1] += outcome.netAfter;
        if(outcome.kind == CoolingLimitKind::Unchanged)
            continue;
        ++counts[outcome.kind == CoolingLimitKind::Hot ? 1 : 2];
        maxChange = std::max(maxChange, std::abs(outcome.temperature - cell.temperature) / cell.temperature);
        // The post-process transport reads only the temperature (it never
        // recomputes it from the internal energy).
        cell.temperature = outcome.temperature;
    }
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, counts, 3, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxChange, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
    summary.applied = true;
    summary.outsideCells = counts[0];
    summary.hotLimitedCells = counts[1];
    summary.coldLimitedCells = counts[2];
    summary.netEmissionBefore = sums[0];
    summary.netEmissionAfter = sums[1];
    summary.maxRelativeTemperatureChange = maxChange;
    runtime.coolingLimit = summary;

    if(runtime.rank == 0)
    {
        std::ostringstream line;
        line << std::setprecision(4)
             << "VOLUME_EMISSION_COOLING_LIMIT hydro_time=" << summary.hydroTime << " s";
        if(summary.hydroTimeFromCourant)
            line << " source=courant cfl=" << cfg.volumeEmissionCoolingLimitCfl
                 << " courant_cell_id=" << summary.courantCellID
                 << " courant_cell_r=" << summary.courantCellRadius
                 << " courant_cell_rho=" << summary.courantCellDensity
                 << " courant_cell_T=" << summary.courantCellTemperature;
        else
            line << " source=explicit";
        line << " outside_cells=" << summary.outsideCells
             << " hot_limited=" << summary.hotLimitedCells
             << " cold_limited=" << summary.coldLimitedCells
             << " net_emission_before=" << summary.netEmissionBefore
             << " net_emission_after=" << summary.netEmissionAfter << " erg/s"
             << " max_relative_dT=" << summary.maxRelativeTemperatureChange;
        std::cout << line.str() << std::endl;
    }
}

void BuildEscapeOriginHotCells(
    Config const& cfg,
    PostprocessRuntime& runtime)
{
    runtime.escapeOriginHotCells.clear();
    if(!cfg.fluxSourceCompare)
        return;
    std::vector<unsigned char> const outside = BuildOutsideFluxSourceMask(cfg, runtime);
    size_t const nCells = runtime.tess.GetPointNo();
    std::vector<uint64_t> localIDs;
    std::vector<unsigned char> localClasses;
    for(size_t i = 0; i < nCells && i < outside.size(); ++i)
    {
        double const T = runtime.cells[i].temperature;
        if(outside[i] == 0 || !(T >= 1e5))
            continue;
        localIDs.push_back(static_cast<uint64_t>(runtime.cells[i].ID));
        localClasses.push_back(static_cast<unsigned char>(T < 1e6 ? 2 : (T < 1e7 ? 3 : 4)));
    }
    std::vector<uint64_t> ids = localIDs;
    std::vector<unsigned char> classes = localClasses;
#ifdef RICH_MPI
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int const localCount = static_cast<int>(localIDs.size());
    std::vector<int> counts(size, 0), displs(size, 0);
    MPI_Allgather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    long long total = 0;
    for(int r = 0; r < size; ++r)
    {
        displs[r] = static_cast<int>(total);
        total += counts[r];
    }
    if(total > static_cast<long long>(std::numeric_limits<int>::max()))
        throw UniversalError("Escape-origin diagnostic: too many hot cells to gather");
    ids.assign(static_cast<size_t>(total), 0);
    classes.assign(static_cast<size_t>(total), 0);
    MPI_Allgatherv(localIDs.data(), localCount, MPI_UINT64_T,
                   ids.data(), counts.data(), displs.data(), MPI_UINT64_T, MPI_COMM_WORLD);
    MPI_Allgatherv(localClasses.data(), localCount, MPI_UNSIGNED_CHAR,
                   classes.data(), counts.data(), displs.data(), MPI_UNSIGNED_CHAR, MPI_COMM_WORLD);
#endif
    runtime.escapeOriginHotCells.reserve(ids.size());
    for(size_t k = 0; k < ids.size(); ++k)
        runtime.escapeOriginHotCells.emplace_back(static_cast<size_t>(ids[k]), classes[k]);
    std::sort(runtime.escapeOriginHotCells.begin(), runtime.escapeOriginHotCells.end());
    runtime.escapeOriginHotCells.shrink_to_fit();
    if(runtime.rank == 0)
    {
        size_t perClass[5] = {0, 0, 0, 0, 0};
        for(auto const& entry : runtime.escapeOriginHotCells)
            ++perClass[entry.second];
        std::cout << "ESCAPE_ORIGIN_SETUP surface_cells=" << runtime.fluxSourceCellIDs.size()
                  << " hot_cells_1e5_1e6K=" << perClass[2]
                  << " hot_cells_1e6_1e7K=" << perClass[3]
                  << " hot_cells_ge1e7K=" << perClass[4] << std::endl;
    }
}

void PrintCompressionReport(
    Config const& cfg,
    RadiationIMC const& physics,
    std::string const& label,
    size_t generation,
    int rank)
{
    if (!cfg.ddmc)
        return;
    double values[9] = {
        static_cast<double>(physics.getDDMCCompressionCutoffCount()),
        physics.getDDMCCompressionCutoffEnergy(),
        static_cast<double>(physics.getDDMCCompressionLimitedStepCount()),
        physics.getDDMCCompressionWorkGain(),
        physics.getDDMCCompressionWorkLoss(),
        static_cast<double>(physics.getDDMCStepCount()),
        static_cast<double>(physics.getDDMCCompressionMovingStepCount()),
        static_cast<double>(physics.getDDMCLeakCount()),
        static_cast<double>(physics.getDDMCCensusCount())};
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, values, 9, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
    if (rank != 0)
        return;
    std::ostringstream line;
    line << std::setprecision(4) << "COMPRESSION " << label << " gen=" << generation
         << " max_dynamical_times=" << cfg.ddmcCompressionMaxDynamicalTimes
         << " band_top_conversions=" << values[0]
         << " conversion_energy=" << values[1] << " erg"
         << " clock_ran_out=" << values[2]
         << " work_gain=" << values[3] << " work_loss=" << values[4] << " erg"
         << " ddmc_steps=" << values[5] << " moving_steps=" << values[6]
         << " leaks=" << values[7] << " census=" << values[8];
    std::cout << line.str() << std::endl;
}

} // namespace imc_postprocess_tde
