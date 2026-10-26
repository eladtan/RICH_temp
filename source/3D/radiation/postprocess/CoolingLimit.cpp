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
#include <unordered_map>
#include <vector>

#include "AdaptiveStatistics.hpp"
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

namespace {

// Global counters of one limit sweep.
struct LimitSweep
{
    uint64_t counts[3] = {0, 0, 0};  // outside, hot limited, cold limited
    double sums[2] = {0.0, 0.0};      // net before, net after
    double maxChange = 0.0;
};

// (ID, value) pairs of every rank, sorted by ID. Collective.
std::vector<std::pair<size_t, double>> GatherSortedByID(
    std::vector<uint64_t> const& ids,
    std::vector<double> const& values,
    char const* what)
{
    std::vector<uint64_t> allIDs = ids;
    std::vector<double> allValues = values;
#ifdef RICH_MPI
    int size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int const localCount = static_cast<int>(ids.size());
    std::vector<int> counts(static_cast<size_t>(size), 0), displs(counts.size(), 0);
    MPI_Allgather(&localCount, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    long long total = 0;
    for(size_t r = 0; r < counts.size(); ++r)
    {
        displs[r] = static_cast<int>(total);
        total += counts[r];
    }
    if(total > static_cast<long long>(std::numeric_limits<int>::max()))
        throw UniversalError(std::string("Cooling limit: too many ") + what + " to gather");
    allIDs.assign(static_cast<size_t>(total), 0);
    allValues.assign(static_cast<size_t>(total), 0.0);
    MPI_Allgatherv(ids.data(), localCount, MPI_UINT64_T,
                   allIDs.data(), counts.data(), displs.data(), MPI_UINT64_T, MPI_COMM_WORLD);
    MPI_Allgatherv(values.data(), localCount, MPI_DOUBLE,
                   allValues.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
#else
    (void)what;
#endif
    std::vector<std::pair<size_t, double>> sorted;
    sorted.reserve(allIDs.size());
    for(size_t k = 0; k < allIDs.size(); ++k)
        sorted.emplace_back(static_cast<size_t>(allIDs[k]), allValues[k]);
    std::sort(sorted.begin(), sorted.end());
    return sorted;
}

// LimitCellCooling on every owned cell at or beyond the flux-source surface
// with the Planck mean of `planckOpacity` (times alpha(T') with Rosseland
// scaling, whose factors the caller has cleared). A cell whose temperature
// changes (every limited cell in temperature mode, cold ones in emission mode)
// takes the limited temperature, and `changedTemperatures`, when given,
// receives its ID and previous temperature. Emission mode: hot-limited cells
// keep their temperature and runtime.coolingLimitEmissionCaps becomes their
// caps. Collective.
LimitSweep LimitOutsideCells(
    Config const& cfg,
    PostprocessRuntime& runtime,
    double hydroTime,
    OpacityCalculator const& planckOpacity,
    bool rosselandScaled,
    std::vector<std::pair<uint64_t, double>>* changedTemperatures)
{
    size_t const nCells = runtime.tess.GetPointNo();
    if(runtime.cells.size() < nCells)
        throw UniversalError("Cooling limit: fewer cells than tessellation points");

    std::vector<unsigned char> const outside = BuildOutsideFluxSourceMask(cfg, runtime);
    LimitSweep sweep;
    bool const emissionMode = cfg.volumeEmissionCoolingLimitMode == CoolingLimitMode::Emission;
    std::vector<uint64_t> capIDs;     // emission mode: hot-limited cells and their caps
    std::vector<double> capValues;
    for(size_t i = 0; i < nCells; ++i)
    {
        if(i >= outside.size() || outside[i] == 0)
            continue;
        ComputationalCell3D& cell = runtime.cells[i];
        double const rho = cell.density;
        CoolingLimitMaterial material;
        // Rosseland scaling: the cell at the trial temperature, copied once.
        ComputationalCell3D atTemperature;
        if(rosselandScaled)
            atTemperature = cell;
        material.planckOpacity = [&](double temperature) {
            double scale = 1.0;
            if(rosselandScaled)
            {
                atTemperature.temperature = temperature;
                if(!ComputeOpacityScaleFactor(*runtime.opacity, *runtime.greyOpacity, atTemperature,
                                              OpacityScaleMode::Rosseland, scale))
                {
                    UniversalError eo("Cooling limit: no Rosseland opacity scale factor");
                    eo.addEntry("Cell ID", cell.ID);
                    eo.addEntry("Temperature", temperature);
                    throw eo;
                }
                if(cfg.opacityScaleAlphaMin > 0.0)
                    scale = std::max(scale, cfg.opacityScaleAlphaMin);
                if(cfg.opacityScaleAlphaMax > 0.0)
                    scale = std::min(scale, cfg.opacityScaleAlphaMax);
            }
            return scale * planckOpacity.CalcPlanckOpacityAtTemperature(cell, temperature);
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
                cell.temperature, rho * cell.Erad, runtime.tess.GetVolume(i), hydroTime, material);
        }
        catch(UniversalError& eo)
        {
            eo.addEntry("Cell ID", cell.ID);
            eo.addEntry("Density", rho);
            throw;
        }
        ++sweep.counts[0];
        sweep.sums[0] += outcome.netBefore;
        sweep.sums[1] += outcome.netAfter;
        if(outcome.kind == CoolingLimitKind::Unchanged)
            continue;
        ++sweep.counts[outcome.kind == CoolingLimitKind::Hot ? 1 : 2];
        if(emissionMode && outcome.kind == CoolingLimitKind::Hot)
        {
            // The cell keeps its temperature; its emission is scaled at each
            // source setup so that its net emission equals the cap.
            capIDs.push_back(static_cast<uint64_t>(cell.ID));
            capValues.push_back(outcome.netAfter);
            continue;
        }
        sweep.maxChange = std::max(sweep.maxChange, std::abs(outcome.temperature - cell.temperature) / cell.temperature);
        if(changedTemperatures)
            changedTemperatures->emplace_back(static_cast<uint64_t>(cell.ID), cell.temperature);
        // The post-process transport reads only the temperature (it never
        // recomputes it from the internal energy).
        cell.temperature = outcome.temperature;
    }
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, sweep.counts, 3, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, sweep.sums, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &sweep.maxChange, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
    if(emissionMode)
    {
        runtime.coolingLimitEmissionCaps = GatherSortedByID(capIDs, capValues, "limited cells");
        for(auto const& cap : runtime.coolingLimitEmissionCaps)
        {
            if(!(cap.second > 0.0) || !std::isfinite(cap.second))
            {
                UniversalError eo("Cooling limit: a hot-limited cell has an invalid cap");
                eo.addEntry("Cell ID", cap.first);
                eo.addEntry("Cap", cap.second);
                throw eo;
            }
        }
    }
    return sweep;
}

void StoreSweep(LimitSweep const& sweep, PostprocessRuntime::CoolingLimitSummary& summary)
{
    summary.applied = true;
    summary.outsideCells = sweep.counts[0];
    summary.hotLimitedCells = sweep.counts[1];
    summary.coldLimitedCells = sweep.counts[2];
    summary.netEmissionBefore = sweep.sums[0];
    summary.netEmissionAfter = sweep.sums[1];
    summary.maxRelativeTemperatureChange = sweep.maxChange;
}

// Rank 0 prints the VOLUME_EMISSION_COOLING_LIMIT line; `pass` is empty for
// the multigroup pass's limit (the line it has always printed).
void PrintCoolingLimit(
    Config const& cfg,
    PostprocessRuntime const& runtime,
    PostprocessRuntime::CoolingLimitSummary const& summary,
    std::string const& pass,
    std::string const& planckOpacity)
{
    if(runtime.rank != 0)
        return;
    std::ostringstream line;
    line << std::setprecision(4) << "VOLUME_EMISSION_COOLING_LIMIT";
    if(!pass.empty())
        line << " pass=" << pass;
    line << " hydro_time=" << summary.hydroTime << " s";
    if(summary.hydroTimeFromCourant)
        line << " source=courant cfl=" << cfg.volumeEmissionCoolingLimitCfl
             << " courant_cell_id=" << summary.courantCellID
             << " courant_cell_r=" << summary.courantCellRadius
             << " courant_cell_rho=" << summary.courantCellDensity
             << " courant_cell_T=" << summary.courantCellTemperature;
    else
        line << " source=explicit";
    line << " planck_opacity=" << planckOpacity
         << " mode=" << (cfg.volumeEmissionCoolingLimitMode == CoolingLimitMode::Emission ? "emission" : "temperature")
         << " outside_cells=" << summary.outsideCells
         << " hot_limited=" << summary.hotLimitedCells
         << " cold_limited=" << summary.coldLimitedCells
         << " net_emission_before=" << summary.netEmissionBefore
         << " net_emission_after=" << summary.netEmissionAfter << " erg/s"
         << " max_relative_dT=" << summary.maxRelativeTemperatureChange;
    std::cout << line.str() << std::endl;
}

} // namespace

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
    if(cfg.opacityScaleMode == OpacityScaleMode::Auto)
        throw UniversalError("Cooling limit: opacity scaling mode auto was not resolved from the snapshot");
    // The cell's net emission is c kappa_P (aT^4 - E) V with the Planck mean
    // the multigroup transport emits with. With opacity scaling the group
    // opacities reproduce the grey Planck mean, so the grey table gives it.
    // Without scaling (post-processing a multigroup hydro run) it is the
    // Planck mean of the unscaled group opacities, typically far below the
    // grey one; the grey table would then over-cool the cells.
    // With Rosseland scaling the scaled group opacities match the grey
    // Rosseland mean, not the Planck one: the Planck mean is the scaled group
    // one, alpha(T') times the unscaled mean at T'. The scale factors are
    // cleared for the limit and rebuilt after it.
    // In the last two cases the grey pass, which emits with the grey table,
    // redoes the limit with it (ApplyGreyPassCoolingLimit).
    bool const multigroupPlanck = ENERGY_GROUPS_NUM > 1 && cfg.opacityScaleMode == OpacityScaleMode::None;
    bool const rosselandScaled = ENERGY_GROUPS_NUM > 1 && cfg.opacityScaleMode == OpacityScaleMode::Rosseland;
    if((multigroupPlanck || rosselandScaled) && !runtime.opacity)
        throw UniversalError("Cooling limit with multigroup Planck opacity needs the multigroup opacity");
    if(rosselandScaled)
    {
        if(!runtime.applyOpacityScaleFactors)
            throw UniversalError("Cooling limit with Rosseland scaling needs the opacity scaling policy");
        runtime.applyOpacityScaleFactors(std::unordered_map<size_t, double>());
    }
    OpacityCalculator const& planckOpacity = (multigroupPlanck || rosselandScaled) ? *runtime.opacity : *runtime.greyOpacity;

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
    summary.greyPlanckOpacity = !(multigroupPlanck || rosselandScaled);

    // The grey pass's limit starts from the snapshot temperatures.
    bool const keepSnapshotTemperatures = cfg.volumeEmissionCoolingLimitPerPass && !summary.greyPlanckOpacity;
    std::vector<std::pair<uint64_t, double>> changed;
    LimitSweep const sweep = LimitOutsideCells(
        cfg, runtime, summary.hydroTime, planckOpacity, rosselandScaled,
        keepSnapshotTemperatures ? &changed : nullptr);
    runtime.coolingLimitSnapshotTemperatures.clear();
    if(keepSnapshotTemperatures)
    {
        std::vector<uint64_t> ids;
        std::vector<double> temperatures;
        ids.reserve(changed.size());
        temperatures.reserve(changed.size());
        for(auto const& entry : changed)
        {
            ids.push_back(entry.first);
            temperatures.push_back(entry.second);
        }
        runtime.coolingLimitSnapshotTemperatures = GatherSortedByID(ids, temperatures, "changed cells");
    }
    StoreSweep(sweep, summary);
    runtime.coolingLimit = summary;
    runtime.greyCoolingLimit = PostprocessRuntime::CoolingLimitSummary();
    PrintCoolingLimit(cfg, runtime, summary, "",
                      multigroupPlanck ? "multigroup" : rosselandScaled ? "multigroup-rosseland-scaled" : "grey");
}

void ApplyGreyPassCoolingLimit(
    Config const& cfg,
    PostprocessRuntime& runtime)
{
    PostprocessRuntime::CoolingLimitSummary const& transport = runtime.coolingLimit;
    if(!transport.applied || transport.greyPlanckOpacity || !cfg.volumeEmissionCoolingLimitPerPass)
        return;
    if(runtime.greyCoolingLimit.applied)
        throw UniversalError("Grey-pass cooling limit applied twice");
    if(!runtime.eos || !runtime.greyOpacity)
        throw UniversalError("Grey-pass cooling limit needs the EOS and the grey opacity");

    // Back to the snapshot temperatures. The cells may have migrated since the
    // multigroup limit, so they are found by ID.
    auto const& snapshot = runtime.coolingLimitSnapshotTemperatures;
    size_t const nCells = runtime.tess.GetPointNo();
    if(runtime.cells.size() < nCells)
        throw UniversalError("Grey-pass cooling limit: fewer cells than tessellation points");
    uint64_t restored = 0;
    for(size_t i = 0; i < nCells; ++i)
    {
        size_t const id = runtime.cells[i].ID;
        auto const it = std::lower_bound(
            snapshot.begin(), snapshot.end(), std::make_pair(id, -std::numeric_limits<double>::infinity()));
        if(it == snapshot.end() || it->first != id)
            continue;
        runtime.cells[i].temperature = it->second;
        ++restored;
    }
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &restored, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
#endif
    if(restored != snapshot.size())
    {
        UniversalError eo("Grey-pass cooling limit: not every cell the multigroup limit changed is owned by a rank");
        eo.addEntry("Changed cells", snapshot.size());
        eo.addEntry("Restored cells", restored);
        throw eo;
    }

    PostprocessRuntime::CoolingLimitSummary summary;
    summary.hydroTime = transport.hydroTime;
    summary.hydroTimeFromCourant = transport.hydroTimeFromCourant;
    summary.courantCellID = transport.courantCellID;
    summary.courantCellRadius = transport.courantCellRadius;
    summary.courantCellDensity = transport.courantCellDensity;
    summary.courantCellTemperature = transport.courantCellTemperature;
    summary.greyPlanckOpacity = true;
    StoreSweep(LimitOutsideCells(cfg, runtime, summary.hydroTime, *runtime.greyOpacity, false, nullptr), summary);
    runtime.greyCoolingLimit = summary;
    runtime.coolingLimitSnapshotTemperatures.clear();
    runtime.coolingLimitSnapshotTemperatures.shrink_to_fit();
    PrintCoolingLimit(cfg, runtime, summary, "grey", "grey");
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

double CoolingLimitEmissionScale(
    PostprocessRuntime const& runtime,
    size_t cellID,
    double gross,
    double absorbed)
{
    auto const& caps = runtime.coolingLimitEmissionCaps;
    if(caps.empty() || !(gross > 0.0))
        return 1.0;
    auto const it = std::lower_bound(caps.begin(), caps.end(), std::make_pair(cellID, -1.0));
    if(it == caps.end() || it->first != cellID)
        return 1.0;
    double const scale = (it->second + std::max(0.0, absorbed)) / gross;
    if(!std::isfinite(scale))
    {
        UniversalError eo("Cooling limit: invalid emission scale");
        eo.addEntry("Cell ID", cellID);
        eo.addEntry("Gross emission", gross);
        eo.addEntry("Absorbed", absorbed);
        throw eo;
    }
    return std::min(1.0, scale);
}

void ReportVolumeSourceGroups(
    Config const& cfg,
    PostprocessRuntime& runtime,
    OpacityCalculator const& opacity,
    std::string const& stage)
{
    if(!cfg.fluxSourceCompare || !cfg.volumeEmissionEnabled)
        return;
    size_t const groups = static_cast<size_t>(ENERGY_GROUPS_NUM);
    std::vector<double> const& edges = opacity.energy_groups_boundary;
    if(groups < 2 || edges.size() != groups + 1 || opacity.energy_groups_center.size() != groups)
        return;
    double constexpr cLight = 2.99792458e10;
    double constexpr hardEdge = 0.999 * 0.3 * 1.602176634e-9;  // 0.3 keV in erg
    size_t constexpr kBins = 5;  // T < 1e4, [1e4, 1e5), [1e5, 1e6), [1e6, 1e7), >= 1e7 K

    std::vector<unsigned char> const outside = BuildOutsideFluxSourceMask(cfg, runtime);
    size_t const nCells = runtime.tess.GetPointNo();
    // [gross per group | snapshot absorbed per group | cells per T bin |
    //  gross per T bin | gross above 0.3 keV per T bin | cells without E_g]
    std::vector<double> sums(2 * groups + 3 * kBins + 1, 0.0);
    double maxT = 0.0;
    for(size_t i = 0; i < nCells && i < outside.size(); ++i)
    {
        if(outside[i] == 0)
            continue;
        ComputationalCell3D const& cell = runtime.cells[i];
        double const T = cell.temperature;
        double const volume = runtime.tess.GetVolume(i);
        size_t const bin = T < 1e4 ? 0 : (T < 1e5 ? 1 : (T < 1e6 ? 2 : (T < 1e7 ? 3 : 4)));
        maxT = std::max(maxT, T);
        bool const hasGroups = cfg.snapshotGroupEnergyUsable && cell.Eg.size() == groups;
        if(!hasGroups)
            sums.back() += 1.0;
        double gross = 0.0;
        double hard = 0.0;
        std::vector<double> emission(groups, 0.0);
        for(size_t g = 0; g < groups; ++g)
        {
            double const kappa = opacity.CalcAbsorptionOpacity(cell, opacity.energy_groups_center[g]);
            emission[g] = cLight * kappa *
                planck_integral::planck_energy_density_group_integral(edges[g], edges[g + 1], T) * volume;
            if(hasGroups)
                sums[groups + g] += cLight * kappa * cell.density * cell.Eg[g] * volume;
            gross += emission[g];
        }
        // Emission-mode cooling limit: the emission is scaled, with the
        // absorption of the snapshot radiation at the same Planck mean.
        double const aT4 = CG::radiation_constant * std::pow(T, 4);
        double const radiationEnergy = cell.density * cell.Erad;
        double const absorbed = aT4 > 0.0 && std::isfinite(radiationEnergy) && radiationEnergy > 0.0
            ? gross * radiationEnergy / aT4 : 0.0;
        double const scale = CoolingLimitEmissionScale(runtime, cell.ID, gross, absorbed);
        gross *= scale;
        for(size_t g = 0; g < groups; ++g)
        {
            sums[g] += emission[g] * scale;
            if(edges[g] >= hardEdge)
                hard += emission[g] * scale;
        }
        sums[2 * groups + bin] += 1.0;
        sums[2 * groups + kBins + bin] += gross;
        sums[2 * groups + 2 * kBins + bin] += hard;
    }
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, sums.data(), static_cast<int>(sums.size()), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maxT, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
    if(runtime.rank != 0)
        return;
    size_t const none = std::numeric_limits<size_t>::max();
    // sums[offset + k], minus sums[subtract + k] unless subtract is none.
    auto list = [&](size_t offset, size_t count, size_t subtract) {
        std::ostringstream out;
        out << std::setprecision(4) << "[";
        for(size_t k = 0; k < count; ++k)
            out << (k ? "," : "")
                << sums[offset + k] - (subtract == none ? 0.0 : sums[subtract + k]);
        out << "]";
        return out.str();
    };
    double outsideCells = 0.0;
    for(size_t b = 0; b < kBins; ++b)
        outsideCells += sums[2 * groups + b];
    std::ostringstream line;
    line << std::setprecision(4)
         << "SOURCE_GROUP_VOLUME stage=" << stage
         << " outside_cells=" << outsideCells
         << " T_max=" << maxT << " K"
         << " cells_without_group_energy=" << sums.back()
         << " gross=" << list(0, groups, none)
         << " snapshot_absorbed="
         << (cfg.snapshotGroupEnergyUsable ? list(groups, groups, none) : std::string("n/a (snapshot type)"))
         << " net=" << (cfg.snapshotGroupEnergyUsable ? list(0, groups, groups) : std::string("n/a (snapshot type)"))
         << " erg/s\n"
         << "SOURCE_GROUP_VOLUME_T stage=" << stage
         << " T_bins=[<1e4,1e4-1e5,1e5-1e6,1e6-1e7,>=1e7]K"
         << " cells=" << list(2 * groups, kBins, none)
         << " gross=" << list(2 * groups + kBins, kBins, none)
         << " gross_ge_0.3keV=" << list(2 * groups + 2 * kBins, kBins, none)
         << " erg/s";
    std::cout << line.str() << std::endl;
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
