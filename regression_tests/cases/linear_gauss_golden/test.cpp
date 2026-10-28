// Records the hydro's outputs on fixed inputs, or checks them bit for bit against a recording.
//   rich write <dir> [points]   record into <dir>
//   rich check <dir> [points]   compare with the recording in <dir> (same rank count)
// Each configuration builds a mesh, calls LinearGauss3D (operator(), BuildSlopes, the dissipation), the fluxes, the
// extensive update, the time step, the point velocities and the cell update, then two time steps of HDSim3D, and
// dumps their results as raw doubles.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <mpi.h>

#include "source/3D/tessellation/Voronoi3D.hpp"
#include "source/misc/universal_error.hpp"
#include "source/mpi/mpi_commands.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/Ghost3D.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/Lagrangian3D.hpp"
#include "source/newtonian/three_dimensional/RoundCells3D.hpp"
#include "source/newtonian/three_dimensional/SourceTerm3D.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/ProgressTracker.hpp"
#if __has_include("source/newtonian/three_dimensional/LinearGaussKernels.hpp")
#define GAUSS_HAS_DEVICE 1 // the portable LinearGauss3D (the recorded reference is built without it)
#endif
#include "source/utils/mpi_utils/mpi_collectives.hpp"

namespace
{

constexpr double gamma53 = 5.0 / 3.0;

struct Config
{
    std::string name;
    bool periodic = false;
    bool slf = true;
    double deltaV = 0.2, theta = 0.5, deltaP = 0.7;
    bool sr = false;
    std::vector<std::string> tracers, calcTracers, stickers;
    std::string skipKey;
    bool pressureCalc = true;
    bool principal = false;
    double velocityScale = 1.0;
    bool jump = true; // a jump in density and pressure (smooth fields otherwise)
    double extensiveCfl = 0.1; // the extensive update's time step, as the largest fraction of a cell's mass or energy a step moves
    double densityScale = 1.0; // scales the density and the pressure
    bool cold = false; // RoundCells3D's cold variant
};

std::vector<Config> Configs()
{
    std::vector<Config> configs(9);
    configs[0].name = "default";
    configs[1].name = "tde";
    configs[1].theta = 0.25;
    configs[1].deltaP = 0.75;
    configs[1].tracers = {"t0", "t1"};
    configs[1].calcTracers = {"t0"};
    configs[1].stickers = {"s0"};
    configs[1].skipKey = "s0";
    configs[1].principal = true;
    configs[1].cold = true;
    configs[2].name = "energy";
    configs[2].pressureCalc = false;
    configs[2].tracers = {"Energy", "Entropy", "t0"};
    configs[2].calcTracers = {"Energy", "t0"};
    configs[3].name = "sr";
    configs[3].sr = true;
    configs[3].velocityScale = 0.3;
    configs[4].name = "nolimit";
    configs[4].slf = false;
    configs[4].jump = false;
    configs[5].name = "periodic";
    configs[5].periodic = true;
    configs[5].tracers = {"t0"};
    configs[5].calcTracers = {"t0"};
    // An unlimited slope across the jump makes a face's pressure negative: checks the error
    configs[6].name = "nolimit_jump";
    configs[6].slf = false;
    // A time step far too large for the extensive update: checks its error
    configs[7].name = "badupdate";
    configs[7].extensiveCfl = 1e3;
    // Fast flow with an entropy tracer, and densities about the cell update's floor (2e-21): its entropy check and
    // its floor
    configs[8].name = "fast_entropy";
    configs[8].tracers = {"Entropy"};
    configs[8].velocityScale = 30.0;
    configs[8].densityScale = 1.5e-21;
    return configs;
}

// A dense ball (a jump in density and pressure) in converging, rotating flow, with smooth variations
ComputationalCell3D State(const Vector3D &r, std::size_t id, const Config &config)
{
    const Vector3D c(0.5, 0.5, 0.5);
    const Vector3D d = r - c;
    const double radius = abs(d);
    const bool inside = config.jump and radius < 0.25;
    ComputationalCell3D cell;
    cell.density = config.densityScale * ((inside ? 4.0 : 1.0) + 0.1 * std::sin(7.0 * r.x) * std::cos(5.0 * r.y));
    cell.pressure = config.densityScale * ((inside ? 10.0 : 1.0) + 0.05 * std::cos(6.0 * r.z));
    cell.internal_energy = cell.pressure / (cell.density * (gamma53 - 1.0));
    cell.velocity = config.velocityScale * (-0.5 * d + 0.2 * Vector3D(-d.y, d.x, 0.1 * std::sin(9.0 * r.z)));
    cell.temperature = cell.pressure / cell.density;
    cell.cs = std::sqrt(gamma53 * cell.pressure / cell.density);
    cell.dt = 1e-3 * (1.0 + r.x);
    cell.Erad = 0.5 + r.y;
    cell.Erad_dt = 0.1 + r.z;
    cell.Erad_dt_dt = 0.2 + 2.0 * r.z;
    for(std::size_t g = 0; g < cell.Eg.size(); ++g)
    {
        cell.Eg[g] = 0.25 + r.x + static_cast<double>(g);
    }
    for(std::size_t t = 0; t < config.tracers.size(); ++t)
    {
        const std::string &name = config.tracers[t];
        cell.tracers[t] = name == "Energy" ? cell.internal_energy : (t == 0 ? 1.0 + r.x : (radius < 0.3 ? 2.0 : 1.0));
    }
    for(std::size_t s = 0; s < config.stickers.size(); ++s)
    {
        cell.stickers[s] = r.x > 0.7;
    }
    cell.ID = id;
    return cell;
}

// Prints a phase's time (with GAUSS_PHASES, on rank 0)
void PrintPhase(int rank, const char *name, double start)
{
    if(std::getenv("GAUSS_PHASES") != nullptr and rank == 0)
    {
        std::cout << name << " " << MPI_Wtime() - start << " s" << std::endl;
    }
}

// Named arrays of doubles, compared by their bits
using Dump = std::map<std::string, std::vector<double>>;

void AppendCell(std::vector<double> &out, const ComputationalCell3D &cell)
{
    out.insert(out.end(), {cell.density, cell.pressure, cell.internal_energy, cell.temperature,
                           cell.velocity.x, cell.velocity.y, cell.velocity.z, cell.dt, cell.Erad,
                           cell.Erad_dt, cell.Erad_dt_dt, cell.cs});
    for(std::size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
    {
        out.push_back(cell.Eg[g]);
    }
    out.insert(out.end(), cell.tracers.begin(), cell.tracers.end());
}

void AppendConserved(std::vector<double> &out, const Conserved3D &c)
{
    out.insert(out.end(), {c.mass, c.momentum.x, c.momentum.y, c.momentum.z, c.energy, c.internal_energy, c.Erad,
                           c.Erad_dt, c.Erad_dt_dt});
    out.insert(out.end(), c.Eg.begin(), c.Eg.end());
    out.insert(out.end(), c.tracers.begin(), c.tracers.end());
}

// The fields a slope carries (the naive slope keeps only these)
void AppendSlopeFields(std::vector<double> &out, const ComputationalCell3D &d)
{
    out.insert(out.end(), {d.density, d.pressure, d.internal_energy, d.velocity.x, d.velocity.y,
                           d.velocity.z, d.cs, d.dt});
    out.insert(out.end(), d.tracers.begin(), d.tracers.end());
}

void WriteDump(const std::string &path, const Dump &dump)
{
    std::ofstream file(path, std::ios::binary);
    for(const auto &[name, values] : dump)
    {
        const std::uint64_t nameSize = name.size(), count = values.size();
        file.write(reinterpret_cast<const char *>(&nameSize), sizeof(nameSize));
        file.write(name.data(), static_cast<std::streamsize>(nameSize));
        file.write(reinterpret_cast<const char *>(&count), sizeof(count));
        file.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(count * sizeof(double)));
    }
    if(!file)
    {
        throw UniversalError("linear_gauss_golden: cannot write " + path);
    }
}

Dump ReadDump(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    if(!file)
    {
        throw UniversalError("linear_gauss_golden: cannot read " + path);
    }
    Dump dump;
    std::uint64_t nameSize = 0;
    while(file.read(reinterpret_cast<char *>(&nameSize), sizeof(nameSize)))
    {
        std::string name(nameSize, ' ');
        std::uint64_t count = 0;
        file.read(name.data(), static_cast<std::streamsize>(nameSize));
        file.read(reinterpret_cast<char *>(&count), sizeof(count));
        std::vector<double> &values = dump[name];
        values.resize(count);
        file.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(count * sizeof(double)));
    }
    return dump;
}

// An error's message and entries, without the stack trace
std::string ErrorText(const UniversalError &error)
{
    std::ostringstream stream;
    reportError(error, stream);
    std::string text = stream.str();
    const std::size_t stack = text.find("Stack trace");
    if(stack != std::string::npos)
    {
        text.erase(text.rfind('\n', stack) + 1);
    }
    return text;
}

void WriteText(const std::string &path, const std::string &text)
{
    std::ofstream(path) << text;
}

// The file's text, or nothing if there is no such file
std::string ReadText(const std::string &path)
{
    std::ifstream file(path);
    std::ostringstream text;
    text << file.rdbuf();
    return file ? text.str() : std::string();
}

// Returns the number of differing values, and prints each differing array
std::size_t Compare(const Dump &expected, const Dump &actual, const std::string &label)
{
    std::size_t differences = 0;
    for(const auto &[name, values] : expected)
    {
        const auto it = actual.find(name);
        if(it == actual.end() or it->second.size() != values.size())
        {
            std::cout << label << " " << name << ": missing or of a different size" << std::endl;
            differences += values.size() + 1;
            continue;
        }
        std::size_t count = 0, first = 0;
        double maxAbs = 0, maxRel = 0;
        for(std::size_t i = 0; i < values.size(); ++i)
        {
            if(std::memcmp(&values[i], &it->second[i], sizeof(double)) != 0)
            {
                if(count == 0)
                {
                    first = i;
                }
                ++count;
                const double diff = std::abs(values[i] - it->second[i]);
                maxAbs = std::max(maxAbs, diff);
                maxRel = std::max(maxRel, diff / std::max(std::abs(values[i]), 1e-300));
            }
        }
        if(count > 0)
        {
            std::cout << label << " " << name << ": " << count << " of " << values.size()
                      << " differ, first at " << first << " (" << values[first] << " vs " << it->second[first]
                      << "), max abs " << maxAbs << ", max rel " << maxRel << std::endl;
        }
        differences += count;
    }
    return differences;
}

// The time of the last LinearGauss3D::operator() call
double interpSeconds = 0;

Dump Run(const Config &config, std::size_t pointsNum, int rank)
{
    ComputationalCell3D::tracerNames = config.tracers;
    ComputationalCell3D::stickerNames = config.stickers;

    const Vector3D ll(0, 0, 0), ur(1, 1, 1);
    std::vector<Vector3D> points;
    if(rank == 0)
    {
        std::mt19937_64 generator(12345);
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        points.resize(pointsNum);
        for(Vector3D &point : points)
        {
            point.x = uniform(generator);
            point.y = uniform(generator);
            point.z = uniform(generator);
        }
    }
    points = MPI_Spread(points, 0, MPI_COMM_WORLD);
    Voronoi3D tess(ll, ur);
    if(config.periodic)
    {
        tess.SetPeriodic(true, true, true);
    }
    tess.BuildParallel(points);

    const std::size_t cellsNum = tess.GetPointNo();
    std::vector<ComputationalCell3D> cells(cellsNum);
    for(std::size_t i = 0; i < cellsNum; ++i)
    {
        cells[i] = State(tess.GetMeshPoint(i), i, config);
    }
    MPI_exchange_data(tess, cells, true);

    IdealGas eos(gamma53);
    RigidWallGenerator3D ghost;
    LinearGauss3D interp(eos, ghost, config.slf, config.deltaV, config.theta, config.deltaP, config.sr,
                         config.calcTracers, config.skipKey, config.pressureCalc, config.principal);
    // GAUSS_DEVICE=1 runs the kernels on the device
#ifdef GAUSS_HAS_DEVICE
    const char *device = std::getenv("GAUSS_DEVICE");
    interp.RunOnDevice(device != nullptr and std::string(device) == "1");
#endif

    Dump dump;
    std::vector<double> &faces = dump["faces"];
    std::vector<std::pair<ComputationalCell3D, ComputationalCell3D>> faceValues;
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    interp(tess, cells, 0.0, faceValues);
    MPI_Barrier(MPI_COMM_WORLD);
    interpSeconds = MPI_Wtime() - start;
    for(const auto &[first, second] : faceValues)
    {
        AppendCell(faces, first);
        AppendCell(faces, second);
    }
    const auto appendSlopes = [&](const std::string &prefix)
    {
        std::vector<double> &limited = dump[prefix + "slopes"];
        std::vector<double> &naive = dump[prefix + "naive"];
        for(std::size_t i = 0; i < cellsNum; ++i)
        {
            const Slope3D &slope = interp.GetSlopes()[i];
            for(const ComputationalCell3D *d : {&slope.xderivative, &slope.yderivative, &slope.zderivative})
            {
                AppendCell(limited, *d);
            }
            const Slope3D &unlimited = interp.GetSlopesUnlimited()[i];
            for(const ComputationalCell3D *d : {&unlimited.xderivative, &unlimited.yderivative, &unlimited.zderivative})
            {
                AppendSlopeFields(naive, *d);
            }
        }
    };
    appendSlopes("");

    // Interp: each cell at its first face's centroid
    std::vector<double> &interpolated = dump["interp"];
    for(std::size_t i = 0; i < cellsNum; ++i)
    {
        ComputationalCell3D value;
        interp.Interp(value, cells[i], i, tess.GetCellCM(i), tess.FaceCM(tess.GetCellFaces(i)[0]), eos);
        AppendCell(interpolated, value);
    }

    // The fluxes of the production chain (rigid walls, HLLC in the bulk), with the default solver and with the
    // iterated wave speeds. The face velocities vary, and some are fast enough for the solver's fast flow.
    const std::size_t facesNum = tess.GetTotalFacesNumber();
    std::vector<Vector3D> faceVelocities(facesNum);
    for(std::size_t f = 0; f < facesNum; ++f)
    {
        const Vector3D &cm = tess.FaceCM(f);
        faceVelocities[f] = 0.2 * Vector3D(std::sin(5.0 * cm.x), std::cos(3.0 * cm.y), std::sin(4.0 * cm.z));
        if(static_cast<long>(1000.0 * (cm.x + cm.y + cm.z)) % 13 == 0)
        {
            faceVelocities[f] *= 1e4;
        }
    }
    IsBoundaryFace3D boundaryFace;
    IsBulkFace3D bulkFace;
    for(const double solverGamma : {-1.0, gamma53})
    {
        Hllc3D rs(solverGamma);
        RigidWallFlux3D wallFlux(rs);
        RegularFlux3D bulkFlux(rs);
        ConditionActionFlux1 flux({{&boundaryFace, &wallFlux}, {&bulkFace, &bulkFlux}}, interp);
        std::vector<Conserved3D> fluxes;
        // As in a run, the face values' vector holds the last step's values (here other ones), all to be rewritten
        ComputationalCell3D stale = State(Vector3D(0.1, 0.2, 0.3), 77, config);
        stale.density = -7.0;
        stale.Erad = 9.0;
        stale.tracers.fill(5.0);
        stale.stickers.fill(true);
        std::vector<std::pair<ComputationalCell3D, ComputationalCell3D>> fluxFaceValues(facesNum, {stale, stale});
        const double fluxStart = MPI_Wtime();
        flux.Calculate(fluxes, tess, faceVelocities, cells, std::vector<Conserved3D>(), eos, 0.0, 0.0, fluxFaceValues);
        if(std::getenv("GAUSS_PHASES") != nullptr and rank == 0)
        {
            std::cout << "CALCULATE " << (solverGamma > 0 ? "iterated" : "default") << " " << MPI_Wtime() - fluxStart << " s" << std::endl;
        }
        if(solverGamma < 0)
        {
            for(const auto &[first, second] : fluxFaceValues)
            {
                for(const ComputationalCell3D *side : {&first, &second})
                {
                    std::vector<double> &out = dump["flux_faces"];
                    AppendCell(out, *side);
                    out.push_back(static_cast<double>(side->ID));
                    out.insert(out.end(), side->stickers.begin(), side->stickers.end());
                }
            }
        }
        std::vector<double> &out = dump[solverGamma > 0 ? "fluxes_iterated" : "fluxes"];
        for(const Conserved3D &c : fluxes)
        {
            AppendConserved(out, c);
        }
        if(solverGamma > 0)
        {
            continue;
        }

        // The extensive update by these fluxes: with no conditions (as RMTV), then with conditions and actions
        std::vector<Conserved3D> initial(cellsNum);
        for(std::size_t i = 0; i < cellsNum; ++i)
        {
            PrimitiveToConserved(cells[i], tess.GetVolume(i), initial[i]);
        }
        // A time step that changes no cell's mass or energy by more than extensiveCfl of it (in absolute fluxes)
        std::vector<double> massOut(cellsNum, 0), energyOut(cellsNum, 0);
        for(std::size_t f = 0; f < fluxes.size(); ++f)
        {
            for(const std::size_t n : {tess.GetFaceNeighbors(f).first, tess.GetFaceNeighbors(f).second})
            {
                if(n < cellsNum)
                {
                    massOut[n] += std::abs(fluxes[f].mass) * tess.GetArea(f);
                    energyOut[n] += std::abs(fluxes[f].energy) * tess.GetArea(f);
                }
            }
        }
        double extensiveDt = std::numeric_limits<double>::max();
        for(std::size_t i = 0; i < cellsNum; ++i)
        {
            extensiveDt = std::min({extensiveDt, initial[i].mass / massOut[i], initial[i].energy / energyOut[i]});
        }
        extensiveDt *= config.extensiveCfl;
        StickerChoose sticker(config.stickers.empty() ? std::string() : config.stickers[0]);
        ChooseAll all;
        NoExtensiveUpdate3D fromCell;
        RegularExtensiveUpdate3D regular;
        std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D *, const ConditionExtensiveUpdater3D::Action3D *>> actions;
        if(not config.stickers.empty())
        {
            actions.emplace_back(&sticker, &fromCell);
        }
        actions.emplace_back(&all, &regular);
        std::vector<Conserved3D> updated; // by the update without actions
        for(const bool withActions : {false, true})
        {
            ConditionExtensiveUpdater3D eu(withActions ? actions : decltype(actions)());
#ifdef GAUSS_HAS_DEVICE
            eu.RunOnDevice(device != nullptr and std::string(device) == "1");
#endif
            // Twice with the same updater (as in a run, where it keeps its buffers): the second's result is dumped
            std::vector<Conserved3D> extensives;
            for(const char *call : {"first", "second"})
            {
                extensives = initial;
                const double updateStart = MPI_Wtime();
                eu(fluxes, tess, extensiveDt, cells, extensives, 0.0, faceVelocities, std::vector<Vector3D>(), fluxFaceValues);
                if(std::getenv("GAUSS_PHASES") != nullptr and rank == 0)
                {
                    std::cout << "EXTENSIVE " << (withActions ? "actions " : "plain ") << call << " " << MPI_Wtime() - updateStart << " s" << std::endl;
                }
            }
            for(const Conserved3D &c : extensives)
            {
                AppendConserved(dump[withActions ? "extensives_actions" : "extensives"], c);
            }
            if(not withActions)
            {
                updated = extensives;
            }
        }

        // The time step, the point velocities (RoundCells3D around Lagrangian3D, as RMTV) and the cell update of the
        // extensives updated above
        ZeroForce3D force;
        CourantFriedrichsLewy cfl(0.3, 1.0, force);
        double phaseStart = MPI_Wtime();
        const double cflDt = cfl(tess, cells, eos, faceVelocities, 0.0);
        PrintPhase(rank, "CFL", phaseStart);
        dump["cfl"] = {cflDt};
        Lagrangian3D lagrangian;
        RoundCells3D round(lagrangian, eos, 1.25, 0.02, config.cold);
        std::vector<Vector3D> pointVelocities;
        phaseStart = MPI_Wtime();
        round(tess, cells, 0.0, pointVelocities);
        round.ApplyFix(tess, cells, 0.0, cflDt, pointVelocities);
        PrintPhase(rank, "POINT_VELOCITIES", phaseStart);
        for(const Vector3D &v : pointVelocities)
        {
            dump["point_velocities"].insert(dump["point_velocities"].end(), {v.x, v.y, v.z});
        }
        DefaultCellUpdater cu(config.sr, gamma53, true, 0);
        std::vector<ComputationalCell3D> updatedCells = cells;
        phaseStart = MPI_Wtime();
        cu(updatedCells, eos, tess, updated);
        PrintPhase(rank, "CELL_UPDATE", phaseStart);
        for(std::size_t i = 0; i < cellsNum; ++i)
        {
            AppendCell(dump["cell_update"], updatedCells[i]);
            AppendConserved(dump["cell_update_extensives"], updated[i]);
        }
    }

    interp.BuildSlopes(tess, cells, 0.0);
    appendSlopes("build_");
    Hllc3D rs;
    dump["dissipation"] = interp.CalcDissipationStreamingFromPreparedSlopes(tess, cells, 0.0, rs, eos);
    dump["counts"] = {static_cast<double>(cellsNum), static_cast<double>(tess.GetTotalFacesNumber())};

    // Two second-order time steps of HDSim3D with these classes (as RMTV, with the CFL time step), on this mesh. Not
    // with periodic boundaries: the points the steps move out of the box are not wrapped back into it.
    if(not config.sr and not config.periodic)
    {
        ZeroForce3D force;
        CourantFriedrichsLewy cfl(0.3, 1.0, force);
        Lagrangian3D lagrangian;
        RoundCells3D round(lagrangian, eos, 1.25, 0.02, config.cold);
        RigidWallFlux3D wallFlux(rs);
        RegularFlux3D bulkFlux(rs);
        ConditionActionFlux1 flux({{&boundaryFace, &wallFlux}, {&bulkFace, &bulkFlux}}, interp);
        DefaultCellUpdater cu(false, 0, true, 0);
        const std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D *, const ConditionExtensiveUpdater3D::Action3D *>> noActions;
        ConditionExtensiveUpdater3D eu(noActions);
#ifdef GAUSS_HAS_DEVICE
        eu.RunOnDevice(device != nullptr and std::string(device) == "1");
#endif
        std::vector<ComputationalCell3D> simCells(cells.begin(), cells.begin() + static_cast<std::ptrdiff_t>(cellsNum));
        std::vector<Conserved3D> simExtensives;
        ProgressTracker tracker;
        HDSim3D sim(tess, simCells, simExtensives, eos, tracker, round, cfl, flux, cu, eu, force,
                    std::make_pair(config.tracers, config.stickers));
        for(int step = 0; step < 2; ++step)
        {
            const double stepStart = MPI_Wtime();
            sim.timeAdvance2();
            PrintPhase(rank, "TIME_ADVANCE2", stepStart);
            dump["sim_dt"].push_back(cfl.GetTimeStep());
        }
        for(std::size_t i = 0; i < tess.GetPointNo(); ++i)
        {
            AppendCell(dump["sim_cells"], sim.getCells()[i]);
            const Vector3D &point = tess.GetMeshPoint(i);
            dump["sim_points"].insert(dump["sim_points"].end(), {point.x, point.y, point.z});
        }
    }
    return dump;
}

} // namespace

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(argc < 3 or (std::string(argv[1]) != "write" and std::string(argv[1]) != "check"))
    {
        if(rank == 0)
        {
            std::cerr << "usage: " << argv[0] << " write|check <dir> [points]" << std::endl;
        }
        MPI_Finalize();
        return 2;
    }
    const bool write = std::string(argv[1]) == "write";
    const std::string dir = argv[2];
    const std::size_t pointsNum = argc > 3 ? std::stoul(argv[3]) : 20000;

    std::size_t localDifferences = 0;
    int localErrors = 0;
    // GAUSS_CONFIGS=name1,name2 runs only these configurations
    const char *only = std::getenv("GAUSS_CONFIGS");
    for(const Config &config : Configs())
    {
        if(only != nullptr and ("," + std::string(only) + ",").find("," + config.name + ",") == std::string::npos)
        {
            continue;
        }
        const std::string path = dir + "/" + config.name + "_" + std::to_string(pointsNum) + "_r" +
                                 std::to_string(rank) + "of" + std::to_string(size) + ".bin";
        const std::string label = config.name + " rank " + std::to_string(rank);
        try
        {
            // A UniversalError is an output too: its text is recorded and compared
            Dump dump;
            std::string error;
            const double start = MPI_Wtime();
            try
            {
                dump = Run(config, pointsNum, rank);
            }
            catch(const UniversalError &thrown)
            {
                error = ErrorText(thrown);
            }
            const double seconds = MPI_Wtime() - start;
            if(write)
            {
                if(error.empty())
                {
                    WriteDump(path, dump);
                }
                else
                {
                    WriteText(path + ".err", error);
                }
            }
            else
            {
                const std::string expectedError = ReadText(path + ".err");
                if(error != expectedError)
                {
                    std::cout << label << ": expected the error\n" << expectedError << "but got\n" << error << std::endl;
                    ++localDifferences;
                }
                else if(error.empty())
                {
                    localDifferences += Compare(ReadDump(path), dump, label);
                }
            }
            if(rank == 0)
            {
                std::cout << config.name << ": " << seconds << " s, operator() " << interpSeconds << " s"
                          << (error.empty() ? "" : " (error)") << std::endl;
            }
        }
        catch(const std::exception &exception)
        {
            std::cerr << label << ": " << exception.what() << std::endl;
            ++localErrors;
        }
        catch(const UniversalError &error)
        {
            std::cerr << label << ":" << std::endl;
            reportError(error, std::cerr);
            ++localErrors;
        }
    }

    unsigned long long differences = localDifferences;
    int errors = localErrors;
    MPI_Allreduce(MPI_IN_PLACE, &differences, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &errors, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if(rank == 0)
    {
        std::cout << "linear_gauss_golden " << (write ? "write" : "check") << " ranks=" << size
                  << " points=" << pointsNum << " differences=" << differences << " errors=" << errors
                  << " PASS=" << (differences == 0 and errors == 0 ? 1 : 0) << std::endl;
    }
    MPI_Finalize();
    return differences == 0 and errors == 0 ? 0 : 1;
}
