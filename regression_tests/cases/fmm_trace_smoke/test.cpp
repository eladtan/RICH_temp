#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/HydroStep.hpp"
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/misc/simple_io.hpp"
#include "source/newtonian/three_dimensional/Lagrangian3D.hpp"
#include "source/newtonian/three_dimensional/RoundCells3D.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/Ghost3D.hpp"
#include "source/newtonian/three_dimensional/ConservativeForce3D.hpp"
#include "source/newtonian/three_dimensional/FastMultipoleAcceleration3D.hpp"
#include <cmath>
#include <fstream>
#include <iostream>

#ifndef RICH_MPI
#error "fmm_trace_smoke exercises the MPI FMM trace and needs an MPI build"
#endif
#include <mpi.h>

// Runs the production self-gravity adapter (FastMultipoleAcceleration3D with
// distributed options) for a few hydro steps on a small self-gravitating
// sphere, so that the per-solve fmm_solve_trace line and its collectives run,
// including the LET reuse diagnostics.  The check function parses the trace
// in fmm_trace_smoke_rank0.log; the physics is not checked.

namespace
{
// Plummer sphere of mass M and scale a, with P = K rho^(5/3).
double plummerDensity(double r, double M, double a)
{
    return 3.0 * M / (4.0 * M_PI * a * a * a) *
        std::pow(1.0 + r * r / (a * a), -2.5);
}

std::vector<ComputationalCell3D> makeCells(const Tessellation3D& tess,
                                           const IdealGas& eos,
                                           double M, double a, double K)
{
    std::vector<ComputationalCell3D> cells(tess.GetPointNo());
    for(std::size_t i = 0; i < cells.size(); ++i)
    {
        const double r = abs(tess.GetMeshPoint(i));
        cells[i].density = std::max(plummerDensity(r, M, a), 1e-6);
        cells[i].pressure = K * std::pow(cells[i].density, 5.0 / 3.0);
        cells[i].internal_energy = eos.dp2e(cells[i].density,
            cells[i].pressure, cells[i].tracers,
            ComputationalCell3D::tracerNames);
    }
    return cells;
}
}

int main(void)
{
    MPI_Init(NULL, NULL);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const double G = 1.0;
    const double M = 1.0;
    const double a = 1.0;
    const double K = 0.3;
    const double width = 8.0;
    const std::size_t pointCount = 20000;
    const int steps = 3;

    const Vector3D ll(-width, -width, -width), ur(width, width, width);
    Voronoi3D tess(ll, ur);
    std::vector<Vector3D> points;
    if(rank == 0)
    {
        points = RandSphereR(pointCount * 3 / 4, ll, ur, 0, 2.0 * a);
        const std::vector<Vector3D> outer =
            RandSphereR2(pointCount - points.size(), ll, ur, 2.0 * a, 0.9 * width);
        points.insert(points.end(), outer.begin(), outer.end());
    }
    points = MPI_Spread(points, 0, MPI_COMM_WORLD);
    points = RoundGrid3D(points, ll, ur, 10);
    tess.BuildParallel(points);

    IdealGas eos(5.0 / 3.0);
    std::vector<ComputationalCell3D> initial = makeCells(tess, eos, M, a, K);

    Hllc3D rs;
    RigidWallGenerator3D ghost;
    LinearGauss3D interp(eos, ghost);
    Lagrangian3D bpm;
    RoundCells3D pm(bpm, eos);
    DefaultCellUpdater cu;
    RigidWallFlux3D rigidflux(rs);
    RegularFlux3D regularFlux(rs);
    IsBoundaryFace3D boundaryFace;
    IsBulkFace3D bulkFace;
    std::vector<std::pair<const ConditionActionFlux1::Condition3D*,
                          const ConditionActionFlux1::Action3D*>> fluxes;
    fluxes.push_back(std::make_pair(&boundaryFace, &rigidflux));
    fluxes.push_back(std::make_pair(&bulkFace, &regularFlux));
    ConditionActionFlux1 fc(fluxes, interp);
    std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D*,
                          const ConditionExtensiveUpdater3D::Action3D*>> updates;
    ConditionExtensiveUpdater3D eu(updates);

    // The production settings of the TDE runs.  The three-argument adapter
    // turns the trace on (emitSolveTrace) without RICH_FMM_TRACE.
    FmmGravityOptions fmmOptions;
    fmmOptions.expansionOrder = 2;
    fmmOptions.thetaCritical = 1.0;
    fmmOptions.leafCapacity = 64;
    FmmDistributedOptions distributed;
    distributed.emitSolveTrace = true;
    distributed.letStalenessDiagnostics = true;
    // The staleness diagnostics require the persistent local tree (the
    // default); stated here because the test depends on it.
    distributed.persistentLocalTreeTopology = true;
    FastMultipoleAcceleration3D acc(fmmOptions, distributed, G);
    ConservativeForce3D force(acc);

    auto tsf = std::make_shared<CourantFriedrichsLewy>(
        0.25, 1, force, std::vector<std::string>(), false);
    Simulation simulation(tess, initial, eos);
    simulation.SetTimeStepFunction(tsf);
    HDSim3D sim(tess, simulation.getCells(), simulation.getExtensives(), eos,
                simulation.getTracker(), pm, *tsf, fc, cu, eu, force,
                std::make_pair(ComputationalCell3D::tracerNames,
                               ComputationalCell3D::stickerNames));
    auto hydroStep = std::make_shared<HydroStep>(sim, HydroStep::TIMEADVANCE_2);
    simulation.addPhysics(hydroStep);
    // An upper bound on the first step; without it the CFL function returns
    // its unset value, -1, and the first step runs backwards in time.
    simulation.SetTimeStep(1.0);
    simulation.PresetLoadBalance("hydro");

    // Rank 0 writes the trace.  Its standard output goes to a file of its own
    // for the steps, so no other rank's output can split a trace record.
    std::ofstream rankZeroLog;
    std::streambuf* standardOutput = nullptr;
    if(rank == 0)
    {
        rankZeroLog.open("fmm_trace_smoke_rank0.log");
        standardOutput = std::cout.rdbuf(rankZeroLog.rdbuf());
    }
    int completed = 0;
    for(int step = 0; step < steps; ++step)
    {
        try
        {
            simulation.step();
            ++completed;
        }
        catch(const UniversalError& error)
        {
            if(rank == 0)
                std::cout.rdbuf(standardOutput);
            reportError(error);
            throw;
        }
        if(rank == 0)
            std::cout << "Cycle " << simulation.GetCycle() << " Time "
                      << simulation.GetTime() << std::endl;
    }
    if(rank == 0)
    {
        std::cout.rdbuf(standardOutput);
        rankZeroLog.close();
    }

    if(rank == 0)
    {
        std::ofstream out("fmm_trace_smoke_metrics.txt");
        out.precision(17);
        out << "steps " << completed << "\n";
        out << "time " << simulation.GetTime() << "\n";
        out << "pass " << (completed == steps ? 1 : 0) << "\n";
    }
    MPI_Finalize();
    return 0;
}
