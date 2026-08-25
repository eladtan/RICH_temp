#include <algorithm>
#include <fstream>
#include <vector>
#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/3D/output/write3D.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/newtonian/three_dimensional/Lagrangian3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/RoundCells3D.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/default_extensive_updater.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/HydroStep.hpp"

int main(void)
{
    Vector3D ll(-1, -1, -1), ur(1, 1, 1);
    int rank = 0;
    int world_size = 1;
#ifdef RICH_MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#endif
    size_t const Np = (world_size > 1) ? static_cast<size_t>(5e6) : static_cast<size_t>(1e5);

    std::vector<Vector3D> points;
    if(rank == 0) {
        points = RandRectangular(Np, ll, ur);
    }
#ifdef RICH_MPI
    points = MPI_Spread(points, 0, MPI_COMM_WORLD);
#endif
    try {
        points = RoundGrid3D(points, ll, ur, 10);
    }
    catch(UniversalError const& e) {
        reportError(e);
        throw;
    }

    Voronoi3D tess(ll, ur);
    try {
#ifdef RICH_MPI
        tess.BuildParallel(points);
#else
        tess.Build(points);
#endif
    } catch (UniversalError const& eo) {
        reportError(eo);
        throw;
    }

    IdealGas eos(5. / 3.);
    size_t const Nlocal = tess.GetPointNo();
    std::vector<ComputationalCell3D> cells(Nlocal);
    ComputationalCell3D inner_cell, outer_cell;
    inner_cell.velocity = Vector3D(0, 0, 0);
    inner_cell.density = 1;
    inner_cell.internal_energy = 8e5;
    inner_cell.pressure = eos.de2p(inner_cell.density, inner_cell.internal_energy, inner_cell.tracers, ComputationalCell3D::tracerNames);
    outer_cell.velocity = Vector3D(0, 0, 0);
    outer_cell.density = 1;
    outer_cell.internal_energy = 0.1;
    outer_cell.pressure = eos.de2p(outer_cell.density, outer_cell.internal_energy, outer_cell.tracers, ComputationalCell3D::tracerNames);
    for(size_t i = 0; i < Nlocal; ++i) {
        if(abs(tess.GetMeshPoint(i)) < 0.1) {
            cells[i] = inner_cell;
        }
        else {
            cells[i] = outer_cell;
        }
    }

    Hllc3D rs;
    RigidWallGenerator3D ghost;
    LinearGauss3D interp(eos, ghost);

    std::vector<std::pair<const ConditionActionFlux1::Condition3D*, const ConditionActionFlux1::Action3D*> > sequence;
    ConditionActionFlux1::Condition3D* isbulk = new IsBulkFace3D();
    ConditionActionFlux1::Condition3D* isboundary = new IsBoundaryFace3D();
    ConditionActionFlux1::Action3D* normal_flux = new RegularFlux3D(rs);
    ConditionActionFlux1::Action3D* rigid_flux = new RigidWallFlux3D(rs);
    sequence.push_back(std::pair<const ConditionActionFlux1::Condition3D*, const ConditionActionFlux1::Action3D*>(isboundary, rigid_flux));
    sequence.push_back(std::pair<const ConditionActionFlux1::Condition3D*, const ConditionActionFlux1::Action3D*>(isbulk, normal_flux));
    ConditionActionFlux1 flux(sequence, interp);

    std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D*, const ConditionExtensiveUpdater3D::Action3D*> > eu_sequence;
    ConditionExtensiveUpdater3D eu(eu_sequence);
    DefaultCellUpdater cu;
    ZeroForce3D force;
    auto tsf = std::make_shared<CourantFriedrichsLewy>(0.3, 1.0, force);
    Lagrangian3D bpm;
    RoundCells3D pm(bpm, eos);

    Simulation simulation(tess, cells, eos);
    simulation.SetTimeStepFunction(tsf);
    HDSim3D sim(tess, simulation.getCells(), simulation.getExtensives(), eos, simulation.getTracker(), pm, *tsf, flux, cu, eu, force, std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));

    auto hydroStep = std::make_shared<HydroStep>(sim, HydroStep::TIMEADVANCE_2);
    simulation.addPhysics(hydroStep);
    simulation.SetTimeStep(1.0);
#ifdef RICH_MPI
    simulation.PresetLoadBalance("hydro");
#endif

    double old_time = simulation.GetTime();
    while(simulation.GetTime() < 0.0075) {
        try {
            if(rank == 0) {
                std::cout << "Cycle " << simulation.GetCycle()
                          << " dt " << simulation.GetTime() - old_time
                          << " time " << simulation.GetTime() << std::endl;
            }
            old_time = simulation.GetTime();
            simulation.step();
        }
        catch(UniversalError const& eo) {
            reportError(eo);
            throw;
        }
    }

    size_t const nbins = 500;
    std::vector<double> r_sum(nbins, 0.0);
    std::vector<double> density_sum(nbins, 0.0);
    std::vector<double> pressure_sum(nbins, 0.0);
    std::vector<double> vr_sum(nbins, 0.0);
    std::vector<double> volume_sum(nbins, 0.0);

    double rmax_local = 0.0;
    for(size_t i = 0; i < tess.GetPointNo(); ++i) {
        double const r = abs(tess.GetCellCM(i));
        rmax_local = std::max(rmax_local, r);
    }
    double rmax = rmax_local;
#ifdef RICH_MPI
    MPI_Allreduce(&rmax_local, &rmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
    if(rmax <= 0.0) {
        rmax = 1.0;
    }

    auto const& final_cells = simulation.getCells();
    for(size_t i = 0; i < tess.GetPointNo(); ++i) {
        Vector3D const cm = tess.GetCellCM(i);
        double const r = abs(cm);
        double vr = 0.0;
        if(r > 0.0) {
            Vector3D const v = final_cells[i].velocity;
            vr = (v.x * cm.x + v.y * cm.y + v.z * cm.z) / r;
        }
        size_t bin = static_cast<size_t>((r / rmax) * static_cast<double>(nbins));
        if(bin >= nbins) {
            bin = nbins - 1;
        }
        double const cell_volume = tess.GetVolume(i);
        r_sum[bin] += r * cell_volume;
        density_sum[bin] += final_cells[i].density * cell_volume;
        pressure_sum[bin] += final_cells[i].pressure * cell_volume;
        vr_sum[bin] += vr * cell_volume;
        volume_sum[bin] += cell_volume;
    }

#ifdef RICH_MPI
    std::vector<double> r_sum_global(nbins, 0.0);
    std::vector<double> density_sum_global(nbins, 0.0);
    std::vector<double> pressure_sum_global(nbins, 0.0);
    std::vector<double> vr_sum_global(nbins, 0.0);
    std::vector<double> volume_sum_global(nbins, 0.0);
    MPI_Reduce(r_sum.data(), r_sum_global.data(), static_cast<int>(nbins), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(density_sum.data(), density_sum_global.data(), static_cast<int>(nbins), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(pressure_sum.data(), pressure_sum_global.data(), static_cast<int>(nbins), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(vr_sum.data(), vr_sum_global.data(), static_cast<int>(nbins), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(volume_sum.data(), volume_sum_global.data(), static_cast<int>(nbins), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        std::ofstream out("sedov_profile.txt");
        for(size_t b = 0; b < nbins; ++b) {
            if(volume_sum_global[b] <= 0.0) {
                continue;
            }
            double const inv = 1.0 / volume_sum_global[b];
            out << r_sum_global[b] * inv << " "
                << density_sum_global[b] * inv << " "
                << pressure_sum_global[b] * inv << " "
                << vr_sum_global[b] * inv << "\n";
        }
        out.close();
    }
#else
    std::ofstream out("sedov_profile.txt");
    for(size_t b = 0; b < nbins; ++b) {
        if(volume_sum[b] <= 0.0) {
            continue;
        }
        double const inv = 1.0 / volume_sum[b];
        out << r_sum[b] * inv << " "
            << density_sum[b] * inv << " "
            << pressure_sum[b] * inv << " "
            << vr_sum[b] * inv << "\n";
    }
    out.close();
#endif

#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return 0;
}
