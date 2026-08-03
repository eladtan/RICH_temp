#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/3D/output/write3D.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/Ghost3D.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/newtonian/three_dimensional/SphericalBackgroundLinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/eulerian_3d.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/default_extensive_updater.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/HydroStep.hpp"
#include "source/newtonian/three_dimensional/spherical_symmetry/SphericalShellProjector3D.hpp"

#ifdef RICH_MPI
#include <mpi.h>
#include "source/mpi/mpi_commands_3d.hpp"
#endif

namespace {

double const R_OUTER = 1.1;
double const R_INNER = 0.05;
#ifdef HIGH_RES
size_t const N_CUBE_EDGE = 82;
#else
size_t const N_CUBE_EDGE = 41;
#endif

SphericalShellMeshOptions build_shell_mesh_options()
{
    SphericalShellMeshOptions options;
    options.center = Vector3D();
    options.inner_radius = R_INNER;
    options.outer_radius = R_OUTER;
    options.angular_edge_count = N_CUBE_EDGE;
    options.guard_shell_count = 2;
    options.fill_inner_core = true;
    options.fill_outer_box = true;
    options.antipodal_directions = true;
    options.centered_cartesian_fill = true;
    return options;
}

std::vector<double> build_bin_edges()
{
    return SphericalShellMeshActiveBinEdges(build_shell_mesh_options());
}

struct DiagResult {
    double shock_r;
    double rho_scatter;
    double vr_scatter;
    double tangential_velocity_rms;
};

DiagResult compute_diagnostics(HDSim3D const& sim,
    std::vector<double> const& bin_edges)
{
    size_t const nbins = bin_edges.size() - 1;
    size_t const N = sim.getTessellation().GetPointNo();

    std::vector<double> vol_sum(nbins, 0.0);
    std::vector<double> rho_vol(nbins, 0.0);
    std::vector<double> vr_vol(nbins, 0.0);
    std::vector<double> ie_vol(nbins, 0.0);
    double tangential_v2_vol = 0.0;
    double tangential_vol = 0.0;

    std::vector<size_t> cell_bin(N, nbins);
    std::vector<double> cell_rho(N);
    std::vector<double> cell_vr(N);
    std::vector<double> cell_vol(N);

    for (size_t i = 0; i < N; ++i) {
        Vector3D const cm = sim.getTessellation().GetCellCM(i);
        double const r = abs(cm);
        if (r < bin_edges.front() || r >= bin_edges.back())
            continue;

        auto it = std::upper_bound(bin_edges.begin(), bin_edges.end(), r);
        if (it == bin_edges.begin())
            continue;
        size_t bin = static_cast<size_t>(it - bin_edges.begin()) - 1;
        if (bin >= nbins)
            continue;

        double const vol = sim.getTessellation().GetVolume(i);
        double const rho = sim.getCells()[i].density;
        double const ie = sim.getCells()[i].internal_energy;
        double vr = 0.0;
        double vt2 = 0.0;
        if (r > 1e-12) {
            Vector3D const v = sim.getCells()[i].velocity;
            vr = (v.x * cm.x + v.y * cm.y + v.z * cm.z) / r;
            Vector3D const vt = v - cm * (vr / r);
            vt2 = ScalarProd(vt, vt);
        }

        cell_bin[i] = bin;
        cell_rho[i] = rho;
        cell_vr[i] = vr;
        cell_vol[i] = vol;

        vol_sum[bin] += vol;
        rho_vol[bin] += rho * vol;
        vr_vol[bin] += vr * vol;
        ie_vol[bin] += ie * vol;
        tangential_v2_vol += vt2 * vol;
        tangential_vol += vol;
    }

#ifdef RICH_MPI
    {
        int n = static_cast<int>(nbins);
        std::vector<double> g(nbins);
        MPI_Allreduce(vol_sum.data(), g.data(), n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        vol_sum = g;
        MPI_Allreduce(rho_vol.data(), g.data(), n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        rho_vol = g;
        MPI_Allreduce(vr_vol.data(), g.data(), n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        vr_vol = g;
        MPI_Allreduce(ie_vol.data(), g.data(), n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        ie_vol = g;
        double gt = 0.0;
        MPI_Allreduce(&tangential_v2_vol, &gt, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        tangential_v2_vol = gt;
        MPI_Allreduce(&tangential_vol, &gt, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        tangential_vol = gt;
    }
#endif

    std::vector<double> rho_mean(nbins, 0.0);
    std::vector<double> vr_mean(nbins, 0.0);
    std::vector<double> ie_mean(nbins, 0.0);
    for (size_t b = 0; b < nbins; ++b) {
        if (vol_sum[b] > 0.0) {
            rho_mean[b] = rho_vol[b] / vol_sum[b];
            vr_mean[b] = vr_vol[b] / vol_sum[b];
            ie_mean[b] = ie_vol[b] / vol_sum[b];
        }
    }

    std::vector<double> rho_mad(nbins, 0.0);
    std::vector<double> vr_mad(nbins, 0.0);

    for (size_t i = 0; i < N; ++i) {
        if (cell_bin[i] >= nbins)
            continue;
        size_t b = cell_bin[i];
        rho_mad[b] += std::abs(cell_rho[i] - rho_mean[b]) * cell_vol[i];
        vr_mad[b] += std::abs(cell_vr[i] - vr_mean[b]) * cell_vol[i];
    }

#ifdef RICH_MPI
    {
        int n = static_cast<int>(nbins);
        std::vector<double> g(nbins);
        MPI_Allreduce(rho_mad.data(), g.data(), n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        rho_mad = g;
        MPI_Allreduce(vr_mad.data(), g.data(), n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        vr_mad = g;
    }
#endif

    double shock_r = bin_edges.back();
    double total_rho_mad = 0.0;
    double total_vr_mad = 0.0;
    double total_vol = 0.0;
    for (size_t b = 0; b < nbins; ++b) {
        if (vol_sum[b] <= 0.0)
            continue;
        double bin_center = 0.5 * (bin_edges[b] + bin_edges[b + 1]);

        if (bin_center >= R_INNER && bin_center <= 1.0) {
            total_rho_mad += rho_mad[b];
            total_vr_mad += vr_mad[b];
            total_vol += vol_sum[b];
        }

        if (ie_mean[b] > 0.2 && bin_center < shock_r)
        {
            shock_r = bin_center;
        }
    }

    DiagResult res;
    res.shock_r = shock_r;
    res.rho_scatter = (total_vol > 0.0) ? total_rho_mad / total_vol : 0.0;
    res.vr_scatter = (total_vol > 0.0) ? total_vr_mad / total_vol : 0.0;
    res.tangential_velocity_rms = (tangential_vol > 0.0)
        ? std::sqrt(tangential_v2_vol / tangential_vol) : 0.0;
    return res;
}

} // namespace

int main(void)
{
    int rank = 0;
    int nprocs = 1;
#ifdef RICH_MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#endif

    double box_half = 1.55;
    Vector3D ll(-box_half, -box_half, -box_half);
    Vector3D ur(box_half, box_half, box_half);

    IdealGas eos(5.0 / 3.0);

    Voronoi3D tess(ll, ur);
    std::vector<ComputationalCell3D> cells;
    SphericalShellMeshOptions const shell_options = build_shell_mesh_options();

    {
        std::vector<Vector3D> points;

        if (rank == 0)
            points = GenerateSphericalShellMesh3D(ll, ur, shell_options);

#ifdef RICH_MPI
        points = MPI_Spread(points, 0, MPI_COMM_WORLD);
#endif
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

        size_t const Nlocal = tess.GetPointNo();
        cells.resize(Nlocal);
        for (size_t i = 0; i < Nlocal; ++i) {
            Vector3D const pos = tess.GetMeshPoint(i);
            Vector3D const cm = tess.GetCellCM(i);
            double const r_mp = abs(pos);
            double const r_cm = abs(cm);
            if (r_mp > 0.9 && r_mp < 1.0) {
                cells[i].density = 10.0;
                cells[i].pressure = 0.1;
                if (r_cm > 1e-12)
                    cells[i].velocity = cm * (-1.0 / r_cm);
            } else {
                cells[i].density = 0.001;
                cells[i].pressure = 1e-5;
                cells[i].velocity = Vector3D(0, 0, 0);
            }
            cells[i].internal_energy = eos.dp2e(cells[i].density, cells[i].pressure,
                                                 cells[i].tracers, ComputationalCell3D::tracerNames);
        }
    }

    Hllc3D rs;
    RigidWallGenerator3D ghost;
    SphericalBackgroundLinearGauss3D interp(eos, ghost, Vector3D(),
        SphericalShellMeshRadii(shell_options), true, 0.2, 0.5, 0.7);
    Eulerian3D pm;
    ZeroForce3D force;
    DefaultCellUpdater cu;

    auto* boundary_face = new IsBoundaryFace3D();
    auto* bulk_face = new IsBulkFace3D();
    RigidWallFlux3D rigidflux(rs);
    auto* regular_flux = new RegularFlux3D(rs);

    std::vector<std::pair<const ConditionActionFlux1::Condition3D*,
        const ConditionActionFlux1::Action3D*>> flux_vector;
    flux_vector.push_back({boundary_face, &rigidflux});
    flux_vector.push_back({bulk_face, regular_flux});
    ConditionActionFlux1 fc(flux_vector, interp);

    std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D*,
        const ConditionExtensiveUpdater3D::Action3D*>> eu_sequence;
    ConditionExtensiveUpdater3D eu(eu_sequence);

    auto tsf = std::make_shared<CourantFriedrichsLewy>(0.3, 1.0, force);

    Simulation simulation(tess, cells, eos);
    simulation.SetTimeStepFunction(tsf);
    HDSim3D sim(tess, simulation.getCells(), simulation.getExtensives(), eos, simulation.getTracker(), pm, *tsf, fc, cu, eu, force,
                std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));
    sim.SetSphericalShellProjector(
        std::make_shared<SphericalShellProjector3D>(
            shell_options.center, SphericalShellMeshRadii(shell_options)));
    sim.SetSphericalPositivityPreservingStage();

    auto hydroStep = std::make_shared<HydroStep>(sim, HydroStep::TIMEADVANCE_2);
    simulation.addPhysics(hydroStep);
    simulation.SetTimeStep(1.0);

    auto global_conserved_totals = [&sim]() {
        double totals[2] = {0.0, 0.0};
        size_t const owned_cell_count = sim.getTessellation().GetPointNo();
        for (size_t i = 0; i < owned_cell_count; ++i) {
            totals[0] += sim.getExtensives()[i].mass;
            totals[1] += sim.getExtensives()[i].energy;
        }
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, totals, 2, MPI_DOUBLE, MPI_SUM,
                      MPI_COMM_WORLD);
#endif
        return std::make_pair(totals[0], totals[1]);
    };
    std::pair<double, double> const initial_totals =
        global_conserved_totals();
    std::pair<double, double> final_totals = initial_totals;
    double max_mass_relative_drift = 0.0;
    double max_energy_relative_drift = 0.0;

    WriteSnapshot3D(sim, "snap_initial.h5");
    if (rank == 0)
        std::cout << "Wrote snap_initial.h5" << std::endl;

    std::vector<double> bin_edges;
    if (rank == 0)
        bin_edges = build_bin_edges();
#ifdef RICH_MPI
    {
        int nedges = static_cast<int>(bin_edges.size());
        MPI_Bcast(&nedges, 1, MPI_INT, 0, MPI_COMM_WORLD);
        bin_edges.resize(static_cast<size_t>(nedges));
        MPI_Bcast(bin_edges.data(), nedges, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
#endif

    if (rank == 0) {
        std::cout << std::scientific << std::setprecision(6);
        std::cout << "Radial bins: " << (bin_edges.size() - 1)
                  << " shells from r=" << bin_edges.front()
                  << " to r=" << bin_edges.back() << std::endl;
    }

    double next_snapshot_r = R_OUTER - 0.1;
    double const innermost_shock_bin_center =
        0.5 * (bin_edges[0] + bin_edges[1]);
    double max_rho_scatter = 0.0;
    double max_vr_scatter = 0.0;
    double max_tangential_velocity_rms = 0.0;

    while (true) {
        try {
            simulation.step();
        } catch (UniversalError const& eo) {
            reportError(eo);
            throw;
        }

        DiagResult diag = compute_diagnostics(sim, bin_edges);
        final_totals = global_conserved_totals();
        double const mass_relative_drift =
            std::abs(final_totals.first - initial_totals.first) /
            std::abs(initial_totals.first);
        double const energy_relative_drift =
            std::abs(final_totals.second - initial_totals.second) /
            std::abs(initial_totals.second);
        max_mass_relative_drift = std::max(max_mass_relative_drift,
            mass_relative_drift);
        max_energy_relative_drift = std::max(max_energy_relative_drift,
            energy_relative_drift);
        max_rho_scatter = std::max(max_rho_scatter, diag.rho_scatter);
        max_vr_scatter = std::max(max_vr_scatter, diag.vr_scatter);
        max_tangential_velocity_rms = std::max(max_tangential_velocity_rms,
                                               diag.tangential_velocity_rms);

        if (rank == 0) {
            std::cout << "Cycle " << simulation.GetCycle()
                      << " t=" << simulation.GetTime()
                      << " dt=" << simulation.GetTimeStep()
                      << " shock_r=" << diag.shock_r
                      << " rho_scatter=" << diag.rho_scatter
                      << " vr_scatter=" << diag.vr_scatter
                      << " tangential_velocity_rms=" << diag.tangential_velocity_rms
                      << " mass_relative_drift=" << mass_relative_drift
                      << " energy_relative_drift=" << energy_relative_drift
                      << std::endl;
        }

        if (diag.shock_r <= next_snapshot_r) {
            int snap_id = static_cast<int>(std::round((R_OUTER - next_snapshot_r) * 10));
            std::string fname = "snap_r" + std::to_string(snap_id) + ".h5";
            WriteSnapshot3D(sim, fname);
            if (rank == 0)
                std::cout << "Wrote " << fname << " at shock_r=" << diag.shock_r << std::endl;
            next_snapshot_r -= 0.1;
        }

        if (diag.shock_r <= innermost_shock_bin_center)
            break;
    }

    WriteSnapshot3D(sim, "snap_final.h5");
    if (rank == 0)
        std::cout << "Wrote snap_final.h5" << std::endl;

    {
        size_t const owned_cell_count = sim.getTessellation().GetPointNo();
        double const box_half = R_OUTER * 1.5;
        double const z_tol = box_half / static_cast<double>(N_CUBE_EDGE);
        std::vector<double> local_x, local_y, local_rho, local_ie;
        for (size_t i = 0; i < owned_cell_count; ++i) {
            Vector3D const cm = sim.getTessellation().GetCellCM(i);
            if (std::abs(cm.z) > z_tol)
                continue;
            local_x.push_back(cm.x);
            local_y.push_back(cm.y);
            local_rho.push_back(sim.getCells()[i].density);
            local_ie.push_back(sim.getCells()[i].internal_energy);
        }
#ifdef RICH_MPI
        int local_count = static_cast<int>(local_x.size());
        std::vector<int> counts(nprocs), displs(nprocs);
        MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            displs[0] = 0;
            for (int p = 1; p < nprocs; ++p)
                displs[p] = displs[p - 1] + counts[p - 1];
        }
        int total = (rank == 0) ? (displs[nprocs - 1] + counts[nprocs - 1]) : 0;
        std::vector<double> all_x(total), all_y(total), all_rho(total), all_ie(total);
        MPI_Gatherv(local_x.data(), local_count, MPI_DOUBLE, all_x.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_y.data(), local_count, MPI_DOUBLE, all_y.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_rho.data(), local_count, MPI_DOUBLE, all_rho.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_ie.data(), local_count, MPI_DOUBLE, all_ie.data(), counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
        std::vector<double>& all_x = local_x;
        std::vector<double>& all_y = local_y;
        std::vector<double>& all_rho = local_rho;
        std::vector<double>& all_ie = local_ie;
#endif
        if (rank == 0) {
            std::ofstream sf("collapse_xy_slice.txt");
            sf << std::scientific << std::setprecision(10);
            for (size_t i = 0; i < all_x.size(); ++i)
                sf << all_x[i] << " " << all_y[i] << " " << all_rho[i] << " " << all_ie[i] << "\n";
            sf.close();
            std::cout << "Wrote collapse_xy_slice.txt (" << all_x.size() << " cells)" << std::endl;
        }
    }

    int exit_code = 0;
    if (rank == 0) {
        int pass = (max_rho_scatter <= 1e-4 && max_vr_scatter <= 1e-4
                    && max_tangential_velocity_rms <= 1e-4
                    && max_mass_relative_drift <= 1e-10
                    && max_energy_relative_drift <= 1e-10) ? 1 : 0;
        std::ofstream mf("collapse_metrics.txt");
        mf << std::scientific << std::setprecision(12);
        mf << "max_density_scatter " << max_rho_scatter << "\n";
        mf << "max_velocity_scatter " << max_vr_scatter << "\n";
        mf << "max_tangential_velocity_rms " << max_tangential_velocity_rms << "\n";
        mf << "minimum_positivity_theta "
           << sim.GetMinimumSphericalPositivityTheta() << "\n";
        mf << "positivity_activation_count "
           << sim.GetSphericalPositivityActivationCount() << "\n";
        mf << "initial_mass " << initial_totals.first << "\n";
        mf << "final_mass " << final_totals.first << "\n";
        mf << "max_mass_relative_drift " << max_mass_relative_drift << "\n";
        mf << "initial_energy " << initial_totals.second << "\n";
        mf << "final_energy " << final_totals.second << "\n";
        mf << "max_energy_relative_drift " << max_energy_relative_drift << "\n";
        mf << "pass " << pass << "\n";
        mf.close();
        std::cout << "Wrote collapse_metrics.txt"
                  << " (max_density_scatter=" << max_rho_scatter
                   << ", max_velocity_scatter=" << max_vr_scatter
                   << ", max_tangential_velocity_rms=" << max_tangential_velocity_rms
                   << ", minimum_positivity_theta="
                   << sim.GetMinimumSphericalPositivityTheta()
                   << ", positivity_activation_count="
                   << sim.GetSphericalPositivityActivationCount()
                   << ", max_mass_relative_drift="
                   << max_mass_relative_drift
                   << ", max_energy_relative_drift="
                   << max_energy_relative_drift
                   << ", pass=" << pass << ")" << std::endl;
        exit_code = pass ? 0 : 1;
    }

#ifdef RICH_MPI
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
#endif
    return exit_code;
}
