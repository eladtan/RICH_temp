#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/newtonian/three_dimensional/eulerian_3d.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/default_extensive_updater.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/ConservativeForce3D.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/Radiation/Diffusion.hpp"
#include <libgen.h>
#include <string.h>
#include "source/Radiation/DiffusionForce.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <numeric>

namespace
{
    class IsPointLeftRightBox3D : public ConditionActionFlux1::Condition3D
    {
    public:
        pair<bool, bool> operator()(size_t face_index, const Tessellation3D& tess,
            const vector<ComputationalCell3D>& cells) const override
        {
            if (!tess.BoundaryFace(face_index))
                return std::pair<bool, bool>(false, false);

            auto const& box = tess.GetBoxCoordinates();
            Vector3D const& first_point = tess.GetMeshPoint(tess.GetFaceNeighbors(face_index).first);
            Vector3D const& second_point = tess.GetMeshPoint(tess.GetFaceNeighbors(face_index).second);

            const bool is_left = first_point.x < box.first.x || first_point.x > box.second.x;
            const bool is_right = second_point.x < box.first.x || second_point.x > box.second.x;

            return std::make_pair(is_left || is_right, is_right);
        }
    };

    class GhostChooser: public SeveralGhostGenerator3D::GhostCriteria3D
    {
    public:
        size_t GhostChoose(Tessellation3D const& tess, size_t index) const
        {
            auto const& box = tess.GetBoxCoordinates();
            Vector3D const& p = tess.GetMeshPoint(index);
            if (p.x < box.first.x)
                return 0;
            if (p.x > box.second.x)
                return 1;
            else
                return 2;
        }
    };
}

int main(void)
{
    size_t const Np = 1024;
    double const box_size = 1e3;
    double const dy = 3 * box_size / (2 * Np);
    Vector3D ll(-box_size, -dy, -dy), ur(2 * box_size, dy, dy);
    int rank = 0;
    int nprocs = 1;
#ifdef RICH_MPI
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#endif

    std::vector<Vector3D> points;
    if (rank == 0)
        points = CartesianMesh(Np, 1, 1, ll, ur);
#ifdef RICH_MPI
    points = MPI_Spread(points, 0, MPI_COMM_WORLD);
#endif

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

    IdealGas eos(5./3., CG::boltzmann_constant / (1.67e-24 * (5.0 / 3.0 - 1)), 1, 0);

    size_t const Nlocal = tess.GetPointNo();
    std::vector<ComputationalCell3D> cells(Nlocal);
    ComputationalCell3D left_cell, right_cell;
    left_cell.velocity = Vector3D(2.3547e5, 0, 0);
    left_cell.density = 5.45887e-13;
    left_cell.temperature = 100;
    left_cell.internal_energy = eos.dT2e(left_cell.density, left_cell.temperature, left_cell.tracers, ComputationalCell3D::tracerNames);
    left_cell.pressure = eos.de2p(left_cell.density, left_cell.internal_energy, left_cell.tracers, ComputationalCell3D::tracerNames);
    left_cell.Erad = CG::radiation_constant * std::pow(left_cell.temperature, 4) / left_cell.density;

    right_cell.velocity = Vector3D(1.03e5, 0, 0);
    right_cell.density = 1.2479e-12;
    right_cell.temperature = 207.757;
    right_cell.Erad = CG::radiation_constant * std::pow(right_cell.temperature, 4) / right_cell.density;
    right_cell.internal_energy = eos.dT2e(right_cell.density, right_cell.temperature, right_cell.tracers, ComputationalCell3D::tracerNames);
    right_cell.pressure = eos.de2p(right_cell.density, right_cell.internal_energy, right_cell.tracers, ComputationalCell3D::tracerNames);

    for (size_t i = 0; i < Nlocal; ++i)
    {
        if (tess.GetMeshPoint(i).x < 0)
            cells[i] = left_cell;
        else
            cells[i] = right_cell;
    }

    Hllc3D rs;

    RigidWallGenerator3D rigid_ghost;
    ConstantPrimitiveGenerator3D left_ghost(left_cell), right_ghost(right_cell);
    std::vector<Ghost3D*> ghost_list = {&left_ghost, &right_ghost, &rigid_ghost};
    GhostChooser ghost_chooser;
    SeveralGhostGenerator3D ghost(ghost_list, ghost_chooser);

    LinearGauss3D interp(eos, ghost);

    std::vector<pair<const ConditionActionFlux1::Condition3D*,
        const ConditionActionFlux1::Action3D*>> sequence;
    ConditionActionFlux1::Condition3D* isbulk = new IsBulkFace3D();
    ConditionActionFlux1::Condition3D* is_side = new IsPointLeftRightBox3D();
    ConditionActionFlux1::Condition3D* isboundary = new IsBoundaryFace3D();
    ConditionActionFlux1::Action3D* normal_flux = new RegularFlux3D(rs);
    ConditionActionFlux1::Action3D* rigid_flux = new RigidWallFlux3D(rs);
    sequence.push_back(std::pair<const ConditionActionFlux1::Condition3D*,
        const ConditionActionFlux1::Action3D*>(is_side, normal_flux));
    sequence.push_back(std::pair<const ConditionActionFlux1::Condition3D*,
        const ConditionActionFlux1::Action3D*>(isboundary, rigid_flux));
    sequence.push_back(std::pair<const ConditionActionFlux1::Condition3D*,
        const ConditionActionFlux1::Action3D*>(isbulk, normal_flux));
    ConditionActionFlux1 flux(sequence, interp);

    std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D*, const ConditionExtensiveUpdater3D::Action3D*>> eu_sequence;
    ConditionExtensiveUpdater3D eu(eu_sequence);

    PowerLawOpacity opacity(CG::speed_of_light / (3 * 0.848902), 0, 0, 3.93e-5, 0, 0);
    DiffusionXInflowBoundary diffusion_boundary(left_cell, right_cell, opacity);
    Diffusion diffusion(opacity, diffusion_boundary, eos);

    DefaultCellUpdater cu(false, 0, true, 0, &diffusion);

    DiffusionForce force(diffusion, eos);

    double const hydro_cfl = 0.3;
    double const force_cfl = 1;
    CourantFriedrichsLewy tsf(hydro_cfl, force_cfl, force);

    Eulerian3D pm;

    HDSim3D sim(tess, cells, eos, pm, tsf, flux, cu, eu, force,
        std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));

    double old_time = sim.getTime();
    double current_dt = 1e-15;
    while (sim.getTime() < 0.01)
    {
        try
        {
            auto step_start = std::chrono::steady_clock::now();

            double new_dt = sim.RadiationTimeStep(current_dt, diffusion);
            tsf.SetTimeStep(new_dt);

            old_time = sim.getTime();
            sim.timeAdvance2();
            current_dt = sim.getTime() - old_time;

            auto step_end = std::chrono::steady_clock::now();
            double wall_sec = std::chrono::duration<double>(step_end - step_start).count();

            if (rank == 0)
            {
                std::cout << std::endl;
                std::cout << "Cycle " << sim.getCycle()
                          << " dt " << std::scientific << std::setprecision(6) << current_dt
                          << " time " << sim.getTime()
                          << " wall_time " << std::fixed << std::setprecision(3) << wall_sec << "s"
                          << std::endl;
            }
        }
        catch (UniversalError const& eo)
        {
            reportError(eo);
            throw;
        }
    }

    // Compute output path in the same directory as this source file
    char file_buf[4096];
    strncpy(file_buf, __FILE__, sizeof(file_buf) - 1);
    file_buf[sizeof(file_buf) - 1] = '\0';
    std::string profile_path = std::string(dirname(file_buf)) + "/mach2_profile.txt";

    // Gather profile data from all MPI ranks and write to file
    {
        std::vector<double> local_x(Nlocal), local_rho(Nlocal), local_T(Nlocal), local_Trad(Nlocal);
        auto const& final_cells = sim.getCells();
        for (size_t i = 0; i < Nlocal; ++i)
        {
            local_x[i] = tess.GetMeshPoint(i).x;
            local_rho[i] = final_cells[i].density;
            local_T[i] = final_cells[i].temperature;
            local_Trad[i] = std::pow(final_cells[i].Erad * final_cells[i].density / CG::radiation_constant, 0.25);
        }

#ifdef RICH_MPI
        int local_n = static_cast<int>(Nlocal);
        std::vector<int> recv_counts(nprocs, 0);
        MPI_Gather(&local_n, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> displs(nprocs, 0);
        int total_n = 0;
        if (rank == 0)
        {
            for (int i = 0; i < nprocs; ++i)
            {
                displs[i] = total_n;
                total_n += recv_counts[i];
            }
        }

        std::vector<double> all_x, all_rho, all_T, all_Trad;
        if (rank == 0)
        {
            all_x.resize(total_n);
            all_rho.resize(total_n);
            all_T.resize(total_n);
            all_Trad.resize(total_n);
        }
        MPI_Gatherv(local_x.data(), local_n, MPI_DOUBLE, all_x.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_rho.data(), local_n, MPI_DOUBLE, all_rho.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_T.data(), local_n, MPI_DOUBLE, all_T.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_Trad.data(), local_n, MPI_DOUBLE, all_Trad.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            std::vector<size_t> idx(total_n);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return all_x[a] < all_x[b]; });

            std::ofstream out(profile_path);
            out << std::scientific << std::setprecision(12);
            for (int i = 0; i < total_n; ++i)
            {
                out << all_x[idx[i]] << " " << all_rho[idx[i]] << " " << all_T[idx[i]] << " " << all_Trad[idx[i]] << "\n";
            }
            out.close();
        }
#else
        std::vector<size_t> idx(Nlocal);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b){ return local_x[a] < local_x[b]; });

        std::ofstream out(profile_path);
        out << std::scientific << std::setprecision(12);
        for (size_t i = 0; i < Nlocal; ++i)
        {
            out << local_x[idx[i]] << " " << local_rho[idx[i]] << " " << local_T[idx[i]] << " " << local_Trad[idx[i]] << "\n";
        }
        out.close();
#endif
    }

#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return 0;
}
