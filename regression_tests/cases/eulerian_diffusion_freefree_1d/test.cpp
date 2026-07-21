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
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/Radiation/Diffusion.hpp"
#include "source/Radiation/DiffusionForce.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <vector>
#include <unistd.h>
 
namespace
{
    class IsPointLeftRightBox3D : public ConditionActionFlux1::Condition3D
    {
    public:
        std::pair<bool, bool> operator()(size_t face_index, const Tessellation3D& tess,
            const std::vector<ComputationalCell3D>& cells) const override
        {
            (void)cells;
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
 
    class GhostChooser : public SeveralGhostGenerator3D::GhostCriteria3D
    {
    public:
        size_t GhostChoose(Tessellation3D const& tess, size_t index) const override
        {
            auto const& box = tess.GetBoxCoordinates();
            Vector3D const& p = tess.GetMeshPoint(index);
            if (p.x < box.first.x)
                return 0;
            if (p.x > box.second.x)
                return 1;
            return 2;
        }
    };
 
    class GrayFreeFreeOpacity : public DiffusionCoefficientCalculator
    {
    public:
        explicit GrayFreeFreeOpacity(double Z)
            : Z_(Z)
        {
        }
 
        double CalcDiffusionCoefficient(ComputationalCell3D const& cell) const override
        {
            const double sigma_a = CalcPlanckOpacity(cell);
            const double sigma_t = std::max(sigma_a + CalcScatteringOpacity(cell), 1e-80);
            return CG::speed_of_light / (3.0 * sigma_t);
        }
 
        double CalcPlanckOpacity(ComputationalCell3D const& cell) const override
        {
            const double T = std::max(cell.temperature, 1.0);
            const double n = std::max(cell.density, 1e-80) * 6.02214076e23;
            const double kT = CG::boltzmann_constant * T;
            const double e_peak = 2.8214393721220787 * kT;
            const double nu_peak = std::max(e_peak / 6.62607015e-27, 1e-80);
            const double stim = 1.0 - std::exp(-e_peak / kT);
            const double gaunt = 1.0;
            return gaunt * 3.7e8 * std::pow(Z_, 3) * n * n / std::sqrt(T) *
                stim * std::pow(nu_peak, -3);
        }
 
        double CalcScatteringOpacity(ComputationalCell3D const& cell) const override
        {
            // Thomson scattering: sigma_s = kappa_T * rho, with kappa_T in cm^2/g.
            constexpr double kappa_thomson = 0.004;
            return kappa_thomson * std::max(cell.density, 0.0);
        }
 
    private:
        double Z_;
    };

    class DiffusionXOpenYZClosedBoundary : public DiffusionBoundaryCalculator
    {
    public:
        void SetBoundaryValues(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
            std::vector<ComputationalCell3D> const& cells, double const Area, double& A, double& b, size_t const face_index) const override
        {
            if (IsXBoundary(tess, index, outside_point))
                open_.SetBoundaryValues(tess, index, outside_point, dt, cells, Area, A, b, face_index);
            else
                closed_.SetBoundaryValues(tess, index, outside_point, dt, cells, Area, A, b, face_index);
        }

        void GetOutSideValues(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, size_t const index, size_t const outside_point,
            std::vector<double> const& new_E, double& E_outside, Vector3D& v_outside) const override
        {
            if (IsXBoundary(tess, index, outside_point))
                open_.GetOutSideValues(tess, cells, index, outside_point, new_E, E_outside, v_outside);
            else
                closed_.GetOutSideValues(tess, cells, index, outside_point, new_E, E_outside, v_outside);
        }

        void SetMomentumTermBoundary(Tessellation3D const& tess, size_t const index, size_t const outside_point, double const dt,
            ComputationalCell3D const& cell, double const Area, double& A, double& b, size_t const face_index,
            double const fleck_factor, double const flux_limiter, double const D, double const sigma_planck) const override
        {
            if (IsXBoundary(tess, index, outside_point))
                open_.SetMomentumTermBoundary(tess, index, outside_point, dt, cell, Area, A, b, face_index,
                    fleck_factor, flux_limiter, D, sigma_planck);
            else
                closed_.SetMomentumTermBoundary(tess, index, outside_point, dt, cell, Area, A, b, face_index,
                    fleck_factor, flux_limiter, D, sigma_planck);
        }

    private:
        static bool IsXBoundary(Tessellation3D const& tess, size_t const index, size_t const outside_point)
        {
            const double R = tess.GetWidth(index);
            const double dx = std::abs(tess.GetMeshPoint(index).x - tess.GetMeshPoint(outside_point).x);
            return dx > R * 1e-4;
        }

        DiffusionOpenBoundary open_;
        DiffusionClosedBox closed_;
    };
 
    double EstimateShockPosition1D(const std::vector<double>& x, const std::vector<double>& density,
        const double density_threshold)
    {
        if (x.empty() || density.size() != x.size())
            return -std::numeric_limits<double>::infinity();

        double shock_x = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < x.size(); ++i)
            if (density[i] > density_threshold)
                shock_x = std::max(shock_x, x[i]);

        return shock_x;
    }
 
    double EstimateShockPositionGlobal(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells,
        int rank, int nprocs, const double density_threshold)
    {
        const size_t nlocal = tess.GetPointNo();
        const int local_n = static_cast<int>(nlocal);
        std::vector<double> local_x(nlocal), local_rho(nlocal);
        for (size_t i = 0; i < nlocal; ++i)
        {
            local_x[i] = tess.GetMeshPoint(i).x;
            local_rho[i] = cells[i].density;
        }
 
#ifdef RICH_MPI
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
 
        std::vector<double> all_x;
        std::vector<double> all_rho;
        if (rank == 0)
        {
            all_x.resize(static_cast<size_t>(total_n));
            all_rho.resize(static_cast<size_t>(total_n));
        }
        MPI_Gatherv(local_x.data(), local_n, MPI_DOUBLE,
            all_x.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_rho.data(), local_n, MPI_DOUBLE,
            all_rho.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
 
        double shock_pos = -std::numeric_limits<double>::infinity();
        if (rank == 0)
            shock_pos = EstimateShockPosition1D(all_x, all_rho, density_threshold);
        MPI_Bcast(&shock_pos, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        return shock_pos;
#else
        (void)rank;
        (void)nprocs;
        return EstimateShockPosition1D(local_x, local_rho, density_threshold);
#endif
    }
}
 
int main(void)
{
#ifndef FREEFREE_NP
#define FREEFREE_NP 512
#endif
#ifndef FREEFREE_COOLING_LIMITER_ON
#define FREEFREE_COOLING_LIMITER_ON false
#endif
    const size_t Np = FREEFREE_NP;
    const double domain_length = 2e12;
    const double dx = domain_length / static_cast<double>(Np);
    const double dy = 0.5 * dx;
    Vector3D ll(0.0, -dy, -dy), ur(domain_length, dy, dy);
 
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
    try
    {
#ifdef RICH_MPI
        tess.BuildParallel(points);
#else
        tess.Build(points);
#endif
    }
    catch (UniversalError const& eo)
    {
        reportError(eo);
        throw;
    }
 
    IdealGas eos(5.0 / 3.0, CG::boltzmann_constant / (1.67e-24 * (5.0 / 3.0 - 1.0)), 1, 0);
    std::vector<ComputationalCell3D> cells(tess.GetPointNo());
 
    ComputationalCell3D left_cell;
    left_cell.density = 2e-13;
    left_cell.temperature = 2e5;
    left_cell.velocity = Vector3D(1e8, 0.0, 0.0);
    left_cell.internal_energy = eos.dT2e(left_cell.density, left_cell.temperature,
        left_cell.tracers, ComputationalCell3D::tracerNames);
    left_cell.pressure = eos.de2p(left_cell.density, left_cell.internal_energy,
        left_cell.tracers, ComputationalCell3D::tracerNames);
    left_cell.Erad = CG::radiation_constant * std::pow(left_cell.temperature, 4) / left_cell.density;
 
    ComputationalCell3D right_cell = left_cell;
    right_cell.velocity = Vector3D(-1e8, 0.0, 0.0);
 
    const double transition_x = 0.5 * domain_length;
    for (size_t i = 0; i < cells.size(); ++i)
    {
        if (tess.GetMeshPoint(i).x <= transition_x)
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
    std::vector<std::pair<const ConditionActionFlux1::Condition3D*, const ConditionActionFlux1::Action3D*>> sequence;
    ConditionActionFlux1::Condition3D* isbulk = new IsBulkFace3D();
    ConditionActionFlux1::Condition3D* is_side = new IsPointLeftRightBox3D();
    ConditionActionFlux1::Condition3D* isboundary = new IsBoundaryFace3D();
    ConditionActionFlux1::Action3D* normal_flux = new RegularFlux3D(rs);
    ConditionActionFlux1::Action3D* rigid_flux = new RigidWallFlux3D(rs);
    sequence.push_back(std::make_pair(is_side, normal_flux));
    sequence.push_back(std::make_pair(isboundary, rigid_flux));
    sequence.push_back(std::make_pair(isbulk, normal_flux));
    ConditionActionFlux1 flux(sequence, interp);
 
    std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D*, const ConditionExtensiveUpdater3D::Action3D*>> eu_sequence;
    ConditionExtensiveUpdater3D eu(eu_sequence);
 
    GrayFreeFreeOpacity opacity(1.0);
    DiffusionXOpenYZClosedBoundary diffusion_boundary;
    Diffusion diffusion(opacity, diffusion_boundary, eos, std::vector<std::string>(), true, true, false, FREEFREE_COOLING_LIMITER_ON);
    DefaultCellUpdater cu(false, 0, true, 0, &diffusion);
    DiffusionForce force(diffusion, eos);
 
    CourantFriedrichsLewy tsf(0.3, 1.0, force);
    Eulerian3D pm;
    HDSim3D sim(tess, cells, eos, pm, tsf, flux, cu, eu, force,
        std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));
 
    const double shock_target = 0.75 * domain_length;
    const size_t max_cycles = 100000;
    const double max_time = 9e4;
    const double density_threshold = 2.0 * left_cell.density;
 
    double current_dt = 1e-20;
    bool aborted_early = false;
    double shock_position = EstimateShockPositionGlobal(tess, sim.getCells(), rank, nprocs, density_threshold);
    while (sim.getCycle() < max_cycles && sim.getTime() < max_time && shock_position < shock_target)
    {
        try
        {
            const auto step_start = std::chrono::steady_clock::now();
            double new_dt = sim.RadiationTimeStep(current_dt, diffusion);
            if (sim.getCycle() >= 1000)
            {
                const double boosted_dt = std::min(5.0, current_dt * 1.05);
                new_dt = std::max(new_dt, boosted_dt);
            }
            tsf.SetTimeStep(new_dt);
            const double old_time = sim.getTime();
            sim.timeAdvance2();
            current_dt = sim.getTime() - old_time;
 
            if (sim.getCycle() % 25 == 0)
                shock_position = EstimateShockPositionGlobal(tess, sim.getCells(), rank, nprocs, density_threshold);
 
            if (rank == 0)
            {
                const auto step_end = std::chrono::steady_clock::now();
                const double wall_sec = std::chrono::duration<double>(step_end - step_start).count();
                std::cout << "\nCycle " << sim.getCycle()
                    << " dt " << current_dt
                    << " time " << sim.getTime()
                    << " shock_x " << shock_position
                    << " wall_time " << wall_sec << "s\n";
            }
        }
        catch (UniversalError const& eo)
        {
            reportError(eo);
            aborted_early = true;
            if (rank == 0)
                std::cerr << "Stopping run after runtime error; writing diagnostics from current state.\n";
            break;
        }
    }
 
    shock_position = EstimateShockPositionGlobal(tess, sim.getCells(), rank, nprocs, density_threshold);
    if (rank == 0 && aborted_early)
        std::cerr << "Run ended early after runtime error.\n";
 
    char cwd_buf[4096];
    if (getcwd(cwd_buf, sizeof(cwd_buf)) == nullptr)
        throw UniversalError("Failed to resolve current working directory");
    const std::string output_dir = std::string(cwd_buf);
    const std::string profile_path = output_dir + "/temperature_profile.txt";
    const std::string shock_path = output_dir + "/shock_position.txt";
 
    const size_t local_n = tess.GetPointNo();
    std::vector<double> local_x(local_n), local_rho(local_n), local_T(local_n), local_Trad(local_n), local_vx(local_n);
    auto const& final_cells = sim.getCells();
    for (size_t i = 0; i < local_n; ++i)
    {
        local_x[i] = tess.GetMeshPoint(i).x;
        local_rho[i] = final_cells[i].density;
        local_T[i] = final_cells[i].temperature;
        local_Trad[i] = std::pow(final_cells[i].Erad * final_cells[i].density / CG::radiation_constant, 0.25);
        local_vx[i] = final_cells[i].velocity.x;
    }
 
#ifdef RICH_MPI
    const int local_n_int = static_cast<int>(local_n);
    std::vector<int> recv_counts(nprocs, 0);
    MPI_Gather(&local_n_int, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
 
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
 
    std::vector<double> all_x, all_rho, all_T, all_Trad, all_vx;
    if (rank == 0)
    {
        all_x.resize(static_cast<size_t>(total_n));
        all_rho.resize(static_cast<size_t>(total_n));
        all_T.resize(static_cast<size_t>(total_n));
        all_Trad.resize(static_cast<size_t>(total_n));
        all_vx.resize(static_cast<size_t>(total_n));
    }
    MPI_Gatherv(local_x.data(), local_n_int, MPI_DOUBLE, all_x.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_rho.data(), local_n_int, MPI_DOUBLE, all_rho.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_T.data(), local_n_int, MPI_DOUBLE, all_T.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_Trad.data(), local_n_int, MPI_DOUBLE, all_Trad.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_vx.data(), local_n_int, MPI_DOUBLE, all_vx.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
 
    if (rank == 0)
    {
        std::vector<size_t> idx(all_x.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return all_x[a] < all_x[b]; });
 
        std::ofstream out(profile_path);
        for (size_t i = 0; i < idx.size(); ++i)
            out << all_x[idx[i]] << " " << all_rho[idx[i]] << " " << all_T[idx[i]] << " " << all_Trad[idx[i]] << " " << all_vx[idx[i]] << "\n";
 
        std::ofstream shock_out(shock_path);
        shock_out << shock_position << "\n";
    }
#else
    std::vector<size_t> idx(local_n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return local_x[a] < local_x[b]; });
 
    std::ofstream out(profile_path);
    for (size_t i = 0; i < idx.size(); ++i)
        out << local_x[idx[i]] << " " << local_rho[idx[i]] << " " << local_T[idx[i]] << " " << local_Trad[idx[i]] << " " << local_vx[idx[i]] << "\n";
 
    std::ofstream shock_out(shock_path);
    shock_out << shock_position << "\n";
#endif
 
#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return 0;
}
