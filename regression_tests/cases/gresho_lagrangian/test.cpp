#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/HydroStep.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/newtonian/three_dimensional/Lagrangian3D.hpp"
#include "source/newtonian/three_dimensional/RoundCells3D.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/Ghost3D.hpp"
#include "source/newtonian/three_dimensional/ConservativeForce3D.hpp"
#include <fstream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <fenv.h>
#include <libgen.h>
#include <string.h>

#ifdef RICH_MPI
#include "mpi/mpi_commands.hpp"
#endif

// Gresho vortex test — Lagrangian + RoundCells, restricted to xy plane
// Domain: [-0.5, 0.5]^2 x [0, dz], Cartesian 50x50x1
// gamma = 5/3, t_end = 5

namespace {

double azimuthal_velocity(double r)
{
    if (r < 0.2)
        return 5.0 * r;
    else if (r > 0.4)
        return 0.0;
    else
        return 2.0 - 5.0 * r;
}

double calc_pressure(double r)
{
    if (r < 0.2)
        return 5.0 + 12.5 * r * r;
    else if (r > 0.4)
        return 3.0 + 4.0 * std::log(2.0);
    else
        return 9.0 + 12.5 * r * r - 20.0 * r + 4.0 * std::log(r / 0.2);
}

class XYPlanePointMotion : public PointMotion3D
{
public:
    XYPlanePointMotion(const PointMotion3D& inner) : inner_(inner) {}

    void operator()(const Tessellation3D& tess,
        const std::vector<ComputationalCell3D>& cells,
        double time, std::vector<Vector3D>& res) const override
    {
        inner_(tess, cells, time, res);
        for (auto& v : res)
            v.z = 0;
    }

    void ApplyFix(Tessellation3D const& tess,
        std::vector<ComputationalCell3D> const& cells,
        double time, double dt,
        std::vector<Vector3D>& velocities) const override
    {
        inner_.ApplyFix(tess, cells, time, dt, velocities);
        for (auto& v : velocities)
            v.z = 0;
    }

private:
    const PointMotion3D& inner_;
};

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
    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

    IdealGas eos(5.0 / 3.0);

    size_t const Nx = 50;
    size_t const Ny = 50;
    double const dz = 1.0 / Ny;
    Vector3D ll(-0.5, -0.5, 0), ur(0.5, 0.5, dz);

    std::vector<Vector3D> points;
    if (rank == 0)
        points = CartesianMesh(Nx, Ny, 1, ll, ur);
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

    size_t Nlocal = tess.GetPointNo();
    std::vector<ComputationalCell3D> cells(Nlocal);
    for (size_t i = 0; i < Nlocal; ++i)
    {
        Vector3D const& pos = tess.GetMeshPoint(i);
        double const r = std::sqrt(pos.x * pos.x + pos.y * pos.y);
        double const vtheta = azimuthal_velocity(r);

        cells[i].density = 1.0;
        cells[i].pressure = calc_pressure(r);
        cells[i].internal_energy = eos.dp2e(cells[i].density, cells[i].pressure,
            cells[i].tracers, ComputationalCell3D::tracerNames);

        if (r > 1e-10)
        {
            cells[i].velocity.x = -vtheta * pos.y / r;
            cells[i].velocity.y = vtheta * pos.x / r;
        }
        else
        {
            cells[i].velocity.x = 0;
            cells[i].velocity.y = 0;
        }
        cells[i].velocity.z = 0;
    }

    Hllc3D rs;
    RigidWallGenerator3D ghost;
    LinearGauss3D interp(eos, ghost);

    Lagrangian3D bpm;
    RoundCells3D round_pm(bpm, eos);
    XYPlanePointMotion pm(round_pm);

    ZeroForce3D force;
    DefaultCellUpdater cu;

    RigidWallFlux3D rigidflux(rs);
    RegularFlux3D *regular_flux = new RegularFlux3D(rs);
    IsBoundaryFace3D *boundary_face = new IsBoundaryFace3D();
    IsBulkFace3D *bulk_face = new IsBulkFace3D();
    std::vector<std::pair<const ConditionActionFlux1::Condition3D *,
        const ConditionActionFlux1::Action3D *>> flux_vector;
    flux_vector.push_back({boundary_face, &rigidflux});
    flux_vector.push_back({bulk_face, regular_flux});
    ConditionActionFlux1 fc(flux_vector, interp);

    std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D *,
        const ConditionExtensiveUpdater3D::Action3D *>> eu_sequence;
    ConditionExtensiveUpdater3D eu(eu_sequence);

    auto tsf = std::make_shared<CourantFriedrichsLewy>(0.3, 1, force);

    Simulation simulation(tess, cells, eos);
    simulation.SetTimeStepFunction(tsf);
    HDSim3D sim(tess, simulation.getCells(), simulation.getExtensives(), eos, simulation.getTracker(), pm, *tsf, fc, cu, eu, force,
        std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));

    auto hydroStep = std::make_shared<HydroStep>(sim, HydroStep::TIMEADVANCE_2);
    simulation.addPhysics(hydroStep);
    simulation.SetTimeStep(1.0);
#ifdef RICH_MPI
    simulation.PresetLoadBalance("hydro");
#endif

    double const tf = 5.0;
    while (simulation.GetTime() < tf)
    {
        try
        {
            simulation.step();
        }
        catch (UniversalError const& eo)
        {
            reportError(eo);
            throw;
        }

        if (rank == 0)
            std::cout << "\nCycle " << simulation.GetCycle() << " Time " << simulation.GetTime()
                << " dt " << simulation.GetTimeStep() << "\n" << std::endl;
    }

    char file_buf[4096];
    strncpy(file_buf, __FILE__, sizeof(file_buf) - 1);
    file_buf[sizeof(file_buf) - 1] = '\0';
    std::string dir_path = std::string(dirname(file_buf));

    // Gather profile data from all MPI ranks
    {
        Nlocal = tess.GetPointNo();
        auto const& final_cells = simulation.getCells();
        std::vector<double> local_x(Nlocal), local_y(Nlocal), local_vol(Nlocal);
        std::vector<double> local_p(Nlocal), local_vx(Nlocal), local_vy(Nlocal);
        for (size_t i = 0; i < Nlocal; ++i)
        {
            local_x[i] = tess.GetCellCM(i).x;
            local_y[i] = tess.GetCellCM(i).y;
            local_vol[i] = tess.GetVolume(i);
            local_p[i] = final_cells[i].pressure;
            local_vx[i] = final_cells[i].velocity.x;
            local_vy[i] = final_cells[i].velocity.y;
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

        std::vector<double> all_x, all_y, all_vol, all_p, all_vx, all_vy;
        if (rank == 0)
        {
            all_x.resize(total_n);
            all_y.resize(total_n);
            all_vol.resize(total_n);
            all_p.resize(total_n);
            all_vx.resize(total_n);
            all_vy.resize(total_n);
        }
        MPI_Gatherv(local_x.data(), local_n, MPI_DOUBLE, all_x.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_y.data(), local_n, MPI_DOUBLE, all_y.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_vol.data(), local_n, MPI_DOUBLE, all_vol.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_p.data(), local_n, MPI_DOUBLE, all_p.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_vx.data(), local_n, MPI_DOUBLE, all_vx.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_vy.data(), local_n, MPI_DOUBLE, all_vy.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            std::ofstream out(dir_path + "/gresho_profile.txt");
            out << std::scientific << std::setprecision(12);
            for (int i = 0; i < total_n; ++i)
            {
                out << all_x[i] << " " << all_y[i] << " " << all_vol[i] << " "
                    << all_p[i] << " " << all_vx[i] << " " << all_vy[i] << "\n";
            }
        }
#else
        std::ofstream out(dir_path + "/gresho_profile.txt");
        out << std::scientific << std::setprecision(12);
        for (size_t i = 0; i < Nlocal; ++i)
        {
            out << local_x[i] << " " << local_y[i] << " " << local_vol[i] << " "
                << local_p[i] << " " << local_vx[i] << " " << local_vy[i] << "\n";
        }
#endif
    }

    if (rank == 0)
    {
        std::ofstream tf_file(dir_path + "/test_type.txt");
        tf_file << "lagrangian\n";
    }

    if (rank == 0)
        std::cout << "Done" << std::endl;

#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return 0;
}
