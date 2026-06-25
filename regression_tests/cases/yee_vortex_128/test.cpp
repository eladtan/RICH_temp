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

namespace {

double const GAMMA = 1.4;
double const BETA = 5.0;

double vortex_temperature(double r2)
{
    return 1.0 - (GAMMA - 1.0) * BETA * BETA
        / (8.0 * GAMMA * M_PI * M_PI) * std::exp(1.0 - r2);
}

double vortex_density(double r2)
{
    double T = vortex_temperature(r2);
    return std::pow(T, 1.0 / (GAMMA - 1.0));
}

double vortex_pressure(double r2)
{
    double T = vortex_temperature(r2);
    return std::pow(T, GAMMA / (GAMMA - 1.0));
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

    IdealGas eos(GAMMA);

    size_t const Nx = 128;
    size_t const Ny = 128;
    double const dz = 10.0 / Ny;
    Vector3D ll(-5.0, -5.0, 0), ur(5.0, 5.0, dz);

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
        double const r2 = pos.x * pos.x + pos.y * pos.y;
        double const exp_half = std::exp(0.5 * (1.0 - r2));
        double const vel_factor = BETA / (2.0 * M_PI) * exp_half;

        cells[i].density = vortex_density(r2);
        cells[i].pressure = vortex_pressure(r2);
        cells[i].internal_energy = eos.dp2e(cells[i].density, cells[i].pressure,
            cells[i].tracers, ComputationalCell3D::tracerNames);
        cells[i].velocity.x = -vel_factor * pos.y;
        cells[i].velocity.y =  vel_factor * pos.x;
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

    double const tf = 10.0;
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

    {
        Nlocal = tess.GetPointNo();
        auto const& final_cells = sim.getCells();
        std::vector<double> local_x(Nlocal), local_y(Nlocal), local_vol(Nlocal);
        std::vector<double> local_rho(Nlocal), local_p(Nlocal);
        std::vector<double> local_vx(Nlocal), local_vy(Nlocal);
        for (size_t i = 0; i < Nlocal; ++i)
        {
            local_x[i]   = tess.GetCellCM(i).x;
            local_y[i]   = tess.GetCellCM(i).y;
            local_vol[i]  = tess.GetVolume(i);
            local_rho[i] = final_cells[i].density;
            local_p[i]   = final_cells[i].pressure;
            local_vx[i]  = final_cells[i].velocity.x;
            local_vy[i]  = final_cells[i].velocity.y;
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

        auto gather = [&](std::vector<double>& local, std::vector<double>& all) {
            if (rank == 0) all.resize(total_n);
            MPI_Gatherv(local.data(), local_n, MPI_DOUBLE,
                         all.data(), recv_counts.data(), displs.data(),
                         MPI_DOUBLE, 0, MPI_COMM_WORLD);
        };

        std::vector<double> all_x, all_y, all_vol, all_rho, all_p, all_vx, all_vy;
        gather(local_x, all_x);
        gather(local_y, all_y);
        gather(local_vol, all_vol);
        gather(local_rho, all_rho);
        gather(local_p, all_p);
        gather(local_vx, all_vx);
        gather(local_vy, all_vy);

        if (rank == 0)
        {
            std::ofstream out(dir_path + "/vortex_profile.txt");
            out << std::scientific << std::setprecision(12);
            for (int i = 0; i < total_n; ++i)
            {
                out << all_x[i] << " " << all_y[i] << " " << all_vol[i] << " "
                    << all_rho[i] << " " << all_p[i] << " "
                    << all_vx[i] << " " << all_vy[i] << "\n";
            }
        }
#else
        std::ofstream out(dir_path + "/vortex_profile.txt");
        out << std::scientific << std::setprecision(12);
        for (size_t i = 0; i < Nlocal; ++i)
        {
            out << local_x[i] << " " << local_y[i] << " " << local_vol[i] << " "
                << local_rho[i] << " " << local_p[i] << " "
                << local_vx[i] << " " << local_vy[i] << "\n";
        }
#endif
    }

    if (rank == 0)
        std::cout << "Done" << std::endl;

#ifdef RICH_MPI
    MPI_Finalize();
#endif
    return 0;
}
