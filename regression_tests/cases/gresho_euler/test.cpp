#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/HydroStep.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/newtonian/three_dimensional/eulerian_3d.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/Ghost3D.hpp"
#include "source/newtonian/three_dimensional/ConservativeForce3D.hpp"
#include <fstream>
#include <iomanip>
#include <cmath>
#include <fenv.h>
#include <libgen.h>
#include <string.h>

// Gresho vortex test — Eulerian mesh
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

} // namespace

int main(void)
{
    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

    IdealGas eos(5.0 / 3.0);

    size_t const Nx = 50;
    size_t const Ny = 50;
    double const dz = 1.0 / Ny;
    Vector3D ll(-0.5, -0.5, 0), ur(0.5, 0.5, dz);
    Voronoi3D tess(ll, ur);

    std::vector<Vector3D> points = CartesianMesh(Nx, Ny, 1, ll, ur);
    try {
        tess.Build(points);
    } catch (UniversalError const& eo) {
        reportError(eo);
        throw;
    }

    std::vector<ComputationalCell3D> cells(tess.GetPointNo());
    for (size_t i = 0; i < tess.GetPointNo(); ++i)
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
    Eulerian3D pm;

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

        std::cout << "\nCycle " << simulation.GetCycle() << " Time " << simulation.GetTime()
            << " dt " << simulation.GetTimeStep() << "\n" << std::endl;
    }

    char file_buf[4096];
    const char *artifact_dir = getenv("THUNDER_ARTIFACT_DIR");
    std::string dir_path;
    if(artifact_dir && artifact_dir[0] != '\0')
    {
        dir_path = artifact_dir;
    }
    else
    {
        getcwd(file_buf, sizeof(file_buf));
        dir_path = file_buf;
    }

    {
        auto const& final_cells = simulation.getCells();
        size_t const N = tess.GetPointNo();
        std::ofstream out(dir_path + "/gresho_profile.txt");
        out << std::scientific << std::setprecision(12);
        for (size_t i = 0; i < N; ++i)
        {
            out << tess.GetCellCM(i).x << " "
                << tess.GetCellCM(i).y << " "
                << tess.GetVolume(i) << " "
                << final_cells[i].pressure << " "
                << final_cells[i].velocity.x << " "
                << final_cells[i].velocity.y << "\n";
        }
    }

    {
        std::ofstream tf_file(dir_path + "/test_type.txt");
        tf_file << "euler\n";
    }

    std::cout << "Done" << std::endl;
    return 0;
}
