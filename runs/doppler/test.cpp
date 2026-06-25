#include "3D/tessellation/Voronoi3D.hpp"
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/SeveralSources3D.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/PCM3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/misc/simple_io.hpp"
#include "source/newtonian/three_dimensional/Lagrangian3D.hpp"
#include "source/newtonian/three_dimensional/RoundCells3D.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/Ghost3D.hpp"
#include "source/newtonian/three_dimensional/OndrejEOS.hpp"
#include "source/3D/output/write3D.hpp"
#include "source/3D/output/read3D.hpp"
#include "source/newtonian/three_dimensional/AMR3D.hpp"
#include "source/newtonian/three_dimensional/GravityAcc3D.hpp"
#include "source/Radiation/Diffusion.hpp"
#include "source/Radiation/MultigroupDiffusionForce.hpp"
#include "source/Radiation/MultigroupDiffusion.hpp"
#include "source/misc/int2str.hpp"
#include <boost/numeric/odeint.hpp>
#include "source/newtonian/three_dimensional/LagrangianExtensiveUpdater3D.hpp"
#include <boost/math/tools/roots.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <exception>
#include <fenv.h>
#include <filesystem>
#include "source/3D/GeometryCommon/UpdateBox.hpp"
namespace fs = std::filesystem;
#include <sys/stat.h>
#include <boost/math/tools/roots.hpp>
#include <sstream>
#include <MeshDecomposer3D/kernels/Rectangle.hpp>
#include "source/newtonian/three_dimensional/Dissipation.hpp"

int main(void)
{
	feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
	Vector3D ll(0, -1, -1), ur(2, 1, 1);
    GrayPowerLawOpacity opacity(1e-100, 0, 0, 1e-100, 0, 0);
    size_t const Ng = ENERGY_GROUPS_NUM;
    double const Emin = units::ev;
    double const Emax = units::kev * 15;
    opacity.energy_groups_boundary = linspace(std::log(Emin), std::log(Emax), Ng + 1);
    for(size_t i = 0; i < Ng + 1; ++i)
        opacity.energy_groups_boundary[i] = std::exp(opacity.energy_groups_boundary[i]);
    opacity.energy_groups_center.resize(Ng);
    for(size_t i = 0; i < Ng; ++i)
        opacity.energy_groups_center[i] = std::sqrt(opacity.energy_groups_boundary[i] * opacity.energy_groups_boundary[i + 1]);
    IdealGas eos(5.0 / 3.0, units::arad * 0.1, 4, 1);
	Voronoi3D tess(ll, ur);
    std::vector<Vector3D> points(2);
    points[0].x = 0.5;
    points[1].x = 1.5;
    tess.Build(points);

	vector<ComputationalCell3D> cells(2);
    cells[0].density = 1.0;
    cells[0].temperature = units::kev_kelvin;
    cells[0].internal_energy = eos.dT2e(cells[0].density, cells[0].temperature, cells[0].tracers, ComputationalCell3D::tracerNames);
    cells[0].pressure = eos.de2p(cells[0].density, cells[0].internal_energy, cells[0].tracers, ComputationalCell3D::tracerNames);
	for(size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
        cells[0].Eg[g] = planck_integral::planck_energy_density_group_integral(opacity.energy_groups_boundary[g], opacity.energy_groups_boundary[g + 1], cells[0].temperature) / cells[0].density;
    for(size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
        cells[0].Erad += cells[0].Eg[g];
    cells[0].velocity = Vector3D(1, 0, 0);
    cells[1] = cells[0];
    // cells[1].velocity = Vector3D(-1, 0, 0);

	Hllc3D rs;

	RigidWallGenerator3D ghost;
    // ComputationalCell3D left_cell(cells[0]), right_cell(cells[0]);
    // left_cell.velocity = Vector3D(1, 0, 0);
    // right_cell.velocity = Vector3D(-1, 0, 0);
    // ConstantPrimitiveGenerator3D left_ghost(left_cell);
    // ConstantPrimitiveGenerator3D right_ghost(right_cell);
    // SeveralGhostGenerator3D ghost(ghosts, gchooser);
	LinearGauss3D interp(eos, ghost, true, 0.2, 0.25, 0.75);
	double Tmin = 1e3;

	Lagrangian3D pm;
	MultigroupDiffusionClosedBoundary D_boundary;
	bool const hydro_on = false;
	bool const compton_on = false;
	bool const flux_limit = true;
	bool const doppler_on = true;
	bool const mixed_frame_on = false;
	MultigroupDiffusion matrix_builder(opacity.energy_groups_center, opacity.energy_groups_boundary, opacity, D_boundary, eos, std::vector<std::string>(), flux_limit, hydro_on, compton_on, doppler_on, mixed_frame_on);

	DefaultCellUpdater cu(false, 0, true, 2000, &matrix_builder);

	RigidWallFlux3D rigidflux(rs);
	RegularFlux3D *regular_flux = new RegularFlux3D(rs);
	IsBoundaryFace3D *boundary_face = new IsBoundaryFace3D();
	IsBulkFace3D *bulk_face = new IsBulkFace3D();
	vector<pair<const ConditionActionFlux1::Condition3D *, const ConditionActionFlux1::Action3D *>> flux_vector;
	flux_vector.push_back(pair<const ConditionActionFlux1::Condition3D *, const ConditionActionFlux1::Action3D *>(boundary_face, &rigidflux));
	flux_vector.push_back(pair<const ConditionActionFlux1::Condition3D *, const ConditionActionFlux1::Action3D *>(bulk_face, regular_flux));
	ConditionActionFlux1 fc(flux_vector, interp);

	vector<pair<const ConditionExtensiveUpdater3D::Condition3D *, const ConditionExtensiveUpdater3D::Action3D *>> eu_sequence;
	ConditionExtensiveUpdater3D eu(eu_sequence);
	ZeroForce3D force;
	CourantFriedrichsLewy tsf(0.3, 1, force);

	std::unique_ptr<HDSim3D> sim;
	sim = std::make_unique<HDSim3D>(tess, cells, eos, pm, tsf, fc, cu, eu, force, std::pair<std::vector<std::string>, std::vector<std::string>> (ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames), false, true);
    double const dt = 0.02;
    std::vector<double> time;
    std::vector<std::vector<double>> Erad_compress(ENERGY_GROUPS_NUM), Erad_expand(ENERGY_GROUPS_NUM);
	while (sim->getTime() < 4)
	{
        try
        {
            time.push_back(sim->getTime());
            for(size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
            {
                Erad_expand[g].push_back(sim->getCells()[0].Eg[g] * sim->getCells()[0].density);
                Erad_compress[g].push_back(sim->getCells()[1].Eg[g] * sim->getCells()[1].density);
            }
            std::cout<<std::endl;
            std::cout << "Cycle " << sim->getCycle() << " Time " << sim->getTime() << std::endl;
            sim->RadiationTimeStep(dt, matrix_builder, true);	
		}
		catch (UniversalError const &eo)
		{
			reportError(eo);
			throw;
		}
	}
    for(size_t g = 0; g < ENERGY_GROUPS_NUM; ++g)
    {
        write_vector(Erad_compress[g], "Erad_compress_" + std::to_string(g) + ".txt");
        write_vector(Erad_expand[g], "Erad_expand_" + std::to_string(g) + ".txt");
    }
	write_vector(time, "time.txt");
    write_vector(opacity.energy_groups_center, "E_center.txt");
    std::cout<<"Done sim"<<std::endl;
	return 0;
}
