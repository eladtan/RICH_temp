#include "3D/tessellation/Voronoi3D.hpp"
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/3D/output/write3D.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/PCM3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/misc/simple_io.hpp"
#include "source/newtonian/three_dimensional/eulerian_3d.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/Ghost3D.hpp"
#include "source/newtonian/three_dimensional/OndrejEOS.hpp"
#include "source/3D/output/write3D.hpp"
#include "source/newtonian/three_dimensional/AMR3D.hpp"
#include "source/Radiation/Diffusion.hpp"
#include "source/Radiation/DiffusionForce.hpp"
#include "source/Radiation/STAgreyOpacity.hpp"
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

int main(void)
{
	int rank = 0;
	int ws = 1;
#ifdef RICH_MPI
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ws);
#endif
	feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
	std::string eos_location("/home/itamarg/workspace/RICH/data/EOS/");
	std::string STA_location("/home/itamarg/workspace/RICH/data/STA/");

	double const lscale = 7e10;
	double const mscale = 2e33;
	double const tscale = 1603;

	OndrejEOS eos(eos_location + "density.txt", eos_location + "Pfile.txt", eos_location + "csfile.txt", eos_location + "Sfile.txt", eos_location + "Ufile.txt", eos_location + "Tfile.txt", eos_location + "CVfile.txt", lscale, mscale, tscale);

	//Radiation
	STAgreyOpacity opacity(STA_location);

	const double width = 1e9 / lscale;
	size_t const Nx = 128;
	size_t const Ny = 3;
	size_t const Nz = 3;
	Vector3D ll(0, -0.5 * width * Ny / Nx, -0.5 * width * Nz / Nx), ur(width, 0.5 * width * Ny / Nx, 0.5 * width * Nz / Nx);
	Voronoi3D tess(ll, ur);
	int counter = 0;
	ComputationalCell3D init_cell;
	double const T = 2000;
	try
	{
		init_cell.density = 0.5 * lscale * lscale * lscale / mscale;
		init_cell.temperature = T;
		init_cell.pressure = eos.dT2p(init_cell.density, init_cell.temperature);
		init_cell.internal_energy = eos.dp2e(init_cell.density, init_cell.pressure);
		init_cell.Erad = CG::radiation_constant * T * T * T * T * tscale * tscale / (init_cell.density * mscale / lscale);
	}
	catch (UniversalError const &eo)
	{
		reportError(eo);
		throw;
	}

	vector<Vector3D> points;
	if(rank == 0)
		points = CartesianMesh(Nx, Ny, Nz, ll, ur);
#ifdef RICH_MPI	
	points = MPI_Spread(points, 0, MPI_COMM_WORLD);	
	tess.BuildParallel(points);
	#else
	tess.Build(points);
	#endif
	vector<ComputationalCell3D> cells(tess.GetPointNo(), init_cell);

	Hllc3D rs;
	RigidWallGenerator3D ghost;
	LinearGauss3D interp(eos, ghost);

	Eulerian3D pm;
	
	DiffusionSideBoundary D_boundary(1.1605e7); // 1 kev
	Diffusion matrix_builder(opacity, D_boundary, eos, std::vector<std::string> (), false, false, false);
	matrix_builder.length_scale_ = lscale;
	matrix_builder.time_scale_ = tscale;
	matrix_builder.mass_scale_ = mscale;
	DiffusionForce force(matrix_builder, eos);

	DefaultCellUpdater cu(false, 0, true, &matrix_builder);

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

	CourantFriedrichsLewy tsf(0.25, 1, force);

	HDSim3D sim(tess, cells, eos, pm, tsf, fc, cu, eu, force, std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));

	double init_dt = 1e-15 / tscale;
	double const dt_output = 1e7 / tscale;
	double const tf = dt_output * 5;
	tsf.SetTimeStep(init_dt);
	double nextT = dt_output;
	double old_dt = init_dt;

	auto t1 = std::chrono::high_resolution_clock::now();
	while (sim.getTime() < tf)
	{
		if (sim.getCycle() % 1 == 0)
		{
			if (rank == 0)
			{
				std::cout<<std::endl;
				std::cout << "Cycle " << sim.getCycle() << " Time " << sim.getTime() << " dt "<<old_dt<<std::endl;
			}
		}
		if (sim.getTime() > nextT)
		{
			WriteSnapshot3D(sim, "snap_" + int2str(counter) + ".h5");
			nextT = sim.getTime() + dt_output;
			++counter;
		}
		try
		{
			double new_dt = sim.RadiationTimeStep(old_dt, matrix_builder, true);
			tsf.SetTimeStep(new_dt);
			old_dt = new_dt;
		}
		catch (UniversalError const &eo)
		{
			reportError(eo);
			throw;
		}
	}
	WriteSnapshot3D(sim, "final.h5");
	auto t2 = std::chrono::high_resolution_clock::now();
	if(rank == 0)
		std::cout<<"run time took "<< std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()<<" seconds"<<std::endl;
	std::cout<<"Done"<<std::endl;
#ifdef RICH_MPI
	MPI_Finalize();
#endif
	return 0;
}
