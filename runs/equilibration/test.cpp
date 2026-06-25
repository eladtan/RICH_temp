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
#include "source/3D/output/write3D.hpp"
#include "source/newtonian/three_dimensional/AMR3D.hpp"
#include "source/Radiation/Diffusion.hpp"
#include "source/Radiation/DiffusionForce.hpp"
#include "source/Radiation/MultigroupDiffusion.hpp"
#include "source/Radiation/MultigroupDiffusionCoefficientCalculator.hpp"
#include "source/Radiation/MultigroupDiffusionBoundaryCalculator.hpp"
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
#include <source/Radiation/CMMC/src/planck_integral/planck_integral.hpp>
#include <algorithm>
#include "boost/math/special_functions/pow.hpp"


typedef std::array<double, 4> state_type;

static constexpr double ev = 1.602176634e-12;
static constexpr double kev = 1e3*ev;

static constexpr double ev_kelvin = ev / CG::boltzmann_constant;
static constexpr double kev_kelvin = 1e3*ev_kelvin;

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

	std::size_t const G = ENERGY_GROUPS_NUM;
	std::vector<double> energy_groups_center(G);
	std::vector<double> energy_groups_boundary(G+1);

	double const Emin = kev*1e-4;
	double const Emax = kev*5e2;
	
	energy_groups_boundary[0] = Emin;
	for(std::size_t g=0; g < G; ++g){
		energy_groups_boundary[g+1] = std::pow(Emax/Emin, 1.0/G)*energy_groups_boundary[g];
		energy_groups_center[g] = 0.5*(energy_groups_boundary[g+1]+energy_groups_boundary[g]);
	}
	double const lscale = 1.;
	double const mscale = 1.;
	double const tscale = 1.;
	// double const lscale = 1;
	// double const mscale = 1;
	// double const tscale = 1;
	if (rank == 0)
		std::cout << "start eos" << std::endl;

	double constexpr m_p = 1.6726231e-24;
	double constexpr cv = 1.0 * CG::boltzmann_constant / (1.4-1.0) / m_p;
    IdealGas eos(/*gamma=*/1.4, /*f=*/cv, /*beta=*/1.0, /*mu=*/0.0);

	if (rank == 0)
		std::cout << "end eos" << std::endl;
	
    if (rank == 0)
		std::cout << "end sta" << std::endl;

	const double width = 3 / lscale;
	size_t const Nx = 1;
	Vector3D ll(0, -0.5 * width / Nx, -0.5 * width / Nx), ur(width, 0.5 * width / Nx, 0.5 * width / Nx);
	Voronoi3D tess(ll, ur);

    using boost::math::pow;

	FreeFreeAbsorptionOpacityMultigroup opacity(1.0, energy_groups_center, energy_groups_boundary);
	int counter = 0;
	ComputationalCell3D init_cell;

	double const T_mat = 10.0*kev_kelvin;
	double const T_rad = 1.0*kev_kelvin;

	try
	{
		init_cell.density = 1. * lscale * lscale * lscale / mscale;
		init_cell.temperature = T_mat;
		init_cell.internal_energy = eos.dT2e(init_cell.density, init_cell.temperature, init_cell.tracers, ComputationalCell3D::tracerNames);
		init_cell.pressure = eos.de2p(init_cell.density, init_cell.internal_energy, init_cell.tracers, ComputationalCell3D::tracerNames);
		init_cell.Erad = CG::radiation_constant * pow<4>(T_rad) * tscale * tscale / (init_cell.density * mscale / lscale);
        for(std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g){
            init_cell.Eg[g] = planck_integral::planck_energy_density_group_integral(energy_groups_boundary[g], energy_groups_boundary[g+1], T_rad);
			init_cell.Eg[g] *= tscale * tscale / (init_cell.density * mscale / lscale); 
			init_cell.Eg[g]  = std::max(init_cell.Eg[g], init_cell.Erad*1e-8);
        }

		std::cout << "Erad=" << init_cell.Erad << ", sumEg=" << std::accumulate(init_cell.Eg.begin(), init_cell.Eg.end(), 0.0) << std::endl; 
	}
	catch (UniversalError const &eo)
	{
		reportError(eo);
		throw;
	}


	vector<Vector3D> points; 
	if(rank == 0)
		points = CartesianMesh(Nx, 1, 1, ll, ur);
#ifdef RICH_MPI
	tess.BuildParallel(points);
#else
	tess.Build(points);
#endif
	vector<ComputationalCell3D> cells(tess.GetPointNo(), init_cell);
	for(size_t i=0; i<cells.size(); ++i)
	{
		if(tess.GetCellCM(i).x < 2.0)
			cells[i].tracers[0] = 1.0;
		else
			cells[i].tracers[1] = 1.0;
	}

	Hllc3D rs;
	RigidWallGenerator3D ghost;
	LinearGauss3D interp(eos, ghost, true, 0.2, 0.25, 0.75);

	Lagrangian3D bpm;
	RoundCells3D pm(bpm, eos, 3.75, 0.01, false, 1.25);
	
	MultigroupDiffusionClosedBoundary D_boundary{};
	MultigroupDiffusion matrix_builder(energy_groups_center, energy_groups_boundary, opacity, D_boundary, eos, std::vector<std::string> (), true, false, false, true, false);
	matrix_builder.length_scale_ = lscale;
	matrix_builder.time_scale_ = tscale;
	matrix_builder.mass_scale_ = mscale;
	ZeroForce3D force = ZeroForce3D();

	DefaultCellUpdater cu(false, 0.0, true, 0.0, &matrix_builder);

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

	HDSim3D sim(tess, cells, eos, pm, tsf, fc, cu, eu, force, std::pair<std::vector<std::string>, std::vector<std::string>> (ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames), false, true);

	double init_dt = 1e-25 / tscale;
	double const tf = 5e-9 / tscale;
	double const dt_output = tf / 100.;
	tsf.SetTimeStep(init_dt);
	double nextT = dt_output;
	double old_dt = init_dt;
	vector<DiagnosticAppendix3D *> appendices;
	WriteSnapshot3D(sim, "init.h5", appendices, true);
	double new_dt = init_dt;
	while (sim.getTime() < tf)
	{
		if (sim.getCycle() % 1 == 0)
		{
			if (rank == 0)
			{
				std::cout<<std::endl;
				std::cout << "Cycle " << sim.getCycle() << " Time " << sim.getTime() << " dt " << new_dt << std::endl;
			}
		}
		if (sim.getTime() > nextT)
		{
			WriteSnapshot3D(sim, "snap_" + int2str(counter) + ".h5", appendices, true);
			nextT = sim.getTime() + dt_output;
			++counter;
		}
		try
		{
			new_dt = sim.RadiationTimeStep(old_dt, matrix_builder, true);
			// tsf.SetTimeStep(new_dt);
			// sim.SetTimeStep(new_dt);
			new_dt=std::min(new_dt, 5e-11);
			if (rank == 0)
				std::cout<<"New time step is "<<new_dt<<std::endl;
			old_dt = new_dt;
		}
		catch (UniversalError const &eo)
		{
			reportError(eo);
			throw;
		}
	}

	WriteSnapshot3D(sim, "snap_" + int2str(counter) + ".h5", appendices, true);
	++counter;
	std::cout<<"Done"<<std::endl;
#ifdef RICH_MPI
	MPI_Finalize();
#endif
	return 0;
}
