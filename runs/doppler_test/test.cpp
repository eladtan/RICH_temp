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
	double const Emax = kev*1e2;
	
	static_assert(ENERGY_GROUPS_NUM > 3, "Energy groups number must be greater than 3");

	energy_groups_boundary[0] = Emin;
	for(std::size_t g=0; g < G; ++g){
		energy_groups_boundary[g+1] = std::pow(Emax/Emin, 1.0/G)*energy_groups_boundary[g];
		energy_groups_center[g] = 0.5*(energy_groups_boundary[g+1]+energy_groups_boundary[g]);
	}

	// energy groups will be filled from energy_groups_boundary[thresh_hold_boundary_left] to energy_groups_boundary[thresh_hold_boundary_right]
	// equivalent to filling only the energy groups `thresh_hold_boundary_left` up to `thresh_hold_boundary_right-1`
	unsigned int const thresh_hold_boundary_left = 70;
	unsigned int const thresh_hold_boundary_right = 85;	

	double const E_thresh_left = energy_groups_boundary[thresh_hold_boundary_left];
	double const E_thresh_right = energy_groups_boundary[thresh_hold_boundary_right];
	
	if(rank == 0) {
		std::cout << "thresh_hold_boundary_left" << thresh_hold_boundary_left << ", thresh_hold_boundary_right: " << thresh_hold_boundary_right <<  std::endl;
		std::cout << "E_thresh_left: " << E_thresh_left/kev << "KeV, E_thresh_right: " << E_thresh_right/kev << "KeV" << std::endl;
	}

	double const lscale = 1.;
	double const mscale = 1.;
	double const tscale = 1.;

	if (rank == 0) std::cout << "start eos" << std::endl;

	// This eos is irrelevent to the test since there is no hydrodynamics
    double const cv = 1e15 / kev_kelvin;
    IdealGas eos(/*gamma=*/1.4, /*f=*/cv, /*beta=*/1.0, /*mu=*/0.0);

	if (rank == 0) std::cout << "end eos" << std::endl;
	
    using boost::math::pow;
	ZeroAbsorptionZeroDiffusionMultigroup opacity{energy_groups_center, energy_groups_boundary};
	
    if (rank == 0) std::cout << "end sta" << std::endl;

	const double width = 10.0 / lscale;
	size_t const Nx = 2;
	Vector3D ll(0, -0.5 * width / Nx, -0.5 * width / Nx), ur(width, 0.5 * width / Nx, 0.5 * width / Nx);
	Voronoi3D tess(ll, ur);

	int counter = 0;
	
	ComputationalCell3D init_cell_left;

	double const T = kev_kelvin;

	try {
		init_cell_left.density = 1. * lscale * lscale * lscale / mscale;
		init_cell_left.temperature = T;
		init_cell_left.internal_energy = eos.dT2e(init_cell_left.density, init_cell_left.temperature, init_cell_left.tracers, ComputationalCell3D::tracerNames);
		init_cell_left.pressure = eos.de2p(init_cell_left.density, init_cell_left.internal_energy, init_cell_left.tracers, ComputationalCell3D::tracerNames);
		init_cell_left.velocity = Vector3D(0.0, 0.0, 0.0);

		// fill only the energy groups between the thresholds
		for(std::size_t g=thresh_hold_boundary_left; g < thresh_hold_boundary_right; ++g){
            init_cell_left.Eg[g] = planck_integral::planck_energy_density_group_integral(energy_groups_boundary[g], energy_groups_boundary[g+1], T);
			init_cell_left.Eg[g] *= tscale * tscale / (init_cell_left.density * mscale / lscale); 
        }

		init_cell_left.Erad = std::accumulate(init_cell_left.Eg.begin(), init_cell_left.Eg.end(), 0.0);

		std::cout << "Erad=" << init_cell_left.Erad << std::endl;
	}
	catch (UniversalError const &eo)
	{
		reportError(eo);
		throw;
	}

	ComputationalCell3D init_cell_right{init_cell_left};

	// right cell has a velocity defined so there will be a velocity divergence
	init_cell_right.velocity = Vector3D(1e9 * lscale / tscale, 0.0, 0.0);

	vector<Vector3D> points; 
	if(rank == 0) points = CartesianMesh(Nx, 1, 1, ll, ur);

#ifdef RICH_MPI
	tess.BuildParallel(points);
#else
	tess.Build(points);
#endif

	size_t const Nlocal = tess.GetPointNo();
	vector<ComputationalCell3D> cells(Nlocal);
    for (size_t i = 0; i < Nlocal; ++i)
    {
		cells[i] = tess.GetMeshPoint(i).x < 0 ? init_cell_left : init_cell_right;
    }

	Hllc3D rs;
	RigidWallGenerator3D ghost;
	LinearGauss3D interp(eos, ghost, true, 0.2, 0.25, 0.75);

	Lagrangian3D bpm;
	RoundCells3D pm(bpm, eos, 3.75, 0.01, false, 1.25);
	
    double const Tb = kev_kelvin;
	MultigroupDiffusionClosedBoundary D_boundary{};

	constexpr bool flux_limiter = false;
	constexpr bool hydro_on = false;
	constexpr bool compton_on = false;
	constexpr bool doppler_on = true;
	// constexpr bool doppler_on = false;

	MultigroupDiffusion matrix_builder(energy_groups_center, energy_groups_boundary, opacity, D_boundary, eos, std::vector<std::string> (), flux_limiter, hydro_on, compton_on, doppler_on);
	matrix_builder.length_scale_ = lscale;
	matrix_builder.time_scale_ = tscale;
	matrix_builder.mass_scale_ = mscale;
	ZeroForce3D force = ZeroForce3D();

	DefaultCellUpdater cu(false, 0, true, 0.0, &matrix_builder);

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

	double init_dt = 1e-13 / tscale;
	double const tf = 1e-8 / tscale;
	double const dt_output = tf / 10.;
	tsf.SetTimeStep(init_dt);
	double nextT = dt_output;
	double old_dt = init_dt;
	double new_dt = init_dt;
	
	vector<DiagnosticAppendix3D *> appendices;
	WriteSnapshot3D(sim, "snap_" + int2str(counter) + ".h5", appendices, true);
	++counter;
	while (sim.getTime() < tf)
	{
		if (sim.getCycle() % 25 == 0)
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
			++counter;
			nextT = sim.getTime() + dt_output;
		}
		try
		{
			new_dt = sim.RadiationTimeStep(old_dt, matrix_builder, true);

			new_dt=std::min(new_dt,tf/1000.0);			
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
