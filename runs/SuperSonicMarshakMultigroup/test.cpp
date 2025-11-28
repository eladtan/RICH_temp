#include "source/3D/tesselation/voronoi/Voronoi3D.hpp"
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
#include "source/Radiation/GrayDiffusion/Diffusion.hpp"
#include "source/Radiation/GrayDiffusion/DiffusionForce.hpp"
#include "source/Radiation/MultigroupDiffusion/MultigroupDiffusion.hpp"
#include "source/Radiation/MultigroupDiffusion/MultigroupDiffusionCoefficientCalculator.hpp"
#include "source/Radiation/MultigroupDiffusion/MultigroupDiffusionBoundaryCalculator.hpp"
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
#include <functional>
#include <charconv>
#include <string_view>


typedef std::array<double, 4> state_type;

const double ev = 1.602176634e-12;
const double kev = 1e3*ev;

const double ev_kelvin = ev / CG::boltzmann_constant;
const double kev_kelvin = 1e3*ev_kelvin;
class OpacityCalc : public MultigroupDiffusionCoefficientCalculator
{
	public:
		OpacityCalc(std::vector<double> const& energy_groups_center, 
                    std::vector<double> const& energy_groups_boundary,
                    double const alpha_,
                    double const lambda_,
                    double const coeff_total_,
                    double const coeff_absorption_) 
                    :   MultigroupDiffusionCoefficientCalculator(energy_groups_center, energy_groups_boundary),
                        alpha(alpha_),
                        lambda(lambda_),
                        coeff_total(coeff_total_),
                        coeff_absorption(coeff_absorption_) {}

	double CalcDiffusionCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const
	{
		return CG::speed_of_light / (3 * coeff_total * std::pow(kev_kelvin / cell.temperature, alpha) * std::pow(cell.density, 1.0+lambda));
		// return CG::speed_of_light / (3 * 2 * std::pow(kev_kelvin / cell.temperature, 4.5) * std::pow(cell.density, 1.9));
		// return CG::speed_of_light / (3 * 100 * std::pow(kev_kelvin / cell.temperature, 3.0));
	}

    double CalcAbsorptionCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const
	{
		double result = coeff_absorption * std::pow(kev_kelvin / cell.temperature, alpha) * std::pow(cell.density, 1.0 + lambda);
		// double result = 0.1 * std::pow(kev_kelvin / cell.temperature, 3.0);
		return result;
	}

    double CalcScatteringCoefficientGroup(ComputationalCell3D const& cell, std::size_t const group) const
	{
		return 0;
	}

    double const alpha;
    double const lambda;

    double const coeff_total;
    double const coeff_absorption;
};

struct Case {
	std::string const description;
	double const alpha;
	double const lambda;
	double const coeff_total;
	double const coeff_absorption;
	double const omega;
	double const beta; 
	double const mu;
	double const f_eos;
	double const t_final;
	bool const cartesian_mesh;
	double const T_initial;
	std::function<double(double)> Tbath;
};

Case get_case(std::string_view const case_num_sv){
	int case_num = -1;
	std::from_chars(case_num_sv.data(), case_num_sv.data() + case_num_sv.size(), case_num);

	switch(case_num){
		// from Menahem's and Nitai's Paper
		case 1:
			return {"Test 1, 2nd Paper", 1.5, 0.2, 40.0, 0.1, -20.0/19.0, 3.4, 0.14, 1e14, 1e-9, true, 0.0005 * kev_kelvin, [](double const time){ return 1.0470478 * std::pow(time/1e-9, 86.0/57.0) * kev_kelvin;} };
		case 3:
			return {"Test 3, 2nd Paper", 4.5, 0.9, 2.0, 1e-3, 40.0/139.0, 6.0, 0.3, 1e14, 1e-9, false, 0.005 * kev_kelvin, [](double const time){ return 1.01008116 * std::pow(time/1e-9, 14.0/139.0) * kev_kelvin;} };
		// From Menahems second paper
		case 22:
			return {"Test 2, 1st Paper", 3.0, -1.0, 100.0, 0.1, 0.0, 4.0, 1.0, 1.372017e14/0.2, 1e-9, true, 0.005 * kev_kelvin, [](double const time){ return 1.008038 * std::pow(time/1e-9, 1.0/3.0) * kev_kelvin;} };
		case 33:
			return {"Test 3, 1st Paper", 3.0, -1.0, 100.0, 100.0, 0.0, 4.0, 1.0, 1.372017e14/0.2, 1e-9, true, 0.005 * kev_kelvin, [](double const time){ return 1.014565 * std::pow(time/1e-9, 1.0/3.0) * kev_kelvin;} };
		
		
		default:
			std::cout << "Error! No Such case as: " << case_num_sv << std::endl;
			std::cout << "Available cases: 1, 3, 22, 33" << std::endl;
			exit(1);
	}
}


int main(int argc, char *argv[])
{
	feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
	int rank = 0;
	int ws = 1;
#ifdef RICH_MPI
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ws);
#endif

	if(argc < 2){
		std::cout << "Not Enough arguments need to give the {case_number}" << std::endl;
		exit(1);
	}
	
	auto const current_case = get_case(argv[1]);

	std::vector<double> energy_groups_boundary = {
		1e-8,
		1e-4,
		std::pow(10.0, -3.5),
		std::pow(10.0, -3.),
		std::pow(10.0, -2.5),
		std::pow(10.0, -2.),
		std::pow(10.0, -1.5),
		std::pow(10.0, -1.),
		std::pow(10.0, -0.5),
		std::pow(10.0, -0.),
		std::pow(10.0, 0.5),
		std::pow(10.0, 1.0),
		std::pow(10.0, 1.5)
    };

	for(size_t i = 0; i < energy_groups_boundary.size(); ++i)
		energy_groups_boundary[i] *= kev;

	std::vector<double> energy_groups_center(energy_groups_boundary.size() - 1);

	for(std::size_t g=0; g < (energy_groups_boundary.size() - 1); ++g)
		energy_groups_center[g] =std::sqrt(energy_groups_boundary[g+1] * energy_groups_boundary[g]);
	
	if(energy_groups_center.size() != ENERGY_GROUPS_NUM){
		std::cout << "Error: energy_groups_center size does not match ENERGY_GROUPS_NUM" << std::endl;
		return 1;
	}

	if(energy_groups_boundary.size() != ENERGY_GROUPS_NUM+1){
		std::cout << "Error: energy_groups_boundries size does not match ENERGY_GROUPS_NUM+1" << std::endl;
		return 1;
	}

	double const lscale = 1;
	double const mscale = 1;
	double const tscale = 1;
	if (rank == 0)
		std::cout << "start eos" << std::endl;

	// double constexpr m_p = 1.6726231e-24;
	// double constexpr cv = CG::radiation_constant / 0.2;

	double beta = current_case.beta;
	double const cv = current_case.f_eos / std::pow(kev_kelvin, beta);
	double mu = current_case.mu;
    
	IdealGas eos(/*gamma=*/1.4, /*f=*/cv, /*beta=*/beta, /*mu=*/mu);

	if (rank == 0)
		std::cout << "end eos" << std::endl;
	
    if (rank == 0)
		std::cout << "end sta" << std::endl;

	size_t const Nx = 512;
	double width = 1;
	Vector3D ll(0, 0, 0), ur(width, width / Nx, width / Nx);
	Voronoi3D tess(ll, ur);

    using boost::math::pow;

    double const alpha = current_case.alpha;
    double const lambda = current_case.lambda;
    double const coeff_tot = current_case.coeff_total;
    double const coeff_abs = current_case.coeff_absorption;

	OpacityCalc opacity(energy_groups_center, 
                        energy_groups_boundary, 
                        alpha,
                        lambda,
                        coeff_tot,
                        coeff_abs);

	int counter = 0;
	ComputationalCell3D init_cell;

	double const T_mat = current_case.T_initial;
	double const T_rad = current_case.T_initial;

	try
	{
		init_cell.density = 1 * lscale * lscale * lscale / mscale;
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
	{
		if(current_case.cartesian_mesh){
			points = CartesianMesh(Nx, 1, 1, ll, ur);
		} else {
			for(size_t i = 0; i < Nx; ++i)
				points.push_back(Vector3D(0.5e-5 + 2.24e-2 * (std::pow(1.0075, 1.0 * i) - 1), 0.5 * width / Nx, 0.5 * width / Nx));
		}
	}
#ifdef RICH_MPI
	tess.BuildParallel(points);
#else
	tess.Build(points);
#endif
	vector<ComputationalCell3D> cells(tess.GetPointNo(), init_cell);
    
    double const omega = current_case.omega;
	for(size_t i = 0; i < tess.GetPointNo(); ++i)
	{
		cells[i].density = std::pow(tess.GetCellCM(i).x, -omega);
		cells[i].internal_energy = eos.dT2e(cells[i].density, init_cell.temperature, init_cell.tracers, ComputationalCell3D::tracerNames);
		cells[i].pressure = eos.de2p(cells[i].density, cells[i].internal_energy, init_cell.tracers, ComputationalCell3D::tracerNames);
		cells[i].Erad = CG::radiation_constant * pow<4>(T_rad) * tscale * tscale / (cells[i].density * mscale / lscale);
        for(std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g){
            cells[i].Eg[g] = planck_integral::planck_energy_density_group_integral(energy_groups_boundary[g], energy_groups_boundary[g+1], T_rad);
			cells[i].Eg[g] *= tscale * tscale / (cells[i].density * mscale / lscale); 
			cells[i].Eg[g]  = std::max(cells[i].Eg[g], cells[i].Erad*1e-8);
        }
	}

	Hllc3D rs;
	RigidWallGenerator3D ghost;
	LinearGauss3D interp(eos, ghost, true, 0.2, 0.25, 0.75);

	Lagrangian3D bpm;
	RoundCells3D pm(bpm, eos, 3.75, 0.01, false, 1.25);
	
	bool const flux_limiter   = false;
	bool const hydro_on       = false;
	bool const compton_on     = false;
	bool const doppler_on     = false;
    bool const protections_on = false;

    double const side_temperature_kev_initial = 1e-3;
	MultigroupDiffusionSideBoundary D_boundary(side_temperature_kev_initial * kev_kelvin, energy_groups_center, energy_groups_boundary);
	
    MultigroupDiffusion matrix_builder(
        energy_groups_center, 
        energy_groups_boundary, 
        opacity, 
        D_boundary, 
        eos, 
        std::vector<std::string> (), 
        flux_limiter, 
        hydro_on, 
        compton_on, 
        doppler_on, 
        -1.0, 
        false
    );
	
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

	double init_dt = 1e-20 / tscale;
	double const tf = 1e-9 / tscale;
	double const dt_output = 0.2*tf;
	tsf.SetTimeStep(init_dt);
	double nextT = dt_output;
	double old_dt = init_dt;
	vector<DiagnosticAppendix3D *> appendices;
	WriteSnapshot3D(sim, "init.h5", appendices, true);
	double new_dt = init_dt;
	std::vector<double> Tgas, Trad, time;
	Tgas.push_back(init_cell.temperature);
	Trad.push_back(std::pow(init_cell.Erad / CG::radiation_constant, 0.25));
	time.push_back(0.0);

	double const Einit = sim.getExtensives()[0].Erad + sim.getExtensives()[0].internal_energy;

	while (sim.getTime() < tf)
	{
		if (sim.getCycle() % 1 == 0)
		{
			if (rank == 0)
			{
				std::cout<<std::endl;
				std::cout << "Cycle " << sim.getCycle() << " Time " << sim.getTime() << " dt " << new_dt << std::endl;
				double const Energy = sim.getExtensives()[0].Erad + sim.getExtensives()[0].internal_energy;
				std::cout<<"Energy "<<Energy<<" dE "<<Energy-Einit<<std::endl;
			}
		}
		if (sim.getTime() > nextT || sim.getCycle() % 1000 == 0)
		{
			WriteSnapshot3D(sim, "snap_" + int2str(counter) + ".h5", appendices, true);
			nextT = sim.getTime() + dt_output;
			++counter;
		}
		try
		{
			// old_dt = 1e-1;
			new_dt = sim.RadiationTimeStep(old_dt, matrix_builder, true);
			// if(sim.getTime() < 1e-11)
			new_dt = std::min(std::max(1e-15, sim.getTime() * 2e-3), std::min(new_dt, 1e-12));
			if(sim.getTime() < 3e-8)
			{
				double const new_T = current_case.Tbath(sim.getTime());

				if(rank == 0) std::cout<<"New temperature is "<< new_T/kev_kelvin<< " KeV" << std::endl;

				D_boundary.SetTemperature(new_T);
			}
			Tgas.push_back(sim.getCells()[0].temperature);
			Trad.push_back(std::pow(sim.getCells()[0].Erad * sim.getCells()[0].density / CG::radiation_constant, 0.25));
			time.push_back(sim.getTime());
			// tsf.SetTimeStep(new_dt);
			// sim.SetTimeStep(new_dt);
			// new_dt=std::min(new_dt, 5e-9);
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
	write_vector(time, "time.txt");
	write_vector(Tgas, "Tgas.txt");
	write_vector(Trad, "Trad.txt");
	WriteSnapshot3D(sim, "snap_" + int2str(counter) + ".h5", appendices, true);
	++counter;
	std::cout<<"Done"<<std::endl;
#ifdef RICH_MPI
	MPI_Finalize();
#endif
	return 0;
}
