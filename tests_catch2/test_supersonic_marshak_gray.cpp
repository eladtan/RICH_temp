#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
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
#include "source/Radiation/GrayDiffusion/DiffusionForce.hpp"
#include "source/Radiation/GrayDiffusion/Diffusion.hpp"
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

#include <catch2/catch_test_macros.hpp>
#include "utils_for_tests.hpp"
#include "snapshot.hpp"

using utils_for_tests::named_vector;
using utils_for_tests::extract_data_from_cells;
using utils_for_tests::make_named_vector;
namespace mpi = utils_for_tests::mpi;

namespace {

typedef std::array<double, 4> state_type;

const double ev = 1.602176634e-12;
const double kev = 1e3*ev;

const double ev_kelvin = ev / CG::boltzmann_constant;
const double kev_kelvin = 1e3*ev_kelvin;

class OpacityCalc : public DiffusionCoefficientCalculator
{
	public:
		OpacityCalc(double const alpha_,
                    double const lambda_,
                    double const coeff_total_,
                    double const coeff_absorption_) 
                    :   DiffusionCoefficientCalculator{},
                        alpha(alpha_),
                        lambda(lambda_),
                        coeff_total(coeff_total_),
                        coeff_absorption(coeff_absorption_) {}

	double CalcDiffusionCoefficient(ComputationalCell3D const& cell) const
	{
		return CG::speed_of_light / (3 * coeff_total * std::pow(kev_kelvin / cell.temperature, alpha) * std::pow(cell.density, 1.0+lambda));
		// return CG::speed_of_light / (3 * 2 * std::pow(kev_kelvin / cell.temperature, 4.5) * std::pow(cell.density, 1.9));
		// return CG::speed_of_light / (3 * 100 * std::pow(kev_kelvin / cell.temperature, 3.0));
	}

    double CalcPlanckOpacity(ComputationalCell3D const& cell) const
	{
		double result = coeff_absorption * std::pow(kev_kelvin / cell.temperature, alpha) * std::pow(cell.density, 1.0 + lambda);
		// double result = 0.1 * std::pow(kev_kelvin / cell.temperature, 3.0);
		return result;
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

auto supersonic_marshak(std::string case_num_str){

	int rank = mpi::get_mpi_rank();
	int ws = mpi::get_mpi_world_size();
	
	auto const current_case = get_case(case_num_str);

	double const lscale = 1;
	double const mscale = 1;
	double const tscale = 1;
	if (rank == 0)
		std::cout << "start eos" << std::endl;

	double beta = current_case.beta;
	double const cv = current_case.f_eos / std::pow(kev_kelvin, beta);
	double mu = current_case.mu;
    
	IdealGas eos(/*gamma=*/1.4, /*f=*/cv, /*beta=*/beta, /*mu=*/mu);

	if (rank == 0)
		std::cout << "end eos" << std::endl;
	
    if (rank == 0)
		std::cout << "end sta" << std::endl;

	// the log grid is taylored to 512 cells
	size_t const Nx = current_case.cartesian_mesh ? 50 : 512;
	double width = 1;
	Vector3D ll(0, 0, 0), ur(width, width / Nx, width / Nx);
	Voronoi3D tess(ll, ur);

    using boost::math::pow;

    double const alpha = current_case.alpha;
    double const lambda = current_case.lambda;
    double const coeff_tot = current_case.coeff_total;
    double const coeff_abs = current_case.coeff_absorption;

	OpacityCalc opacity(alpha,
                        lambda,
                        coeff_tot,
                        coeff_abs);

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
	DiffusionSideBoundary D_boundary(side_temperature_kev_initial * kev_kelvin);
	
    Diffusion matrix_builder(
        opacity, 
        D_boundary, 
        eos, 
        std::vector<std::string> (), 
        flux_limiter, 
        hydro_on, 
        compton_on 
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
	double const tf = 0.5e-9 / tscale;
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

		try
		{
			new_dt = sim.RadiationTimeStep(old_dt, matrix_builder, true);
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

	std::vector<ComputationalCell3D> cells_end_sim = sim.getCells();
	cells_end_sim.resize(sim.getTesselation().GetPointNo());
	auto time_output = make_named_vector<double>("time", {}); 

	if(mpi::get_mpi_rank() == mpi::rank_root) time_output.vec.push_back(sim.getTime());

	return std::make_tuple(
		extract_data_from_cells("ID", cells_end_sim, &ComputationalCell3D::ID), 
		extract_data_from_cells("temperature", cells_end_sim, &ComputationalCell3D::temperature),
		extract_data_from_cells("density", cells_end_sim, &ComputationalCell3D::density),
		extract_data_from_cells("Erad", cells_end_sim, &ComputationalCell3D::Erad),
		time_output
	);

}

} // namespace

TEST_CASE_METHOD(mpi::RichMpiFixture,"Supersonic_marshak_Gray_Diffusion_Test_1_2nd_Paper", "[supersonic_marshak][gray_diffusion]"){
	snapshot::SnapShot snap;

	auto const success = std::apply(
		[&snap](auto const& ID, auto const&... run_info){
			return snap.CompareOrSaveGather(
				std::nullopt,
				ID,
				run_info...
			);
		},
		supersonic_marshak("1")
	);

	REQUIRE(success);
}

TEST_CASE_METHOD(mpi::RichMpiFixture,"Supersonic_marshak_Gray_Diffusion_Test_3_2nd_Paper", "[supersonic_marshak][gray_diffusion]"){
	snapshot::SnapShot snap;

	auto const success = std::apply(
		[&snap](auto const& ID, auto const&... run_info){
			return snap.CompareOrSaveGather(
				std::nullopt,
				ID,
				run_info...
			);
		},
		supersonic_marshak("3")
	);

	REQUIRE(success);
}

TEST_CASE_METHOD(mpi::RichMpiFixture,"Supersonic_marshak_Gray_Diffusion_Test_2_1st_Paper", "[supersonic_marshak][gray_diffusion]"){
	snapshot::SnapShot snap;

	auto const success = std::apply(
		[&snap](auto const& ID, auto const&... run_info){
			return snap.CompareOrSaveGather(
				std::nullopt,
				ID,
				run_info...
			);
		},
		supersonic_marshak("22")
	);

	REQUIRE(success);
}

TEST_CASE_METHOD(mpi::RichMpiFixture,"Supersonic_marshak_Gray_Diffusion_Test_3_1st_Paper", "[supersonic_marshak][gray_diffusion]"){
	snapshot::SnapShot snap;

	auto const success = std::apply(
		[&snap](auto const& ID, auto const&... run_info){
			return snap.CompareOrSaveGather(
				std::nullopt,
				ID,
				run_info...
			);
		},
		supersonic_marshak("33")
	);

	REQUIRE(success);
}