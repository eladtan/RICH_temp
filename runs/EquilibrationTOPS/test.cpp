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
#include <cstdlib>
#include "boost/math/special_functions/pow.hpp"
#include <string_view>
#include <charconv>
#include <optional>

namespace {

static constexpr double ev = 1.602176634e-12;
static constexpr double kev = 1e3*ev;

static constexpr double ev_kelvin = ev / CG::boltzmann_constant;
static constexpr double kev_kelvin = 1e3*ev_kelvin;

struct Case {
	std::string const description;
	double const T_mat;
	double const T_rad;
	bool   const compton_on;
	bool   const absorption_on;	 
};

Case get_case(std::string_view const case_num_sv){
	int case_num = -1;
	std::from_chars(case_num_sv.data(), case_num_sv.data() + case_num_sv.size(), case_num);

	switch(case_num){
		case 0:
			return {"Winslow (TOPS)", 20.0*kev_kelvin, 1.0*kev_kelvin, true, true};
		case 2:
			return {"Winslow, no compton (TOPS)", 20.0*kev_kelvin, 1.0*kev_kelvin, false, true};
		case 3: 
			return {"Till (TOPS)", 1.0*kev_kelvin, 10.0*kev_kelvin, true, true};
		case 4:
			return {"Till, no compton (TOPS)", 1.0*kev_kelvin, 10.0*kev_kelvin, false, true};
		default:
			std::cout << "Error! Only cases 0, 2, 3, 4 are supported for TOPS." << std::endl;
			exit(1);
	}
}

class TOPSopacity : public MultigroupDiffusionCoefficientCalculator
{
private:
	std::vector<double> rho_, T_;
	std::vector<std::vector<std::vector<double>>> rossland_, planck_, scatter_;
public:
	TOPSopacity(std::string const& file_directory)
	{
		energy_groups_boundary = read_vector(file_directory + "frequency_edges.txt");
		for(double& Egb : energy_groups_boundary)
			Egb *= 11604.5 * CG::boltzmann_constant;
		energy_groups_center.resize(energy_groups_boundary.size() - 1, std::numeric_limits<double>::quiet_NaN());
		for(size_t i = 0; i < energy_groups_boundary.size() - 1; ++i)
			energy_groups_center[i] = std::sqrt(energy_groups_boundary[i] * energy_groups_boundary[i + 1]);
		size_t const Ng = energy_groups_boundary.size() - 1;
		T_ = read_vector(file_directory + "T.txt");
		for(size_t i = 0; i < T_.size(); ++i)
		{
			T_[i] *= 11604.5;
			T_[i] = std::log(T_[i]);
		}
		size_t const Nt = T_.size();
		rho_ = read_vector(file_directory + "rho.txt");
		size_t const Nrho = rho_.size();
		for(size_t i = 0; i < Nrho; ++i)
			rho_[i] = std::log(rho_[i]);
		rossland_.resize(Ng);
		planck_.resize(Ng);
		scatter_.resize(Ng);
		for(size_t i = 0; i < Ng; ++i)
		{
			auto temp_ross = read_vector(file_directory + "sigma_rossland_" + std::to_string(i + 1) + ".txt");
			auto temp_ross_abs = read_vector(file_directory + "sigma_absorption_rossland_" + std::to_string(i + 1) + ".txt");
			auto temp_scattering = read_vector(file_directory + "sigma_scattering_planck_" + std::to_string(i + 1) + ".txt");
			rossland_[i].resize(Nrho);
			planck_[i].resize(Nrho);
			scatter_[i].resize(Nrho);
			for(size_t j = 0; j < Nrho; ++j)
			{
				rossland_[i][j].resize(Nt);
				planck_[i][j].resize(Nt);
				scatter_[i][j].resize(Nt);
				for(size_t k = 0; k < Nt; ++k)
				{
					rossland_[i][j][k] = std::log(std::max(temp_ross[j * Nt + k], 1e-30)) + rho_[j];
					planck_[i][j][k] = std::log(std::max(temp_ross_abs[j * Nt + k], 1e-30)) + rho_[j];
					scatter_[i][j][k] = std::log(std::max(temp_scattering[j * Nt + k], 1e-30)) + rho_[j];
				}
			}
		}
		std::cout << "TOPSopacity: loaded " << Ng << " groups, " << Nt << " temperatures, " << Nrho << " densities" << std::endl;
	}

	double CalcDiffusionCoefficientGroup(ComputationalCell3D const& cell, size_t const group) const override
	{
		double T = std::log(cell.temperature);
		double d = std::log(cell.density);
		if(T < T_[0]) T = T_[0];
		if(T > T_.back()) T = T_.back();
		if(d < rho_[0]) d = rho_[0];
		if(d > rho_.back()) d = rho_.back();
		double const sig = std::exp(BiLinearInterpolation(rho_, T_, rossland_[group], d, T));
		return CG::speed_of_light / (3 * sig);
	}

	double CalcAbsorptionCoefficientGroup(ComputationalCell3D const& cell, size_t group) const override
	{
		double T = std::log(cell.temperature);
		double d = std::log(cell.density);
		if(T < T_[0]) T = T_[0];
		if(T > T_.back()) T = T_.back();
		if(d < rho_[0]) d = rho_[0];
		if(d > rho_.back()) d = rho_.back();
		double const sig = std::exp(BiLinearInterpolation(rho_, T_, planck_[group], d, T));
		return sig;
	}

	double CalcScatteringCoefficientGroup(ComputationalCell3D const& cell, size_t group) const override
	{
		double T = std::log(cell.temperature);
		double d = std::log(cell.density);
		if(T < T_[0]) T = T_[0];
		if(T > T_.back()) T = T_.back();
		if(d < rho_[0]) d = rho_[0];
		if(d > rho_.back()) d = rho_.back();
		double const sig = std::exp(BiLinearInterpolation(rho_, T_, scatter_[group], d, T));
		return sig;
	}
};

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
		std::cout << "Usage: " << argv[0] << " {case_number} [init_dt max_dt growth tf]" << std::endl;
		exit(1);
	}

	auto const current_case = get_case(argv[1]);
	std::cout << "Running case: " << current_case.description << std::endl;
	std::cout << "T_mat = " << current_case.T_mat/kev_kelvin << " KeV, T_rad = " << current_case.T_rad/kev_kelvin << " KeV" << std::endl;

	std::optional<double> force_time_step{};
	std::optional<double> custom_init_dt{};
	std::optional<double> custom_max_dt{};
	double dt_growth_factor = 1.2;
	std::optional<double> custom_tf{};

	if(argc >= 3){
		double dt_arg = -1.0;
		std::string_view dt_sv = argv[2];
		std::from_chars(dt_sv.data(), dt_sv.data() + dt_sv.size(), dt_arg);

		if(argc == 3){
			force_time_step = dt_arg;
			std::cout << "Force Time Step ON = " << *force_time_step << std::endl;
		} else {
			custom_init_dt = dt_arg;
			double max_dt_arg = -1.0;
			std::string_view max_dt_sv = argv[3];
			std::from_chars(max_dt_sv.data(), max_dt_sv.data() + max_dt_sv.size(), max_dt_arg);
			custom_max_dt = max_dt_arg;

			if(argc >= 5){
				std::string_view gf_sv = argv[4];
				std::from_chars(gf_sv.data(), gf_sv.data() + gf_sv.size(), dt_growth_factor);
			}
			if(argc >= 6){
				double tf_arg = -1.0;
				std::string_view tf_sv = argv[5];
				std::from_chars(tf_sv.data(), tf_sv.data() + tf_sv.size(), tf_arg);
				custom_tf = tf_arg;
			}
			std::cout << "Custom adaptive dt: init_dt = " << *custom_init_dt
			          << ", max_dt = " << *custom_max_dt
			          << ", growth = " << dt_growth_factor
			          << (custom_tf ? ", tf = " + std::to_string(*custom_tf) : "")
			          << std::endl;
		}
	}

	// Read opacity tables from TOPS_OPACITY_DIR environment variable
	std::string opacity_dir;
	{
		char const* env = std::getenv("TOPS_OPACITY_DIR");
		if (!env || std::string(env).empty()) {
			std::cerr << "Error: TOPS_OPACITY_DIR environment variable must be set" << std::endl;
			return 1;
		}
		opacity_dir = env;
		if (opacity_dir.back() != '/') opacity_dir += '/';
	}

	TOPSopacity opacity(opacity_dir);

	std::size_t const G = opacity.energy_groups_boundary.size() - 1;
	if(G != ENERGY_GROUPS_NUM){
		std::cerr << "Error: TOPS opacity has " << G << " groups but compiled with ENERGY_GROUPS_NUM=" << ENERGY_GROUPS_NUM << std::endl;
		return 1;
	}

	auto const& energy_groups_center = opacity.energy_groups_center;
	auto const& energy_groups_boundary = opacity.energy_groups_boundary;

	double const lscale = 1.;
	double const mscale = 1.;
	double const tscale = 1.;
	if (rank == 0)
		std::cout << "start eos" << std::endl;

	double constexpr m_p = 1.6726231e-24;
	double constexpr gamma = 5.0/3.0;
	double constexpr N_avogadro = 6.0221408e23;
	double constexpr cv = 2.0 * CG::boltzmann_constant * N_avogadro / (gamma-1.0);

	IdealGas eos(gamma, cv, 1.0, 0.0);

	if (rank == 0)
		std::cout << "end eos" << std::endl;

	const double width = 1 / lscale;
	size_t const Nx = 1;
	Vector3D ll(0, -0.5 * width / Nx, -0.5 * width / Nx), ur(width, 0.5 * width / Nx, 0.5 * width / Nx);
	Voronoi3D tess(ll, ur);

	using boost::math::pow;

	int counter = 0;
	ComputationalCell3D init_cell;

	double const T_mat = current_case.T_mat;
	double const T_rad = current_case.T_rad;

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
			init_cell.Eg[g] = std::max(init_cell.Eg[g], init_cell.Erad*1e-8);
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

	constexpr bool flux_limiter = true;
	constexpr bool hydro_on = false;
	const     bool compton_on = current_case.compton_on;
	constexpr bool doppler_on = false;
	constexpr bool protections_on = false;
	constexpr bool clamp_coupling_strength = false;

	std::string compton_table_dir;
	{
		char const* env = std::getenv("COMPTON_TABLE_DIR");
		if (compton_on) {
			if (!env || std::string(env).empty()) {
				std::cerr << "Error: COMPTON_TABLE_DIR environment variable must be set when compton is on" << std::endl;
				return 1;
			}
			compton_table_dir = env;
		}
	}

	MultigroupDiffusion matrix_builder{
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
		protections_on,
		false,
		compton_table_dir,
		clamp_coupling_strength};

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

	double init_dt = custom_init_dt ? *custom_init_dt : 5e-14 / tscale;
	double const max_dt_cap = custom_max_dt ? *custom_max_dt : 5e-12;
	double const tf = custom_tf ? *custom_tf : 3e-8 / tscale;
	double const dt_output = tf / 100.;
	tsf.SetTimeStep(init_dt);
	double nextT = dt_output;
	double old_dt = init_dt;
	vector<DiagnosticAppendix3D *> appendices;

	WriteSnapshot3D(sim, "snap_" + int2str(counter) + ".h5", appendices, true);
	++counter;

	double new_dt = force_time_step ? *force_time_step : init_dt;
	bool dt_still_growing = true;

	while (sim.getTime() < tf)
	{
		if (rank == 0)
		{
			std::cout<<std::endl;
			std::cout << "Cycle " << sim.getCycle() << " Time " << sim.getTime() << " dt " << new_dt << std::endl;
		}

		int output_interval = dt_still_growing ? 1 : 10;
		if (sim.getTime() > nextT or sim.getCycle() % output_interval == 0 or sim.getCycle() < 10)
		{
			WriteSnapshot3D(sim, "snap_" + int2str(counter) + ".h5", appendices, true);
			nextT = sim.getTime() + dt_output;
			++counter;
		}
		try
		{
			new_dt = sim.RadiationTimeStep(old_dt, matrix_builder, true);

			if (force_time_step) {
				new_dt = force_time_step.value();
			} else {
				new_dt = std::min(old_dt * dt_growth_factor, max_dt_cap);
				if (new_dt >= max_dt_cap) dt_still_growing = false;
			}

			if (rank == 0) std::cout<<"New time step is "<<new_dt<<std::endl;

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
