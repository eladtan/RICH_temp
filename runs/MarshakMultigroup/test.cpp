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
#include "source/Radiation/MultigroupDiffusion.hpp"
#include "source/Radiation/OpacityCalculator.hpp"
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


static constexpr double ev = 1.602176634e-12;
static constexpr double kev = 1e3*ev;

class STAMGopacity: public OpacityCalculator
	{
	private:
		std::vector<double> rho_, T_;
		std::vector<std::vector<std::vector<double>>> rossland_, planck_, scatter_;
	public:
		STAMGopacity(std::string file_directory) 
		{
			energy_groups_boundary = read_vector(file_directory + "frequency_edges.txt");   
			for(double& Egb : energy_groups_boundary)
			    Egb *= 11604.5 * CG::boltzmann_constant;
			energy_groups_center.resize(energy_groups_boundary.size() - 1, std::numeric_limits<double>::quiet_NaN());
			for(size_t i = 0; i < energy_groups_boundary.size() - 1; ++i)
				energy_groups_center[i] = std::sqrt(energy_groups_boundary[i] * energy_groups_boundary[i + 1]);
			size_t const Ng = energy_groups_boundary.size() - 1;
			T_ = read_vector(file_directory +"T.txt");
			// Convert from ev to kelvin
			for(size_t i = 0; i < T_.size(); ++i)
			{
				T_[i] *= 11604.5;
				T_[i] = std::log(T_[i]);
			}
			size_t const Nt = T_.size();
			rho_ = read_vector(file_directory +"rho.txt");
			size_t const Nrho = rho_.size();
			for(size_t i = 0; i < Nrho; ++i)
				rho_[i] = std::log(rho_[i]);
			rossland_.resize(Ng);
			planck_.resize(Ng);
			scatter_.resize(Ng);
			for(size_t i = 0; i < Ng; ++i)
			{
				auto temp_ross = read_vector(file_directory +"sigma_rossland_" + std::to_string(i + 1) + ".txt");
				auto temp_ross_abs = read_vector(file_directory +"sigma_absorption_rossland_" + std::to_string(i + 1) + ".txt");
				auto temp_scattering = read_vector(file_directory +"sigma_scattering_planck_" + std::to_string(i + 1) + ".txt");
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
						rossland_[i][j][k] = std::log(temp_ross[j * Nt + k]) + rho_[j];
						planck_[i][j][k] = std::log(temp_ross_abs[j * Nt + k]) + rho_[j];
						scatter_[i][j][k] = std::log(temp_scattering[j * Nt + k]) + rho_[j];
					}
				}
			}
		}

		double CalcDiffusionCoefficient(ComputationalCell3D const& cell, double energy) const override
		{
			std::size_t const group = findGroup(energy);
			double T = std::log(cell.temperature);
			double d = std::log(cell.density);
			double d_ratio = 1;
			if(d < rho_[0])
			{
				d_ratio = rho_[0] / d;
				d = rho_[0];
			}
			if(d > rho_.back())
			{
				d_ratio = rho_.back() / d;
				d = rho_.back();
			}
			if(T < T_[0])
				T = T_[0];
			if(T > T_.back())
			    T = T_.back();
			double const sig = std::exp(BiLinearInterpolation(rho_, T_, rossland_[group], d, T)) * d_ratio;
			return CG::speed_of_light / (3 * sig);
		}

		double CalcAbsorptionOpacity(ComputationalCell3D const& cell, double energy) const override
		{
			std::size_t const group = findGroup(energy);
			double T = std::log(cell.temperature);
			double d = std::log(cell.density);
			double d_ratio = 1, T_ratio = 1;
			if(d < rho_[0])
			{
				d_ratio = rho_[0] / d;
				d = rho_[0];
			}
			if(d > rho_.back())
			{
				d_ratio = rho_.back() / d;
				d = rho_.back();
			}
			if(T < T_[0])
				T = T_[0];
			if(T > T_.back())
			{
				T_ratio = std::exp(-1.5 * (T - T_.back()));
			    T = T_.back();
			}
			double const sig = std::exp(BiLinearInterpolation(rho_, T_, planck_[group], d, T)) * d_ratio * T_ratio;
			return sig;
		}

		double CalcScatteringOpacity(ComputationalCell3D const& cell, double energy) const override
		{
			std::size_t const group = findGroup(energy);
			double T = std::log(cell.temperature);
			double d = std::log(cell.density);
			double d_ratio = 1;
			if(d < rho_[0])
			{
				d_ratio = rho_[0] / d;
				d = rho_[0];
			}
			if(d > rho_.back())
			{
				d_ratio = rho_.back() / d;
				d = rho_.back();
			}
			if(T < T_[0])
				T = T_[0];
			if(T > T_.back())
			    T = T_.back();
			double const sig = std::exp(BiLinearInterpolation(rho_, T_, scatter_[group], d, T)) * d_ratio;
			return sig;
		}
	};


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
	double const Emax = kev*1e3;
	
	energy_groups_boundary[0] = Emin;
	for(std::size_t g=0; g < G; ++g){
		energy_groups_boundary[g+1] = std::pow(Emax/Emin, 1.0/G)*energy_groups_boundary[g];
		energy_groups_center[g] = 0.5*(energy_groups_boundary[g+1]+energy_groups_boundary[g]);
	}

	// std::vector<double> energy_groups_center =    {  0.1, 0.4, 1.0, 3.0, 10., 100., 500.};
    // std::vector<double> energy_groups_boundary = {1e-7, 0.2, 0.6, 1.4, 4.6, 15.4, 200., 1000.};

    
    // for(auto& val : energy_groups_center) val *= kev;
    // for(auto& val : energy_groups_boundary) val *= kev;

	std::string eos_location("/home/itamarg/workspace/RICH/data/EOS/");
	std::string STA_location("/home/itamarg/workspace/RICH/data/STA/");

	double const lscale = 7e10;
	double const mscale = 2e33;
	double const tscale = 1603;
	if (rank == 0)
		std::cout << "start eos" << std::endl;
	OndrejEOS eos(eos_location + "density.txt", eos_location + "Pfile.txt", eos_location + "csfile.txt", eos_location + "Sfile.txt", eos_location + "Ufile.txt", eos_location + "Tfile.txt", eos_location + "CVfile.txt", lscale, mscale, tscale);
	if (rank == 0)
		std::cout << "end eos" << std::endl;
	//Radiation
	// GrayPowerLawOpacity opacity(CG::speed_of_light / (3 * 0.848902), 0, 0, 3.93e19, 0, -3);
	STAMGopacity opacity("/home/elads/RICH_itamar/data/STA/MG/");
	
    if (rank == 0)
		std::cout << "end sta" << std::endl;

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
		init_cell.density = 1e-8 * 0.5 * lscale * lscale * lscale / mscale;
		init_cell.temperature = T;
		init_cell.pressure = eos.dT2p(init_cell.density, init_cell.temperature);
		init_cell.internal_energy = eos.dp2e(init_cell.density, init_cell.pressure);
		init_cell.Erad = CG::radiation_constant * T * T * T * T * tscale * tscale / (init_cell.density * mscale / lscale);
        for(std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g){
            init_cell.Eg[g] = planck_integral::planck_energy_density_group_integral(opacity.energy_groups_boundary[g], opacity.energy_groups_boundary[g+1], T);
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
	
	MultigroupDiffusionSideBoundary D_boundary(1.1605e7, energy_groups_center, energy_groups_boundary);
	MultigroupDiffusion matrix_builder(energy_groups_center, energy_groups_boundary, opacity, D_boundary, eos, std::vector<std::string> (), false, false, false, false);
	matrix_builder.length_scale_ = lscale;
	matrix_builder.time_scale_ = tscale;
	matrix_builder.mass_scale_ = mscale;
	ZeroForce3D force = ZeroForce3D();

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

	double init_dt = 1e8 * 1e-13 / tscale;
	double const dt_output = 1e-9 / tscale;
	double const tf = 1e8 * 1e-8 / tscale;
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
