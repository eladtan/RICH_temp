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
#include "source/Radiation/OpacityCalculator.hpp"
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

namespace
{
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
			for (size_t i = 0; i < energy_groups_boundary.size() - 1; ++i)
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
}

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
	double const lscale = 7e10;
	double const mscale = 2e33;
	double const tscale = 1603;
    std::string eos_location("/home/elads/RICH_itamar/data/EOS/");
	std::string STA_location("/home/elads/RICH_itamar/data/STA/MG/");
	OndrejEOS eos(eos_location + "density.txt", eos_location + "Pfile.txt", eos_location + "csfile.txt", eos_location + "Sfile.txt", eos_location + "Ufile.txt", eos_location + "Tfile.txt", eos_location + "CVfile.txt", lscale, mscale, tscale);
	if (rank == 0)
		std::cout << "end eos" << std::endl;
	//Radiation
	STAMGopacity opacity(STA_location);

	ComputationalCell3D c_opac;
	c_opac.density = 0.1;
	c_opac.temperature = 10000;
	double d_temp = opacity.CalcAbsorptionOpacity(c_opac, opacity.energy_groups_center[1]);
	c_opac.temperature = 1e7;
	d_temp = opacity.CalcAbsorptionOpacity(c_opac, opacity.energy_groups_center[1]);
	c_opac.density = 1e-7;
	d_temp = opacity.CalcAbsorptionOpacity(c_opac, opacity.energy_groups_center[1]);
	if (rank == 0)
		std::cout << "end sta" << std::endl;

	const double width = 5;
	Vector3D ll(-width, -width, -width), ur(width, width, width);
	Voronoi3D tess(ll, ur);

	vector<ComputationalCell3D> cells;
	std::vector<Vector3D> ptemp;
    int np = 1e3;
    if(rank == 0)
    {
        ptemp = RandSphereR1(np, ll, ur, 0, 1.1, Vector3D());
        vector<Vector3D> ptemp2 = RandSphereR(np / 2, ll, ur, 0.8 , 1.05, Vector3D());
        vector<Vector3D> ptemp3 = RandSphereR2(np / 4, ll, ur, 1, 1.4 * width, Vector3D());
        ptemp.insert(ptemp.end(), ptemp2.begin(), ptemp2.end());
        ptemp.insert(ptemp.end(), ptemp3.begin(), ptemp3.end());
    }
#ifdef RICH_MPI
    ptemp = MPI_Spread(ptemp, 0, MPI_COMM_WORLD);
#endif
    try
    {
        vector<Vector3D> points = RoundGrid3D(ptemp, ll, ur, 15);
#ifdef RICH_MPI
        tess.BuildParallel(points);
#else
        tess.Build(points);
#endif
        size_t const N = tess.GetPointNo();
        cells.resize(N);
        for(size_t i = 0; i < N; ++i)
        {
            double const r = abs(tess.GetMeshPoint(i));
            cells[i].density = 1e-9 / (r * r + 0.01);
            cells[i].temperature = r > 0.2 ? 1e4 : 1e6;
            cells[i].internal_energy = eos.dT2e(cells[i].density, cells[i].temperature);
            cells[i].pressure = eos.dT2p(cells[i].density, cells[i].temperature);
            double const T = cells[i].temperature;
            cells[i].Erad = 7.5657e-15 * T * T * T * T * tscale * tscale * lscale / (mscale * cells[i].density);
            size_t const Ng = ENERGY_GROUPS_NUM;
            for(size_t g = 0; g < Ng; ++g)
                cells[i].Eg[g] = std::max(cells[i].Erad * 1e-12, planck_integral::planck_energy_density_group_integral(opacity.energy_groups_boundary[g], opacity.energy_groups_boundary[g+1], T) * tscale * tscale * lscale / (mscale * cells[i].density));
        }       
    }
    catch (UniversalError const &eo)
    {
        reportError(eo);
        throw;
    }
	if (rank == 0)
		std::cout << "Finished build" << std::endl;
	std::cout<<"Rank "<<rank<<" has "<<tess.GetPointNo()<<" points "<<" and "<<cells.size()<<" cells "<<std::endl;

	Hllc3D rs;
	RigidWallGenerator3D ghost;
	LinearGauss3D interp(eos, ghost, true, 0.2, 0.25, 0.75);

	Lagrangian3D bpm;
	RoundCells3D pm(bpm, eos, 1.75, 0.005, false, 1.25);

	MultigroupDiffusionClosedBoundary D_boundary;
	MultigroupDiffusion matrix_builder(opacity.energy_groups_center, opacity.energy_groups_boundary, opacity, D_boundary, eos, std::vector<std::string>(), true, true/*false*/, false, true, false);
	matrix_builder.length_scale_ = lscale;
	matrix_builder.time_scale_ = tscale;
	matrix_builder.mass_scale_ = mscale;

	std::shared_ptr<MultigroupDiffusionForce> rad_force = std::make_shared<MultigroupDiffusionForce>(matrix_builder, eos);
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
	std::vector<std::shared_ptr<SourceTerm3D>> forces;

	forces.push_back(rad_force);
	SeveralSources3D force(forces);
	CourantFriedrichsLewy tsf(0.3, 1, force, std::vector<std::string> (),	false);

	std::unique_ptr<HDSim3D> sim;
	sim = std::make_unique<HDSim3D>(tess, cells, eos, pm, tsf, fc, cu, eu, force, std::pair<std::vector<std::string>, std::vector<std::string>> (ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames), false, true);
	double init_dt = 1e-6;
    double output_dt = 1e-3;
    double nextT = output_dt;
	tsf.SetTimeStep(init_dt);
    double old_dt = init_dt;
    double old_t = 0;
	double step_time = 0;
    int counter = 0;
	while (sim->getTime() < output_dt * 100)
	{
		if (sim->getCycle() % 1 == 0)
		{
			int ntotal = tess.GetPointNo();
#ifdef RICH_MPI
			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Allreduce(MPI_IN_PLACE, &ntotal, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
#endif
			if (rank == 0)
			{
				std::cout<<std::endl;
				std::cout << "Point num " << ntotal << " dt " << old_dt << " run time " << step_time << std::endl;
				std::cout << "Cycle " << sim->getCycle() << " Time " << sim->getTime() <<" next time output "<<nextT<<std::endl;
			}
		}
		if (sim->getTime() > nextT)
		{
			WriteSnapshot3D(*sim, "snap_" + int2str(counter) + ".h5");
			nextT = sim->getTime() + output_dt;
			++counter;
		}
		try
		{
#ifdef RICH_MPI
            double step_tstart = MPI_Wtime();
#endif
			double new_dt = sim->RadiationTimeStep(old_dt, matrix_builder, /*true*/ false);
			tsf.SetTimeStep(new_dt);
			if (rank == 0)
				std::cout << "Finished rad step" << std::endl;
			sim->timeAdvance2();
			if (rank == 0)
				std::cout << "Finished hydro step" << std::endl;
			old_dt = sim->getTime() - old_t;

			// old_dt = new_dt;

			old_t = sim->getTime();
#ifdef RICH_MPI
			step_time = MPI_Wtime() - step_tstart;
#endif
		}
		catch (UniversalError const &eo)
		{
			reportError(eo);
			throw;
		}
	}
#ifdef RICH_MPI
    if(rank == 0)
	   std::cout<<"Done sim"<<std::endl;
	MPI_Finalize();
#endif
	return 0;
}
