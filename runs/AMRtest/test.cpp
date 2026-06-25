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
#include "source/3D/output/hdf_write.hpp"
#include "source/newtonian/three_dimensional/AMR3D.hpp"
#include "source/newtonian/three_dimensional/GravityAcc3D.hpp"
#include "source/Radiation/Diffusion.hpp"
#include "source/Radiation/DiffusionForce.hpp"
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

namespace
{
    double InterpolateTable(double const T, double const d, std::vector<double> const& T_, std::vector<double> const& rho_, std::vector<std::vector<double>> const& data,
		double const T_high_slope = 0)
	{
		size_t const slope_length = 7;
		if(T < T_[0])
		{
			if(d < rho_[0])
			{
				double const d_slope = (data[0][slope_length - 1] -data[0][0]) / (rho_[slope_length - 1] - rho_[0]);
				double const T_slope = (data[slope_length - 1][0] -data[0][0]) / (T_[slope_length - 1] - T_[0]);
				return std::exp(data[0][0] + d_slope * (d - rho_[0]) + T_slope * (T - T_[0]));
			}
			else
			{	
				double const data_T0 = BiLinearInterpolation(T_, rho_, data, T_[0] * 1.00001, d);
				double const T_slope = (BiLinearInterpolation(T_, rho_, data, T_[slope_length - 1], d) - data_T0) / (T_[slope_length - 1] - T_[0]);
				return std::exp(BiLinearInterpolation(T_, rho_, data, T_[0] * 1.00001, d) + T_slope * (T - T_[0]));
			}
		}
		if(T > T_.back())
		{
			if(d < rho_[0])
			{
				double const d_slope = (data[T_.size() - 1][slope_length - 1] -data[T_.size() - 1][0]) / (rho_[slope_length - 1] - rho_[0]);
				return std::exp(data[T_.size() - 1][0] + d_slope * (d - rho_[0]) + T_high_slope * (T - T_.back()));
			}
			else
				return std::exp(BiLinearInterpolation(T_, rho_, data, T_.back() * 0.99999, d) + T_high_slope * (T - T_.back()));
		}
		if(d < rho_[0])
		{
			double const data_d0 = BiLinearInterpolation(T_, rho_, data, T, rho_[0] * 0.9999);
			double const d_slope =(BiLinearInterpolation(T_, rho_, data, T, rho_[slope_length - 1]) - data_d0) / (rho_[slope_length - 1] - rho_[0]);
			return std::exp(BiLinearInterpolation(T_, rho_, data, T, rho_[0] * 0.9999) + d_slope * (d - rho_[0]));
		}
		return std::exp(BiLinearInterpolation(T_, rho_, data, T, d));
	}

	class STAopacity: public OpacityCalculator
	{
	private:
		std::vector<double> rho_, T_;
		std::vector<std::vector<double>> rossland_, planck_, scatter_;
	public:
		STAopacity(std::string file_directory)   
		{
			size_t const Nmatrix = 128;
			std::vector<double> temp = read_vector(file_directory + "ross.txt");
			rossland_.resize(Nmatrix);
			for(size_t i = 0; i < Nmatrix; ++i)
			{
				rossland_[i].resize(Nmatrix);
				for(size_t j = 0; j < Nmatrix; ++j)
					rossland_[i][j] = temp[i * Nmatrix + j];
			}
			temp = read_vector(file_directory +"planck.txt");
			planck_.resize(Nmatrix);
			for(size_t i = 0; i < Nmatrix; ++i)
			{
				planck_[i].resize(Nmatrix);
				for(size_t j = 0; j < Nmatrix; ++j)
					planck_[i][j] = temp[i * Nmatrix + j];
			}
			temp = read_vector(file_directory +"scatter.txt");
			scatter_.resize(Nmatrix);
			for(size_t i = 0; i < Nmatrix; ++i)
			{
				scatter_[i].resize(Nmatrix);
				for(size_t j = 0; j < Nmatrix; ++j)
					scatter_[i][j] = temp[i * Nmatrix + j];
			}
			T_ = read_vector(file_directory +"T.txt");
			rho_ = read_vector(file_directory +"rho.txt");
		}

		double CalcDiffusionCoefficient(ComputationalCell3D const& cell) const override
		{
			double const T = std::log(cell.temperature);
			double const d = std::log(cell.density);
			double const sig = InterpolateTable(T, d, T_, rho_, rossland_);
			if(sig < 1e-100)
				std::cout<<"Small rossland, d "<<d<<" T "<<T<<" ID "<<cell.ID<<std::endl;
			return CG::speed_of_light / (3 * sig);
		}

		double CalcPlanckOpacity(ComputationalCell3D const& cell) const override
		{
			double const T = std::log(cell.temperature);
			double const d = std::log(cell.density);
			return InterpolateTable(T, d, T_, rho_, planck_, -3.5);
		}

		double CalcScatteringOpacity(ComputationalCell3D const& cell) const override
		{
			double const T = std::log(cell.temperature);
			double const d = std::log(cell.density);
			return InterpolateTable(T, d, T_, rho_, scatter_);
		}
	};

	class MassRefine : public CellsToRefine3D
	{
	public:

		MassRefine() {;}

		std::pair<vector<size_t>, vector<Vector3D>> ToRefine(Tessellation3D const &tess, vector<ComputationalCell3D> const &/*cells*/, double time) const
		{
            int rank = 0;
#ifdef RICH_MPI
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
            boost::mt19937_64 base_generator_type(rank + 1e6 * std::abs(std::log(std::abs(time) * 1e20 + 1e-10)));
            boost::random::uniform_real_distribution<> dist;
			size_t Norg = tess.GetPointNo();
			vector<size_t> res;
			for (size_t i = 0; i < Norg; ++i)
			{
				if (fastabs(tess.GetCellCM(i) - tess.GetMeshPoint(i)) > (tess.GetWidth(i) * 0.15))
					continue;
				
				if (dist(base_generator_type)<0.01)
					res.push_back(i);
			}
			return std::pair<vector<size_t>, vector<Vector3D>>(res, vector<Vector3D>());
		}
	};

	class RemoveBig : public CellsToRemove3D
	{
	public:

		RemoveBig() {;}

		std::pair<vector<size_t>, vector<double>> ToRemove(Tessellation3D const &tess, vector<ComputationalCell3D> const &/*cells*/, double time) const
		{
			
			vector<size_t> res;
			vector<double> merits;
			int rank = 0;
#ifdef RICH_MPI
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
            boost::mt19937_64 base_generator_type(12 + rank + 1e7 * std::abs(std::log(std::abs(time) * 1e20 + 1e-10)));
            boost::random::uniform_real_distribution<> dist;
			size_t Norg = tess.GetPointNo();
			for (size_t i = 0; i < Norg; ++i)
			{
				if (fastabs(tess.GetCellCM(i) - tess.GetMeshPoint(i)) > (tess.GetWidth(i) * 0.15))
					continue;
				
				if (dist(base_generator_type)<0.01)
                {
					res.push_back(i);
                    merits.push_back(dist(base_generator_type));
			    }
            }
			return std::pair<vector<size_t>, vector<double>>(res, merits);
		}
	};

}

int main(void)
{
	int rank = 0;
	int ws = 1;
#ifdef RICH_MPI
	MPI_Init(NULL, NULL);
	double last_start = MPI_Wtime();
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ws);
#endif
	feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
	double const dmin_eos = -22;
	double const dmax_eos = 1.1;
	double const dd_eos = 0.05;
	double const Tmin_eos = 0.2;
	double const Tmax_eos = 8.0;
	double const dT_eos = 0.01;
	double const lscale = 1;
	double const mscale = 1;
	double const tscale = 1;
	std::string eos_location("/home/esternberg/RICH/data/EOS/");
	std::string STA_location("/home/esternberg/RICH/data/STA/");
	if (rank == 0)
		std::cout << "start eos" << std::endl;
	OndrejEOS eos(dmin_eos, dmax_eos, dd_eos, eos_location + "Pfile.txt", eos_location + "csfile.txt", eos_location + "Sfile.txt", eos_location + "Ufile.txt", eos_location + "Tfile.txt", eos_location + "CVfile.txt", 1, 1, 1);
	if (rank == 0)
		std::cout << "end eos" << std::endl;
	//Radiation
	STAopacity opacity(STA_location);
	if (rank == 0)
		std::cout << "end sta" << std::endl;

	const double width = 5;
	Vector3D ll(-width, -width, -width), ur(width, width, width);
	Voronoi3D tess(ll, ur);
#ifdef RICH_MPI
	Voronoi3D tproc(ll, ur);
#endif
    double const R = 1;
	vector<ComputationalCell3D> cells;
	
#ifdef RICH_MPI
		vector<Vector3D> procpoints = RoundGrid3DSingle(RandSphereR2(ws, ll, ur, 0, width), ll, ur);
		tproc.Build(procpoints);
#endif
		size_t const np = 1e6;
		vector<Vector3D> ptemp = RandSphereR(np, ll, ur, 0, R * 1.1, Vector3D()
#ifdef RICH_MPI
		, &tproc
#endif
		);
		vector<Vector3D> ptemp2 = RandSphereR(np / 2, ll, ur, 0.8 * R, R * 1.05, Vector3D()
#ifdef RICH_MPI
		, &tproc
#endif
		);
		vector<Vector3D> ptemp3 = RandSphereR2(np / 4, ll, ur, R, 1.4 * width, Vector3D()
#ifdef RICH_MPI
		, &tproc
#endif
		);
		ptemp.insert(ptemp.end(), ptemp2.begin(), ptemp2.end());
		ptemp.insert(ptemp.end(), ptemp3.begin(), ptemp3.end());
		vector<Vector3D> points = RoundGrid3D(ptemp, ll, ur, 15);
		tess.BuildHilbert(points);
		ComputationalCell3D::tracerNames.push_back("Entropy");
        ComputationalCell3D dummy_cell;
        dummy_cell.density = 1;
        dummy_cell.temperature = 3e4;
        dummy_cell.Erad = CG::radiation_constant * std::pow(dummy_cell.temperature, 4.0);
        dummy_cell.internal_energy = eos.dT2e(dummy_cell.density, dummy_cell.temperature);
        dummy_cell.pressure = eos.dT2p(dummy_cell.density, dummy_cell.temperature);
        dummy_cell.tracers[0] = eos.dp2s(dummy_cell.density, dummy_cell.pressure);
        cells.resize(tess.GetPointNo(), dummy_cell);
	if (rank == 0)
		std::cout << "Finished build" << std::endl;
	std::cout<<"Rank "<<rank<<" has "<<tess.GetPointNo()<<" points "<<" and "<<cells.size()<<" cells "<<std::endl;

	Hllc3D rs;
	RigidWallGenerator3D ghost;
	LinearGauss3D interp(eos, ghost, true, 0.2, 0.25, 0.75);
	double Tmin = 1e3;

	Lagrangian3D bpm;
	RoundCells3D pm(bpm, eos, 3.25, 0.01, false, 1.25);

	DiffusionOpenBoundary D_boundary;
	Diffusion matrix_builder(opacity, D_boundary, eos, std::vector<std::string> (), true, true, true);
	matrix_builder.length_scale_ = lscale;
	matrix_builder.time_scale_ = tscale;
	matrix_builder.mass_scale_ = mscale;
	std::shared_ptr<DiffusionForce> rad_force = std::make_shared<DiffusionForce>(matrix_builder, eos);
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
	GravityAcceleration3D sg(0.95, true, 1.0);
	std::shared_ptr<ConservativeForce3D> gravity_force = std::make_shared<ConservativeForce3D>(sg, false);
	std::vector<std::shared_ptr<SourceTerm3D>> forces;

	forces.push_back(gravity_force);
	forces.push_back(rad_force);
	SeveralSources3D force(forces);
	CourantFriedrichsLewy tsf(0.225, 1, force, std::vector<std::string> (),	false);

	Simulation simulation(tess, cells, eos);
	std::unique_ptr<HDSim3D> sim;
	sim = std::make_unique<HDSim3D>(tess, simulation.getCells(), simulation.getExtensives(), eos, simulation.getTracker(), pm, tsf, fc, cu, eu, force, std::pair<std::vector<std::string>, std::vector<std::string>> (ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));

	double init_dt = 1e-5;
    double old_dt = init_dt;
	tsf.SetTimeStep(init_dt);
    double old_t = sim->getTime();
    double step_time = 0;
	

	RemoveBig remove;
	MassRefine refine;
	AMR3D amr(eos, refine, remove, interp);
		
	for(size_t i = 0; i < 20; ++i)
	{
		if (sim->getCycle() % 1 == 0)
		{
#ifdef RICH_MPI
			int ntotal = tess.GetPointNo();
			int total_ntotal = 0;
			MPI_Reduce(&ntotal, &total_ntotal, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
			if (rank == 0)
			{
				std::cout<<std::endl;
				std::cout << "Point num " << total_ntotal << " dt " << old_dt << " run time " << step_time << std::endl;
#endif
				std::cout << "Cycle " << sim->getCycle() << " Time " << sim->getTime() << std::endl;
			}
		}
		
		try
		{
#ifdef RICH_MPI
			double step_tstart = MPI_Wtime();
#endif
			double new_dt = sim->RadiationTimeStep(old_dt * 1, matrix_builder);
			tsf.SetTimeStep(new_dt);
			if (rank == 0)
				std::cout << "Finished rad step" << std::endl;
			sim->timeAdvance2();
			if (rank == 0)
				std::cout << "Finished hydro step" << std::endl;
			if (sim->getCycle() % 5 == 0)
			{
				if(rank == 0)
					std::cout<<"Doing AMR"<<std::endl;
				amr(simulation);
			}
			old_dt = sim->getTime() - old_t;
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
	MPI_Finalize();
#endif
	return 0;
}
