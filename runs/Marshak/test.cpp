#include "source/3D/tesselation/voronoi/Voronoi3D.hpp"
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/3D/output/write_vtu_3d.hpp"
#include "source/3D/output/hdf_write.hpp"
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
#include "source/newtonian/three_dimensional/AMR3D.hpp"
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

typedef std::array<double, 4> state_type;

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

	class STAopacity: public DiffusionCoefficientCalculator
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
			return CG::speed_of_light / (3 * InterpolateTable(T, d, T_, rho_, rossland_));
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
}

int main(void)
{
	feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
	int rank = 0;
	int ws = 1;
#ifdef RICH_MPI
	MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ws);
#endif
	std::string eos_location("/home/elads/RICH/data/EOS/");
	std::string STA_location("/home/elads/RICH/data/STA/");
	double const dmin_eos = -22;
	double const dmax_eos = 1.1;
	double const dd_eos = 0.05;
	double const Tmin_eos = 0.2;
	double const Tmax_eos = 8.0;
	double const dT_eos = 0.01;
	double const lscale = 7e10;
	double const mscale = 2e33;
	double const tscale = 1603;
	// double const lscale = 1;
	// double const mscale = 1;
	// double const tscale = 1;
	if (rank == 0)
		std::cout << "start eos" << std::endl;
	OndrejEOS eos(dmin_eos, dmax_eos, dd_eos, eos_location + "Pfile.txt", eos_location + "csfile.txt", eos_location + "Sfile.txt", eos_location + "Ufile.txt", eos_location + "Tfile.txt", eos_location + "CVfile.txt", lscale, mscale, tscale);
	if (rank == 0)
		std::cout << "end eos" << std::endl;
	//Radiation
	STAopacity opacity(STA_location);
	if (rank == 0)
		std::cout << "end sta" << std::endl;

	const double width = 10 / lscale;
	size_t const Nx = 128 * 5;//*128;
	size_t const Ny = 10;
	size_t const Nz = 10;
	Vector3D ll(0, -0.5 * width * Ny / Nx, -0.5 * width * Nz / Nx), ur(width, 0.5 * width * Ny / Nx, 0.5 * width * Nz / Nx);
	Voronoi3D tess(ll, ur);
	int counter = 0;
	ComputationalCell3D init_cell;
	double const T = 300;
	try
	{
		init_cell.density = 1 * lscale * lscale * lscale / mscale;
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
	tess.BuildHilbert(points);
	#else
	tess.Build(points);
	#endif
	vector<ComputationalCell3D> cells(tess.GetPointNo(), init_cell);

	Hllc3D rs;
	RigidWallGenerator3D ghost;
	LinearGauss3D interp(eos, ghost, true, 0.2, 0.25, 0.75);

	Lagrangian3D bpm;
	RoundCells3D pm(bpm, eos, 3.75, 0.01, false, 1.25);
	
	DiffusionSideBoundary D_boundary(1.1605e7);
	Diffusion matrix_builder(opacity, D_boundary, eos, std::vector<std::string> (), true, false);
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

	HDSim3D sim(tess, cells, eos, pm, tsf, fc, cu, eu, force, std::pair<std::vector<std::string>, std::vector<std::string>> (ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames), false, true);

	double init_dt = 1e-13 / tscale;
	double const dt_output = 1e-9 / tscale;
	double const tf = 1e-8 / tscale;
	tsf.SetTimeStep(init_dt);
	double nextT = dt_output;
	double old_dt = init_dt;
	vector<DiagnosticAppendix3D *> appendices;
	WriteSnapshot3D(sim, "init.h5", appendices, true);
	auto t1 = std::chrono::high_resolution_clock::now();
	while (sim.getTime() < tf)
	{
		if (sim.getCycle() % 100 == 0)
		{
			if (rank == 0)
			{
				std::cout<<std::endl;
				std::cout << "Cycle " << sim.getCycle() << " Time " << sim.getTime() << " dt "<<old_dt<<std::endl;
			}
		}
		if (sim.getTime() > nextT)
		{
			// WriteSnapshot3D(sim, "snapnew_" + int2str(counter) + ".h5", appendices, true);
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
	auto t2 = std::chrono::high_resolution_clock::now();
	if(rank == 0)
		std::cout<<"run time took "<< std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()<<" seconds"<<std::endl;
	std::cout<<"Done"<<std::endl;
#ifdef RICH_MPI
	MPI_Finalize();
#endif
	return 0;
}
