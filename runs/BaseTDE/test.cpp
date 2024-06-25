#include "source/3D/GeometryCommon/Voronoi3D.hpp"
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
#include "source/3D/GeometryCommon/hdf_write.hpp"
#include "source/newtonian/three_dimensional/AMR3D.hpp"
#include "source/newtonian/three_dimensional/ANNSelfGravity.hpp"
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
#ifdef RICH_MPI
#include "source/mpi/mpi_commands.hpp"
#include "source/mpi/ConstNumberPerProc3D.hpp"
#include "source/mpi/SetLoad3D.hpp"
#endif
#include <sys/stat.h>
#include <boost/math/tools/roots.hpp>
#include <sstream>

typedef std::array<double, 4> state_type;

#define smooth_factor 0.6
namespace
{
	class PaczynskiOrbit
	{
	private:
		double M_, Rg_;

	public:
		PaczynskiOrbit(double M) : M_(M), Rg_(0)
		{
			Rg_ = 4.21 * M / 1e6;
		}

		void operator()(const state_type &x, state_type &dxdt, const double /* t */)
		{
			double r = std::sqrt(x[0] * x[0] + x[1] * x[1]);
			dxdt[0] = x[2];
			dxdt[1] = x[3];
			dxdt[2] = -x[0] * M_ / (r * (r - Rg_) * (r - Rg_));
			dxdt[3] = -x[1] * M_ / (r * (r - Rg_) * (r - Rg_));
		}
	};

	state_type GetTrueAnomaly(double t, double M, double Rp, double const dE = 0)
	{
		double Rg = 4.21 * M / 1e6;
		double vp = std::sqrt(2 * (M / (Rp - Rg) + dE));
		typedef boost::numeric::odeint::runge_kutta_cash_karp54<state_type> error_stepper_type;
		PaczynskiOrbit orbit(M);
		state_type x0;
		x0[0] = Rp;
		x0[1] = 0;
		x0[2] = 0;
		x0[3] = -vp;
		boost::numeric::odeint::integrate_adaptive(boost::numeric::odeint::make_controlled<error_stepper_type>(1.0e-11, 1.0e-8), orbit,
												   x0, 0.0, t, t * 1e-5);
		return x0;
	}

	void UpdateReferenceFrame(HDSim3D &sim, double const Rstar, double const Mstar, double const MBH, 
		double const beta)
	{
		double const Rt = Rstar * std::pow(MBH / Mstar, 0.333333333);
		double const Rp = Rt / beta;
		state_type x0 = GetTrueAnomaly(sim.getTime(), MBH, Rp);
		int rank = 0;
#ifdef RICH_MPI
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
		if(rank == 0)
		{
			std::cout<<"Updating reference frame ";
			for(size_t i = 0; i < 4; ++i)
				std::cout<<x0[i]<<" ";
			std::cout<<std::endl;
		}
		std::vector<Vector3D> points = sim.getTesselation().accessMeshPoints();
#ifdef RICH_MPI
		std::vector<Vector3D> proc_points = sim.getProcTesselation().accessMeshPoints();
		size_t const Nproc = sim.getProcTesselation().GetPointNo();
		for(size_t i = 0; i < Nproc; ++i)
		{
			proc_points[i].x += x0[0];
			proc_points[i].y += x0[1];
		}
		proc_points.resize(Nproc);
#endif
		std::vector<Conserved3D> &extensives = sim.getExtensives();
		std::vector<ComputationalCell3D> &cells = sim.getCells();
		size_t const N = sim.getTesselation().GetPointNo();
		std::pair<Vector3D, Vector3D> box_points = sim.getTesselation().GetBoxCoordinates();
		double const reference_density = 1e-4 * Mstar / ((box_points.second.x - box_points.first.x) * (box_points.second.y - box_points.first.y) * (box_points.second.z - box_points.first.z));
		for(size_t i = 0; i < N; ++i)
		{
			points[i].x += x0[0];
			points[i].y += x0[1];
			if(cells[i].density > reference_density)
			{
				cells[i].velocity.x += x0[2];
				cells[i].velocity.y += x0[3];
			}
			else
				cells[i].velocity = Vector3D();
			extensives[i].momentum = extensives[i].mass * cells[i].velocity;
			extensives[i].energy = extensives[i].internal_energy + 0.5 * ScalarProd(extensives[i].momentum, extensives[i].momentum) / extensives[i].mass;
		}
		points.resize(N);		
		box_points.first.x += x0[0];
		box_points.first.y += x0[1];
		box_points.second.x += x0[0];
		box_points.second.y += x0[1];
#ifdef RICH_MPI
		sim.getProcTesselation().SetBox(box_points.first, box_points.second);
		sim.getProcTesselation().Build(proc_points);
#endif
		sim.getTesselation().SetBox(box_points.first, box_points.second);
		sim.getTesselation().Build(points
#ifdef RICH_MPI
		, sim.getProcTesselation()
#endif
		);
	}

	void CheckIfFullGravityIsNeeded(HDSim3D &sim, std::string const& gravity_name, double const Rstar,
		double const Mstar, double const MBH, double const beta, std::string const& restart_name)
	{
		if(sim.getTime() > 10)
		{
			double const Rt = Rstar * std::pow(MBH / Mstar, 0.333333333);
			double const Rp = Rt / beta;
			state_type x0 = GetTrueAnomaly(sim.getTime(), MBH, Rp, -3 * Mstar * std::pow(MBH / Mstar, 0.3333333) / Rstar);
			std::cout<<x0[0]<<","<<x0[1]<<std::endl;
			if(x0[1] > 0.1 && x0[2] > 0.1)
			{
				UpdateReferenceFrame(sim, Rstar, Mstar, MBH, beta);
#ifdef RICH_MPI
				int rank = 0;
				MPI_Barrier(MPI_COMM_WORLD);
				MPI_Comm_rank(MPI_COMM_WORLD, &rank);
				std::cout<<"Point number "<<sim.getTesselation().GetPointNo()<<std::endl;
				if(rank == 0)
#endif
				write_number(1, gravity_name);
				vector<DiagnosticAppendix3D *> appendices;
				WriteSnapshot3D(sim, restart_name, appendices, true);
#ifdef RICH_MPI
				if(rank == 0)
					std::cout<<"Done Gravity change"<<std::endl;
				MPI_Barrier(MPI_COMM_WORLD);
#endif
				exit(0);
			}
		}
	}
	
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

	class MassRefine : public CellsToRefine3D
	{
	private:
		double domain_size_, Mbh_, Mstar_, Rstar_;

	public:
		void SetSize(double s)
		{
			domain_size_ = s;
		}

		MassRefine(double domainsize, double Mbh, double Mstar, double Rstar) : domain_size_(domainsize), Mbh_(Mbh), Mstar_(Mstar), Rstar_(Rstar) {}

		std::pair<vector<size_t>, vector<Vector3D>> ToRefine(Tessellation3D const &tess, vector<ComputationalCell3D> const &cells, double time) const
		{
			std::vector<std::vector<double>> maxr;
			std::vector<std::vector<double>> phi;
			std::vector<double> theta;
			size_t Norg = tess.GetPointNo();
			vector<size_t> res;
			double MaxMass = 1.5e-7;
			std::vector<size_t> neigh;
			std::vector<double> volumes = tess.GetAllVolumes();
#ifdef RICH_MPI
			MPI_exchange_data2(tess, volumes, true);
#endif
			double const apocenter = Rstar_ * std::pow(Mbh_ / Mstar_, 2.0 / 3.0);
			double const Rt = Rstar_ * std::pow(Mbh_ / Mstar_, 1.0 / 3.0);
			double const min_cell_size = Rt * 1e-2;
			double const apocenter_time = 1.25 * std::sqrt(apocenter * apocenter * apocenter / Mbh_);

			for (size_t i = 0; i < Norg; ++i)
			{
				if (fastabs(tess.GetCellCM(i) - tess.GetMeshPoint(i)) > (tess.GetWidth(i) * 0.15))
					continue;
				double r_dist = std::max(fastabs(tess.GetMeshPoint(i)), Rt * smooth_factor);
				if (tess.GetWidth(i) < min_cell_size * (r_dist < 0.65 * Rt ? smooth_factor / 0.6 : 1))
					continue;
				
				if ((r_dist > 0.65 * Rt && r_dist < 2 * Rt) || r_dist > apocenter || r_dist < smooth_factor * Rt)
					continue;

				double MaxMass2 = (tess.GetMeshPoint(i).x > (-apocenter * 2.5)) ? MaxMass : MaxMass * 30;
				MaxMass2 *= std::max(1e-1, std::min(1.0, std::pow(std::abs(time) / apocenter_time, 3.0)));

				double V = tess.GetVolume(i);
				tess.GetNeighbors(i, neigh);
				size_t Nneigh = neigh.size();
				bool good = true, good2 = false;
				for (size_t j = 0; j < Nneigh; ++j)
				{
					if (!tess.IsPointOutsideBox(neigh[j]))
					{
						if (fastabs(tess.GetCellCM(neigh[j]) - tess.GetMeshPoint(neigh[j])) > (0.09 * std::pow(volumes[neigh[j]], 0.33333333333)))
						{
							good = false;
							break;
						}
						if ((5 * volumes[neigh[j]]) < V)
							good2 = true;
					}
				}
				if (!good)
					continue;
				if (good2)
				{
					res.push_back(i);
					continue;
				}
				if ((V * cells[i].density) > (MaxMass2 * std::min(r_dist * r_dist / (50 * Rt * Rt), 1.0)) || V > domain_size_ * 1e-5)
				{
					{
						res.push_back(i);
						continue;
					}
				}
			}
			return std::pair<vector<size_t>, vector<Vector3D>>(res, vector<Vector3D>());
		}
	};

	class RemoveBig : public CellsToRemove3D
	{
	private:
		double domain_size_, Mbh_, Mstar_, Rstar_;
		OndrejEOS const &eos_;

	public:
		void SetSize(double s)
		{
			domain_size_ = s;
		}

		RemoveBig(double domain_size, OndrejEOS const &eos, double Mbh, double Mstar, double Rstar) : domain_size_(domain_size), eos_(eos), Mbh_(Mbh), Mstar_(Mstar), Rstar_(Rstar) {}

		std::pair<vector<size_t>, vector<double>> ToRemove(Tessellation3D const &tess, vector<ComputationalCell3D> const &cells, double time) const
		{
			std::vector<std::vector<double>> maxr;
			std::vector<std::vector<double>> phi;
			std::vector<double> theta;
			vector<size_t> res;
			vector<double> merits;
			vector<size_t> neigh;
			size_t Norg = tess.GetPointNo();
			std::vector<double> volumes = tess.GetAllVolumes();
#ifdef RICH_MPI
			MPI_exchange_data2(tess, volumes, true);
#endif
			double const apocenter = Rstar_ * std::pow(Mbh_ / Mstar_, 2.0 / 3.0);
			double const Rt = Rstar_ * std::pow(Mbh_ / Mstar_, 1.0 / 3.0);
			double const time_Rt = std::sqrt(Rt * Rt * Rt / Mbh_);
			double const min_cell_size = Rt * 1e-2;
			double const apocenter_time = 1.25 * std::sqrt(apocenter * apocenter * apocenter / Mbh_);

			double MaxMass = 3e-8;
			for (size_t i = 0; i < Norg; ++i)
			{
				bool good = true;
				// Do we have little mass amount?
				if (Norg < 500)
					continue;
				double Vol = tess.GetVolume(i);
				double w = tess.GetWidth(i);
				double MaxMass2 = (tess.GetMeshPoint(i).x > -Rt * apocenter * 2.5) ? MaxMass : MaxMass * 30;
				double const r_org = fastabs(tess.GetMeshPoint(i));
				double r_i = std::max(Rt * smooth_factor, r_org);
				MaxMass2 *= std::max(1e-1, std::min(1.0, std::pow(std::abs(time) / apocenter_time, 3.0)));
				MaxMass2 = MaxMass2 * std::min(r_i * r_i / (50 * Rt * Rt), 1.0);
				double const dt = w / eos_.dp2c(cells[i].density, cells[i].pressure, cells[i].tracers);
				double const in_factor = r_i < 0.65 * Rt ? smooth_factor / 0.6 : 1;
				MaxMass2 *= std::max(1.0, std::pow(r_i / r_org, 2.0));
				if (Vol * cells[i].density > MaxMass2 && w > (in_factor * 0.7 * min_cell_size) && dt > (0.02 * time_Rt * in_factor))
					continue;
				if (Vol > domain_size_ * 0.5e-5)
					continue;
				// Make sure we are not that much bigger than smallest neighbor
				tess.GetNeighbors(i, neigh);
				size_t Nneigh = neigh.size();
				for (size_t j = 0; j < Nneigh; ++j)
				{
					if (!tess.IsPointOutsideBox(neigh[j]))
						if (volumes[neigh[j]] < Vol * 0.5)
						{
							good = false;
							break;
						}
				}
				if (good)
				{
					// Make sure we are not too high aspect ratio
					if (fastabs(tess.GetMeshPoint(i) - tess.GetCellCM(i)) > 0.15 * tess.GetWidth(i))
						good = false;
				}
				if (good)
				{
					res.push_back(i);
					merits.push_back(1.0 / Vol);
				}
			}
			return std::pair<vector<size_t>, vector<double>>(res, merits);
		}
	};

	vector<ComputationalCell3D> GetCells(Tessellation3D const &tess, double M, double R, OndrejEOS const &eos, double const Punits)
	{
		vector<double> xsi = read_vector("/home/esternberg/RICH/data/xsi.txt");
		vector<double> theta = read_vector("/home/esternberg/RICH/data/theta.txt");
		xsi[0] = 0;

		/*double n = 3;
		double endfactor = 2.01824;*/

		double n = 1.5;
		double endfactor = 2.714;

		double alpha = R / xsi.back();
		double rho_c = M / (4 * M_PI * alpha * alpha * alpha * endfactor);
		double K = alpha * alpha * 4 * M_PI / ((n + 1) * std::pow(rho_c, 1.0 / n - 1));

		size_t N = tess.GetPointNo();
		vector<ComputationalCell3D> res(N);
		for (size_t i = 0; i < N; ++i)
		{
			Vector3D const &point = tess.GetMeshPoint(i);
			double r = abs(point);
			double t = 0;
			if (r < R)
			{
				t = LinearInterpolation(xsi, theta, r / alpha);
				res[i].tracers[1] = (1);
				res[i].density = std::max(rho_c * std::pow(t, n), 1e-5);
			}
			else
			{
				t = theta.back() * 10;
				res[i].density = rho_c * std::pow(t, n);
				res[i].tracers[1] = (0);
			}
			res[i].tracers[4] = 0;
			double const P = K * std::pow(res[i].density, 1 + 1.0 / n);
			double const a = CG::radiation_constant;
			double const d= res[i].density;
			auto f = [&eos, d, P, a, Punits](double const x){return P - eos.dT2p(d, x) - Punits * a * x * x * x * x / 3;};
			boost::math::tools::eps_tolerance<double> tol(10);
			std::uintmax_t it = 150;
			std::pair<double, double> Tres = boost::math::tools::bracket_and_solve_root(f, 1e4, 2.0, false, tol, it);
			double const T = 0.5 * (Tres.first + Tres.second);
			double const wrongT = eos.dp2T(d, P);
			res[i].internal_energy = eos.dT2e(res[i].density, T, res[i].tracers);
			res[i].pressure = eos.de2p(res[i].density, res[i].internal_energy);
			res[i].Erad = 7.5657e-15 * T * T * T * T * 1603 * 1603 / (7e10 * 7e10 * res[i].density);
			res[i].temperature = T;
			res[i].tracers[0] = (eos.dp2s(res[i].density, res[i].pressure, res[i].tracers));
			res[i].tracers[2] = (0);
			res[i].tracers[3] = (0);
		}
		return res;
	}

	ComputationalCell3D GetReferenceCell(OndrejEOS const &eos, Tessellation3D const &tess, double time)
	{
		double R = 1;
		double M = 1;
		/*double n = 3;
		double endfactor = 2.01824;
		double alpha = R / 6.89684;*/
		double n = 1.5;
		double endfactor = 2.714;
		double alpha = R / 3.65375;
		double rho_c = M / (4 * M_PI * alpha * alpha * alpha * endfactor);
		double K = alpha * alpha * 4 * M_PI / ((n + 1) * std::pow(rho_c, 1.0 / n - 1));
		ComputationalCell3D reference;
		std::pair<Vector3D, Vector3D> box = tess.GetBoxCoordinates();
		double dfactor = std::min(1.0, std::max(0.01, std::exp(-(time - 900) / 100.0)));
		double mindensity = dfactor * 1e-6 * M / ((box.second.x - box.first.x) * (box.second.z - box.first.z) * (box.second.y - box.first.y));
		mindensity = std::max(mindensity, 1e-20);
		reference.density = mindensity;
		double const Tref = 1e3;
		reference.Erad = 7.5657e-15 * Tref * Tref * Tref * Tref * 1603 * 1603 / (7e10 * 7e10 * reference.density);
		reference.pressure = eos.dT2p(reference.density, Tref, reference.tracers);
		reference.velocity = Vector3D();
		reference.internal_energy = eos.dp2e(reference.density, reference.pressure, reference.tracers);
		reference.temperature = Tref;
		reference.tracers[0] = (eos.dp2s(reference.density, reference.pressure, reference.tracers));
		reference.tracers[1] = (0);
		reference.tracers[2] = (0);
		reference.tracers[3] = (0);
		return reference;
	}

	class TDEGravity : public Acceleration3D
	{
	private:
		Acceleration3D const &selfgravity_;
		const double Mbh_, M_, R_, beta_;

	public:
		const bool tide_on_;

		TDEGravity(double Mbh, double M, double R, double beta, Acceleration3D const &sg, bool tide) : selfgravity_(sg), Mbh_(Mbh), M_(M), R_(R), beta_(beta), tide_on_(tide) {}

		void operator()(const Tessellation3D &tess, const vector<ComputationalCell3D> &cells,
						const vector<Conserved3D> &fluxes, const double time, vector<Vector3D> &acc) const
		{
			// Calc self gravity
			selfgravity_(tess, cells, fluxes, time, acc);

			// Calc the force on the CM
			Vector3D Acm, Rcm;
			double Rg = 4.21 * Mbh_ / 1e6;
			double const Rt = R_ * std::pow(Mbh_ / M_, 0.333333333);
			if (tide_on_)
			{
				double Rp = Rt / beta_;
				state_type x0 = GetTrueAnomaly(time, Mbh_, Rp);
				double r = std::sqrt(x0[0] * x0[0] + x0[1] * x0[1]);
				Acm = -Mbh_ * Vector3D(x0[0] / (r * (r - Rg) * (r - Rg)), x0[1] / (r * (r - Rg) * (r - Rg)), 0);
				Rcm = Vector3D(x0[0], x0[1], 0);
			}
			std::pair<Vector3D, Vector3D> box = tess.GetBoxCoordinates();
			double mindensity = std::max(1e-20, 1e-5 * M_ / ((box.second.x - box.first.x) * (box.second.z - box.first.z) * (box.second.y - box.first.y)));
			// Calc the tidal force
			size_t N = acc.size();
			double smooth = Rt * smooth_factor;
			for (size_t i = 0; i < N; ++i)
			{
				Vector3D const &point = tess.GetCellCM(i);
				Vector3D full_point = point + Rcm;
				double r_i = std::max(abs(full_point), Rg * 4);
				if (r_i > smooth)
					acc[i] += -(Mbh_ / (r_i * (r_i - Rg) * (r_i - Rg))) * full_point - Acm;
				else
				{
					double h = smooth;
					acc[i] += -(Mbh_ / (h * (h - Rg) * (h - Rg))) * full_point - Acm;
				}
				if (cells[i].density < mindensity || cells[i].tracers[1] < 0.1)
					acc[i] = Vector3D(0, 0, 0);
			}
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
	double last_start = MPI_Wtime();
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ws);
#endif
	std::string run_directory("/scratch-shared/esternberg/");
	double const R = read_number("Rstar.txt");
	double const M = read_number("Mstar.txt");
	double const Mbh = read_number("Mbh.txt");
	double const beta =  read_number("beta.txt");
	std::stringstream ss;
	ss<<"R"<<R<<"M"<<M<<"BH"<<Mbh<<"beta"<<beta<<"S"<<static_cast<size_t>(smooth_factor*100);
	std::string const run_name = ss.str();
	run_directory += run_name + "/";
	fs::create_directory(run_directory.c_str());
	double const Rt = R * std::pow(Mbh / M, 0.333333);
	double const Rp = Rt / beta;
	double const apocenter = Rp * std::pow(Mbh / M, 0.333333);
	std::string file_name = run_directory + "snap_";
	std::string restart_name = run_directory + "restart.h5";
	std::string counter_name = run_directory + "counter.txt";
	int counter = 0;
	bool const restart = fs::exists(counter_name);
	if(rank == 0)
		std::cout<<"restart "<<restart<<std::endl;
	if(restart)
		counter = read_int(counter_name);
	std::string gravity_name = run_directory + "gravity.txt";
	std::string eos_location("/home/esternberg/RICH/data/EOS/");
	std::string STA_location("/home/esternberg/RICH/data/STA/");
	bool const full_gravity = fs::exists(gravity_name);
	if(restart && full_gravity && (not fs::exists(file_name + int2str(counter) + ".h5")))
	{
		file_name += "full_";
		if(rank == 0)
			std::cout<<"Adding full to filename"<<std::endl;
	}
	if(rank == 0)
		std::cout<<"Full gravity "<<full_gravity<<std::endl;
	double const dmin_eos = -22;
	double const dmax_eos = 1.1;
	double const dd_eos = 0.05;
	double const Tmin_eos = 0.2;
	double const Tmax_eos = 8.0;
	double const dT_eos = 0.01;
	double const lscale = 7e10;
	double const mscale = 2e33;
	double const tscale = 1603;
	if (rank == 0)
		std::cout << "start eos" << std::endl;
	OndrejEOS eos(dmin_eos, dmax_eos, dd_eos, eos_location + "Pfile.txt", eos_location + "csfile.txt", eos_location + "Sfile.txt", eos_location + "Ufile.txt", eos_location + "Tfile.txt", eos_location + "CVfile.txt", lscale, mscale, tscale);
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

	vector<ComputationalCell3D> cells;
	double tstart = 0, t_restart = -100;
	Snapshot3D snap;
	if (restart)
	{
		snap = ReadSnapshot3D(file_name + int2str(counter) + ".h5"
#ifdef RICH_MPI
		, true
#endif
		);
		t_restart = snap.time;
		if(fs::exists(restart_name))
		{
			auto last_time_restart = std::filesystem::last_write_time(restart_name);
			auto last_time_snap = std::filesystem::last_write_time(file_name + int2str(counter) + ".h5");
			if(last_time_snap < last_time_restart)
				snap = ReadSnapshot3D(restart_name
		#ifdef RICH_MPI
					, true
		#endif
				);
		}
		if (full_gravity && file_name.find(std::string("full")) == std::string::npos)
			file_name += "full_";
		++counter;
		ll = snap.ll;
		ur = snap.ur;
#ifdef RICH_MPI
		tproc.SetBox(snap.ll, snap.ur);
		tproc.Build(snap.proc_points);
#endif
		tess.SetBox(snap.ll, snap.ur);
		tess.Build(snap.mesh_points
#ifdef RICH_MPI
			, tproc
#endif
		);
		cells = snap.cells;
		ComputationalCell3D::tracerNames = snap.tracerstickernames.first;
	}
	else
	{
		double startfactor = 3;
		double fstart = -acos(2 * Rp / (startfactor * Rt) - 1);
		tstart = 0.3333333 * sqrt(2 * Rp * Rp * Rp / Mbh) * tan(0.5 * fstart) * (3 + tan(0.5 * fstart) * tan(0.5 * fstart));
#ifdef RICH_MPI
		vector<Vector3D> procpoints = RoundGrid3DSingle(RandSphereR2(ws, ll, ur, 0, width), ll, ur);
		tproc.Build(procpoints);
#endif
		size_t const np = std::min(1e7, 1e6 * std::sqrt(Mbh / 1e4));
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
		vector<Vector3D> points = RoundGrid3D(ptemp, ll, ur, 15
#ifdef RICH_MPI
		, &tproc
#endif
		);
		tess.Build(points
#ifdef RICH_MPI
		, tproc
#endif
		);
		cells = GetCells(tess, M, R, eos, tscale * tscale * lscale / mscale);
		ComputationalCell3D::tracerNames.push_back("Entropy");
		ComputationalCell3D::tracerNames.push_back("Star");
	}
	if (rank == 0)
		std::cout << "Finished build" << std::endl;

	Hllc3D rs;
	RigidWallGenerator3D ghost;
	LinearGauss3D interp(eos, ghost, true, 0.2, 0.25, 0.75);
	double Tmin = 1e3;

	Lagrangian3D bpm;
	RoundCells3D pm(bpm, eos, 3.25, 0.01, false, 1.25);

	DiffusionOpenBoundary D_boundary;
	Diffusion matrix_builder(opacity, D_boundary, eos);
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
	ANNSelfGravity sg(1.05
#ifdef RICH_MPI
	, &tproc
#endif
	);
	TDEGravity acc(Mbh, M, R, beta, sg, not full_gravity);
	std::shared_ptr<ConservativeForce3D> gravity_force = std::make_shared<ConservativeForce3D>(acc, false);
	std::vector<std::shared_ptr<SourceTerm3D>> forces;

	forces.push_back(gravity_force);
	forces.push_back(rad_force);
	SeveralSources3D force(forces);
	CourantFriedrichsLewy tsf(0.225, 1, force, std::vector<std::string> (),	false);
#ifdef RICH_MPI
	ConstNumberPerProc3D procupdate(0.00005, 0.275, 2);
#endif
	std::unique_ptr<HDSim3D> sim;
	if(restart)
	{
		sim = std::make_unique<HDSim3D>(tess, tproc, snap.cells, eos, pm, tsf, fc, cu, eu, force, std::pair<std::vector<std::string>, std::vector<std::string>> (ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames), false
	#ifdef RICH_MPI
		, &procupdate
	#endif
		, false);
		sim->SetTime(snap.time);
		sim->SetCycle(snap.cycle);
	}
else
{
	sim = std::make_unique<HDSim3D>(tess, 
	#ifdef RICH_MPI
		tproc, 
	#endif
		cells, eos, pm, tsf, fc, cu, eu, force, std::pair<std::vector<std::string>, std::vector<std::string>> (ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames), false
	#ifdef RICH_MPI
		, &procupdate
	#endif
		, true);
	sim->SetTime(tstart);
}
	double init_dt = 1e-4;
	tsf.SetTimeStep(init_dt);
	if (rank == 0)
		std::cout << "Restart time " << sim->getTime() << std::endl;
	ComputationalCell3D reference_cell = GetReferenceCell(eos, tess, sim->getTime());
	double tf = 4 * std::sqrt(apocenter * apocenter * apocenter / Mbh);
	double mindt = 0.001;
	double nextT = 0;
	nextT = (t_restart < -20) ? sim->getTime() : t_restart;
	nextT += std::min(8.0, mindt + 0.05 * std::pow(std::abs(sim->getTime()), 0.666666));

	RemoveBig remove(8 * width * width * width, eos, Mbh, M, R);
	MassRefine refine(8 * width * width * width, Mbh, M, R);
	PCM3D ainterp(ghost);
	AMR3D amr(eos, refine, remove, interp);
	std::pair<Vector3D, Vector3D> box2 = sim->getTesselation().GetBoxCoordinates();
	double newvol2 = (box2.second.x - box2.first.x) * (box2.second.y - box2.first.y) * (box2.second.z - box2.first.z);
	refine.SetSize(newvol2);
	remove.SetSize(newvol2);
	vector<DiagnosticAppendix3D *> appendices;
	double old_t = sim->getTime();
	double old_dt = init_dt;
	double step_time = 0;
	double const restart_wtime = 20000;
	double const min_dt_output = 0.02 * std::sqrt(std::pow(R, 3.0) * Mbh / M);
	if(not restart)
		WriteSnapshot3D(*sim, "init.h5", appendices, true);
	while (sim->getTime() < tf)
	{
#ifdef RICH_MPI
		int ntotal = 0;
#endif
		if (sim->getCycle() % 1 == 0)
		{
#ifdef RICH_MPI
			double load = procupdate.GetLoadImbalance(tess, ntotal);
			if (rank == 0)
			{
				std::cout<<std::endl;
				std::cout << "Load = " << load << " Point num " << ntotal << " dt " << old_dt << " run time " << step_time << std::endl;
			}
			if (load > 1.9)
			{
				if (rank == 0)
					std::cout << "Redoing load balance" << std::endl;
				SetLoad(*sim, 50, 0.005, 2, 0.275);
				SetLoad(*sim, 50, 0.003, 2, 0.275);
				SetLoad(*sim, 35, 0.001, 2, 0.275);
				SetLoad(*sim, 30, 0.0005, 2, 0.275);
				SetLoad(*sim, 20, 0.0001, 2, 0.275);
				SetLoad(*sim, 10, 0.00005, 2, 0.275, true, true);
			}
			if (rank == 0)
#endif
				std::cout << "Cycle " << sim->getCycle() << " Time " << sim->getTime() << std::endl;
		}
		if (sim->getTime() > nextT)
		{
			WriteSnapshot3D(*sim, file_name + int2str(counter) + ".h5", appendices, true);
#ifdef RICH_MPI
			if (rank == 0)
#endif
			write_int(counter, counter_name);
			nextT = sim->getTime() + std::min(min_dt_output, mindt + 0.1 * std::pow(std::abs(sim->getTime()), 0.666666));
			++counter;
		}
		try
		{
			int restart_dump = 0;
#ifdef RICH_MPI
			if (rank == 0)
			{
				if (MPI_Wtime() - last_start > restart_wtime)
					restart_dump = 1;
			}
			MPI_Bcast(&restart_dump, 1, MPI_INT, 0, MPI_COMM_WORLD);
			if (restart_dump == 1)
			{
				WriteSnapshot3D(*sim, run_directory + "restart.h5", appendices, true);
				last_start = MPI_Wtime();
			}
			double step_tstart = MPI_Wtime();
#endif
			// sim->RadiationTimeStep(old_dt * 1e-2, matrix_builder);
			// sim->RadiationTimeStep(old_dt * 1e-1, matrix_builder);
			// double new_dt = sim->RadiationTimeStep(old_dt * 0.89, matrix_builder);
			// sim->RadiationTimeStep(old_dt * 1e-3, matrix_builder);
			// sim->RadiationTimeStep(old_dt * 1e-3, matrix_builder);
			double new_dt = sim->RadiationTimeStep(old_dt * 1, matrix_builder);
			tsf.SetTimeStep(new_dt);
			if (rank == 0)
				std::cout << "Finished rad step" << std::endl;
			sim->timeAdvance2();
			if (rank == 0)
				std::cout << "Finished hydro step" << std::endl;
			if (full_gravity && sim->getCycle() % 10 == 0)
			{
				if(rank == 0)
					std::cout<<"Doing AMR"<<std::endl;
				amr(*sim);
			}
			old_dt = sim->getTime() - old_t;
			old_t = sim->getTime();
			if(not full_gravity)
				CheckIfFullGravityIsNeeded(*sim, gravity_name, R, M, Mbh, beta, restart_name);
			reference_cell = GetReferenceCell(eos, tess, sim->getTime());
			if (sim->getCycle() % 7 == 0)
			{
				UpdateBox(sim->getTesselation(), *sim, 0.5, 1e-5, reference_cell
#ifdef RICH_MPI
				, sim->getProcTesselation()
#endif
				);
				std::pair<Vector3D, Vector3D> box = sim->getTesselation().GetBoxCoordinates();
				double newvol = (box.second.x - box.first.x) * (box.second.y - box.first.y) * (box.second.z - box.first.z);
				refine.SetSize(newvol);
				remove.SetSize(newvol);
			}
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
