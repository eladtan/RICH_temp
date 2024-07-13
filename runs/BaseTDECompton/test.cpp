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
#include "source/newtonian/three_dimensional/OndrejEOS.hpp"
#include "source/3D/output/write3D.hpp"
#include "source/3D/output/read3D.hpp"
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
#include "source/Radiation/STAgreyOpacity.hpp"
#include "source/newtonian/three_dimensional/OndrejEOS.hpp"
#include "source/newtonian/three_dimensional/Dissipation.hpp"

typedef std::array<double, 4> state_type;

#define smooth_factor 0.6
// #define hi_res 1

namespace
{
	class DissipationDiag: public DiagnosticAppendix3D
	{
		private:
			Dissipation const& dissipation_;
		public:
		DissipationDiag(Dissipation const& dissipation):dissipation_(dissipation){}

		std::vector<double> operator()(const HDSim3D& sim) const
		{
			int rank = 0;
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			if(rank == 0)
				std::cout<<"Starting dissipation diag"<<std::endl;
			return dissipation_.CalcDissipation(sim);
		}

		std::string getName(void) const
		{
			return std::string("Dissipation");
		}
	};

	class GradDiag: public DiagnosticAppendix3D
	{
		private:
			LinearGauss3D const& interp_;
			size_t const direction_, value_;
		public:
		GradDiag(size_t const direction, size_t const value, LinearGauss3D const& interp): direction_(direction), value_(value), interp_(interp){}

		std::vector<double> operator()(const HDSim3D& sim) const
		{
		    std::vector<Slope3D> slopes = interp_.GetSlopesUnlimited();
			size_t const N = sim.getTesselation().GetPointNo();
			std::vector<double> res(N, 0);
			switch(value_)
			{
				case 0:
					switch(direction_)
					{
						case 0:
							for(size_t i = 0; i < N; ++i)
								res[i] = slopes[i].xderivative.internal_energy;
							break;
						case 1:
							for(size_t i = 0; i < N; ++i)
								res[i] = slopes[i].yderivative.internal_energy;
							break;
						case 2:
							for(size_t i = 0; i < N; ++i)
								res[i] = slopes[i].zderivative.internal_energy;
							break;
					}
					break;
				case 1:
					switch(direction_)
					{
						case 0:
							for(size_t i = 0; i < N; ++i)
								res[i] = slopes[i].xderivative.pressure;
							break;
						case 1:
							for(size_t i = 0; i < N; ++i)
								res[i] = slopes[i].yderivative.pressure;
							break;
						case 2:
							for(size_t i = 0; i < N; ++i)
								res[i] = slopes[i].zderivative.pressure;
							break;
					}
					break;
				case 2:
					switch(direction_)
					{
						case 0:
							for(size_t i = 0; i < N; ++i)
								res[i] = slopes[i].xderivative.density;
							break;
						case 1:
							for(size_t i = 0; i < N; ++i)
								res[i] = slopes[i].yderivative.density;
							break;
						case 2:
							for(size_t i = 0; i < N; ++i)
								res[i] = slopes[i].zderivative.density;
							break;
					}
					break;
				case 3:
					for(size_t i = 0; i < N; ++i)
						res[i] = slopes[i].xderivative.velocity.x + slopes[i].yderivative.velocity.y + slopes[i].zderivative.velocity.z;
					break;
			}
			return res;
		}

		std::string getName(void) const
		{
			switch(value_)
			{
				case 0:
					switch(direction_)
					{
						case 0:
							return std::string("DsieDx");
							break;
						case 1:
							return std::string("DsieDy");
							break;
						case 2:
							return std::string("DsieDz");
							break;
					}
					break;
				case 1:
					switch(direction_)
					{
						case 0:
							return std::string("DpDx");
							break;
						case 1:
							return std::string("DpDy");
							break;
						case 2:
							return std::string("DpDz");
							break;
					}
					break;
				case 2:
					switch(direction_)
					{
						case 0:
							return std::string("DrhoDx");
							break;
						case 1:
							return std::string("DrhoDy");
							break;
						case 2:
							return std::string("DrhoDz");
							break;
					}
					break;
				case 3:
					return std::string("divV");
					break;
			}
		}
	};

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
		std::vector<Conserved3D> &extensives = sim.getExtensives();
		std::vector<ComputationalCell3D> &cells = sim.getCells();
		size_t const N = sim.getTesselation().GetPointNo();
		std::pair<Vector3D, Vector3D> box_points = sim.getTesselation().GetBoxCoordinates();
		double const reference_density = 1e-8 * Mstar / ((box_points.second.x - box_points.first.x) * (box_points.second.y - box_points.first.y) * (box_points.second.z - box_points.first.z));
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
		sim.getTesselation().SetBox(box_points.first, box_points.second);
		sim.getTesselation().BuildParallel(points);
		ComputationalCell3D cdummy;
		MPI_exchange_data(sim.getTesselation(), cells, false, &cdummy);;
	}

	void CheckIfFullGravityIsNeeded(HDSim3D &sim, std::string const& gravity_name, double const Rstar,
		double const Mstar, double const MBH, double const beta, std::string const& restart_name)
	{
		int rank = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		if(sim.getTime() > 5)
		{
			double const Rt = Rstar * std::pow(MBH / Mstar, 0.333333333);
			double const Rp = Rt / beta;
			state_type x0 = GetTrueAnomaly(sim.getTime(), MBH, Rp, -3 * Mstar * std::pow(MBH / Mstar, 0.3333333) / Rstar);
			Tessellation3D const& tess = sim.getTesselation();
			std::vector<ComputationalCell3D> const& cells = sim.getCells();
			int need_update = 0;
			for(size_t i = 0; i < tess.GetPointNo(); ++i)
			{
				if(cells[i].density > 1e-14 && (cells[i].velocity.x + x0[2]) > 0 && (cells[i].velocity.y + x0[3]) < 0 && x0[0] < -2 * Rt)
				{
					need_update = 1;
					break;
				}
			}
			MPI_Allreduce(MPI_IN_PLACE, &need_update, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
			if(rank == 0)
				std::cout<<x0[0]<<","<<x0[1]<<std::endl;
			if((x0[1] > 0.1 && x0[2] > 0.1) || need_update == 1)
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
	
	class MassRefine : public CellsToRefine3D
	{
	private:
		double domain_size_, Mbh_, Mstar_, Rstar_, beta_;

	public:
		void SetSize(double s)
		{
			domain_size_ = s;
		}

		MassRefine(double domainsize, double Mbh, double Mstar, double Rstar, double beta) : domain_size_(domainsize), Mbh_(Mbh), Mstar_(Mstar), Rstar_(Rstar), beta_(beta) {}

		std::pair<vector<size_t>, vector<Vector3D>> ToRefine(Tessellation3D const &tess, vector<ComputationalCell3D> const &cells, double time) const
		{
			std::vector<std::vector<double>> maxr;
			std::vector<std::vector<double>> phi;
			std::vector<double> theta;
			size_t Norg = tess.GetPointNo();
			vector<size_t> res;
			double MaxMass = 1.5e-7 * Mstar_;
			double const Rt = Rstar_ * std::pow(Mbh_ / Mstar_, 1.0 / 3.0) / beta_;
			double min_cell_size = 0.5 * Rt * 1e-2;
#ifdef hi_res
			MaxMass *= 0.25;
			min_cell_size *= std::pow(0.25, 0.33333);
#endif
			std::vector<size_t> neigh;
			std::vector<double> volumes = tess.GetAllVolumes();
#ifdef RICH_MPI
			MPI_exchange_data2(tess, volumes, true);
#endif
			double const apocenter = Rstar_ * std::pow(Mbh_ / Mstar_, 2.0 / 3.0);
			double const apocenter_time = 1.25 * std::sqrt(apocenter * apocenter * apocenter / Mbh_);

			for (size_t i = 0; i < Norg; ++i)
			{
				if (fastabs(tess.GetCellCM(i) - tess.GetMeshPoint(i)) > (tess.GetWidth(i) * 0.15))
					continue;
				double r_dist = std::max(fastabs(tess.GetMeshPoint(i)), Rt * smooth_factor);
				if (tess.GetWidth(i) < min_cell_size * (r_dist < (0.65 * Rt) ? smooth_factor / 0.6 : 1))
					continue;
				
				if (r_dist < (1.25 * Rt))
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
						if ((6 * volumes[neigh[j]]) < V)
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
				if ((V * cells[i].density) > (MaxMass2 * std::min(std::pow(0.05 * r_dist / Rt, 2.5), 1.0)) || V > domain_size_ * 1e-5)
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
		double domain_size_, Mbh_, Mstar_, Rstar_, beta_;
		OndrejEOS const &eos_;

	public:
		void SetSize(double s)
		{
			domain_size_ = s;
		}

		RemoveBig(double domain_size, OndrejEOS const &eos, double Mbh, double Mstar, double Rstar, double beta) : domain_size_(domain_size), eos_(eos), Mbh_(Mbh), Mstar_(Mstar), Rstar_(Rstar), beta_(beta) {}

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
			double const Rt = Rstar_ * std::pow(Mbh_ / Mstar_, 1.0 / 3.0) / beta_;
			double const time_Rt = std::sqrt(Rt * Rt * Rt / Mbh_);
			double min_cell_size = Rt * 1e-2;
			double const apocenter_time = 1.25 * std::sqrt(apocenter * apocenter * apocenter / Mbh_);
			int rank = -1;
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			if(rank == 0)
				std::cout<<"Remove apocenter: "<<apocenter<<" time_Rt "<<time_Rt<<" min_cell_size "<<min_cell_size<<std::endl;
			double MaxMass = 3.5e-8 * Mstar_;
#ifdef hi_res
			MaxMass *= 0.25;
			min_cell_size *= std::pow(0.25, 0.33333);
#endif
			for (size_t i = 0; i < Norg; ++i)
			{
				bool good = true;
				// Do we have little mass amount?
				if (Norg < 500)
					continue;
				double Vol = tess.GetVolume(i);
				double w = tess.GetWidth(i);
				double MaxMass2 = (tess.GetMeshPoint(i).x > -apocenter * 2.5) ? MaxMass : MaxMass * 30;
				double const r_org = fastabs(tess.GetMeshPoint(i));
				double r_i = std::max(Rt * smooth_factor, r_org);
				MaxMass2 *= std::max(1e-1, std::min(1.0, std::pow(std::abs(time) / apocenter_time, 3.0)));
				MaxMass2 = MaxMass2 * std::min(std::pow(0.05 * r_i / Rt, 2.5), 1.0);
				double const dt = w / (eos_.dp2c(cells[i].density, cells[i].pressure, cells[i].tracers) + 0.5 * fastabs(cells[i].velocity));
				double const in_factor = r_i < 0.65 * Rt ? smooth_factor / 0.6 : 1;
				MaxMass2 *= std::max(1.0, std::pow(r_i / r_org, 2.0));
				if (Vol * cells[i].density > MaxMass2 && dt > (0.025 * time_Rt * in_factor))//w > (in_factor * 0.75 * min_cell_size)
					continue;
				if (Vol > domain_size_ * 0.5e-5)
					continue;
				// Make sure we are not that much bigger than smallest neighbor
				tess.GetNeighbors(i, neigh);
				size_t Nneigh = neigh.size();
				for (size_t j = 0; j < Nneigh; ++j)
				{
					if (!tess.IsPointOutsideBox(neigh[j]))
						if (volumes[neigh[j]] < Vol * 0.4)
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

	ComputationalCell3D GetReferenceCell(OndrejEOS const &eos, Tessellation3D const &tess, double time)
	{
		double M = 1;
		ComputationalCell3D reference;
		std::pair<Vector3D, Vector3D> box = tess.GetBoxCoordinates();
		double dfactor = 1;
		double mindensity = dfactor * 1e-11 * M / ((box.second.x - box.first.x) * (box.second.z - box.first.z) * (box.second.y - box.first.y));
		mindensity = std::max(mindensity, 1e-20);
		reference.density = mindensity;
		double const Tref = 500;
		double const Tgas = 1e7;
		reference.Erad = 7.5657e-15 * Tref * Tref * Tref * Tref * 1603 * 1603 * 7e10 / (2e33 * reference.density);
		reference.pressure = eos.dT2p(reference.density, Tgas, reference.tracers);
		reference.velocity = Vector3D();
		reference.internal_energy = eos.dp2e(reference.density, reference.pressure, reference.tracers);
		reference.temperature = Tgas;
		reference.tracers[0] = (eos.dp2s(reference.density, reference.pressure, reference.tracers));
		reference.tracers[1] = (0);
		reference.tracers[2] = (0);
		reference.tracers[3] = (0);
		return reference;
	}

	vector<ComputationalCell3D> GetCells(Tessellation3D const &tess, double M, double R, OndrejEOS const &eos, double const Punits, double const n)
	{
		double endfactor = 0;
		vector<double> xsi;
		vector<double> theta;
		if(n > 2)
		{
			xsi = read_vector("/home/esternberg/RICH/data/xsi3.txt");
			theta = read_vector("/home/esternberg/RICH/data/theta3.txt");
			endfactor = 2.0182359;
		}
		else
		{
			xsi = read_vector("/home/esternberg/RICH/data/xsi32.txt");
			theta = read_vector("/home/esternberg/RICH/data/theta32.txt");
			endfactor = 2.714055;
		}
		xsi[0] = 0;

		double alpha = R / xsi.back();
		double rho_c = M / (4 * M_PI * alpha * alpha * alpha * endfactor);
		double K = alpha * alpha * 4 * M_PI / ((n + 1) * std::pow(rho_c, 1.0 / n - 1));

		size_t N = tess.GetPointNo();
		vector<ComputationalCell3D> res(N);
		ComputationalCell3D reference = GetReferenceCell(eos, tess, 0);
		for (size_t i = 0; i < N; ++i)
		{
			Vector3D const &point = tess.GetMeshPoint(i);
			double r = abs(point);
			double t = 0;
			if (r < R)
			{
				t = LinearInterpolation(xsi, theta, r / alpha);
				res[i].tracers[1] = (1);
				res[i].density = std::max(rho_c * std::pow(t, n), 1e-6);
				double const P = K * std::pow(res[i].density, 1 + 1.0 / n);
				double const a = CG::radiation_constant;
				double const d= res[i].density;
				auto f = [&eos, d, P, a, Punits](double const x){return P - eos.dT2p(d, x) - Punits * a * x * x * x * x / 3;};
				boost::math::tools::eps_tolerance<double> tol(10);
				std::uintmax_t it = 150;
				std::pair<double, double> Tres = boost::math::tools::bracket_and_solve_root(f, 1e4, 2.0, false, tol, it);
				double const T = 0.5 * (Tres.first + Tres.second);
				res[i].internal_energy = eos.dT2e(res[i].density, T, res[i].tracers);
				res[i].tracers[4] = 0;
				res[i].pressure = eos.de2p(res[i].density, res[i].internal_energy);
				res[i].Erad = 7.5657e-15 * T * T * T * T * 1603 * 1603 * 7e10 / (2e33 * res[i].density);
				res[i].temperature = T;
			}
			else
			{
				res[i] = reference;
				res[i].tracers[1] = (0);
			}
			res[i].tracers[0] = (eos.dp2s(res[i].density, res[i].pressure, res[i].tracers));
			res[i].tracers[2] = (0);
			res[i].tracers[3] = (0);
		}
		return res;
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
			double mindensity = std::max(1e-19, 1e-10 * M_ / ((box.second.x - box.first.x) * (box.second.z - box.first.z) * (box.second.y - box.first.y)));
			// Calc the tidal force
			size_t N = acc.size();
			double smooth = Rt * smooth_factor / beta_;
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
	// set signals for floating point exceptions
	feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
	// Get basic MPI info
	int rank = 0;
	int ws = 1;
#ifdef RICH_MPI
	MPI_Init(NULL, NULL);
	double last_start = MPI_Wtime();
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ws);
#endif

	// Set paths
	std::string run_directory("/scratch-shared/esternberg/");
	double const R = read_number("Rstar.txt");
	double const M = read_number("Mstar.txt");
	double const n = read_number("n.txt");
	double const Mbh = read_number("Mbh.txt");
	double const beta =  read_number("beta.txt");
	std::stringstream ss;
	ss<<"R"<<R<<"M"<<M<<"BH"<<Mbh<<"beta"<<beta<<"S"<<static_cast<size_t>(smooth_factor*100)<<"n"<<n<<"Compton";
#ifdef hi_res
	ss<<"HiRes";
#endif
	if(rank == 0)
		std::cout<<"Creating directory "<<ss.str()<<std::endl;
	std::string const run_name = ss.str();
	run_directory += run_name + "/";
	fs::create_directory(run_directory.c_str());

	// set physical parameters of run
	double const Rt = R * std::pow(Mbh / M, 0.333333);
	double const Rp = Rt / beta;
	double const apocenter = Rt * std::pow(Mbh / M, 0.333333);
	if(rank == 0)
		std::cout<<"Apocenter is "<<apocenter<<std::endl;
	std::string file_name = run_directory + "snap_";
	std::string restart_name = run_directory + "restart.h5";
	std::string counter_name = run_directory + "counter.txt";
	int counter = 0;
	// check if this is a restart run
	bool const restart = fs::exists(counter_name);
	if(rank == 0)
		std::cout<<"restart "<<restart<<std::endl;
	if(restart)
	{
		counter = read_int(counter_name);
		std::filesystem::last_write_time(counter_name, std::filesystem::file_time_type::clock::now());
	}
	std::string gravity_name = run_directory + "gravity.txt";
	std::string eos_location("/home/esternberg/RICH/data/EOS/");
	std::string STA_location("/home/esternberg/RICH/data/STA/");
	// check if full gravity is needed, we are no longer in CM frame
	bool const full_gravity = fs::exists(gravity_name);
	if(full_gravity)
		std::filesystem::last_write_time(gravity_name, std::filesystem::file_time_type::clock::now());
	if(restart && full_gravity && (not fs::exists(file_name + int2str(counter) + ".h5")))
	{
		file_name += "full_";
		if(rank == 0)
			std::cout<<"Adding full to filename"<<std::endl;
	}
	if(rank == 0)
		std::cout<<"Full gravity "<<full_gravity<<std::endl;
	double const lscale = 7e10;
	double const mscale = 2e33;
	double const tscale = 1603;
	// The EOS
	OndrejEOS eos(eos_location +"density.txt", eos_location + "Pfile.txt", eos_location + "csfile.txt", eos_location + "Sfile.txt", eos_location + "Ufile.txt", eos_location + "Tfile.txt", eos_location + "CVfile.txt", lscale, mscale, tscale);

	//Radiation
	STAgreyOpacity opacity(STA_location);

	const double width = 5;
	Vector3D ll(-width, -width, -width), ur(width, width, width);
	Voronoi3D tess(ll, ur);

	vector<ComputationalCell3D> cells;
	double tstart = 0, t_restart = -100;
	Snapshot3D snap;
	if (restart)
	{
		int hdf5_rank = -1;
		int NranksInFile = GetNumberOfRanksInHDF(file_name + int2str(counter) + ".h5");
		if(rank == 0)
			std::cout<<"Reading from file "<<file_name + int2str(counter) + ".h5 file has "<<NranksInFile<<" ranks"<<std::endl;
		snap = ReadSnapshot3D(file_name + int2str(counter) + ".h5"
#ifdef RICH_MPI
		, true, hdf5_rank
#endif
		);
		// Check if we are restaring with less cpus then last time
		if(ws < NranksInFile && rank == 0)
		{
			for(int j = ws; j < NranksInFile; ++j)
			{
				Snapshot3D snap_temp = ReadSnapshot3D(file_name + int2str(counter) + ".h5"
#ifdef RICH_MPI
		, true, j
#endif
				);
				snap.cells.insert(snap.cells.end(), snap_temp.cells.begin(), snap_temp.cells.end());
				snap.mesh_points.insert(snap.mesh_points.end(), snap_temp.mesh_points.begin(), snap_temp.mesh_points.end());
			}
		}
		t_restart = snap.time;
		if(fs::exists(restart_name))
		{
			auto last_time_restart = std::filesystem::last_write_time(restart_name);
			auto last_time_snap = std::filesystem::last_write_time(file_name + int2str(counter) + ".h5");
			if(last_time_snap < last_time_restart)
			{
				if(rank == 0)
					std::cout<<"Reading from restart file"<<std::endl;
				snap = ReadSnapshot3D(restart_name
		#ifdef RICH_MPI
					, true, hdf5_rank
		#endif
				);
				if(ws < NranksInFile && rank == 0)
				{
					for(int j = ws; j < NranksInFile; ++j)
					{
						Snapshot3D snap_temp = ReadSnapshot3D(restart_name
		#ifdef RICH_MPI
				, true, j
		#endif
						);
						snap.cells.insert(snap.cells.end(), snap_temp.cells.begin(), snap_temp.cells.end());
						snap.mesh_points.insert(snap.mesh_points.end(), snap_temp.mesh_points.begin(), snap_temp.mesh_points.end());
					}
				}
			}
		}
		std::cout<<"Rank "<<rank<<" has "<<snap.mesh_points.size()<<" points, hdf5_rank "<<hdf5_rank<<std::endl;
		if (full_gravity && file_name.find(std::string("full")) == std::string::npos)
			file_name += "full_";
		++counter;
		ll = snap.ll;
		ur = snap.ur;
		if(rank == 0)
			std::cout<<"Box is ll="<<ll<<" ur="<<ur<<std::endl;
		tess.SetBox(snap.ll, snap.ur);
		tess.BuildParallel(snap.mesh_points);
		if(rank == 0)
			std::cout<<"Done build"<<ur<<std::endl;
		ComputationalCell3D cdummy;
		MPI_exchange_data(tess, snap.cells, false, &cdummy);
		cells = snap.cells;
		ComputationalCell3D::tracerNames = snap.tracerstickernames.first;
	}
	else
	{
		double startfactor = 3;
		double fstart = -acos(2 * Rp / (startfactor * Rt) - 1);
		tstart = 0.3333333 * sqrt(2 * Rp * Rp * Rp / Mbh) * tan(0.5 * fstart) * (3 + tan(0.5 * fstart) * tan(0.5 * fstart));
		size_t const np = std::min(1e7, 1e6 * std::sqrt(Mbh / 1e4));
		vector<Vector3D> ptemp = RandSphereR1(np, ll, ur, 0, R * 1.1);
		vector<Vector3D> ptemp2 = RandSphereR(np / 2, ll, ur, 0.8 * R, R * 1.05);
		vector<Vector3D> ptemp3 = RandSphereR2(np / 4, ll, ur, R, 1.4 * width);
		ptemp.insert(ptemp.end(), ptemp2.begin(), ptemp2.end());
		ptemp.insert(ptemp.end(), ptemp3.begin(), ptemp3.end());
		try
		{
			vector<Vector3D> points = RoundGrid3D(ptemp, ll, ur, 15);
			tess.BuildParallel(points);
			cells = GetCells(tess, M, R, eos, tscale * tscale * lscale / mscale, n);
		}
		catch (UniversalError const &eo)
		{
			reportError(eo);
			throw;
		}
		ComputationalCell3D::tracerNames.push_back("Entropy");
		ComputationalCell3D::tracerNames.push_back("Star");
	}
	if (rank == 0)
		std::cout << "Finished build" << std::endl;
	std::cout<<"Rank "<<rank<<" has "<<tess.GetPointNo()<<" points "<<" and "<<cells.size()<<" cells "<<std::endl;

	Hllc3D rs;
	RigidWallGenerator3D ghost;
	LinearGauss3D interp(eos, ghost, true, 0.2, 0.25, 0.75);
	double Tmin = 1e3;

	Lagrangian3D bpm;
	RoundCells3D pm(bpm, eos, 1.75, 0.005, false, 1.25);

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
	GravityAcceleration3D sg(1.05, true, 1.0);
	TDEGravity acc(Mbh, M, R, beta, sg, not full_gravity);
	std::shared_ptr<ConservativeForce3D> gravity_force = std::make_shared<ConservativeForce3D>(acc, false);
	std::vector<std::shared_ptr<SourceTerm3D>> forces;

	forces.push_back(gravity_force);
	forces.push_back(rad_force);
	SeveralSources3D force(forces);
	CourantFriedrichsLewy tsf(0.3, 1, force, std::vector<std::string> (),	false);

	std::unique_ptr<HDSim3D> sim;
	if(restart)
	{
		sim = std::make_unique<HDSim3D>(tess, cells, eos, pm, tsf, fc, cu, eu, force, std::pair<std::vector<std::string>, std::vector<std::string>> (ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames), false, false);
		sim->SetTime(snap.time);
		sim->SetCycle(snap.cycle);
	}
	else
	{
		sim = std::make_unique<HDSim3D>(tess, cells, eos, pm, tsf, fc, cu, eu, force, std::pair<std::vector<std::string>, std::vector<std::string>> (ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames), false, true);
		sim->SetTime(tstart);
	}
	double init_dt = 1e-4;
	tsf.SetTimeStep(init_dt);
	if (rank == 0)
		std::cout << "Restart time " << sim->getTime() << std::endl;
	ComputationalCell3D reference_cell = GetReferenceCell(eos, tess, sim->getTime());
	double tf = 6 * std::sqrt(apocenter * apocenter * apocenter / Mbh);
	double mindt = 0.001;
	double nextT = 0;
	nextT = (t_restart < -20) ? sim->getTime() : t_restart;
	nextT += std::min(8.0, mindt + 0.05 * std::pow(std::abs(sim->getTime()), 0.666666));
	nextT = std::max(nextT, sim->getTime() + 0.01);

	RemoveBig remove(8 * width * width * width, eos, Mbh, M, R, beta);
	MassRefine refine(8 * width * width * width, Mbh, M, R, beta);
	AMR3D amr(eos, refine, remove, interp);
	std::pair<Vector3D, Vector3D> box2 = sim->getTesselation().GetBoxCoordinates();
	double newvol2 = (box2.second.x - box2.first.x) * (box2.second.y - box2.first.y) * (box2.second.z - box2.first.z);
	refine.SetSize(newvol2);
	remove.SetSize(newvol2);
	vector<DiagnosticAppendix3D *> appendices;
	GradDiag diag00(0, 0, interp);
	GradDiag diag01(0, 1, interp);
	GradDiag diag02(0, 2, interp);
	GradDiag diag10(1, 0, interp);
	GradDiag diag11(1, 1, interp);
	GradDiag diag12(1, 2, interp);
	GradDiag diag20(2, 0, interp);
	GradDiag diag21(2, 1, interp);
	GradDiag diag22(2, 2, interp);
	GradDiag diag3(0, 3, interp);
	Dissipation dissipation(rs, eos);
	DissipationDiag DissDiag(dissipation);
	appendices.push_back(&diag00);
	appendices.push_back(&diag01);
	appendices.push_back(&diag02);
	appendices.push_back(&diag10);
	appendices.push_back(&diag11);
	appendices.push_back(&diag12);
	appendices.push_back(&diag20);
	appendices.push_back(&diag21);
	appendices.push_back(&diag22);
	appendices.push_back(&diag3);
	appendices.push_back(&DissDiag);
	
	double old_t = sim->getTime();
	double old_dt = init_dt;
	double step_time = 0;
	double const restart_wtime = 25000;
	double const min_dt_output = 0.02 * std::sqrt(std::pow(R, 3.0) * Mbh / M);
	if(not restart)
	{
		interp(tess, sim->getCells(), 0, dissipation.face_values);
		WriteSnapshot3D(*sim, "init.h5", appendices, true);
		dissipation.face_values.clear();
		dissipation.face_values.shrink_to_fit();
	}
	
	while (sim->getTime() < tf)
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
		if (sim->getTime() > nextT)
		{
			if(rank == 0)
				std::cout<<"Starting writing file "<<file_name + int2str(counter) + ".h5"<<std::endl;
			interp(tess, sim->getCells(), 0, dissipation.face_values);
			WriteSnapshot3D(*sim, file_name + int2str(counter) + ".h5", appendices, true);
#ifdef RICH_MPI
			if (rank == 0)
#endif
			write_int(counter, counter_name);
			nextT = sim->getTime() + std::min(min_dt_output, mindt + 0.1 * std::pow(std::abs(sim->getTime()), 0.666666));
			++counter;
			dissipation.face_values.clear();
			dissipation.face_values.shrink_to_fit();
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
				if(rank == 0)
					std::cout<<"Starting writing file "<<run_directory + "restart.h5"<<std::endl;
				interp(tess, sim->getCells(), 0, dissipation.face_values);
				WriteSnapshot3D(*sim, run_directory + "restart.h5", appendices, true);
				dissipation.face_values.clear();
				dissipation.face_values.shrink_to_fit();
				last_start = MPI_Wtime();
			}
			double step_tstart = MPI_Wtime();
#endif
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
				UpdateBox(tess, *sim, 0.5, 1e-5, reference_cell);
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
    if(rank == 0)
	   std::cout<<"Done sim"<<std::endl;
	MPI_Finalize();
#endif
	return 0;
}
