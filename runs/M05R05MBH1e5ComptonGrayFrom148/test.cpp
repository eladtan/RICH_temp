#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/HydroStep.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationStep.hpp"
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
#include "source/newtonian/three_dimensional/FastMultipoleAcceleration3D.hpp"
#include "source/Radiation/Diffusion.hpp"
#include "source/Radiation/DiffusionForce.hpp"
// #include "source/Radiation/MultigroupDiffusionForce.hpp"
#include "source/Radiation/MultigroupDiffusion.hpp"
#include <boost/numeric/odeint.hpp>
#include "source/newtonian/three_dimensional/LagrangianExtensiveUpdater3D.hpp"
#include <boost/math/tools/roots.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <exception>
#include <random>
#include <fenv.h>
#include "source/3D/GeometryCommon/UpdateBox.hpp"
#include "source/Radiation/STAgreyOpacity.hpp"
#include <sys/stat.h>
#include <boost/math/tools/roots.hpp>
#include <exception>
#include "source/misc/utils.hpp"

typedef std::array<double, 4> state_type;

#define smooth_factor 0.6
#define hi_res 1
// #define low_res 1
// #define remove_center 1
namespace
{
	void RemoveCenter(HDSim3D& sim, double MBH, double Mstar, double Rstar,
		EquationOfState const& eos, double beta)
	{
		double const Rt = Rstar * std::pow(MBH / Mstar, 0.333333333) / beta;
		double Rsmooth = std::max(Rt * 0.4, std::min(Rt - Rstar * 15, Rt * smooth_factor));
		std::vector<Conserved3D> &extensives = sim.getExtensives();
		std::vector<ComputationalCell3D> &cells = sim.getCells();
		size_t const N = sim.getTessellation().GetPointNo();
		for(size_t i = 0; i < N; ++i)
		{
			if(fastabs(sim.getTessellation().GetCellCM(i)) < Rsmooth)
			{
				double new_density = std::max(1e-20, cells[i].density * 0.5);
				double density_ratio = cells[i].density / new_density;
				double old_T = cells[i].temperature;
				double new_T = std::max(1e4, cells[i].temperature * 0.8);
				cells[i].tracers[2] *= cells[i].density;
				cells[i].tracers[2] += cells[i].density - new_density;
				cells[i].density = new_density;
				cells[i].tracers[2] /= new_density;
				cells[i].temperature = new_T;
				if(fastabs(cells[i].velocity) > 1e-4)
					cells[i].velocity *= 0.95;
				cells[i].internal_energy = eos.dT2e(new_density, new_T, cells[i].tracers, ComputationalCell3D::tracerNames);
				cells[i].pressure = eos.de2p(new_density, cells[i].internal_energy, cells[i].tracers, ComputationalCell3D::tracerNames);
				cells[i].tracers[0] = eos.dp2s(new_density, cells[i].pressure, cells[i].tracers, ComputationalCell3D::tracerNames);
				double Erad_ratio = 1;//boost::math::pow<4>(new_T / old_T);
				cells[i].Erad *= density_ratio * Erad_ratio;
				cells[i].Erad_dt *= density_ratio * Erad_ratio;
				cells[i].Erad_dt_dt *= density_ratio * Erad_ratio;
				PrimitiveToConserved(cells[i], sim.getTessellation().GetVolume(i), extensives[i]);
			}
		}
		MPI_exchange_data(sim.getTessellation(), cells, true);
		MPI_exchange_data(sim.getTessellation(), extensives, true);
	}

	class DissipationDiag: public DiagnosticAppendix3D
	{
		private:
			LinearGauss3D const& interp_;
			Hllc3D const& rs_;
			EquationOfState const& eos_;
		public:
		DissipationDiag(LinearGauss3D const& interp, Hllc3D const& rs, EquationOfState const& eos)
			:interp_(interp), rs_(rs), eos_(eos){}

		std::vector<double> operator()(const HDSim3D& sim) const
		{
			return interp_.CalcDissipationStreamingFromPreparedSlopes(sim.getTessellation(), sim.getCells(), sim.getTime(), rs_, eos_);
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
		    const std::vector<Slope3D>& slopes = interp_.GetSlopesUnlimited();
			size_t const N = sim.getTessellation().GetPointNo();
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
			return std::string("Unknown");
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
		std::vector<Vector3D> points = sim.getTessellation().accessMeshPoints();
		std::vector<Conserved3D> &extensives = sim.getExtensives();
		std::vector<ComputationalCell3D> &cells = sim.getCells();
		size_t const N = sim.getTessellation().GetPointNo();
		std::pair<Vector3D, Vector3D> box_points = sim.getTessellation().GetBoxCoordinates();
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
		sim.getTessellation().SetBox(box_points.first, box_points.second);
#ifdef RICH_MPI
		sim.getTessellation().BuildParallel(points);
		ComputationalCell3D cdummy;
		MPI_exchange_data(sim.getTessellation(), cells, false);
#else
		sim.getTessellation().Build(points);
#endif
	}

	void CheckIfFullGravityIsNeeded(HDSim3D &sim, std::string const& gravity_name, double const Rstar,
		double const Mstar, double const MBH, double const beta, std::string const& restart_name)
	{
		int rank = 0;
#ifdef RICH_MPI
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
		if(sim.getTime() > 5)
		{
			double const Rt = Rstar * std::pow(MBH / Mstar, 0.333333333);
			double const Rp = Rt / beta;
			state_type x0 = GetTrueAnomaly(sim.getTime(), MBH, Rp, 0/*-3 * Mstar * std::pow(MBH / Mstar, 0.3333333) / Rstar*/);
			Tessellation3D const& tess = sim.getTessellation();
			std::vector<ComputationalCell3D> const& cells = sim.getCells();
			int need_update = 0;
			double max_x = -1e20, max_y = -1e20;
			for(size_t i = 0; i < tess.GetPointNo(); ++i)
			{
				if(cells[i].density * MBH > 1e-11 && (cells[i].velocity.x + x0[2]) > 0 && x0[0] < -2 * Rt)
				{
					max_x = std::max(max_x, tess.GetMeshPoint(i).x);
					max_y = std::max(max_y, tess.GetMeshPoint(i).y);
				}
				if(cells[i].density * MBH > 1e-11 && (cells[i].velocity.x + x0[2]) > 0 && (cells[i].velocity.y + x0[3]) < 0 && x0[0] < -2 * Rt)
				{
					need_update = 1;
					break;
				}
			}
#ifdef RICH_MPI
			MPI_Allreduce(MPI_IN_PLACE, &need_update, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
			MPI_Allreduce(MPI_IN_PLACE, &max_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			MPI_Allreduce(MPI_IN_PLACE, &max_y, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
			if(rank == 0)
				std::cout<<x0[0]<<","<<x0[1]<<" max values "<<max_x + x0[0]<<", "<<max_y + x0[1]<<std::endl;
			if((x0[1] > 0.1 && x0[2] > 0.1) || need_update == 1)
			{
				UpdateReferenceFrame(sim, Rstar, Mstar, MBH, beta);
#ifdef RICH_MPI
				int rank = 0;
				MPI_Barrier(MPI_COMM_WORLD);
				MPI_Comm_rank(MPI_COMM_WORLD, &rank);
				std::cout<<"Point number "<<sim.getTessellation().GetPointNo()<<std::endl;
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
	
// 	 class MassRefine : public CellsToRefine3D
//         {
//         private:
//                 double domain_size_, Mbh_, Mstar_, Rstar_, beta_;

//         public:
//                 void SetSize(double s)
//                 {
//                         domain_size_ = s;
//                 }

//                 MassRefine(double domainsize, double Mbh, double Mstar, double Rstar, double beta) : domain_size_(domainsize), Mbh_(Mbh), Mstar_(Mstar), Rst\
// ar_(Rstar), beta_(beta) {}

//                 std::pair<vector<size_t>, vector<Vector3D>> ToRefine(Tessellation3D const &tess, vector<ComputationalCell3D> const &cells, double time) cons\
// t
//                 {
//                         std::vector<std::vector<double>> maxr;
//                         std::vector<std::vector<double>> phi;
//                         std::vector<double> theta;
//                         size_t Norg = tess.GetPointNo();
//                         vector<size_t> res;
//                         double MaxMass = 1.5e-7 * Mstar_;
//                         double const Rt = Rstar_ * std::pow(Mbh_ / Mstar_, 1.0 / 3.0) / beta_;
//                         double min_cell_size = 0.5 * Rt * 1e-2;
// #ifdef hi_res
//                         MaxMass *= 0.25;
//                         min_cell_size *= std::pow(0.25, 0.33333);
// #endif
// #ifdef low_res
//                         MaxMass *= 4;
//                         min_cell_size *= std::pow(4.0, 0.333333);
// 						#endif
//                         std::vector<size_t> neigh;
//                         std::vector<double> volumes = tess.GetAllVolumes();
// #ifdef RICH_MPI
//                         MPI_exchange_data(tess, volumes, true);
// #endif
//                         double const apocenter = Rstar_ * std::pow(Mbh_ / Mstar_, 2.0 / 3.0);
//                         double const apocenter_time = 1.25 * std::sqrt(apocenter * apocenter * apocenter / Mbh_);

//                         for (size_t i = 0; i < Norg; ++i)
//                         {
//                                 if (fastabs(tess.GetCellCM(i) - tess.GetMeshPoint(i)) > (tess.GetWidth(i) * 0.15))
//                                         continue;
//                                 double r_dist = std::max(fastabs(tess.GetMeshPoint(i)), Rt * smooth_factor);
//                                 if (tess.GetWidth(i) < min_cell_size * (r_dist < (0.65 * Rt) ? smooth_factor / 0.6 : 1))
//                                         continue;

//                                 if (r_dist < (1.25 * Rt))
//                                         continue;

//                                 double MaxMass2 = (tess.GetMeshPoint(i).x > (-apocenter * 2.5)) ? MaxMass : MaxMass * 30;
//                                 MaxMass2 *= std::max(1e-1, std::min(1.0, std::pow(std::abs(time) / apocenter_time, 3.0)));

//                                 double V = tess.GetVolume(i);
//  tess.GetNeighbors(i, neigh);
//                                 size_t Nneigh = neigh.size();
//                                 bool good = true, good2 = false;
//                                 for (size_t j = 0; j < Nneigh; ++j)
//                                 {
//                                         if (!tess.IsPointOutsideBox(neigh[j]))
//                                         {
//                                                 if (fastabs(tess.GetCellCM(neigh[j]) - tess.GetMeshPoint(neigh[j])) > (0.09 * std::pow(volumes[neigh[j]], 0.33333333333)))
//                                                 {
//                                                         good = false;
//                                                         break;
//                                                 }
//                                                 if ((6 * volumes[neigh[j]]) < V)
//                                                         good2 = true;
//                                         }
//                                 }
//                                 if (!good)
//                                         continue;
//                                 if (good2)
//                                 {
//                                         res.push_back(i);
//                                         continue;
// 										 }
//                                 if ((V * cells[i].density) > (MaxMass2 * std::min(std::pow(0.05 * r_dist / Rt, 2.5), 1.0)) || V > domain_size_ * 1e-5)
//                                 {
//                                         {
//                                                 res.push_back(i);
//                                                 continue;
//                                         }
//                                 }
//                         }
//                         return std::pair<vector<size_t>, vector<Vector3D>>(res, vector<Vector3D>());
//                 }
//         };

//  class RemoveBig : public CellsToRemove3D
//         {
//         private:
//                 double domain_size_, Mbh_, Mstar_, Rstar_, beta_;
//                 OndrejEOS const &eos_;

//         public:
//                 void SetSize(double s)
//                 {
//                         domain_size_ = s;
//                 }

//                 RemoveBig(double domain_size, OndrejEOS const &eos, double Mbh, double Mstar, double Rstar, double beta) : domain_size_(domain_size), eos_(e\
// os), Mbh_(Mbh), Mstar_(Mstar), Rstar_(Rstar), beta_(beta) {}

//                 std::pair<vector<size_t>, vector<double>> ToRemove(Tessellation3D const &tess, vector<ComputationalCell3D> const &cells, double time) const
//                 {
//                         std::vector<std::vector<double>> maxr;
//                         std::vector<std::vector<double>> phi;
//                         std::vector<double> theta;
//                         vector<size_t> res;
//                         vector<double> merits;
//                         vector<size_t> neigh;
//                         size_t Norg = tess.GetPointNo();
//                         std::vector<double> volumes = tess.GetAllVolumes();
// #ifdef RICH_MPI
//                         MPI_exchange_data(tess, volumes, true);
// #endif
//                         double const apocenter = Rstar_ * std::pow(Mbh_ / Mstar_, 2.0 / 3.0);
//                         double const Rt = Rstar_ * std::pow(Mbh_ / Mstar_, 1.0 / 3.0) / beta_;
//                         double const time_Rt = std::sqrt(Rt * Rt * Rt / Mbh_);
//                         double min_cell_size = Rt * 1e-2;
//                         double const apocenter_time = 1.25 * std::sqrt(apocenter * apocenter * apocenter / Mbh_);
// 						 int rank = -1;
//                         MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//                         if(rank == 0)
//                                 std::cout<<"Remove apocenter: "<<apocenter<<" time_Rt "<<time_Rt<<" min_cell_size "<<min_cell_size<<std::endl;
//                         double MaxMass = 3.5e-8 * Mstar_;
// #ifdef hi_res
//                         MaxMass *= 0.25;
//                         min_cell_size *= std::pow(0.25, 0.33333);
// #endif
// #ifdef low_res
//                         MaxMass *= 4;
//                         min_cell_size *= std::pow(4.0, 0.33333);
// #endif
//                         for (size_t i = 0; i < Norg; ++i)
//                         {
//                                 bool good = true;
//                                 // Do we have little mass amount?
//                                 if (Norg < 500)
//                                         continue;
//                                 double Vol = tess.GetVolume(i);
//                                 double w = tess.GetWidth(i);
//                                 double MaxMass2 = (tess.GetMeshPoint(i).x > -apocenter * 2.5) ? MaxMass : MaxMass * 30;
//                                 double const r_org = fastabs(tess.GetMeshPoint(i));
//  								double r_i = std::max(Rt * smooth_factor, r_org);
//                                 MaxMass2 *= std::max(1e-1, std::min(1.0, std::pow(std::abs(time) / apocenter_time, 3.0)));
//                                 MaxMass2 = MaxMass2 * std::min(std::pow(0.05 * r_i / Rt, 2.5), 1.0);
//                                 double const dt = w / (eos_.dp2c(cells[i].density, cells[i].pressure, cells[i].tracers) + 0.5 * fastabs(cells[i].velocity));
//                                 double const in_factor = r_i < 0.65 * Rt ? smooth_factor / 0.6 : 1;
//                                 MaxMass2 *= std::max(1.0, std::pow(r_i / r_org, 2.0));
//                                 if (Vol * cells[i].density > MaxMass2 && dt > (0.025 * time_Rt * in_factor))//w > (in_factor * 0.75 * min_cell_size)
//                                         continue;
//                                 if (Vol > domain_size_ * 0.5e-5)
//                                         continue;
//                                 // if(tess.GetMeshPoint(i).x > -8000 && (Vol * cells[i].density) > 5e-8)
//                                 //      std::cout<<"Removing ID "<<cells[i].ID<<" m "<<cells[i].density * Vol <<" w "<<w<<" dt "<<dt<<" MaxMass2 "<<MaxMass2\
// <<std::endl;
//                                 // Make sure we are not that much bigger than smallest neighbor
//                                 tess.GetNeighbors(i, neigh);
//                                 size_t Nneigh = neigh.size();
//                                 for (size_t j = 0; j < Nneigh; ++j)
//                                 {
//                                         if (!tess.IsPointOutsideBox(neigh[j]))
//                                                 if (volumes[neigh[j]] < Vol * 0.4)
//                                                 {
//                                                         good = false;
//                                                         break;
// 												}
// 								}
//                                 if (good)
//                                 {
//                                         // Make sure we are not too high aspect ratio
//                                         if (fastabs(tess.GetMeshPoint(i) - tess.GetCellCM(i)) > 0.15 * tess.GetWidth(i))
//                                                 good = false;
//                                 }
//                                 if (good)
//                                 {
//                                         res.push_back(i);
//                                         merits.push_back(1.0 / Vol);
//                                 }
//                         }
//                         return std::pair<vector<size_t>, vector<double>>(res, merits);
//                 }
//         };




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
			int rank = 0;
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			std::vector<std::vector<double>> maxr;
			std::vector<std::vector<double>> phi;
			std::vector<double> theta;
			size_t Norg = tess.GetPointNo();
			vector<size_t> res;
			double MaxMass = 1.5e-7 * Mstar_;
			double const Rt = Rstar_ * std::pow(Mbh_ / Mstar_, 1.0 / 3.0) / beta_;
			double min_cell_size = Rt * 1e-2;
#ifdef hi_res
			MaxMass *= 0.25;
			min_cell_size *= std::pow(0.25, 0.33333);
#endif
#ifdef low_res
			MaxMass *= 4;
			min_cell_size *= std::pow(4.0, 0.33333);
#endif
			std::vector<size_t> neigh;
			std::vector<double> volumes = tess.GetAllVolumes();
#ifdef RICH_MPI
			MPI_exchange_data(tess, volumes, true);
#endif
			double const apocenter = Rstar_ * std::pow(Mbh_ / Mstar_, 2.0 / 3.0);
			// double const apocenter_time = std::pow(Mbh_ / 1e4, 0.166666) * 1.25 * std::sqrt(apocenter * apocenter * apocenter / Mbh_);
			double rho_s = Mstar_ / (apocenter * apocenter * 10);
			double rho_x = rho_s * 1e-6;
			double target_volume = 4 * M_PI * std::pow(min_cell_size * 2, 3.0) / 3;
			for (size_t i = 0; i < Norg; ++i)
			{
				if(tess.GetMeshPoint(i).x > 0.85 * Rt && cells[i].velocity.x > 0 && cells[i].temperature < 1e7)
					rho_x = std::max(rho_x, cells[i].density);
			}
			MPI_Allreduce(MPI_IN_PLACE, &rho_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			if(rank == 0)
				std::cout << "rho_x = " << rho_x << std::endl;
			for (size_t i = 0; i < Norg; ++i)
			{
				if (fastabs(tess.GetCellCM(i) - tess.GetMeshPoint(i)) > (tess.GetWidth(i) * 0.15))
					continue;
				double r_dist = std::max(fastabs(tess.GetMeshPoint(i)), Rt * smooth_factor);
				if (tess.GetWidth(i) < min_cell_size * (r_dist < (0.65 * Rt) ? smooth_factor / 0.6 : 1))
					continue;
				double const z_abs = std::abs(tess.GetCellCM(i).z);
				double V = tess.GetVolume(i);
				bool first_refine = false;
				if(cells[i].density < 1e-19 && r_dist < 0.5 * apocenter && r_dist > 0.6 * Rt && ((V > 0.01 * z_abs * z_abs * z_abs) || (z_abs < 20)))
				{
					if(V > 4*target_volume * std::max(1.0, std::pow(r_dist / Rt, 1.5)))
						first_refine = true;
				}
				if ((r_dist < (1.75 * Rt) || r_dist > 3 * apocenter) && (not first_refine))
					continue;

				double MaxMass2 = (tess.GetMeshPoint(i).x > (-apocenter * 4.5)) ? MaxMass : MaxMass * 30;
				
				tess.GetNeighbors(i, neigh);
				size_t Nneigh = neigh.size();
				bool good = true, good2 = false;
				for (size_t j = 0; j < Nneigh; ++j)
				{
					if (!tess.IsPointOutsideBox(neigh[j]))
					{
						if (fastabs(tess.GetCellCM(neigh[j]) - tess.GetMeshPoint(neigh[j])) > (0.15 * std::pow(volumes[neigh[j]], 0.33333333333)))
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
				
				if((r_dist < 1.25 * apocenter && cells[i].density > rho_x * 0.01) || (r_dist < 0.5 * apocenter && ((V > 0.01 * z_abs * z_abs * z_abs) || (z_abs < 20))))
				{
					if(V > target_volume * std::pow(r_dist / Rt, 1.5))
					{
						res.push_back(i);
						continue;
					}
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
			MPI_exchange_data(tess, volumes, true);
#endif
			double const apocenter = Rstar_ * std::pow(Mbh_ / Mstar_, 2.0 / 3.0);
			double const Rt = Rstar_ * std::pow(Mbh_ / Mstar_, 1.0 / 3.0) / beta_;
			double const time_Rt = std::sqrt(Rt * Rt * Rt / Mbh_);
			double min_cell_size = Rt * 1e-2;
			double rho_s = Mstar_ / (apocenter * apocenter * 10);
			double rho_x = rho_s * 1e-6;
			double MaxMass = 3.5e-8 * Mstar_;
#ifdef hi_res
			MaxMass *= 0.25;
			min_cell_size *= std::pow(0.25, 0.33333);
#endif
#ifdef low_res
			MaxMass *= 4;
			min_cell_size *= std::pow(4.0, 0.33333);
#endif
			double target_volume = 4 * M_PI * std::pow(1.2 * min_cell_size, 3.0) / 3;
			for (size_t i = 0; i < Norg; ++i)
			{
				if(tess.GetMeshPoint(i).x > Rt * 0.85 && cells[i].velocity.x > 0 && cells[i].temperature < 1e7)
					rho_x = std::max(rho_x, cells[i].density);
			}
			MPI_Allreduce(MPI_IN_PLACE, &rho_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			for (size_t i = 0; i < Norg; ++i)
			{
				bool good = true;
				// Do we have little mass amount?
				if (Norg < 500)
					continue;
				double const r_org = fastabs(tess.GetMeshPoint(i));
				double w = tess.GetWidth(i);
				double Vol = tess.GetVolume(i);
				if(w < 0.6 * min_cell_size || (w < min_cell_size && r_org < 0.58 * Rt))
				{
					res.push_back(i);
					merits.push_back(1.0 / Vol);
					continue;
				}
				if(r_org < 1.75 * Rt && r_org > 0.6 * Rt)
					continue;
				// if(r_org < 3 * Rt && cells[i].temperature < 1e7 && cells[i].velocity.x < -10)
				//     continue;
				double MaxMass2 = (tess.GetMeshPoint(i).x > -apocenter * 4.5) ? MaxMass : MaxMass * 30;
				double r_i = std::max(Rt * smooth_factor, r_org);
				if(r_i < apocenter)
					MaxMass2 = MaxMass2 * std::min(std::pow(0.05 * r_i / Rt, 2.5), 1.0);
				double const dt = w / (eos_.dp2c(cells[i].density, cells[i].pressure, cells[i].tracers) + 0.5 * fastabs(cells[i].velocity));
				double const in_factor = r_i < (0.65 * Rt) ? (smooth_factor / 0.6) : 1;
				// MaxMass2 *= std::max(1.0, std::pow(r_i / r_org, 2.0));
				if (Vol * cells[i].density > MaxMass2 && w > (in_factor * 0.75 * min_cell_size) && dt > (0.02 * time_Rt * in_factor))
					continue;
				double const z_abs = std::abs(tess.GetCellCM(i).z);
				if((r_i < 1.25 * apocenter && cells[i].density > rho_x * 0.01 && r_i > Rt)  || (r_i < 0.5 * apocenter && ((Vol > 0.01 * z_abs * z_abs * z_abs) || z_abs < 20)))
					if(Vol > 4 * target_volume * std::pow(r_i / Rt, 1.5))
					{
						continue;
					}
				if (Vol > domain_size_ * 0.5e-5)
					continue;
				// Make sure we are not that much bigger than smallest neighbor
				tess.GetNeighbors(i, neigh);
				size_t Nneigh = neigh.size();
				for (size_t j = 0; j < Nneigh; ++j)
				{
					if (!tess.IsPointOutsideBox(neigh[j]))
						if (volumes[neigh[j]] < Vol * 0.3)
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
			xsi = read_vector("/home/elads/RICH/data/xsi3.txt");
			theta = read_vector("/home/elads/RICH/data/theta3.txt");
			endfactor = 2.0182359;
		}
		else
		{
			xsi = read_vector("/home/elads/RICH/data/xsi32.txt");
			theta = read_vector("/home/elads/RICH/data/theta32.txt");
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
			double smooth = std::max(Rt * 0.4 / beta_, std::min(Rt / beta_ - R_ * 15, Rt * smooth_factor / beta_));
			// double smooth = Rt * smooth_factor / beta_;
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
	// std::cout<<"Here1"<<std::endl;
	feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
	// std::cout<<"Here2"<<std::endl;
	int rank = 0;
	int ws = 1;
#ifdef RICH_MPI
	MPI_Init(NULL, NULL);
	// std::cout<<"Here3"<<std::endl;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ws);
#endif
	// std::cout<<"Here5"<<std::endl;
	double const R = read_number("Rstar.txt");
	// std::cout<<"Here7"<<std::endl;
	double const M = read_number("Mstar.txt");
	double const n = read_number("n.txt");
	// std::cout<<"Here8"<<std::endl;
	double const Mbh = read_number("Mbh.txt");
	double const beta =  read_number("beta.txt");
	// std::cout<<"Here9"<<std::endl;
	double const Rt = R * std::pow(Mbh / M, 0.333333);
	double const Rp = Rt / beta;
	double const apocenter = Rt * std::pow(Mbh / M, 0.333333);
	std::string const input_snapshot = "/scratch-shared/yhe/rich-data/R0.47M0.5BH100000beta1S60n1.5ComptonHiResNewAMR/snap_full_148.h5";
	bool const restart = true;
	bool const full_gravity = true;
	if(rank == 0)
		std::cout<<"restart from "<<input_snapshot<<std::endl;
	std::string eos_location("/home/esternberg/RICH/data/EOS/");
	std::string STA_location("/home/esternberg/RICH/data/STA/");
	if(rank == 0)
		std::cout<<"Full gravity "<<full_gravity<<std::endl;
	double const lscale = 7e10;
	double const mscale = 2e33;
	double const tscale = 1603;
	if (rank == 0)
		std::cout << "start eos" << std::endl;
	OndrejEOS eos(eos_location + "density.txt", eos_location + "Pfile.txt", eos_location + "csfile.txt", eos_location + "Sfile.txt", eos_location + "Ufile.txt", eos_location + "Tfile.txt", eos_location + "CVfile.txt", lscale, mscale, tscale);
	if (rank == 0)
		std::cout << "end eos" << std::endl;
	//Radiation
	STAgreyOpacity opacity(STA_location);
	// STAMGopacity opacity("/home/elads/RICH/data/STA/MG/");
	if (rank == 0)
		std::cout << "end sta" << std::endl;

	const double width = 5;
	Vector3D ll(-width, -width, -width), ur(width, width, width);
	Voronoi3D tess(ll, ur);

	vector<ComputationalCell3D> cells;
	double tstart = 0;
	Snapshot3D snap;
	if (restart)
	{
		int hdf5_rank = -1;
		int NranksInFile = 1;
#ifdef RICH_MPI
		NranksInFile = GetNumberOfRanksInHDF(input_snapshot);
#endif
		if(rank == 0)
			std::cout<<"Reading from file "<<input_snapshot<<" file has "<<NranksInFile<<" ranks"<<std::endl;
#ifdef RICH_MPI
		if(ws > NranksInFile && rank >= NranksInFile)
		{
			// fake_rank selects an existing file shard for global metadata only
			hdf5_rank = rank % NranksInFile;
			snap = ReadSnapshot3D(input_snapshot, true, hdf5_rank);
			snap.mesh_points.clear();
			snap.cells.clear();
		}
		else
		{
			snap = ReadSnapshot3D(input_snapshot, true, hdf5_rank);
		}
#else
		snap = ReadSnapshot3D(input_snapshot);
#endif

		if(ws < NranksInFile)
		{
			std::mt19937_64 eng(0); // Seed the generator
    		std::uniform_int_distribution<> distr(0, ws - 1); // Define the range
			for(int j = ws; j < NranksInFile; ++j)
			{
				if(distr(eng) != rank)
					continue;
				Snapshot3D snap_temp = ReadSnapshot3D(input_snapshot
#ifdef RICH_MPI
		, true, j
#endif
				);
				snap.cells.insert(snap.cells.end(), snap_temp.cells.begin(), snap_temp.cells.end());
				snap.mesh_points.insert(snap.mesh_points.end(), snap_temp.mesh_points.begin(), snap_temp.mesh_points.end());
			}
		}
		std::cout<<"Rank "<<rank<<" has "<<snap.mesh_points.size()<<" points, hdf5_rank "<<hdf5_rank<<std::endl;
		ll = snap.ll;
		ur = snap.ur;
		if(rank == 0)
			std::cout<<"Box is ll="<<ll<<" ur="<<ur<<std::endl;
		tess.SetBox(snap.ll, snap.ur);
		// tess.SetKernel(new Rectangle(ll, ur));
#ifdef RICH_MPI
		tess.BuildParallel(snap.mesh_points);
		MPI_exchange_data(tess, snap.cells, false);
#else
	tess.Build(snap.mesh_points);
#endif
		cells = snap.cells;
		ComputationalCell3D::tracerNames = snap.tracerstickernames.first;
#ifdef remove_center
		if(ComputationalCell3D::tracerNames.size() < 3)
			ComputationalCell3D::tracerNames.push_back("WasRemoved");
#endif
	}
	else
	{
		double startfactor = 3;
		double fstart = -acos(2 * Rp / (startfactor * Rt) - 1);
		tstart = 0.3333333 * sqrt(2 * Rp * Rp * Rp / Mbh) * tan(0.5 * fstart) * (3 + tan(0.5 * fstart) * tan(0.5 * fstart));
		size_t const np = std::min(1e7, 1e6 * std::sqrt(Mbh / 1e4));
		vector<Vector3D> ptemp;
		if(rank == 0)
		{
			ptemp = RandSphereR1(np, ll, ur, 0, R * 1.1, Vector3D());
			vector<Vector3D> ptemp2 = RandSphereR(np / 2, ll, ur, 0.8 * R, R * 1.05, Vector3D());
			vector<Vector3D> ptemp3 = RandSphereR2(np / 4, ll, ur, R, 1.4 * width, Vector3D());
			ptemp.insert(ptemp.end(), ptemp2.begin(), ptemp2.end());
			ptemp.insert(ptemp.end(), ptemp3.begin(), ptemp3.end());
		}
#ifdef RICH_MPI
		ptemp = MPI_Spread(ptemp, 0, MPI_COMM_WORLD);
#endif
		// try
		{
			vector<Vector3D> points = RoundGrid3D(ptemp, ll, ur, 15);
			if (rank == 0)
				std::cout << "Starting build" << std::endl;
#ifdef RICH_MPI
			tess.BuildParallel(points);
#else
			tess.Build(points);
#endif
			if (rank == 0)
				std::cout << "Finished build" << std::endl;
			cells = GetCells(tess, M, R, eos, tscale * tscale * lscale / mscale, n);
		}
		// catch (UniversalError const &eo)
		// {
		// 	reportError(eo);
		// 	throw;
		// }
		ComputationalCell3D::tracerNames.push_back("Entropy");
		ComputationalCell3D::tracerNames.push_back("Star");
		ComputationalCell3D::tracerNames.push_back("WasRemoved");
	}
	std::cout<<"Rank "<<rank<<" has "<<tess.GetPointNo()<<" points "<<" and "<<cells.size()<<" cells "<<std::endl;

	Hllc3D rs;
	RigidWallGenerator3D ghost;
	LinearGauss3D interp(eos, ghost, true, 0.2, 0.25, 0.75);
	double Tmin = 1e3;

	Lagrangian3D bpm;
	RoundCells3D pm(bpm, eos, 1.75, 0.005, false, 1.75);

	MultigroupDiffusionOpenBoundary D_boundary;
	bool const hydro_on = true;
	bool const compton_on = true;
	bool const flux_limit = true;
	bool const doppler_on = true;
	bool const mixed_frame_on = false;
	bool const protection_on = true;
	DiffusionOpenBoundary d_boundary;
	// MultigroupDiffusion matrix_builder(opacity.energy_groups_center, opacity.energy_groups_boundary, opacity, D_boundary, eos, std::vector<std::string>(), flux_limit, hydro_on, compton_on, doppler_on, mixed_frame_on, 2000, protection_on);
	Diffusion matrix_builder(opacity, d_boundary, eos, std::vector<std::string>(), flux_limit, hydro_on, compton_on);
	matrix_builder.length_scale_ = lscale;
	matrix_builder.time_scale_ = tscale;
	matrix_builder.mass_scale_ = mscale;
	

	// std::shared_ptr<MultigroupDiffusionForce> rad_force = std::make_shared<MultigroupDiffusionForce>(matrix_builder, eos);
	DefaultCellUpdater cu(false, 0, true, 2000, &matrix_builder);

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
	// GravityAcceleration3D sg(1.05, true, 1.0);
	FmmGravityOptions fmmOptions;
	fmmOptions.expansionOrder = 2;
	fmmOptions.thetaCritical = 1.0;
	fmmOptions.leafCapacity = 64;
	FmmDistributedOptions fmmDistributed;
	// The leaf half-size bound was measured not to reduce the LET payload at
	// levels 7, 8 or 9, while costing the sparse ranks ~14x more tree nodes.
	// Bounded LET waves address the payload instead, so leave the bound off.
	fmmDistributed.maxLeafHalfSizeLevel = 0;
	fmmDistributed.enableLeafM2P = false;
	fmmDistributed.maxLetWaveBytes =
		static_cast<std::size_t>(128) * 1024u * 1024u;
	fmmDistributed.persistentLocalTreeTopology = true;
	fmmDistributed.maxRemoteBytes =
		static_cast<std::size_t>(32) * 1024u * 1024u * 1024u;
	FastMultipoleAcceleration3D sg(fmmOptions, fmmDistributed, 1.0);
	TDEGravity acc(Mbh, M, R, beta, sg, not full_gravity);
	std::shared_ptr<DiffusionForce> dforce = std::make_shared<DiffusionForce>(matrix_builder, eos, true);
	std::shared_ptr<ConservativeForce3D> gravity_force = std::make_shared<ConservativeForce3D>(acc, false);
	std::vector<std::shared_ptr<SourceTerm3D>> forces;
	std::shared_ptr<ZeroForce3D> zero_force = std::make_shared<ZeroForce3D>();

	forces.push_back(gravity_force);
	forces.push_back(dforce);
	SeveralSources3D force(forces);
	auto tsf = std::make_shared<CourantFriedrichsLewy>(0.275, 1, force, std::vector<std::string> (), false);

	Simulation simulation(tess, cells, eos, !restart);
	simulation.SetTimeStepFunction(tsf);
	std::unique_ptr<HDSim3D> sim;
	if(restart)
	{
		sim = std::make_unique<HDSim3D>(tess, simulation.getCells(), simulation.getExtensives(), eos, simulation.getTracker(), pm, *tsf, fc, cu, eu, force, std::pair<std::vector<std::string>, std::vector<std::string>> (ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));
		simulation.SetTime(snap.time);
		simulation.SetCycle(snap.cycle);
	}
	else
	{
		sim = std::make_unique<HDSim3D>(tess, simulation.getCells(), simulation.getExtensives(), eos, simulation.getTracker(), pm, *tsf, fc, cu, eu, force, std::pair<std::vector<std::string>, std::vector<std::string>> (ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));
		simulation.SetTime(tstart);
	}
	auto hydroStep = std::make_shared<HydroStep>(*sim, HydroStep::TIMEADVANCE_2);
	auto radStep = std::make_shared<RadiationStep>(tess, simulation.getCells(), simulation.getExtensives(),
		simulation.getTracker(),
#ifdef RICH_MPI
		hydroStep->getCost(),
#endif
		matrix_builder, false);
	simulation.addPhysics(hydroStep);
	simulation.addPhysics(radStep);
	double init_dt = 1e-4;
	simulation.SetTimeStep(init_dt);
	if (rank == 0)
		std::cout << "Restart time " << simulation.GetTime() << std::endl;
	ComputationalCell3D reference_cell = GetReferenceCell(eos, tess, simulation.GetTime());
	double tf = 6 * std::sqrt(apocenter * apocenter * apocenter / Mbh);

	RemoveBig remove(8 * width * width * width, eos, Mbh, M, R, beta);
	MassRefine refine(8 * width * width * width, Mbh, M, R, beta);
	PCM3D ainterp(ghost);
	AMR3D amr(eos, refine, remove, interp);
	std::pair<Vector3D, Vector3D> box2 = sim->getTessellation().GetBoxCoordinates();
	double newvol2 = (box2.second.x - box2.first.x) * (box2.second.y - box2.first.y) * (box2.second.z - box2.first.z);
	refine.SetSize(newvol2);
	remove.SetSize(newvol2);
	
	double old_t = simulation.GetTime();
	double old_dt = init_dt;
	double step_time = 0;
	size_t const end_cycle = snap.cycle + 55;
	
	while (simulation.GetTime() < tf && simulation.GetCycle() < end_cycle)
	{
		if (simulation.GetCycle() % 1 == 0)
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
				std::cout << "Cycle " << simulation.GetCycle() << " Time " << simulation.GetTime() << std::endl;
			}
		}
		try
		{
#ifdef RICH_MPI
			double step_tstart = MPI_Wtime();
			MPI_Barrier(MPI_COMM_WORLD);
#endif
			simulation.step();
			double new_dt = simulation.GetTimeStep();
			new_dt = std::min(0.01, std::max(2.01e-4, new_dt));
			if(new_dt < 0.5 * old_dt)
			    new_dt = 0.5 * old_dt;
			simulation.SetTimeStep(new_dt);
			if (rank == 0)
				std::cout << "Finished rad step" << std::endl;
			if (rank == 0)
				std::cout << "Finished hydro step" << std::endl;
			// if (full_gravity && simulation.GetCycle() % 10 == 0)
			// {
			// 	if(rank == 0)
			// 		std::cout<<"Doing AMR"<<std::endl;
			// 	amr(simulation);
			// }
#ifdef remove_center
			if(full_gravity)
				RemoveCenter(*sim, Mbh, M, R, eos, beta);
#endif
			old_dt = simulation.GetTime() - old_t;
			old_t = simulation.GetTime();
			reference_cell = GetReferenceCell(eos, tess, simulation.GetTime());
			if (simulation.GetCycle() % 7 == 0)
			{
				UpdateBox(tess, simulation, 0.5, 1e-5, reference_cell);
				std::pair<Vector3D, Vector3D> box = sim->getTessellation().GetBoxCoordinates();
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
