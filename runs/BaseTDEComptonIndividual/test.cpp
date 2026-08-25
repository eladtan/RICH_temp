#include "3D/tessellation/Voronoi3D.hpp"
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/simulation/ActiveMeshView.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/HydroStep.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationStep.hpp"
#include "source/newtonian/three_dimensional/SeveralSources3D.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
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
#include "source/Radiation/MultigroupDiffusion.hpp"
#include "source/misc/int2str.hpp"
#include <boost/numeric/odeint.hpp>
#include <boost/math/tools/roots.hpp>
#include <algorithm>
#include <cstdlib>
#include <fenv.h>
#include <filesystem>
namespace fs = std::filesystem;
#include <sstream>
#include "source/newtonian/three_dimensional/Dissipation.hpp"
#include <memory>
#include <limits>
#include <stdexcept>

typedef std::array<double, 4> state_type;

#define smooth_factor 0.5
namespace
{
	void RequireOnEveryRank(bool valid, std::string const& message)
	{
#ifdef RICH_MPI
		int valid_on_every_rank = valid ? 1 : 0;
		MPI_Allreduce(MPI_IN_PLACE, &valid_on_every_rank, 1, MPI_INT, MPI_MIN,
			MPI_COMM_WORLD);
		valid = valid_on_every_rank != 0;
#endif
		if(!valid)
			throw std::logic_error(message);
	}

	std::unique_ptr<ActiveMeshView> MakeActiveMeshView(
		Tessellation3D const& tess, std::size_t canonical_count,
		std::string const& message)
	{
		std::unique_ptr<ActiveMeshView> view;
		bool valid = true;
		try
		{
			view = std::make_unique<ActiveMeshView>(tess, canonical_count);
		}
		catch(...)
		{
			valid = false;
		}
		RequireOnEveryRank(valid, message);
		return view;
	}

	class MeshAlignedStateGuard
	{
	public:
		MeshAlignedStateGuard(HDSim3D& sim, Simulation const& simulation):
			sim_(sim)
		{
			if(!simulation.IndividualStateSynchronized())
				throw std::logic_error(
					"Legacy snapshot requires one synchronized individual state");
			std::vector<ComputationalCell3D>& cells = sim_.getCells();
			std::vector<Conserved3D>& extensives = sim_.getExtensives();
			Tessellation3D const& tess = sim_.getTessellation();
			RequireOnEveryRank(cells.size() == extensives.size(),
				"Snapshot cell and extensive counts differ");
			std::unique_ptr<ActiveMeshView> view = MakeActiveMeshView(
				tess, cells.size(),
				"Cannot map synchronized snapshot state onto the full mesh");
			RequireOnEveryRank(view->localSize() == cells.size(),
				"Synchronized snapshot mesh omits owned cells");

			std::vector<ComputationalCell3D> aligned_cells;
			std::vector<Conserved3D> aligned_extensives;
			view->gatherOwnedInto(cells, aligned_cells);
			view->gatherOwnedInto(extensives, aligned_extensives);
			std::vector<ComputationalCell3D> exchange_cells = cells;
			std::vector<Conserved3D> exchange_extensives = extensives;
			tess.SyncPartialBuildData(aligned_cells, exchange_cells);
			tess.SyncPartialBuildData(aligned_extensives, exchange_extensives);
			cells.swap(aligned_cells);
			extensives.swap(aligned_extensives);
			canonical_cells_.swap(aligned_cells);
			canonical_extensives_.swap(aligned_extensives);
		}

		~MeshAlignedStateGuard()
		{
			sim_.getCells().swap(canonical_cells_);
			sim_.getExtensives().swap(canonical_extensives_);
		}

		MeshAlignedStateGuard(MeshAlignedStateGuard const&) = delete;
		MeshAlignedStateGuard& operator=(MeshAlignedStateGuard const&) = delete;

	private:
		HDSim3D& sim_;
		std::vector<ComputationalCell3D> canonical_cells_;
		std::vector<Conserved3D> canonical_extensives_;
	};

	class RemoveCenter
	{
	public:
		RemoveCenter(HDSim3D& sim, EquationOfState const& eos,
			double MBH, double Mstar, double Rstar, double beta, bool enabled):
			sim_(sim), eos_(eos), enabled_(enabled),
			rt_(Rstar * std::pow(MBH / Mstar, 0.333333333) / beta),
			rsmooth_(std::max(rt_ * 0.4,
				std::min(rt_ - Rstar * 15, rt_ * smooth_factor))),
			sticker_index_(binary_index_find(ComputationalCell3D::stickerNames,
				std::string("InsideRemoveCenter")))
		{}

		void Apply(IndividualStepContext const& context)
		{
			if(!enabled_)
				return;
			Tessellation3D const& tess = sim_.getTessellation();
			std::vector<ComputationalCell3D>& cells = sim_.getCells();
			std::vector<Conserved3D>& extensives = sim_.getExtensives();
			RequireOnEveryRank(
				cells.size() == extensives.size() &&
				context.active_mask.size() == cells.size() &&
				context.cached_accelerations.size() == cells.size() &&
				context.gravity_half_kick_pending.size() == cells.size(),
				"Center sink has inconsistent individual-step arrays");
			std::unique_ptr<ActiveMeshView> view = MakeActiveMeshView(
				tess, cells.size(),
				"Cannot map center-sink targets onto the individual event mesh");
			bool mass_changed = false;
			for(std::size_t local = 0; local < view->localSize(); ++local)
			{
				std::size_t const global = view->localToGlobal(local);
				if(context.isActive(global))
					mass_changed = applyCell(cells[global], extensives[global],
						tess.GetCellCM(local), tess.GetVolume(local)) || mass_changed;
			}
#ifdef RICH_MPI
			int mass_changed_on_any_rank = mass_changed ? 1 : 0;
			MPI_Allreduce(MPI_IN_PLACE, &mass_changed_on_any_rank, 1, MPI_INT,
				MPI_MAX, MPI_COMM_WORLD);
			mass_changed = mass_changed_on_any_rank != 0;
#endif
			if(mass_changed)
			{
				std::fill(context.cached_accelerations.begin(),
					context.cached_accelerations.end(), Vector3D());
				std::fill(context.gravity_half_kick_pending.begin(),
					context.gravity_half_kick_pending.end(), 0);
			}
		}
	private:
		bool applyCell(ComputationalCell3D& cell, Conserved3D& extensive,
			Vector3D const& centroid, double volume) const
		{
			double const radius = fastabs(centroid);
			if(radius < rsmooth_)
			{
				cell.stickers[sticker_index_] = true;
				double const old_density = cell.density;
				double const new_density = std::max(1e-20, old_density * 0.8);
				double const density_ratio = old_density / new_density;
				double const new_temperature =
					std::min(1e7, std::max(1e4, cell.temperature * 0.8));
				cell.tracers[2] *= old_density;
				cell.tracers[2] += old_density - new_density;
				cell.density = new_density;
				cell.tracers[2] /= new_density;
				cell.temperature = new_temperature;
				double const smoothing_fraction =
					std::min(1.0, radius / rsmooth_);
				cell.velocity *=
					1.0 - 0.1 * smoothing_fraction * smoothing_fraction;
				cell.internal_energy = eos_.dT2e(new_density, new_temperature,
					cell.tracers, ComputationalCell3D::tracerNames);
				cell.pressure = eos_.de2p(new_density, cell.internal_energy,
					cell.tracers, ComputationalCell3D::tracerNames);
				cell.tracers[0] = eos_.dp2s(new_density, cell.pressure,
					cell.tracers, ComputationalCell3D::tracerNames);
				cell.Erad *= density_ratio;
				for(std::size_t group = 0; group < ENERGY_GROUPS_NUM; ++group)
					cell.Eg[group] *= density_ratio;
				cell.Erad_dt *= density_ratio;
				cell.Erad_dt_dt *= density_ratio;
				PrimitiveToConserved(cell, volume, extensive);
				return new_density != old_density;
			}

			cell.stickers[sticker_index_] = false;
			if(radius < std::min(rt_ * 0.8, rsmooth_ * 1.5) &&
				cell.temperature > 1e9)
			{
				cell.temperature *= 0.8;
				cell.internal_energy = eos_.dT2e(cell.density, cell.temperature,
					cell.tracers, ComputationalCell3D::tracerNames);
				cell.pressure = eos_.de2p(cell.density, cell.internal_energy,
					cell.tracers, ComputationalCell3D::tracerNames);
				cell.tracers[0] = eos_.dp2s(cell.density, cell.pressure,
					cell.tracers, ComputationalCell3D::tracerNames);
				PrimitiveToConserved(cell, volume, extensive);
			}
			return false;
		}

		HDSim3D& sim_;
		EquationOfState const& eos_;
		bool enabled_;
		double const rt_;
		double const rsmooth_;
		std::size_t const sticker_index_;
	};

	class DissipationDiag: public DiagnosticAppendix3D
	{
		private:
			Dissipation const& dissipation_;
		public:
		DissipationDiag(Dissipation const& dissipation):dissipation_(dissipation){}

		std::vector<double> operator()(const HDSim3D& sim) const
		{
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
		double const beta, Simulation &simulation)
	{
		double const Rt = Rstar * std::pow(MBH / Mstar, 0.333333333);
		double const Rp = Rt / beta;
		state_type x0 = GetTrueAnomaly(simulation.GetTime(), MBH, Rp);
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
		Tessellation3D& tess = sim.getTessellation();
		std::vector<Vector3D> points = tess.accessMeshPoints();
		std::vector<Conserved3D>& canonical_extensives = sim.getExtensives();
		std::vector<ComputationalCell3D>& canonical_cells = sim.getCells();
		std::size_t const N = tess.GetPointNo();
		RequireOnEveryRank(
			points.size() >= N && canonical_cells.size() == N &&
			canonical_extensives.size() == N,
			"Reference-frame change requires one full synchronized mesh");
		points.resize(N);
		std::unique_ptr<ActiveMeshView> view = MakeActiveMeshView(
			tess, N, "Cannot map the synchronized reference-frame state");
		RequireOnEveryRank(view->localSize() == N,
			"Reference-frame mesh omits owned cells");
		std::vector<ComputationalCell3D> cells;
		std::vector<Conserved3D> extensives;
		view->gatherOwnedInto(canonical_cells, cells);
		view->gatherOwnedInto(canonical_extensives, extensives);
		std::pair<Vector3D, Vector3D> box_points = tess.GetBoxCoordinates();
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
		box_points.first.x += x0[0];
		box_points.first.y += x0[1];
		box_points.second.x += x0[0];
		box_points.second.y += x0[1];
		tess.SetBox(box_points.first, box_points.second);
#ifdef RICH_MPI
		tess.BuildParallel(points);
		MPI_exchange_data(tess, extensives, false);
		MPI_exchange_data(tess, cells, false);
#else
		tess.Build(points);
#endif
		RequireOnEveryRank(
			cells.size() == tess.GetPointNo() && extensives.size() == cells.size(),
			"Reference-frame rebuild produced misaligned committed state");
		canonical_cells.swap(cells);
		canonical_extensives.swap(extensives);
		sim.ResetIndividualMeshState();
	}

	void ResetIndividualSchedulerAfterReferenceFrameChange(Simulation& simulation)
	{
		IndividualTimeStepScheduler* scheduler = simulation.GetIndividualTimeStepScheduler();
		if(scheduler == nullptr || !scheduler->initialized())
			throw std::logic_error("Reference-frame change requires initialized individual-timestep state");

		IndividualTimeStepOptions options = scheduler->options();
		double const time_quantum = scheduler->timeQuantum();
#ifdef RICH_MPI
		double minimum_time_quantum = time_quantum;
		double maximum_time_quantum = time_quantum;
		MPI_Allreduce(MPI_IN_PLACE, &minimum_time_quantum, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &maximum_time_quantum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		if(minimum_time_quantum != maximum_time_quantum)
			throw std::logic_error("Reference-frame change requires one shared timestep quantum");
#endif
		double const maximum_reset_dt = std::ldexp(time_quantum, static_cast<int>(options.initial_bin));
		double target_dt = std::min(simulation.GetTimeStep(), maximum_reset_dt);
		int valid_target_dt = std::isfinite(target_dt) && target_dt > 0 ? 1 : 0;
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, &valid_target_dt, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
		if(valid_target_dt == 0)
			throw std::logic_error("Reference-frame change requires a positive finite timestep");
#ifdef RICH_MPI
		// GetTimeStep() is rank-local in individual mode. All ranks must restart
		// the rebuilt mesh at the same event time.
		MPI_Allreduce(MPI_IN_PLACE, &target_dt, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif

		std::uint8_t reset_bin = 0;
		while(reset_bin < options.maximum_bin &&
			std::ldexp(time_quantum, static_cast<int>(reset_bin) + 1) <= target_dt)
			++reset_bin;
#ifdef RICH_MPI
		unsigned int minimum_reset_bin = reset_bin;
		unsigned int maximum_reset_bin = reset_bin;
		MPI_Allreduce(MPI_IN_PLACE, &minimum_reset_bin, 1, MPI_UNSIGNED, MPI_MIN, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &maximum_reset_bin, 1, MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
		if(minimum_reset_bin != maximum_reset_bin)
			throw std::logic_error("Reference-frame change produced inconsistent timestep bins");
		reset_bin = static_cast<std::uint8_t>(minimum_reset_bin);
#endif
		options.initial_bin = reset_bin;
		options.time_quantum = time_quantum;
		double const reset_dt = std::ldexp(time_quantum, static_cast<int>(reset_bin));

		auto const repair_accounting = scheduler->radiationRepairAccounting();
		auto const defect_accounting = scheduler->radiationDefectAccounting();
		int force_all_active = scheduler->forceAllActiveLatched() ? 1 : 0;
#ifdef RICH_MPI
		int minimum_force_all_active = force_all_active;
		int maximum_force_all_active = force_all_active;
		MPI_Allreduce(MPI_IN_PLACE, &minimum_force_all_active, 1, MPI_INT,
			MPI_MIN, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &maximum_force_all_active, 1, MPI_INT,
			MPI_MAX, MPI_COMM_WORLD);
		if(minimum_force_all_active != maximum_force_all_active)
			throw std::logic_error(
				"Reference-frame change found inconsistent force-all-active latches");
		force_all_active = minimum_force_all_active;
#endif
		*scheduler = IndividualTimeStepScheduler(options);
		scheduler->initialize(simulation.getCells(), simulation.GetTime(), reset_dt);
		scheduler->radiationRepairAccounting() = repair_accounting;
		scheduler->radiationDefectAccounting() = defect_accounting;
		scheduler->setForceAllActiveLatched(force_all_active != 0);
	}

	void CheckIfFullGravityIsNeeded(HDSim3D &sim, std::string const& gravity_name, double const Rstar,
		double const Mstar, double const MBH, double const beta, std::string const& restart_name,
		std::string const& individual_restart_name, Simulation &simulation,
		bool& reference_frame_change_pending)
	{
		int rank = 0;
#ifdef RICH_MPI
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
		if(simulation.GetTime() > 5)
		{
			double const Rt = Rstar * std::pow(MBH / Mstar, 0.333333333);
			double const Rp = Rt / beta;
			state_type x0 = GetTrueAnomaly(simulation.GetTime(), MBH, Rp, -3 * Mstar * std::pow(MBH / Mstar, 0.3333333) / Rstar);
			std::vector<ComputationalCell3D> const& cells = sim.getCells();
			IndividualTimeStepScheduler* scheduler =
				simulation.GetIndividualTimeStepScheduler();
			int scheduler_ready = scheduler != nullptr && scheduler->initialized();
#ifdef RICH_MPI
			MPI_Allreduce(MPI_IN_PLACE, &scheduler_ready, 1, MPI_INT, MPI_MIN,
				MPI_COMM_WORLD);
#endif
			if(scheduler_ready == 0)
				throw std::logic_error(
					"Gravity reference-frame change requires initialized individual scheduling");
			std::vector<CellTimeState> const& states = scheduler->states();
			RequireOnEveryRank(states.size() == cells.size(),
				"Gravity reference-frame scan has inconsistent scheduler state");
			std::uint64_t const current_tick = scheduler->currentTick();
			int need_update = 0;
			for(size_t i = 0; i < cells.size(); ++i)
			{
				if(states[i].last_primitive_tick != current_tick)
					continue;
				if(cells[i].density > 1e-14 && (cells[i].velocity.x + x0[2]) > 0 && (cells[i].velocity.y + x0[3]) < 0 && x0[0] < -2 * Rt)
				{
					need_update = 1;
					break;
				}
			}
#ifdef RICH_MPI
			MPI_Allreduce(MPI_IN_PLACE, &need_update, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
			if(rank == 0)
				std::cout<<x0[0]<<","<<x0[1]<<std::endl;
			bool const transition_requested =
				reference_frame_change_pending ||
				(x0[1] > 0.1 && x0[2] > 0.1) || need_update == 1;
			if(transition_requested)
			{
				if(!simulation.IndividualStateSynchronized())
				{
					simulation.RequestSynchronizedIndividualEvent();
					reference_frame_change_pending = true;
					if(rank == 0)
						std::cout << "Deferring gravity reference-frame change until "
							"a synchronized individual event" << std::endl;
					return;
				}

				UpdateReferenceFrame(sim, Rstar, Mstar, MBH, beta, simulation);
				ResetIndividualSchedulerAfterReferenceFrameChange(simulation);
#ifdef RICH_MPI
				MPI_Barrier(MPI_COMM_WORLD);
				std::cout<<"Point number "<<sim.getTessellation().GetPointNo()<<std::endl;
#endif
				vector<DiagnosticAppendix3D *> appendices;
				{
					MeshAlignedStateGuard aligned_state(sim, simulation);
					WriteSnapshot3D(sim, restart_name, appendices, true);
				}
				WriteSimulation(simulation, individual_restart_name);
#ifdef RICH_MPI
				MPI_Barrier(MPI_COMM_WORLD);
				if(rank == 0)
#endif
					write_number(1, gravity_name);
#ifdef RICH_MPI
				if(rank == 0)
					std::cout<<"Done Gravity change"<<std::endl;
				MPI_Barrier(MPI_COMM_WORLD);
#endif
				exit(0);
			}
		}
	}

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
			if(T < T_[0])
				T = T_[0];
			if(T > T_.back())
				T = T_.back();
			if(d < rho_[0])
			{
				d_ratio = cell.density / std::exp(rho_[0]);
				d = rho_[0];
				double const scattering = CalcScatteringOpacity(cell, energy);
				double const sig = std::exp(BiLinearInterpolation(rho_, T_, rossland_[group], d, T)) * d_ratio;
				return CG::speed_of_light / (3 * std::max(sig, scattering));
			}
			if(d > rho_.back())
			{
				d_ratio = cell.density / std::exp(rho_.back());
				d = rho_.back();
			}
			double const sig = std::exp(BiLinearInterpolation(rho_, T_, rossland_[group], d, T)) * d_ratio;
			return CG::speed_of_light / (3 * sig);
		}

		double CalcAbsorptionOpacity(ComputationalCell3D const& cell, double energy) const override
		{
			std::size_t const group = findGroup(energy);
			double T = std::log(cell.temperature);
			double d = std::log(cell.density);
			double d_ratio = 1;
			double d_slope = 2;
			double T_ratio = 1;
			if(d < rho_[0])
			{
				if(T > T_[0] && T < T_.back())
				{
					auto it = std::lower_bound(T_.begin(), T_.end(), T);
					auto idx = std::distance(T_.begin(), it);
					d_slope = (planck_[group][idx][10] - planck_[group][idx][0]) / (rho_[10] - rho_[0]);
				}
				d_ratio = cell.density / std::exp(rho_[0]);
				d = rho_[0];
			}
			if(d > rho_.back())
			{
				d_ratio = cell.density / std::exp(rho_.back());
				d = rho_.back();
			}
			if(T < T_[0])
				T = T_[0];
			if(T > T_.back())
			{
				T_ratio = std::pow(cell.temperature / std::exp(T_.back()), -1.5);
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
				d_ratio = cell.density / std::exp(rho_[0]);
				d = rho_[0];
			}
			if(d > rho_.back())
			{
				d_ratio = cell.density / std::exp(rho_.back());
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
#ifdef RICH_MPI
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
			std::vector<std::vector<double>> maxr;
			std::vector<std::vector<double>> phi;
			std::vector<double> theta;
			size_t Norg = tess.GetPointNo();
			vector<size_t> res;
			double MaxMass = 1.5e-7 * Mstar_;
			double const Rt = Rstar_ * std::pow(Mbh_ / Mstar_, 1.0 / 3.0) / beta_;
			double min_cell_size = Rt * 1e-2;
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
			double rho_s = Mstar_ / (apocenter * apocenter * 10);
			double rho_x = rho_s * 1e-6;
			double target_volume = 4 * M_PI * std::pow(min_cell_size * 2, 3.0) / 3;
			for (size_t i = 0; i < Norg; ++i)
			{
				if(tess.GetMeshPoint(i).x > 0.85 * Rt && cells[i].velocity.x > 0 && cells[i].temperature < 1e7)
					rho_x = std::max(rho_x, cells[i].density);
			}
#ifdef RICH_MPI
			MPI_Allreduce(MPI_IN_PLACE, &rho_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
			double min_low_density_volume = std::min(8000.0, std::max(2000.0, 2000.0 * (rho_x / (rho_s * 5e-4))));
			double min_density_factor = min_low_density_volume / 2000.0;
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
				if ((r_dist < (1.5 * Rt) || r_dist > 3 * apocenter) && (not first_refine))
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

				if((r_dist < 1.25 * apocenter && cells[i].density > rho_x * 0.01))
				{
					if(V > std::min(200.0, target_volume * std::max(1.0, std::pow(0.5 * r_dist / Rt, 1.0))))
					{
						res.push_back(i);
						continue;
					}
				}
				if((r_dist < 0.5 * apocenter && ((V > 0.01 * z_abs * z_abs * z_abs) || (z_abs < 20))))
				{
					if(V > std::min(min_low_density_volume, min_density_factor * 4 * target_volume * std::pow(r_dist / Rt, 1.5)))
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
			double const smooth = Rt * smooth_factor / beta_;
			double const time_Rt = std::sqrt(Rt * Rt * Rt / Mbh_);
			double const apocenter_time = std::sqrt(apocenter * apocenter * apocenter / Mbh_);
			double min_cell_size = Rt * 1e-2;
			double rho_s = Mstar_ / (apocenter * apocenter * 10);
			double rho_x = rho_s * 1e-6;
			double MaxMass = 3.5e-8 * Mstar_ * std::min(1.0, std::pow(time / apocenter_time, 2.0));
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
#ifdef RICH_MPI
			MPI_Allreduce(MPI_IN_PLACE, &rho_x, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
			double min_low_density_volume = std::min(2000.0, std::max(500.0, 500.0 * (rho_x / (rho_s * 5e-4))));
			double min_density_factor = min_low_density_volume / 500.0;
			for (size_t i = 0; i < Norg; ++i)
			{
				bool good = true;
				// Do we have little mass amount?
				if (Norg < 500)
					continue;
				double d_CM = fastabs(tess.GetMeshPoint(i) - tess.GetCellCM(i));
				double const r_org = fastabs(tess.GetMeshPoint(i));
				double w = tess.GetWidth(i);
				double Vol = tess.GetVolume(i);
				bool shape_ok = (d_CM < 0.2 * w);
				if(!shape_ok && r_org < smooth && cells[i].density < 1e-18)
					shape_ok = true;
				if((w < 0.6 * min_cell_size || (w < min_cell_size && r_org < 0.58 * Rt)) && shape_ok)
				{
					res.push_back(i);
					merits.push_back(1.0 / Vol);
					continue;
				}
				if(r_org < 1.75 * Rt && r_org > 0.6 * Rt)
					continue;
				double MaxMass2 = (tess.GetMeshPoint(i).x > -apocenter * 4.5) ? MaxMass : MaxMass * 30;
				double r_i = std::max(Rt * smooth_factor, r_org);
				if(r_i < apocenter)
					MaxMass2 = MaxMass2 * std::min(std::pow(0.05 * r_i / Rt, 2.5), 1.0);
				double const dt = w / (eos_.dp2c(cells[i].density, cells[i].pressure, cells[i].tracers) + 0.3 * fastabs(cells[i].velocity));
				double const in_factor = r_i < (0.65 * Rt) ? (smooth_factor / 0.6) : 1;
				if (Vol * cells[i].density > MaxMass2 && w > (in_factor * 0.25 * min_cell_size) && dt > (0.02 * time_Rt * in_factor))
					continue;
				double const z_abs = std::abs(tess.GetCellCM(i).z);
				if((r_i < 1.25 * apocenter && cells[i].density > rho_x * 0.01 && r_i > Rt))
				{
					if(Vol > std::min(50.0, target_volume * std::max(1.0, std::pow(0.5 * r_i / Rt, 1.0))))
					{
						continue;
					}
				}
				if((r_i < 0.5 * apocenter && ((Vol > 0.01 * z_abs * z_abs * z_abs) || z_abs < 20)))
				{
					if(Vol > std::min(min_low_density_volume, min_density_factor * 4 * target_volume * std::pow(r_i / Rt, 1.5)))
					{
						continue;
					}
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
					if (d_CM > 0.15 * w && !(r_org < smooth && cells[i].density < 1e-18))
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

	ComputationalCell3D GetReferenceCell(OndrejEOS const &eos, Tessellation3D const &tess, double time, std::vector<double> const& energy_groups_boundary)
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
		size_t const Ng = energy_groups_boundary.size() - 1;
		double const Erad_factor = boost::math::pow<4>(Tref / Tgas) / Ng;
		for(size_t g = 0; g < Ng; ++g)
			reference.Eg[g] = Erad_factor * planck_integral::planck_energy_density_group_integral(energy_groups_boundary[g], energy_groups_boundary[g+1], Tgas) * 1603 * 1603 * 7e10 / (2e33 * reference.density);
		reference.Erad = 7.5657e-15 * Tref * Tref * Tref * Tref * 1603 * 1603 * 7e10 / (2e33 * reference.density);
		reference.pressure = eos.dT2p(reference.density, Tgas, reference.tracers);
		reference.velocity = Vector3D();
		reference.internal_energy = eos.dp2e(reference.density, reference.pressure, reference.tracers);
		reference.temperature = Tgas;
		reference.tracers[0] = (eos.dp2s(reference.density, reference.pressure, reference.tracers));
		reference.tracers[1] = (0);
		reference.tracers[3] = (0);
		return reference;
	}

	vector<ComputationalCell3D> GetCells(Tessellation3D const &tess, double M, double R, OndrejEOS const &eos, double const Punits, double const n, std::vector<double> const& energy_groups_boundary)
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
		ComputationalCell3D reference = GetReferenceCell(eos, tess, 0, energy_groups_boundary);
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
				size_t const Ng = energy_groups_boundary.size() - 1;
				for(size_t g = 0; g < Ng; ++g)
					res[i].Eg[g] = planck_integral::planck_energy_density_group_integral(energy_groups_boundary[g], energy_groups_boundary[g+1], T) * 1603 * 1603 * 7e10 / (2e33 * res[i].density);
				res[i].temperature = T;
			}
			else
			{
				res[i] = reference;
				res[i].tracers[1] = (0);
			}
			res[i].tracers[0] = (eos.dp2s(res[i].density, res[i].pressure, res[i].tracers));
			res[i].tracers[3] = (0);
		}
		return res;
	}

	class TDEGravity : public Acceleration3D
	{
	private:
		struct TidalFieldState
		{
			Vector3D center_position;
			Vector3D center_acceleration;
			double gravitational_radius;
			double smooth_radius;
			double transition_lower;
			double transition_upper;
		};

		Acceleration3D const &selfgravity_;
		const double Mbh_, M_, R_, beta_;

		static void RequireOnEveryRank(bool valid, char const* message)
		{
#ifdef RICH_MPI
			int valid_on_every_rank = valid ? 1 : 0;
			MPI_Allreduce(MPI_IN_PLACE, &valid_on_every_rank, 1, MPI_INT,
				MPI_MIN, MPI_COMM_WORLD);
			valid = valid_on_every_rank != 0;
#endif
			if(!valid)
				throw std::logic_error(message);
		}

		TidalFieldState MakeTidalFieldState(double time) const
		{
			TidalFieldState state;
			state.center_position = Vector3D(0, 0, 0);
			state.center_acceleration = Vector3D(0, 0, 0);
			state.gravitational_radius = 4.21 * Mbh_ / 1e6;
			double const tidal_radius = R_ * std::pow(Mbh_ / M_, 0.333333333);
			state.smooth_radius = tidal_radius * smooth_factor / beta_;
			double const transition_half_width = 0.2 * state.smooth_radius;
			state.transition_lower = state.smooth_radius - transition_half_width;
			state.transition_upper = state.smooth_radius + transition_half_width;
			if(tide_on_)
			{
				double const pericenter = tidal_radius / beta_;
				state_type const anomaly = GetTrueAnomaly(time, Mbh_, pericenter);
				double const radius = std::sqrt(
					anomaly[0] * anomaly[0] + anomaly[1] * anomaly[1]);
				state.center_position = Vector3D(anomaly[0], anomaly[1], 0);
				state.center_acceleration = -Mbh_ * Vector3D(
					anomaly[0] / (radius * (radius - state.gravitational_radius) *
						(radius - state.gravitational_radius)),
					anomaly[1] / (radius * (radius - state.gravitational_radius) *
						(radius - state.gravitational_radius)), 0);
			}
			return state;
		}

		Vector3D TidalAcceleration(
			Vector3D const& point, TidalFieldState const& state) const
		{
			Vector3D const full_point = point + state.center_position;
			double const radius = std::max(
				abs(full_point), 4 * state.gravitational_radius);
			auto field_at_radius = [&](double evaluation_radius)
			{
				return -(Mbh_ / (evaluation_radius *
					(evaluation_radius - state.gravitational_radius) *
					(evaluation_radius - state.gravitational_radius))) * full_point;
			};
			if(radius >= state.transition_upper)
				return field_at_radius(radius) - state.center_acceleration;
			if(radius <= state.transition_lower)
				return field_at_radius(state.smooth_radius) -
					state.center_acceleration;

			double const fraction = (radius - state.transition_lower) /
				(state.transition_upper - state.transition_lower);
			double const blend = fraction * fraction * (3.0 - 2.0 * fraction);
			return (1.0 - blend) * field_at_radius(state.smooth_radius) +
				blend * field_at_radius(radius) - state.center_acceleration;
		}

		double MinimumDensity(
			std::pair<Vector3D, Vector3D> const& bounds) const
		{
			Vector3D const extent = bounds.second - bounds.first;
			return std::max(1e-19, 1e-10 * M_ /
				(extent.x * extent.z * extent.y));
		}

	public:
		const bool tide_on_;

		TDEGravity(double Mbh, double M, double R, double beta, Acceleration3D const &sg, bool tide) : selfgravity_(sg), Mbh_(Mbh), M_(M), R_(R), beta_(beta), tide_on_(tide) {}

		void operator()(const Tessellation3D &tess, const vector<ComputationalCell3D> &cells,
						const vector<Conserved3D> &fluxes, const double time, vector<Vector3D> &acc) const
		{
			selfgravity_(tess, cells, fluxes, time, acc);
			RequireOnEveryRank(cells.size() >= acc.size(),
				"TDE gravity cell array is smaller than its acceleration array");
			std::pair<Vector3D, Vector3D> const bounds = tess.GetBoxCoordinates();
			TidalFieldState const field = MakeTidalFieldState(time);
			double const minimum_density = MinimumDensity(bounds);
			for (size_t i = 0; i < acc.size(); ++i)
			{
				acc[i] += TidalAcceleration(tess.GetCellCM(i), field);
				if (cells[i].density < minimum_density || cells[i].tracers[1] < 0.1)
					acc[i] = Vector3D(0, 0, 0);
			}
		}

		bool SupportsIndividualTargetEvaluation(void) const override
		{
			return selfgravity_.SupportsIndividualTargetEvaluation();
		}

		void EvaluateIndividualTargets(
			std::pair<Vector3D, Vector3D> const& bounds,
			vector<Vector3D> const& source_points,
			vector<double> const& source_masses,
			vector<std::uint64_t> const& source_ids,
			vector<Vector3D> const& target_points,
			vector<ComputationalCell3D> const& target_cells,
			double time,
			vector<Vector3D>& acc) const override
		{
			RequireOnEveryRank(target_cells.size() == target_points.size(),
				"TDE gravity target cell and point counts differ");
			selfgravity_.EvaluateIndividualTargets(bounds, source_points,
				source_masses, source_ids, target_points, target_cells, time, acc);
			RequireOnEveryRank(acc.size() == target_points.size(),
				"TDE self-gravity returned the wrong target count");

			TidalFieldState const field = MakeTidalFieldState(time);
			double const minimum_density = MinimumDensity(bounds);
			for(size_t target = 0; target < target_points.size(); ++target)
			{
				acc[target] += TidalAcceleration(target_points[target], field);
				if(target_cells[target].density < minimum_density ||
					target_cells[target].tracers[1] < 0.1)
					acc[target] = Vector3D(0, 0, 0);
			}
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
	char const* configured_run_directory = std::getenv("RICH_TDE_RUN_DIRECTORY");
	std::string run_directory =
		configured_run_directory != nullptr && configured_run_directory[0] != '\0'
		? configured_run_directory : ".";
	if(run_directory.back() != '/')
		run_directory += '/';
	double const R = read_number("Rstar.txt");
	double const M = read_number("Mstar.txt");
	double const n = read_number("n.txt");
	double const Mbh = read_number("Mbh.txt");
	double const beta =  read_number("beta.txt");
	std::stringstream ss;
	ss<<"R"<<R<<"M"<<M<<"BH"<<Mbh<<"beta"<<beta<<"S"<<static_cast<size_t>(smooth_factor*100)<<"n"<<n<<"Compton";
	if(rank == 0)
		std::cout<<"Creating directory "<<ss.str()<<std::endl;
	std::string const run_name = ss.str();
	run_directory += run_name + "/";
	fs::create_directories(run_directory.c_str());
	double const Rt = R * std::pow(Mbh / M, 0.333333);
	double const Rp = Rt / beta;
	double const apocenter = Rt * std::pow(Mbh / M, 0.333333);
	std::string file_name = run_directory + "snap_";
	std::string restart_name = run_directory + "restart.h5";
	std::string const normal_individual_restart_name = run_directory + "individual_restart.h5";
	std::string const full_individual_restart_name = run_directory + "individual_full_restart.h5";
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
	std::string eos_location("../../data/EOS/");
	bool const full_gravity = fs::exists(gravity_name);
	if(full_gravity)
		std::filesystem::last_write_time(gravity_name, std::filesystem::file_time_type::clock::now());
	std::string const individual_restart_name = full_gravity ?
		full_individual_restart_name : normal_individual_restart_name;
	if(restart && !fs::exists(individual_restart_name))
		throw UniversalError("Missing individual-timestep restart checkpoint: " + individual_restart_name);
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
	if (rank == 0)
		std::cout << "start eos" << std::endl;
	OndrejEOS eos(eos_location + "density.txt", eos_location + "Pfile.txt", eos_location + "csfile.txt", eos_location + "Sfile.txt", eos_location + "Ufile.txt", eos_location + "Tfile.txt", eos_location + "CVfile.txt", lscale, mscale, tscale);
	if (rank == 0)
		std::cout << "end eos" << std::endl;
	//Radiation
	STAMGopacity opacity("/home/elads/RICH/data/STA/MG/");
	if (rank == 0)
		std::cout << "end sta" << std::endl;

	const double width = 5;
	Vector3D ll(-width, -width, -width), ur(width, width, width);
	Voronoi3D tess(ll, ur);

	vector<ComputationalCell3D> cells;
	double tstart = 0, t_restart = -100;
	Snapshot3D snap;
	if (restart)
	{
		int hdf5_rank = -1;
		int NranksInFile = 1;
#ifdef RICH_MPI
		NranksInFile = GetNumberOfRanksInHDF(file_name + int2str(counter) + ".h5");
#endif
		if(rank == 0)
			std::cout<<"Reading from file "<<file_name + int2str(counter) + ".h5 file has "<<NranksInFile<<" ranks"<<std::endl;
		snap = ReadSnapshot3D(file_name + int2str(counter) + ".h5"
#ifdef RICH_MPI
		, true, hdf5_rank
#endif
		);
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
#ifdef RICH_MPI
		tess.BuildParallel(snap.mesh_points);
		MPI_exchange_data(tess, snap.cells, false);
#else
	tess.Build(snap.mesh_points);
#endif
		cells = snap.cells;
		ComputationalCell3D::tracerNames = snap.tracerstickernames.first;
		ComputationalCell3D::stickerNames = snap.tracerstickernames.second;
		if(ComputationalCell3D::tracerNames.size() < 3)
			ComputationalCell3D::tracerNames.push_back("WasRemoved");
		if(std::find(ComputationalCell3D::stickerNames.begin(),
			ComputationalCell3D::stickerNames.end(), "InsideRemoveCenter") ==
			ComputationalCell3D::stickerNames.end())
		{
			ComputationalCell3D::stickerNames.push_back("InsideRemoveCenter");
			std::size_t const index = ComputationalCell3D::stickerNames.size() - 1;
			for(ComputationalCell3D& cell : cells)
				cell.stickers[index] = false;
		}
	}
	else
	{
		double startfactor = 3;
		double fstart = -acos(2 * Rp / (startfactor * Rt) - 1);
		tstart = 0.3333333 * sqrt(2 * Rp * Rp * Rp / Mbh) * tan(0.5 * fstart) * (3 + tan(0.5 * fstart) * tan(0.5 * fstart));
		size_t const np = std::max(1e6, std::min(1e7, 1e6 * std::sqrt(Mbh / 1e4)));
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
		try
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
			cells = GetCells(tess, M, R, eos, tscale * tscale * lscale / mscale, n, opacity.energy_groups_boundary);
		}
		catch (UniversalError const &eo)
		{
			reportError(eo);
			throw;
		}
		ComputationalCell3D::tracerNames.push_back("Entropy");
		ComputationalCell3D::tracerNames.push_back("Star");
		ComputationalCell3D::tracerNames.push_back("WasRemoved");
		ComputationalCell3D::stickerNames.push_back("InsideRemoveCenter");
		{
			std::size_t const index = ComputationalCell3D::stickerNames.size() - 1;
			for(ComputationalCell3D& cell : cells)
				cell.stickers[index] = false;
		}
	}
	std::cout<<"Rank "<<rank<<" has "<<tess.GetPointNo()<<" points "<<" and "<<cells.size()<<" cells "<<std::endl;

	Hllc3D rs;
	RigidWallGenerator3D ghost;
	LinearGauss3D interp(eos, ghost, true, 0.2, 0.25, 0.75);
	Lagrangian3D bpm;
	RoundCells3D pm(bpm, eos, 1.75, 0.005, false, 1.25, 0.01, std::vector<std::string>(), 150);

	MultigroupDiffusionOpenBoundary D_boundary;
	bool const hydro_on = true;
	bool const compton_on = true;
	bool const flux_limit = true;
	bool const doppler_on = true;
	bool const protection_on = true;
	std::vector<std::string> rad_zero_cells({"InsideRemoveCenter"});
	MultigroupDiffusion matrix_builder(opacity.energy_groups_center, opacity.energy_groups_boundary, opacity, D_boundary, eos, rad_zero_cells, flux_limit, hydro_on, compton_on, doppler_on, 2000, protection_on, true);
	matrix_builder.length_scale_ = lscale;
	matrix_builder.time_scale_ = tscale;
	matrix_builder.mass_scale_ = mscale;

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
	FmmGravityOptions fmm_options;
	fmm_options.expansionOrder = 2;
	fmm_options.thetaCritical = 1.0;
	fmm_options.leafCapacity = 64;
	FastMultipoleAcceleration3D sg(fmm_options, 1.0);
	TDEGravity acc(Mbh, M, R, beta, sg, not full_gravity);
	std::shared_ptr<ConservativeForce3D> gravity_force = std::make_shared<ConservativeForce3D>(acc, false);
	std::vector<std::shared_ptr<SourceTerm3D>> forces;
	forces.push_back(gravity_force);
	SeveralSources3D force(forces);
	auto tsf = std::make_shared<CourantFriedrichsLewy>(0.4, 1, force, std::vector<std::string> (), false);

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
	RemoveCenter center_sink(*sim, eos, Mbh, M, R, beta, full_gravity);
	simulation.addPhysics(hydroStep);
	simulation.addPhysics(radStep);
	simulation.SetIndividualPostPhysics(
		[&](IndividualStepContext const& context)
		{
			center_sink.Apply(context);
		});
	double init_dt = 1e-4;
	simulation.SetTimeStep(init_dt);
	if(restart)
	{
		ReadSimulation(individual_restart_name, simulation);
		if(simulation.GetTimeIntegrationMode() != TimeIntegrationMode::Individual)
			throw UniversalError("Restart checkpoint has no individual-timestep state: " + individual_restart_name);
		t_restart = simulation.GetTime();
	}
	else
	{
		IndividualTimeStepOptions individual_options;
		individual_options.initial_bin = 30;
		individual_options.maximum_bin = 40;
		individual_options.maximum_neighbor_bin_difference = 2;
		individual_options.mesh_build_policy = IndividualMeshBuildPolicy::AutoPartial;
		simulation.EnableIndividualTimeSteps(individual_options);
	}
	if (rank == 0)
		std::cout << "Restart time " << simulation.GetTime() << std::endl;
	double tf = 6 * std::sqrt(apocenter * apocenter * apocenter / Mbh);
	double mindt = 0.001;
	double nextT = 0;
	nextT = (t_restart < -20) ? simulation.GetTime() : t_restart;
	nextT += std::min(50.0, mindt + 0.2 * std::pow(std::abs(simulation.GetTime()), 0.666666));
	nextT = std::max(nextT, simulation.GetTime() + 0.01);

	RemoveBig remove(8 * width * width * width, eos, Mbh, M, R, beta);
	MassRefine refine(8 * width * width * width, Mbh, M, R, beta);
	AMR3D amr(eos, refine, remove, interp);
	std::pair<Vector3D, Vector3D> box2 = sim->getTessellation().GetBoxCoordinates();
	double newvol2 = (box2.second.x - box2.first.x) * (box2.second.y - box2.first.y) * (box2.second.z - box2.first.z);
	refine.SetSize(newvol2);
	remove.SetSize(newvol2);
	// Legacy UpdateBox rebuilds conserved state outside the individual scheduler;
	// keep this run's domain fixed until box growth has a conservative remap.
	simulation.SetIndividualAMR(
		[&](IndividualStepContext const& context)
		{
			if(!full_gravity || (simulation.GetCycle() + 1) % 10 != 0)
				return IndividualAMRChangeSet();
			if(rank == 0)
				std::cout << "Doing individual AMR" << std::endl;
			return amr.ApplyIndividual(simulation, context);
		});
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

	double old_t = simulation.GetTime();
	double old_dt = init_dt;
	bool reference_frame_change_pending = false;
	bool regular_output_pending = false;
#ifdef RICH_MPI
	bool restart_output_pending = false;
#endif
	double step_time = 0;
	double const restart_wtime = 10000;
	double const min_dt_output = 0.025 * std::sqrt(std::pow(R, 3.0) * Mbh / M);
	auto write_synchronized_snapshot = [&](std::string const& output_name)
	{
		MeshAlignedStateGuard aligned_state(*sim, simulation);
		interp(tess, sim->getCells(), 0, dissipation.face_values);
		WriteSnapshot3D(*sim, output_name, appendices, true);
	};
	while (simulation.GetTime() < tf)
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
		if(!reference_frame_change_pending &&
			(regular_output_pending || simulation.GetTime() > nextT))
		{
			regular_output_pending = true;
			if(!simulation.IndividualStateSynchronized())
			{
				simulation.RequestSynchronizedIndividualEvent();
				if(rank == 0)
					std::cout << "Deferring output until a synchronized individual event"
						<< std::endl;
			}
			else
			{
				std::string const output_name =
					file_name + int2str(counter) + ".h5";
				if(rank == 0)
					std::cout << "Starting writing file " << output_name << std::endl;
				write_synchronized_snapshot(output_name);
				WriteSimulation(simulation, individual_restart_name);
				if(rank == 0)
					write_int(counter, counter_name);
				nextT = simulation.GetTime() + std::min(min_dt_output,
					mindt + 0.2 * std::pow(std::abs(simulation.GetTime()), 0.666666));
				++counter;
				regular_output_pending = false;
				dissipation.face_values.clear();
				dissipation.face_values.shrink_to_fit();
			}
		}
		try
		{
			int restart_dump = 0;
#ifdef RICH_MPI
			if(!reference_frame_change_pending && rank == 0)
			{
				if (MPI_Wtime() - last_start > restart_wtime)
					restart_dump = 1;
			}
			MPI_Bcast(&restart_dump, 1, MPI_INT, 0, MPI_COMM_WORLD);
			if(restart_dump == 1)
				restart_output_pending = true;
			if(!reference_frame_change_pending && restart_output_pending)
			{
				if(!simulation.IndividualStateSynchronized())
				{
					simulation.RequestSynchronizedIndividualEvent();
					if(rank == 0)
						std::cout << "Deferring restart until a synchronized individual event"
							<< std::endl;
				}
				else
				{
					std::string const output_name = run_directory + "restart.h5";
					if(rank == 0)
						std::cout << "Starting writing file " << output_name << std::endl;
					write_synchronized_snapshot(output_name);
					WriteSimulation(simulation, individual_restart_name);
					dissipation.face_values.clear();
					dissipation.face_values.shrink_to_fit();
					restart_output_pending = false;
					last_start = MPI_Wtime();
				}
			}
			double step_tstart = MPI_Wtime();
#endif
#ifdef RICH_MPI
			MPI_Barrier(MPI_COMM_WORLD);
			#endif
				simulation.step();
				old_dt = simulation.GetTime() - old_t;
			old_t = simulation.GetTime();
			if(not full_gravity)
				CheckIfFullGravityIsNeeded(*sim, gravity_name, R, M, Mbh, beta, restart_name,
					full_individual_restart_name, simulation,
					reference_frame_change_pending);
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
