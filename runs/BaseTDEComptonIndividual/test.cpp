#include <chrono>
#include "3D/tessellation/Voronoi3D.hpp"
#include "source/3D/GeometryCommon/RoundGrid3D.hpp"
#include "source/3D/GeometryCommon/UpdateBox.hpp"
#include <iomanip>
#include <functional>
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/simulation/RuntimeLog.hpp"
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
#include "source/Radiation/Diffusion.hpp"
#include "source/Radiation/STAgreyOpacity.hpp"
#include "source/misc/int2str.hpp"
#include <boost/numeric/odeint.hpp>
#include <boost/math/tools/roots.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fenv.h>
#include <filesystem>
namespace fs = std::filesystem;
#include <fstream>
#include <sstream>
#include "source/newtonian/three_dimensional/Dissipation.hpp"
#include <memory>
#include <limits>
#include <stdexcept>

typedef std::array<double, 4> state_type;

#define smooth_factor 0.5
namespace
{
	void WriteIntegerControlFileAtomically(int value, std::string const& filename)
	{
		fs::path const destination(filename);
		fs::path temporary(destination);
		temporary += ".tmp";
		std::ofstream output(temporary, std::ios::trunc);
		if(!output)
			throw std::runtime_error("Could not open control-file temporary: " +
				temporary.string());
		output << value << '\n';
		output.close();
		if(!output)
			throw std::runtime_error("Could not write control-file temporary: " +
				temporary.string());
		fs::rename(temporary, destination);
	}

	int ReadIntegerControlFile(std::string const& filename)
	{
		std::ifstream input(filename);
		int value = 0;
		if(!(input >> value))
			throw std::runtime_error("Could not read integer control file: " + filename);
		input >> std::ws;
		if(!input.eof())
			throw std::runtime_error("Invalid trailing data in integer control file: " +
				filename);
		return value;
	}

	bool IsNumberedSnapshotArtifact(fs::path const& path)
	{
		std::string stem = path.filename().string();
		for(std::string const& extension : {std::string(".h5"),
			std::string(".vtu"), std::string(".pvtu")})
		{
			if(stem.size() > extension.size() &&
				stem.compare(stem.size() - extension.size(), extension.size(), extension) == 0)
			{
				stem.resize(stem.size() - extension.size());
				break;
			}
		}
		std::string const prefix = stem.rfind("snap_full_", 0) == 0 ?
			"snap_full_" : "snap_";
		if(stem.rfind(prefix, 0) != 0 || stem.size() == prefix.size())
			return false;
		return std::all_of(stem.begin() + prefix.size(), stem.end(),
			[](char character) { return character >= '0' && character <= '9'; });
	}

	bool IsOldRunArtifact(fs::path const& path)
	{
		std::string const name = path.filename().string();
		return IsNumberedSnapshotArtifact(path) ||
			name == "initial" || name == "initial.h5" ||
			name == "initial.vtu" || name == "initial.pvtu" ||
			name == "restart" || name == "restart.h5" ||
			name == "restart.vtu" || name == "restart.pvtu" ||
			name == "individual_restart" ||
			name == "individual_restart.h5" ||
			name == "individual_full_restart" ||
			name == "individual_full_restart.h5";
	}

	void RemoveOldRunArtifacts(std::string const& run_directory)
	{
		for(fs::directory_entry const& entry : fs::directory_iterator(run_directory))
		{
			if(IsOldRunArtifact(entry.path()))
				fs::remove_all(entry.path());
		}
	}

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

	// RICH_TDE_WRITE_VTU=0 skips the per-rank ParaView files (snap_N/*.vtu and
	// snap_N.pvtu, about 2.5x the size of the HDF5 snapshot) written next to
	// every snapshot; unset, empty or 1 keeps them.  Parsed collectively on the first
	// call, which main makes at start-up.
	bool TdeWriteVtu(void)
	{
		static int state = -1;
		if(state < 0)
		{
			char const* const configured = std::getenv("RICH_TDE_WRITE_VTU");
			std::string const value = configured == nullptr ? std::string() :
				std::string(configured);
			RequireOnEveryRank(value.empty() || value == "0" || value == "1",
				"RICH_TDE_WRITE_VTU must be 0 or 1");
			int lowest = value == "0" ? 0 : 1;
			int highest = lowest;
#ifdef RICH_MPI
			MPI_Allreduce(MPI_IN_PLACE, &lowest, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
			MPI_Allreduce(MPI_IN_PLACE, &highest, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
			RequireOnEveryRank(lowest == highest,
				"RICH_TDE_WRITE_VTU differs across MPI ranks");
			state = lowest;
		}
		return state == 1;
	}

	// A strict 0/1 run switch (unset or empty = 0) that must agree on every rank.
	// Collective.
	bool TdeSwitch(char const* name)
	{
		char const* const configured = std::getenv(name);
		std::string const value = configured == nullptr ? std::string() :
			std::string(configured);
		RequireOnEveryRank(value.empty() || value == "0" || value == "1",
			std::string(name) + " must be 0 or 1");
		int lowest = value == "1" ? 1 : 0;
		int highest = lowest;
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, &lowest, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &highest, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
		RequireOnEveryRank(lowest == highest,
			std::string(name) + " differs across MPI ranks");
		return highest == 1;
	}

	// Mesh quality of near-vacuum cells (plan 2026-09-28, step 0c): for owned cells within 10x of the lowest
	// density (floor cells, whose CFL limit is set by geometry and mesh motion), rank-0 percentiles of
	// q = V / (Amax x width), of the CFL effective radius min(width, V / Amax) and of the generator-centroid
	// offset / width, in bands of distance from the origin.  Values outside a histogram's range are counted as
	// censored tails (below/above) and a percentile falling in a tail is printed as "<lo" / ">hi".  Collective;
	// skipped unless every rank's mesh holds all of its owned cells.
	void ReportFloorMeshQuality(HDSim3D const& sim, char const* label)
	{
		Tessellation3D const& tess = sim.getTessellation();
		vector<ComputationalCell3D> const& cells = sim.getCells();
		size_t const owned = sim.getExtensives().size();
		int complete = tess.GetPointNo() == owned && cells.size() >= owned ? 1 : 0;
		double lowest = std::numeric_limits<double>::infinity();
		if(complete != 0)
			for(size_t i = 0; i < owned; ++i)
				lowest = std::min(lowest, cells[i].density);
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, &complete, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &lowest, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
		int rank = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
		int const rank = 0;
#endif
		if(complete == 0)
		{
			if(rank == 0)
				std::cout << "RICH_TDE_MESH_QUALITY label=" << label << " time=" << sim.getTime()
					<< " skipped=partial_mesh" << std::endl;
			return;
		}
		// Per band and metric: bins + 2 tail slots (below, above).  log10 ranges: q [-4, 1], effective radius
		// [-4, 2], offset/width [-6, 0].
		constexpr int bands = 4;
		constexpr int metrics = 3;
		constexpr int bins = 100;
		constexpr int slots = bins + 2;
		double const band_edges[bands + 1] = {0, 2, 5, 20, std::numeric_limits<double>::infinity()};
		double const range_lo[metrics] = {-4, -4, -6};
		double const range_hi[metrics] = {1, 2, 0};
		vector<double> histogram(static_cast<size_t>(bands * metrics * slots + bands), 0.0);
		auto add = [&](int band, int metric, double value)
		{
			size_t const base = static_cast<size_t>((band * metrics + metric) * slots);
			double const x = value > 0 ? std::log10(value) : -std::numeric_limits<double>::infinity();
			if(x < range_lo[metric])
				histogram[base + bins] += 1;
			else if(x >= range_hi[metric])
				histogram[base + bins + 1] += 1;
			else
				histogram[base + static_cast<size_t>((x - range_lo[metric]) / (range_hi[metric] - range_lo[metric]) *
					bins)] += 1;
		};
		for(size_t i = 0; i < owned; ++i)
		{
			if(!(cells[i].density <= 10 * lowest))
				continue;
			double amax = 0;
			for(size_t const face : tess.GetCellFaces(i))
				amax = std::max(amax, tess.GetArea(face));
			double const width = tess.GetWidth(i);
			double const volume = tess.GetVolume(i);
			if(!(amax > 0 && width > 0 && volume > 0))
				continue;
			double const r = abs(tess.GetMeshPoint(i));
			int band = 0;
			while(band < bands - 1 && r >= band_edges[band + 1])
				++band;
			add(band, 0, volume / (amax * width));
			add(band, 1, std::min(width, volume / amax));
			add(band, 2, abs(tess.GetCellCM(i) - tess.GetMeshPoint(i)) / width);
			histogram[static_cast<size_t>(bands * metrics * slots + band)] += 1;
		}
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, histogram.data(), static_cast<int>(histogram.size()), MPI_DOUBLE, MPI_SUM,
			MPI_COMM_WORLD);
#endif
		if(rank != 0)
			return;
		auto percentile = [&](int band, int metric, double fraction) -> std::string
		{
			size_t const base = static_cast<size_t>((band * metrics + metric) * slots);
			double total = 0;
			for(int b = 0; b < slots; ++b)
				total += histogram[base + static_cast<size_t>(b)];
			if(!(total > 0))
				return "nan";
			double running = histogram[base + bins];
			std::ostringstream text;
			text << std::setprecision(4);
			if(running >= fraction * total)
			{
				text << "<" << std::pow(10.0, range_lo[metric]);
				return text.str();
			}
			double const width_dex = (range_hi[metric] - range_lo[metric]) / bins;
			for(int b = 0; b < bins; ++b)
			{
				running += histogram[base + static_cast<size_t>(b)];
				if(running >= fraction * total)
				{
					text << std::pow(10.0, range_lo[metric] + (b + 0.5) * width_dex);
					return text.str();
				}
			}
			text << ">" << std::pow(10.0, range_hi[metric]);
			return text.str();
		};
		std::ostringstream line;
		line << std::setprecision(4) << "RICH_TDE_MESH_QUALITY label=" << label << " time=" << sim.getTime()
			<< " floor_density=" << lowest;
		for(int band = 0; band < bands; ++band)
		{
			line << " band" << band << "_r=" << band_edges[band] << "-" << band_edges[band + 1]
				<< ",n=" << histogram[static_cast<size_t>(bands * metrics * slots + band)];
			char const* const names[metrics] = {"q", "eff", "offset"};
			for(int metric = 0; metric < metrics; ++metric)
			{
				size_t const base = static_cast<size_t>((band * metrics + metric) * slots);
				line << "," << names[metric] << "_p1=" << percentile(band, metric, 0.01)
					<< "," << names[metric] << "_p5=" << percentile(band, metric, 0.05)
					<< "," << names[metric] << "_p50=" << percentile(band, metric, 0.5)
					<< "," << names[metric] << "_below=" << histogram[base + bins]
					<< "," << names[metric] << "_above=" << histogram[base + bins + 1];
			}
		}
		std::cout << line.str() << std::endl;
	}

	void WriteTdeSnapshot(HDSim3D const& sim, std::string const& name,
		vector<DiagnosticAppendix3D*> const& appendices)
	{
		ReportFloorMeshQuality(sim, "snapshot");
#ifdef RICH_MPI
		WriteSnapshot3D(sim, name, appendices, true, TdeWriteVtu());
#else
		WriteSnapshot3D(sim, name, appendices, TdeWriteVtu());
#endif
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
			if(simulation.GetTimeIntegrationMode() != TimeIntegrationMode::Individual)
			{
				// Global stepping keeps the primitives mesh-aligned (owned cells
				// followed by ghost copies on the full mesh); nothing to realign.
				RequireOnEveryRank(simulation.StateSynchronized(),
					"Snapshot requires a synchronized state");
				return;
			}
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
			realigned_ = true;
		}

		~MeshAlignedStateGuard()
		{
			if(!realigned_)
				return;
			sim_.getCells().swap(canonical_cells_);
			sim_.getExtensives().swap(canonical_extensives_);
		}

		MeshAlignedStateGuard(MeshAlignedStateGuard const&) = delete;
		MeshAlignedStateGuard& operator=(MeshAlignedStateGuard const&) = delete;

	private:
		HDSim3D& sim_;
		bool realigned_ = false;
		std::vector<ComputationalCell3D> canonical_cells_;
		std::vector<Conserved3D> canonical_extensives_;
	};

	class RemoveCenter
	{
	public:
		// `masked(cell)`: whether the gravity evaluates this cell's acceleration
		// to exactly zero (TDEGravity::MasksAcceleration on the current box).
		RemoveCenter(HDSim3D& sim, EquationOfState const& eos,
			double MBH, double Mstar, double Rstar, double beta, bool enabled,
			std::function<bool(ComputationalCell3D const&)> masked):
			sim_(sim), eos_(eos), enabled_(enabled), masked_(std::move(masked)),
			rt_(Rstar * std::pow(MBH / Mstar, 0.333333333) / beta),
			rsmooth_(std::max(rt_ * 0.4,
				std::min(rt_ - Rstar * 15, rt_ * smooth_factor))),
			sticker_index_(binary_index_find(ComputationalCell3D::stickerNames,
				std::string("InsideRemoveCenter")))
		{}

		// Global stepping (adaptive controller) advances every owned cell on
		// the same step, so the sink sweeps the owned range, as in
		// runs/BaseTDEComptonGlobal; there is no per-event acceleration cache.
		void Apply(void)
		{
			// A global step makes the individual global-step reference stale:
			// the first sink application after re-entering individual mode
			// falls back to the legacy per-event factors until the next
			// time-step suggestion refreshes it.
			sim_.ResetIndividualGlobalStepReference();
			if(!enabled_)
				return;
			Tessellation3D const& tess = sim_.getTessellation();
			std::vector<ComputationalCell3D>& cells = sim_.getCells();
			std::vector<Conserved3D>& extensives = sim_.getExtensives();
			std::size_t const N = tess.GetPointNo();
			RequireOnEveryRank(cells.size() >= N && extensives.size() >= N,
				"Center sink has inconsistent committed arrays");
			for(std::size_t i = 0; i < N; ++i)
				applyCell(cells[i], extensives[i], tess.GetCellCM(i),
					tess.GetVolume(i));
		}

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
			// The acceleration cache survives the sink (user's choice,
			// 2026-09-24).  A cell's own mass does not enter its own
			// acceleration, so the cells the sink changes keep valid caches.
			// Every cache is set by its cell's second-half evaluation at the end
			// of an interval and used by the first half of the next interval at
			// that same time, so the only mass that kick's acceleration misses
			// is what this event's sink removes after that evaluation: at most
			// G dM / d^2 from one event's dM (logged below; d the distance to
			// the nearest changed cell; FMM approximation error aside).  The
			// same cache also predicts passive cells at later face times
			// (ConditionActionFlux1), which can span several sink events; that
			// prediction error is not bounded by one event.  A cell taken below
			// the gravity mask would evaluate to exactly zero, so its cache
			// becomes zero.  An AMR pass refreshes the caches of the cells whose
			// interval opens now (Simulation::refreshCurrentAccelerationCaches),
			// which also absorbs this event's removal for them.  Flushing every
			// cache instead forced a full first-half gravity solve in 119 of
			// 126 events (job 10205130).
			unsigned long long counts[3] = {0, 0, 0};  // changed, zeroed, applied
			double masses[2] = {0, 0};  // removed, total before
			// Replicated: the guard floor's global-step reference (zero before
			// the first suggestion, then the sink acts once per event as before).
			double const reference_dt = sim_.GetIndividualGlobalStepReference();
			double exponents[2] = {std::numeric_limits<double>::infinity(), 0};
			std::size_t example_id = 0;
			double example_removed = 0;
			for(std::size_t local = 0; local < view->localSize(); ++local)
			{
				std::size_t const global = view->localToGlobal(local);
				if(!context.isActive(global))
					continue;
				double const old_mass = extensives[global].mass;
				// Per unit time: the factors act for this cell's closed interval
				// in units of the step a global step would take (see applyCell).
				// Not clipped (velocity damping has no floor, so a clip would
				// make it depend on the split); the division cannot overflow.
				// A huge exponent only underflows the factors to zero, which the
				// density and temperature floors absorb (underflow does not trap).
				double exponent = 1.0;
				double const interval = context.cellTimeStep(global);
				if(reference_dt > 0 && std::isfinite(interval) && interval >= 0)
					exponent = reference_dt > interval /
						std::numeric_limits<double>::max() ?
						interval / reference_dt : std::numeric_limits<double>::max();
				bool touched = false;
				bool const mass_changed = applyCell(cells[global], extensives[global],
					tess.GetCellCM(local), tess.GetVolume(local), exponent, &touched);
				if(touched)
				{
					++counts[2];
					exponents[0] = std::min(exponents[0], exponent);
					exponents[1] = std::max(exponents[1], exponent);
				}
				if(!mass_changed)
					continue;
				++counts[0];
				double const removed = old_mass - extensives[global].mass;
				masses[0] += removed;
				if(removed > example_removed)
				{
					example_removed = removed;
					example_id = cells[global].ID;
				}
				if(masked_ && masked_(cells[global]))
				{
					context.cached_accelerations[global] = Vector3D();
					++counts[1];
				}
			}
			// Total before the removal, for the removed fraction.
			for(Conserved3D const& extensive : extensives)
				masses[1] += extensive.mass;
			masses[1] += masses[0];
			unsigned long long example_cell = static_cast<unsigned long long>(example_id);
#ifdef RICH_MPI
			MPI_Allreduce(MPI_IN_PLACE, counts, 3, MPI_UNSIGNED_LONG_LONG,
				MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(MPI_IN_PLACE, masses, 2, MPI_DOUBLE, MPI_SUM,
				MPI_COMM_WORLD);
			double exponent_extrema[2] = {-exponents[0], exponents[1]};
			MPI_Allreduce(MPI_IN_PLACE, exponent_extrema, 2, MPI_DOUBLE, MPI_MAX,
				MPI_COMM_WORLD);
			exponents[0] = -exponent_extrema[0];
			exponents[1] = exponent_extrema[1];
			struct { double value; int rank; } local_example = {example_removed, 0}, largest = {0, 0};
			MPI_Comm_rank(MPI_COMM_WORLD, &local_example.rank);
			MPI_Allreduce(&local_example, &largest, 1, MPI_DOUBLE_INT, MPI_MAXLOC,
				MPI_COMM_WORLD);
			example_removed = largest.value;
			MPI_Bcast(&example_cell, 1, MPI_UNSIGNED_LONG_LONG, largest.rank,
				MPI_COMM_WORLD);
#endif
			cumulative_removed_mass_ += masses[0];
			int rank = 0;
#ifdef RICH_MPI
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
			if((counts[0] > 0 || counts[2] > 0) && rank == 0)
				std::cout << std::setprecision(6) << "TDE_SINK event_tick=" << context.event_tick
					<< " changed_cells=" << counts[0] << " applied_cells=" << counts[2]
					<< " zeroed_caches=" << counts[1]
					<< " removed_mass=" << masses[0]
					<< " removed_fraction=" << (masses[1] > 0 ? masses[0] / masses[1] : 0.0)
					<< " cumulative_individual_removed_mass=" << cumulative_removed_mass_
					<< " example_id=" << example_cell
					<< " example_removed=" << example_removed
					<< " reference_dt=" << reference_dt
					<< " exponent_min=" << exponents[0]
					<< " exponent_max=" << exponents[1] << std::endl;
		}
	private:
		// One application of the sink.  `exponent` scales it to a duration:
		// 1 on a global step (the legacy per-step factors); on an individual
		// event the cell's closed interval over the step a global step would
		// take, so a cell in a fine bin loses the same fraction per unit time as
		// one in a coarse bin (applied per event, the removal rate grew with
		// the activation rate: sink cells drained faster, got finer bins and
		// drained faster still; TDE trial job 10208596).  The density, the
		// temperature and the velocity damping factors are raised to it; the
		// floors and clamps are unchanged.
		bool applyCell(ComputationalCell3D& cell, Conserved3D& extensive,
			Vector3D const& centroid, double volume, double exponent = 1.0,
			bool* touched = nullptr) const
		{
			// Exactly the legacy factor at exponent 1 (std::pow(f, 1.0) is
			// not always f in the Intel math library), so a global step is
			// unchanged bit for bit.
			auto const scaled = [exponent](double factor)
			{
				return exponent == 1.0 ? factor : std::pow(factor, exponent);
			};
			double const radius = fastabs(centroid);
			if(radius < rsmooth_)
			{
				cell.stickers[sticker_index_] = true;
				double const old_density = cell.density;
				double const new_density = std::max(1e-20,
					old_density * scaled(0.8));
				double const density_ratio = old_density / new_density;
				if(touched != nullptr)
					*touched = true;
				// Legacy (exponent 1): min(1e7, max(1e4, 0.8 T)).  Otherwise
				// max(1e4, min(1e7 x 0.8^(x - 1), T x 0.8^x)): equal to the
				// legacy map at x = 1 and composing exactly under any split of
				// the interval, fractional ones included (two applications at
				// x = 0.5 equal one at x = 1); the floor is applied last, so the
				// decaying cap cannot undercut it.
				double const new_temperature = exponent == 1.0 ?
					std::min(1e7, std::max(1e4, cell.temperature * 0.8)) :
					std::max(1e4, std::min(1e7 * std::pow(0.8, exponent - 1.0),
						cell.temperature * scaled(0.8)));
				cell.tracers[2] *= old_density;
				cell.tracers[2] += old_density - new_density;
				cell.density = new_density;
				cell.tracers[2] /= new_density;
				cell.temperature = new_temperature;
				double const smoothing_fraction =
					std::min(1.0, radius / rsmooth_);
				cell.velocity *=
					scaled(1.0 - 0.1 * smoothing_fraction * smoothing_fraction);
				cell.internal_energy = eos_.dT2e(new_density, new_temperature,
					cell.tracers, ComputationalCell3D::tracerNames);
				cell.pressure = eos_.de2p(new_density, cell.internal_energy,
					cell.tracers, ComputationalCell3D::tracerNames);
				cell.tracers[0] = eos_.dp2s(new_density, cell.pressure,
					cell.tracers, ComputationalCell3D::tracerNames);
				cell.Erad *= density_ratio;
				cell.Erad_dt *= density_ratio;
				cell.Erad_dt_dt *= density_ratio;
				PrimitiveToConserved(cell, volume, extensive);
				return new_density != old_density;
			}

			cell.stickers[sticker_index_] = false;
			if(radius < std::min(rt_ * 0.8, rsmooth_ * 1.5) &&
				cell.temperature > 1e9)
			{
				if(touched != nullptr)
					*touched = true;
				// Legacy: x0.8 per application while above 1e9, so it stops at
				// the first value <= 1e9; an exponent covers at most as many
				// applications as reach it: the whole part of the exponent runs
				// as legacy steps (the same multiplications, stopping where they
				// stop), the fractional part as one partial step if still above
				// the cutoff.  For a fractional exponent the result depends on the
				// split (no map that stops at a threshold can match the legacy
				// step at 1 and compose exactly).
				double const whole = std::floor(exponent);
				for(double step = 0; step < whole && cell.temperature > 1e9; ++step)
					cell.temperature *= 0.8;
				if(exponent > whole && cell.temperature > 1e9)
					cell.temperature *= std::pow(0.8, exponent - whole);
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
		std::function<bool(ComputationalCell3D const&)> masked_;
		// Mass removed on individual events since this process started (not
		// persisted across restarts; global steps not included).
		double cumulative_removed_mass_ = 0;
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
		std::pair<Vector3D, Vector3D> box_points = tess.GetBoxCoordinates();
		double const reference_density = 1e-8 * Mstar / ((box_points.second.x - box_points.first.x) * (box_points.second.y - box_points.first.y) * (box_points.second.z - box_points.first.z));
		// Shift the owned cells [0, N), the points and the box, then rebuild.
		auto shift = [&](std::vector<ComputationalCell3D>& cells, std::vector<Conserved3D>& extensives)
		{
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
		};
		// Stepping globally (the adaptive controller) the state is the global
		// path's: owned cells followed by ghost copies, shifted in place exactly
		// as runs/BaseTDEComptonGlobal does.
		if(simulation.GetTimeIntegrationMode() != TimeIntegrationMode::Individual)
		{
			RequireOnEveryRank(
				points.size() >= N && canonical_cells.size() >= N &&
				canonical_extensives.size() >= N,
				"Reference-frame change requires one full mesh");
			points.resize(N);
			shift(canonical_cells, canonical_extensives);
			RequireOnEveryRank(
				canonical_cells.size() >= tess.GetPointNo() &&
				canonical_extensives.size() >= tess.GetPointNo(),
				"Reference-frame rebuild produced misaligned committed state");
			return;
		}
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
		shift(cells, extensives);
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
			// Under the adaptive controller the run may be stepping globally;
			// every owned cell is then current and there is no scheduler.
			bool const individual_mode =
				simulation.GetTimeIntegrationMode() == TimeIntegrationMode::Individual;
			int scheduler_ready = !individual_mode ||
				(scheduler != nullptr && scheduler->initialized());
#ifdef RICH_MPI
			MPI_Allreduce(MPI_IN_PLACE, &scheduler_ready, 1, MPI_INT, MPI_MIN,
				MPI_COMM_WORLD);
#endif
			// The adaptive controller switched into individual mode at this step
			// boundary: the new scheduler initializes at its first event.  The
			// state is the completed global step's, so it is scanned as a global
			// state; a trigger is latched (reference_frame_change_pending) and
			// the transition itself waits for that first event.
			bool const scheduler_uninitialized = scheduler_ready == 0;
			bool const scan_individual = individual_mode && !scheduler_uninitialized;
			std::vector<CellTimeState> const* states =
				scan_individual ? &scheduler->states() : nullptr;
			RequireOnEveryRank(states == nullptr || states->size() == cells.size(),
				"Gravity reference-frame scan has inconsistent scheduler state");
			std::uint64_t const current_tick =
				scan_individual ? scheduler->currentTick() : 0;
			int need_update = 0;
			std::size_t const owned_cells = scan_individual ?
				cells.size() : sim.getTessellation().GetPointNo();
			for(size_t i = 0; i < owned_cells && i < cells.size(); ++i)
			{
				if(states != nullptr && (*states)[i].last_primitive_tick != current_tick)
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
			if(rank == 0 && RuntimeLogDetailed())
				std::cout<<x0[0]<<","<<x0[1]<<std::endl;
			bool const transition_requested =
				reference_frame_change_pending ||
				(x0[1] > 0.1 && x0[2] > 0.1) || need_update == 1;
			if(scheduler_uninitialized)
			{
				if(transition_requested)
				{
					reference_frame_change_pending = true;
					if(rank == 0)
						std::cout << "Gravity reference-frame change latched; it waits for "
							"the first event of the new individual scheduler" << std::endl;
				}
				return;
			}
			if(transition_requested)
			{
				if(!simulation.StateSynchronized())
				{
					simulation.RequestSynchronizedIndividualEvent();
					reference_frame_change_pending = true;
					if(rank == 0)
						std::cout << "Deferring gravity reference-frame change until "
							"a synchronized individual event" << std::endl;
					return;
				}
				bool gravity_transition_marked = true;
				if(rank == 0)
				{
					try
					{
						WriteIntegerControlFileAtomically(-1, gravity_name);
					}
					catch(std::exception const& error)
					{
						gravity_transition_marked = false;
						std::cerr << error.what() << std::endl;
					}
				}
				RequireOnEveryRank(gravity_transition_marked,
					"Could not mark the gravity transition in progress");

				UpdateReferenceFrame(sim, Rstar, Mstar, MBH, beta, simulation);
				if(simulation.GetTimeIntegrationMode() == TimeIntegrationMode::Individual)
					ResetIndividualSchedulerAfterReferenceFrameChange(simulation);
#ifdef RICH_MPI
				MPI_Barrier(MPI_COMM_WORLD);
				if(rank == 0 && RuntimeLogDetailed())
					std::cout<<"Point number "<<sim.getTessellation().GetPointNo()<<std::endl;
#endif
				vector<DiagnosticAppendix3D *> appendices;
				{
					MeshAlignedStateGuard aligned_state(sim, simulation);
					WriteTdeSnapshot(sim, restart_name, appendices);
				}
				WriteSimulation(simulation, individual_restart_name);
#ifdef RICH_MPI
				MPI_Barrier(MPI_COMM_WORLD);
#endif
				bool gravity_marker_written = true;
				if(rank == 0)
				{
					try
					{
						WriteIntegerControlFileAtomically(1, gravity_name);
					}
					catch(std::exception const& error)
					{
						gravity_marker_written = false;
						std::cerr << error.what() << std::endl;
					}
				}
				RequireOnEveryRank(gravity_marker_written,
					"Could not commit the full-gravity control file");
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
		// The rule both evaluation paths apply: no acceleration for a cell
		// below the density floor or without enough stellar material.
		bool MasksAcceleration(ComputationalCell3D const& cell,
			std::pair<Vector3D, Vector3D> const& bounds) const
		{
			return cell.density < MinimumDensity(bounds) || cell.tracers[1] < 0.1;
		}

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

int main(int argc, char* argv[])
{
	int rank = 0;
	int ws = 1;
#ifdef RICH_MPI
	MPI_Init(&argc, &argv);
	double last_start = MPI_Wtime();
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &ws);
#endif
	bool start_fresh = false;
	bool arguments_valid = true;
	std::string invalid_argument;
	for(int argument = 1; argument < argc; ++argument)
	{
		if(std::string(argv[argument]) == "--fresh")
			start_fresh = true;
		else
		{
			arguments_valid = false;
			invalid_argument = argv[argument];
		}
	}
	if(!arguments_valid)
	{
		if(rank == 0)
			std::cerr << "Unknown argument '" << invalid_argument
				<< "'; supported argument: --fresh" << std::endl;
#ifdef RICH_MPI
		MPI_Finalize();
#endif
		return 2;
	}
	feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
	char const* configured_run_directory = std::getenv("RICH_TDE_RUN_DIRECTORY");
	std::string run_directory =
		configured_run_directory != nullptr && configured_run_directory[0] != '\0'
		? configured_run_directory : "/data/users/elads/TDE_individual_dt";
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
	std::string const initial_snapshot_name = run_directory + "initial.h5";
	std::string restart_name = run_directory + "restart.h5";
	std::string const normal_individual_restart_name = run_directory + "individual_restart.h5";
	std::string const full_individual_restart_name = run_directory + "individual_full_restart.h5";
	std::string counter_name = run_directory + "counter.txt";
	std::string gravity_name = run_directory + "gravity.txt";
	if(start_fresh)
	{
		// Mark the new generation incomplete until its first snapshot and scheduler
		// checkpoint have both been written.  This prevents a later ordinary launch
		// from accidentally resuming an older snap_0 after an early fresh-run crash.
		bool fresh_directory_ready = true;
		if(rank == 0)
		{
			try
			{
				WriteIntegerControlFileAtomically(-1, counter_name);
				WriteIntegerControlFileAtomically(0, gravity_name);
				RemoveOldRunArtifacts(run_directory);
			}
			catch(std::exception const& error)
			{
				fresh_directory_ready = false;
				std::cerr << error.what() << std::endl;
			}
		}
		RequireOnEveryRank(fresh_directory_ready,
			"Could not prepare the fresh-run output directory");
	}
	int counter = 0;
	// check if this is a restart run
	bool const restart = !start_fresh && fs::exists(counter_name);
	if(rank == 0)
		std::cout << "start_fresh " << start_fresh << std::endl;
	if(rank == 0)
		std::cout<<"restart "<<restart<<std::endl;
	if(restart)
	{
		counter = ReadIntegerControlFile(counter_name);
		if(counter < 0)
			throw UniversalError(
				"Previous --fresh run ended before its first complete checkpoint; rerun with --fresh");
		std::filesystem::last_write_time(counter_name, std::filesystem::file_time_type::clock::now());
	}
	std::string eos_location("../../data/EOS/");
	int const gravity_state = !start_fresh && fs::exists(gravity_name) ?
		ReadIntegerControlFile(gravity_name) : 0;
	if(gravity_state == -1)
		throw UniversalError(
			"Previous gravity transition ended before its checkpoint was committed; rerun with --fresh");
	if(gravity_state != 0 && gravity_state != 1)
		throw UniversalError("Invalid gravity control value in " + gravity_name);
	bool const full_gravity = gravity_state == 1;
	if(full_gravity)
		std::filesystem::last_write_time(gravity_name, std::filesystem::file_time_type::clock::now());
	std::string const individual_restart_name = full_gravity ?
		full_individual_restart_name : normal_individual_restart_name;
	// RICH_TDE_RESTART_FROM_SNAPSHOT=1 (see the switch notes below): a restart reads
	// only the snapshot, so no scheduler/global checkpoint is required.
	bool const restart_from_snapshot = TdeSwitch("RICH_TDE_RESTART_FROM_SNAPSHOT");
	if(restart && !restart_from_snapshot && !fs::exists(individual_restart_name))
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
	STAgreyOpacity opacity("/home/elads/RICH/data/STA/");
	if (rank == 0)
		std::cout << "end sta" << std::endl;

	const double width = 5;
	Vector3D ll(-width, -width, -width), ur(width, width, width);
	Voronoi3D tess(ll, ur);

	vector<ComputationalCell3D> cells;
	double const startfactor = 3;
	double const fstart = -acos(2 * Rp / (startfactor * Rt) - 1);
	double const tstart = 0.3333333 * sqrt(2 * Rp * Rp * Rp / Mbh) *
		tan(0.5 * fstart) *
		(3 + tan(0.5 * fstart) * tan(0.5 * fstart));
	double t_restart = -100;
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
			cells = GetCells(tess, M, R, eos, tscale * tscale * lscale / mscale, n);
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

	bool const hydro_on = true;
	bool const compton_on = true;
	bool const flux_limit = true;
	std::vector<std::string> rad_zero_cells({"InsideRemoveCenter"});
	DiffusionOpenBoundary d_boundary;
	Diffusion matrix_builder(opacity, d_boundary, eos, rad_zero_cells,
		flux_limit, hydro_on, compton_on);
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
	// RICH_TDE_CFL_DEBUG=1 (diagnostic, default 0): CourantFriedrichsLewy's
	// verbose dump of the raw-CFL winning cell and its faces whenever an
	// evaluation falls below 0.9999 of the previous one; that cell need not set
	// the accepted step (RICH_CFL_DECISION_TRACE records which criterion does).
	bool cfl_debug = false;
	{
		char const* const configured = std::getenv("RICH_TDE_CFL_DEBUG");
		std::string const value = configured == nullptr ? std::string() :
			std::string(configured);
		RequireOnEveryRank(value.empty() || value == "0" || value == "1",
			"RICH_TDE_CFL_DEBUG must be 0 or 1");
		cfl_debug = value == "1";
		int debug_min = cfl_debug ? 1 : 0;
		int debug_max = debug_min;
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, &debug_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &debug_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
		RequireOnEveryRank(debug_min == debug_max,
			"RICH_TDE_CFL_DEBUG differs across MPI ranks");
	}
	auto tsf = std::make_shared<CourantFriedrichsLewy>(0.4, 1, force,
		std::vector<std::string> (), cfl_debug);

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
	RemoveCenter center_sink(*sim, eos, Mbh, M, R, beta, full_gravity,
		[&acc, &tess](ComputationalCell3D const& cell)
		{
			return acc.MasksAcceleration(cell, tess.GetBoxCoordinates());
		});
	simulation.addPhysics(hydroStep);
	simulation.addPhysics(radStep);
	simulation.SetIndividualPostPhysics(
		[&](IndividualStepContext const& context)
		{
			center_sink.Apply(context);
		});
	double init_dt = 1e-4;
	simulation.SetTimeStep(init_dt);
	IndividualTimeStepOptions individual_options;
	individual_options.initial_bin = 30;
	individual_options.maximum_bin = 40;
	individual_options.maximum_neighbor_bin_difference = 1;
	individual_options.mesh_build_policy = IndividualMeshBuildPolicy::AutoPartial;
	// RICH_TDE_INDIVIDUAL_SCHEME (diagnostic): "partial" (default) variable bins on
	// AutoPartial meshes; "full-variable" variable bins on full meshes; "full"
	// one shared, adaptively selected bin on full meshes (the synchronized oracle
	// of the regression drivers' RICH_INDIVIDUAL_MODE).
	{
		char const* const configured = std::getenv("RICH_TDE_INDIVIDUAL_SCHEME");
		std::string const scheme = configured == nullptr ? std::string() :
			std::string(configured);
		RequireOnEveryRank(scheme.empty() || scheme == "partial" ||
			scheme == "full-variable" || scheme == "full",
			"RICH_TDE_INDIVIDUAL_SCHEME must be partial, full-variable or full");
		int const code = scheme == "full" ? 2 : (scheme == "full-variable" ? 1 : 0);
		int code_min = code;
		int code_max = code;
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, &code_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &code_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
		RequireOnEveryRank(code_min == code_max,
			"RICH_TDE_INDIVIDUAL_SCHEME differs across MPI ranks");
		if(code > 0)
		{
			individual_options.mesh_build_policy = IndividualMeshBuildPolicy::FullReference;
			individual_options.force_synchronized = code == 2;
			if(rank == 0)
				std::cout << "RICH_TDE_INDIVIDUAL_SCHEME=" << scheme << std::endl;
		}
	}
	// RICH_TDE_START_MODE: how a fresh run starts.  "individual" enables the
	// scheduler here; "global" starts on the global path, as a restart from a
	// global-mode checkpoint does, and the adaptive controller enters
	// individual stepping only on its own criteria.  Unset: global when the
	// controller is on (a 77-event individual start of the TDE left a faster,
	// hotter surface layer that cost 2.07x the global steps to t=-1.20, jobs
	// 10204483 and 10204569 arm B4; started global the controller logged the
	// global driver's step sequence, arm B3), individual when it is off.
	// A restart takes its mode from the checkpoint.  With this problem's
	// parameters no restart precedes the early snapshot: the fresh run's counter
	// stays -1 until its first numbered output, and the first regular target
	// (tstart + 0.265) lies after the early one (tstart + 0.05).
	if(!TdeWriteVtu() && rank == 0)
		std::cout << "RICH_TDE_WRITE_VTU=0: snapshots without ParaView files" << std::endl;
	// RICH_TDE_UPDATE_BOX=1: grow the box with the legacy driver's UpdateBox
	// (runs/BaseTDECompton: every 7 cycles, speed threshold 0.5, new cells at
	// volume fraction 1e-5 in the reference state).  The fixed +-5 box with
	// rigid walls lets the debris pile into the corners from t ~ 0.6, where
	// slivers of width ~1e-4 pin the step near 3e-5 (jobs 10204588/10204589).
	// UpdateBox rebuilds the mesh and recomputes every extensive from the
	// primitives, so in individual mode it runs only at a synchronized event
	// (UpdateBoxSynchronized, see the main loop).  RICH_TDE_RESTART_FROM_SNAPSHOT=1: a restart reads only
	// snap_<counter> (or a newer restart.h5), as runs/BaseTDEComptonGlobal does,
	// and steps globally from init_dt instead of loading the scheduler/global
	// checkpoint, which exists only for the newest output.
	bool const update_box = TdeSwitch("RICH_TDE_UPDATE_BOX");
	if(rank == 0 && (update_box || restart_from_snapshot))
		std::cout << "RICH_TDE_UPDATE_BOX=" << (update_box ? 1 : 0)
			<< " RICH_TDE_RESTART_FROM_SNAPSHOT=" << (restart_from_snapshot ? 1 : 0)
			<< std::endl;
	bool const adaptive_requested = true;
	bool start_global = false;
	bool start_mode_explicit = false;
	{
		char const* const configured_start_mode = std::getenv("RICH_TDE_START_MODE");
		std::string const start_mode = configured_start_mode == nullptr ?
			std::string() : std::string(configured_start_mode);
		RequireOnEveryRank(start_mode.empty() || start_mode == "individual" ||
			start_mode == "global",
			"RICH_TDE_START_MODE must be individual or global");
		bool const adaptive_on =
			Simulation::AdaptiveIntegrationModeWillEnable(adaptive_requested);
		start_mode_explicit = !start_mode.empty();
		start_global = start_mode_explicit ? start_mode == "global" : adaptive_on;
		int start_global_min = start_global ? 1 : 0;
		int start_global_max = start_global_min;
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, &start_global_min, 1, MPI_INT, MPI_MIN,
			MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &start_global_max, 1, MPI_INT, MPI_MAX,
			MPI_COMM_WORLD);
#endif
		RequireOnEveryRank(start_global_min == start_global_max,
			"RICH_TDE_START_MODE differs across MPI ranks");
	}
	if(restart && restart_from_snapshot)
	{
		// The snapshot already set time and cycle; no checkpoint is read, so the
		// run steps globally from init_dt under the adaptive controller.
		if(rank == 0)
			std::cout << "Restart from the snapshot only (RICH_TDE_RESTART_FROM_SNAPSHOT=1): "
				"global stepping from t=" << simulation.GetTime() << std::endl;
		t_restart = simulation.GetTime();
	}
	else if(restart)
	{
		ReadSimulation(individual_restart_name, simulation);
		// A checkpoint written while the adaptive controller was stepping
		// globally carries no scheduler state; the controller re-enters
		// individual mode from the global step when it pays.
		if(simulation.GetTimeIntegrationMode() != TimeIntegrationMode::Individual &&
			rank == 0)
			std::cout << "Restart checkpoint was written in global mode; "
				"resuming global stepping under the adaptive controller"
				<< std::endl;
		if(start_mode_explicit && rank == 0)
			std::cout << "RICH_TDE_START_MODE ignored on restart: the checkpoint "
				"sets the integration mode" << std::endl;
		t_restart = simulation.GetTime();
	}
	else
	{
		if(rank == 0)
			std::cout << "Fresh run starts on the "
				<< (start_global ? "global" : "individual") << " path ("
				<< (start_mode_explicit ? "RICH_TDE_START_MODE" :
					"default for the adaptive controller's state") << ")" << std::endl;
		if(!start_global)
			simulation.EnableIndividualTimeSteps(individual_options);
	}
	// Let the run choose between global and individual stepping from measured
	// throughput and the potential gain of the current timestep distribution.
	simulation.SetAdaptiveIntegrationMode(adaptive_requested, individual_options);
	if (rank == 0)
		std::cout << "Restart time " << simulation.GetTime() << std::endl;
	ReportFloorMeshQuality(*sim, "start");
	double tf = 6 * std::sqrt(apocenter * apocenter * apocenter / Mbh);
	// RICH_TDE_FINAL_TIME overrides the stopping time so a probe ends on the
	// "Done sim" marker at a chosen t instead of on the job time limit (same
	// override as runs/BaseTDEComptonGlobal/test.cpp).  Unset: full-orbit tf.
	{
		char const* const configured_final_time =
			std::getenv("RICH_TDE_FINAL_TIME");
		if(configured_final_time != nullptr && configured_final_time[0] != '\0')
		{
			char* end = nullptr;
			double const parsed = std::strtod(configured_final_time, &end);
			bool const valid = end != configured_final_time &&
				*end == '\0' && std::isfinite(parsed);
			RequireOnEveryRank(valid,
				"RICH_TDE_FINAL_TIME must be a finite number");
			tf = parsed;
			if(rank == 0)
				std::cout << "Final time overridden by RICH_TDE_FINAL_TIME: "
					<< tf << std::endl;
		}
	}
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
	// Full AMR passes in individual mode.  The every-10th-event pass sees only
	// that event's active cells, and the all-active events can alias with it
	// (TDE job 10222679, t=35.32-35.77: 68 all-active events, none on the
	// gate), so cells went unrefined and above all unremoved: 8% more cells
	// than the global run and ~290k-cell catch-up bursts in every global probe.
	// A full pass is therefore also due, at the first all-active event, once
	// RICH_TDE_FULL_AMR_INTERVAL_BINS (default 10, as the global cadence of 10
	// steps; 0 = off) finest-bin intervals have passed since the last full pass
	// in either mode.  Parsed once and agreed across ranks.
	double full_amr_interval_bins = 10;
	{
		char const* const configured = std::getenv("RICH_TDE_FULL_AMR_INTERVAL_BINS");
		bool valid = true;
		if(configured != nullptr && configured[0] != '\0')
		{
			char* end = nullptr;
			full_amr_interval_bins = std::strtod(configured, &end);
			valid = end != configured && *end == '\0' && std::isfinite(full_amr_interval_bins) &&
				full_amr_interval_bins >= 0;
		}
		// Unconditional: RequireOnEveryRank is collective, and ranks may differ in whether it is set.
		RequireOnEveryRank(valid, "RICH_TDE_FULL_AMR_INTERVAL_BINS must be a finite number >= 0");
		double extremes[2] = {full_amr_interval_bins, -full_amr_interval_bins};
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, extremes, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
		RequireOnEveryRank(extremes[0] == -extremes[1], "RICH_TDE_FULL_AMR_INTERVAL_BINS differs across MPI ranks");
	}
	// Time of the last AMR pass over every cell (global, or all-active
	// individual), whether or not it changed the mesh.  Replicated.
	double last_full_amr_time = simulation.GetTime();
	simulation.SetIndividualAMR(
		[&](IndividualStepContext const& context)
		{
			if(!full_gravity)
				return IndividualAMRChangeSet();
			IndividualTimeStepScheduler const* const scheduler = simulation.GetIndividualTimeStepScheduler();
			// Due and overdue are replicated (time, quantum and finest bin are
			// the same on every rank), so the collective all-active test below
			// is too.  With full passes on, the every-10th-event active-only
			// pass is redundant for cadence (every cell, the finest-bin ones
			// included, is evaluated every ~10 finest intervals, ~10 global
			// steps) and cost ~0.3 s per late partial event (a full canonical
			// build from partial geometry); it remains only as a fallback once
			// no full pass has run for twice the interval (all-active events
			// sparse).  Interval 0 keeps the old gate alone.
			double const since_full = simulation.GetTime() - last_full_amr_time;
			double const full_interval = scheduler != nullptr && scheduler->initialized() ? full_amr_interval_bins *
				std::ldexp(scheduler->timeQuantum(), static_cast<int>(scheduler->minimumOccupiedBin())) : 0;
			bool const old_gate = (simulation.GetCycle() + 1) % 10 == 0;
			bool const due = full_amr_interval_bins > 0 && full_interval > 0 && since_full >= full_interval;
			bool const gate = full_amr_interval_bins > 0 ? old_gate && due && since_full >= 2 * full_interval : old_gate;
			int all_active = 0;
			if(gate || due)
			{
				all_active = scheduler != nullptr &&
					context.active_indices.size() == scheduler->states().size() ? 1 : 0;
#ifdef RICH_MPI
				MPI_Allreduce(MPI_IN_PLACE, &all_active, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
			}
			if(!gate && !(due && all_active != 0))
				return IndividualAMRChangeSet();
			if(rank == 0 && RuntimeLogDetailed())
				std::cout << "Doing individual AMR" << std::endl;
			if(rank == 0 && full_amr_interval_bins > 0)
				std::cout << std::setprecision(12) << "RICH_TDE_FULL_AMR cycle=" << simulation.GetCycle()
					<< " time=" << simulation.GetTime() << " since_last=" << since_full
					<< " interval=" << full_interval << " pass=" << (all_active != 0 ? "full" : "overdue_active_only")
					<< std::endl;
			if(all_active != 0)
				last_full_amr_time = simulation.GetTime();
			return amr.ApplyIndividual(simulation, context);
		});
	// The same operations on a global step (adaptive controller), with the
	// cadence of runs/BaseTDEComptonGlobal: AMR every 10 cycles once full
	// gravity is on, the center sink after every step.
	simulation.SetGlobalPostStep(
		[&]()
		{
			if(full_gravity && simulation.GetCycle() % 10 == 0)
			{
				if(rank == 0 && RuntimeLogDetailed())
					std::cout << "Doing AMR" << std::endl;
				amr(simulation);
				last_full_amr_time = simulation.GetTime();
			}
			center_sink.Apply();
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

	bool reference_frame_change_pending = false;
	// An individual box growth waiting for a synchronized state (process-wide
	// the same value on every rank; not persisted: a restart re-evaluates it).
	bool box_growth_pending = false;
	bool regular_output_pending = false;
	bool early_output_pending = false;
	// One extra numbered snapshot, 0.05 code-time units after the original start.
	// RICH_TDE_TERMINAL_OUTPUT_TIME=<t> moves it to t: a synchronized snapshot
	// at t in either stepping mode, after which the run stops, to compare runs
	// from one restart.  On the individual timeline t is rounded to the nearest
	// tick of the current timeline (always from the requested value), so arms
	// that step differently land within one tick quantum of each other.  Unlike
	// the early snapshot it holds the adaptive controller on the global path
	// only within four global steps of t (see SetAdaptiveDecisionsDeferred
	// below).  Presence and value are agreed across ranks; t must lie after the
	// restart time and before the final time.
	double early_output_time = tstart + 0.05;
	bool terminal_output_configured = false;
	double requested_terminal_time = 0;
	// The written snapshot must lie within this of the requested time: exact
	// landing on the global path, one tick quantum on the individual timeline.
	double terminal_output_tolerance = 0;
	{
		char const* const configured = std::getenv("RICH_TDE_TERMINAL_OUTPUT_TIME");
		bool const present = configured != nullptr && configured[0] != '\0';
		double parsed = 0;
		bool valid = true;
		if(present)
		{
			char* end = nullptr;
			parsed = std::strtod(configured, &end);
			valid = end != configured && *end == '\0' && std::isfinite(parsed);
			if(!valid)
				parsed = 0;
		}
		// Startup agreement, unconditional like TdeSwitch/TdeWriteVtu; the final
		// time joins it because the range check below depends on it.
		double extrema[8] = {present ? 1.0 : 0.0, valid ? 1.0 : 0.0, parsed, tf,
			present ? -1.0 : -0.0, valid ? -1.0 : -0.0, -parsed, -tf};
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, extrema, 8, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
		if(extrema[3] != -extrema[7])
			throw std::invalid_argument("RICH_TDE_FINAL_TIME differs across MPI ranks");
		if(extrema[0] != -extrema[4] || extrema[1] != -extrema[5] ||
		   extrema[2] != -extrema[6])
			throw std::invalid_argument(
				"RICH_TDE_TERMINAL_OUTPUT_TIME differs across MPI ranks");
		if(extrema[0] > 0)
		{
			if(!(extrema[1] > 0))
				throw std::invalid_argument(
					"RICH_TDE_TERMINAL_OUTPUT_TIME must be a finite number");
			if(!(extrema[2] > simulation.GetTime()) || !(extrema[2] < tf))
				throw std::invalid_argument(
					"RICH_TDE_TERMINAL_OUTPUT_TIME must lie after the restart time and before the final time");
			requested_terminal_time = extrema[2];
			early_output_time = requested_terminal_time;
			terminal_output_configured = true;
			terminal_output_tolerance = 64 * std::numeric_limits<double>::epsilon() *
				std::max(1.0, std::abs(requested_terminal_time));
		}
	}
	bool early_output_written = simulation.GetTime() >= early_output_time;
#ifdef RICH_MPI
	bool restart_output_pending = false;
#endif
	double const restart_wtime = 10000;
	double const min_dt_output = 0.025 * std::sqrt(std::pow(R, 3.0) * Mbh / M);
	auto write_synchronized_snapshot = [&](std::string const& output_name)
	{
		MeshAlignedStateGuard aligned_state(*sim, simulation);
		interp(tess, sim->getCells(), 0, dissipation.face_values);
		WriteTdeSnapshot(*sim, output_name, appendices);
	};
	// A global start writes no initial file, as runs/BaseTDEComptonGlobal does:
	// the reconstruction below fills the interpolator's slope cache, and a
	// global start must follow that driver's step sequence to compare with it.
	if(!restart && !start_global)
	{
		if(rank == 0)
			std::cout << "Starting writing initial file " << initial_snapshot_name
				<< std::endl;
		{
			// Reconstruction needs current primitive values for MPI ghost cells;
			// keep the synchronized copy only for this initial output.
			std::vector<ComputationalCell3D> snapshot_cells = sim->getCells();
			std::vector<ComputationalCell3D> exchange_cells = sim->getCells();
			tess.SyncPartialBuildData(snapshot_cells, exchange_cells);
			interp(tess, snapshot_cells, 0, dissipation.face_values);
		}
		WriteTdeSnapshot(*sim, initial_snapshot_name, appendices);
		dissipation.face_values.clear();
		dissipation.face_values.shrink_to_fit();
	}
	// A pending configured terminal snapshot is written even at or after tf.
	while (simulation.GetTime() < tf ||
		(terminal_output_configured && !early_output_written))
	{
		if(!early_output_written && simulation.GetTime() < early_output_time)
		{
			IndividualTimeStepScheduler* const scheduler =
				simulation.GetIndividualTimeStepScheduler();
			double const target_tolerance = 64 * std::numeric_limits<double>::epsilon() *
				std::max(1.0, std::abs(early_output_time));
			if(scheduler == nullptr)
			{
				if(simulation.GetTimeIntegrationMode() != TimeIntegrationMode::Global)
					throw std::logic_error(
						"Early snapshot requires an individual scheduler");
				// Global stepping: land the next step on the snapshot time.
				double const remaining = early_output_time - simulation.GetTime();
				if(remaining > 0)
					simulation.SetTimeStep(
						std::min(simulation.GetTimeStep(), remaining));
			}
			else if(!scheduler->initialized())
			{
				// A configured terminal snapshot before the scheduler starts:
				// land the next step on it, a global step (the controller is held
				// near the target below) or the first individual event, whose
				// length is this time step and which synchronizes every cell.
				if(terminal_output_configured)
				{
					double const remaining = early_output_time - simulation.GetTime();
					if(remaining > 0)
						simulation.SetTimeStep(
							std::min(simulation.GetTimeStep(), remaining));
				}
				else
				{
					double const first_event_time = simulation.GetTime() +
						simulation.GetTimeStep();
					if(first_event_time >= early_output_time - target_tolerance)
						throw std::logic_error(
							"Initial individual event would cross the early snapshot time");
				}
			}
			else
			{
				long double const target_tick_coordinate =
					(static_cast<long double>(terminal_output_configured ?
						requested_terminal_time : early_output_time) -
					 static_cast<long double>(scheduler->timeOrigin())) /
					static_cast<long double>(scheduler->timeQuantum());
				// Scheduler state is the same on every rank, so is this check.
				if(!(target_tick_coordinate >= 0) ||
				   !(target_tick_coordinate < 4.0e18L))
					throw std::logic_error(
						"Early snapshot time is outside the individual timeline");
				std::uint64_t const target_tick = static_cast<std::uint64_t>(
					std::llround(target_tick_coordinate));
				double const represented_target_time = scheduler->timeOrigin() +
					scheduler->timeQuantum() * static_cast<double>(target_tick);
				if(terminal_output_configured)
				{
					early_output_time = represented_target_time;
					terminal_output_tolerance = std::max(terminal_output_tolerance,
						scheduler->timeQuantum());
				}
				else if(std::abs(represented_target_time - early_output_time) > target_tolerance)
					throw std::logic_error(
						"Early snapshot time is not representable on the individual timeline");
				if(target_tick > scheduler->currentTick())
					scheduler->clampToTerminalTick(target_tick);
			}
		}
		bool const early_output_due = !early_output_written &&
			(early_output_pending || simulation.GetTime() >= early_output_time);
		bool const regular_output_due = regular_output_pending ||
			simulation.GetTime() > nextT;
		if(!reference_frame_change_pending &&
			(early_output_due || regular_output_due))
		{
			early_output_pending = early_output_due;
			regular_output_pending = regular_output_due;
			if(!simulation.StateSynchronized())
			{
				// A configured terminal snapshot that missed its tick (its time
				// rounded to the current one) waits for the next synchronized
				// event instead; RICH_OUTPUT reports the time it was written at.
				if(early_output_pending && !terminal_output_configured)
					throw std::logic_error(
						"Early snapshot terminal event is not synchronized");
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
				auto const output_start = std::chrono::steady_clock::now();
				write_synchronized_snapshot(output_name);
				auto const snapshot_end = std::chrono::steady_clock::now();
				WriteSimulation(simulation, individual_restart_name);
				auto const checkpoint_end = std::chrono::steady_clock::now();
				if(rank == 0)
					std::cout << "RICH_OUTPUT file=" << output_name
						<< " time=" << simulation.GetTime()
						<< " snapshot_s=" << std::chrono::duration<double>(
							snapshot_end - output_start).count()
						<< " checkpoint_s=" << std::chrono::duration<double>(
							checkpoint_end - snapshot_end).count()
						<< " vtu=" << (TdeWriteVtu() ? 1 : 0)
						<< " (rank-0 wall)" << std::endl;
				bool counter_written = true;
				if(rank == 0)
				{
					try
					{
						WriteIntegerControlFileAtomically(counter, counter_name);
					}
					catch(std::exception const& error)
					{
						counter_written = false;
						std::cerr << error.what() << std::endl;
					}
				}
				RequireOnEveryRank(counter_written,
					"Could not commit the restart counter");
				if(early_output_pending)
				{
					early_output_written = true;
					early_output_pending = false;
				}
				else
				{
					nextT = simulation.GetTime() + std::min(min_dt_output,
						mindt + 0.2 * std::pow(std::abs(simulation.GetTime()), 0.666666));
					regular_output_pending = false;
				}
				++counter;
				dissipation.face_values.clear();
				dissipation.face_values.shrink_to_fit();
			}
		}
		// A configured terminal snapshot ends the run once written (the same
		// decision on every rank: time and flags are replicated).  Written away
		// from the requested time (a missed tick served at a later synchronized
		// event), the run fails after the snapshot so the comparison is not
		// silently misaligned.
		if(terminal_output_configured && early_output_written)
		{
			double const error = std::abs(simulation.GetTime() - requested_terminal_time);
			bool const aligned = error <= terminal_output_tolerance;
			if(rank == 0)
				std::cout << "RICH_TDE_TERMINAL_OUTPUT time=" << std::setprecision(17)
					<< simulation.GetTime() << " requested=" << requested_terminal_time
					<< " error=" << error << " tolerance=" << terminal_output_tolerance
					<< " aligned=" << (aligned ? 1 : 0) << std::endl;
			if(!aligned)
				throw std::logic_error(
					"Terminal snapshot written away from RICH_TDE_TERMINAL_OUTPUT_TIME");
			break;
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
				if(!simulation.StateSynchronized())
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
#endif
			// No controller decision on a global step while the early snapshot is
			// pending: entering individual mode there would give the snapshot
			// time to a fresh scheduler that cannot land on it.  A configured
			// terminal snapshot is rounded to the scheduler's tick instead and
			// holds the controller only within four global steps of its time,
			// so no first individual event (one global step long) reaches it.
			bool const terminal_output_near = terminal_output_configured &&
				!early_output_written && simulation.GetTime() +
				4 * simulation.GetTimeStep() >= requested_terminal_time;
			simulation.SetAdaptiveDecisionsDeferred(!early_output_written &&
				(!terminal_output_configured || terminal_output_near));
			bool const stepped_globally =
				simulation.GetTimeIntegrationMode() == TimeIntegrationMode::Global;
				simulation.step();
			if(not full_gravity)
				CheckIfFullGravityIsNeeded(*sim, gravity_name, R, M, Mbh, beta, restart_name,
					full_individual_restart_name, simulation,
					reference_frame_change_pending);
			// Box growth, after the step and outside it, as in the legacy driver:
			// the controller has already decided on this step's consistent state.
			// Global stepping checks every 7 cycles (runs/BaseTDECompton), skipped
			// when the step was individual or the controller has just entered
			// individual mode.  Individual stepping checks the committed state
			// after every event and grows only at a synchronized event: the exact
			// check runs at every one, and a due growth requests one.  A request
			// the controller answers by switching to global is served on that
			// global boundary.  Individual growth waits while a frame change is
			// pending (the change shifts every position at the next event).
			if(update_box)
			{
				bool const individual_now =
					simulation.GetTimeIntegrationMode() == TimeIntegrationMode::Individual;
				bool grow_now = false;
				if(individual_now && reference_frame_change_pending)
					grow_now = false;
				else if(individual_now)
				{
					if(simulation.IndividualStateSynchronized())
						grow_now = true;
					else if(box_growth_pending || BoxGrowthDue(simulation, 0.5))
					{
						if(!box_growth_pending && rank == 0)
							std::cout << "RICH_UPDATE_BOX_REQUEST cycle=" << simulation.GetCycle()
								<< " time=" << simulation.GetTime()
								<< " (requesting a synchronized individual event)" << std::endl;
						box_growth_pending = true;
						simulation.RequestSynchronizedIndividualEvent();
					}
				}
				else
					grow_now = box_growth_pending ||
						(stepped_globally && simulation.GetCycle() % 7 == 0);
				if(grow_now)
				{
					ComputationalCell3D const reference_cell =
						GetReferenceCell(eos, tess, simulation.GetTime());
					unsigned long long cells_counts[2] = {
						static_cast<unsigned long long>(tess.GetPointNo()), 0};
					auto const resize_start = std::chrono::steady_clock::now();
					Simulation::DomainGrowthReport growth;
					bool const grew = UpdateBoxSynchronized(tess, simulation, 0.5, 1e-5,
						reference_cell, &growth);
					double const resize_seconds = std::chrono::duration<double>(
						std::chrono::steady_clock::now() - resize_start).count();
					box_growth_pending = false;
					if(grew)
					{
						std::pair<Vector3D, Vector3D> const box = tess.GetBoxCoordinates();
						double const newvol = (box.second.x - box.first.x) *
							(box.second.y - box.first.y) * (box.second.z - box.first.z);
						refine.SetSize(newvol);
						remove.SetSize(newvol);
						// The individual growth has told the controller already.
						if(!individual_now)
							simulation.NotifyDomainChanged();
						// Resizes run outside Simulation::step, so their cost is
						// reported here and is missing from RICH_STEP step_s.
						cells_counts[1] = static_cast<unsigned long long>(tess.GetPointNo());
#ifdef RICH_MPI
						MPI_Allreduce(MPI_IN_PLACE, cells_counts, 2, MPI_UNSIGNED_LONG_LONG,
							MPI_SUM, MPI_COMM_WORLD);
#endif
						if(rank == 0)
						{
							std::cout << "RICH_UPDATE_BOX cycle=" << simulation.GetCycle()
								<< " time=" << simulation.GetTime()
								<< " seconds=" << resize_seconds << " (rank-0 wall)"
								<< " cells_before=" << cells_counts[0]
								<< " cells_after=" << cells_counts[1]
								<< " box_ll=" << box.first.x << "," << box.first.y << "," << box.first.z
								<< " box_ur=" << box.second.x << "," << box.second.y << "," << box.second.z
								<< " mode=" << (individual_now ? "individual" : "global");
							if(individual_now)
								std::cout << std::setprecision(12)
									<< " growth_seconds_max=" << growth.seconds
									<< " added_cells=" << growth.added_cells
									<< " reseeded_cells=" << growth.reseeded_cells
									<< " shortened_cells=" << growth.shortened_cells
									<< " seed_bin=" << growth.seed_bin
									<< " smallest_limit=" << growth.smallest_limit
									<< " accelerations_refreshed=" << (growth.accelerations_refreshed ? 1 : 0)
									<< " full_builds=" << growth.mesh_build_timing.full_builds
									<< " build_seconds_rank0=" << growth.mesh_build_timing.seconds
									<< " mass_before=" << growth.mass_before
									<< " mass_after=" << growth.mass_after
									<< " inserted_mass=" << growth.inserted_mass
									<< " energy_before=" << growth.energy_before
									<< " energy_after=" << growth.energy_after
									<< " inserted_energy=" << growth.inserted_energy;
							std::cout << std::endl;
						}
					}
				}
			}
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
