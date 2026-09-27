#include "hdsim_3d.hpp"
#include "newtonian/three_dimensional/simulation/RuntimeLog.hpp"
#include "CourantFriedrichsLewy.hpp"
#include "CFL1D.hpp"
#include "misc/memory_debug.hpp"
#include "misc/memory_profile.hpp"
#include "misc/mpi_wait_profiler.hpp"
#ifdef RICH_MPI
#include <MeshDecomposer3D/load_balancing/HilbertLoadBalancer.hpp>
#endif
#include "3D/tessellation/voronoi/exception/MadVoroException.hpp"
#include "newtonian/three_dimensional/simulation/ActiveMeshView.hpp"
#include "spherical_symmetry/SphericalShellProjector3D.hpp"
#include <algorithm>
#include <tuple>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <cstring>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>


namespace
{
	struct StrictBooleanEnvironment
	{
		bool value;
		bool valid;
	};

	StrictBooleanEnvironment ReadStrictBooleanEnvironment(char const* name)
	{
		char const* const value = std::getenv(name);
		if(value == nullptr || value[0] == '\0' ||
		   std::strcmp(value, "0") == 0 ||
		   std::strcmp(value, "false") == 0 ||
		   std::strcmp(value, "off") == 0 ||
		   std::strcmp(value, "no") == 0)
			return {false, true};
		if(std::strcmp(value, "1") == 0 ||
		   std::strcmp(value, "true") == 0 ||
		   std::strcmp(value, "on") == 0 ||
		   std::strcmp(value, "yes") == 0)
			return {true, true};
		return {false, false};
	}

	// Runtime request for full-versus-partial geometry parity.  The scheduler
	// option `verify_partial_build` is fixed at setup and restored from a
	// checkpoint, so a restart probe cannot switch it on; this honours the
	// documented environment switch at the point of use instead.
	bool VerifyPartialBuildRequested()
	{
		static StrictBooleanEnvironment const parsed =
			ReadStrictBooleanEnvironment("RICH_VERIFY_PARTIAL_BUILD");
		if(!parsed.valid)
			throw std::invalid_argument(
				"RICH_VERIFY_PARTIAL_BUILD must be a strict boolean");
		return parsed.value;
	}

	// RICH_INDIVIDUAL_REEXPAND_VERIFY (debug, strict boolean): after a closure
	// that kept its mesh through re-expansion, rebuild the same targets, as the
	// path without re-expansion would, compare the two meshes before physics
	// runs, and continue on the rebuilt one.
	bool ReexpandVerifyRequested()
	{
		// First read at a collective point (a closure every rank agreed was
		// closed after re-expansion), and it selects collectives, so it must agree.
		static bool const value = []()
		{
			StrictBooleanEnvironment const parsed =
				ReadStrictBooleanEnvironment("RICH_INDIVIDUAL_REEXPAND_VERIFY");
			int code[2] = {parsed.valid ? (parsed.value ? 1 : 0) : -1, 0};
			code[1] = -code[0];
#ifdef RICH_MPI
			MPI_Allreduce(MPI_IN_PLACE, code, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
			if(code[0] < 0 || code[0] != -code[1])
				throw std::invalid_argument(
					"RICH_INDIVIDUAL_REEXPAND_VERIFY must be one strict boolean on every rank");
			return code[0] == 1;
		}();
		return value;
	}

	// Default-on switch: unset or a true spelling enables, a false spelling
	// disables, anything else is rejected.
	bool ReadDefaultOnBooleanEnvironment(char const* name)
	{
		char const* const value = std::getenv(name);
		if(value == nullptr || value[0] == '\0')
			return true;
		StrictBooleanEnvironment const parsed =
			ReadStrictBooleanEnvironment(name);
		if(!parsed.valid)
			throw std::invalid_argument(
				std::string(name) + " must be a strict boolean");
		return parsed.value;
	}

	enum class IndividualEventMeshReuseMode
	{
		Off = 0,
		Selected = 1,
		Shadow = 2,
		Conflict = 3,
		Invalid = 4
	};

	char const* IndividualEventMeshReuseModeName(
		IndividualEventMeshReuseMode mode)
	{
		switch(mode)
		{
		case IndividualEventMeshReuseMode::Off:
			return "off";
		case IndividualEventMeshReuseMode::Selected:
			return "selected";
		case IndividualEventMeshReuseMode::Shadow:
			return "shadow";
		case IndividualEventMeshReuseMode::Conflict:
			return "conflict";
		case IndividualEventMeshReuseMode::Invalid:
			return "invalid";
		}
		return "invalid";
	}

	struct IndividualEventMeshReuseValidationState
	{
		size_t epoch = 0;
		size_t cached_epoch = std::numeric_limits<size_t>::max();
		vector<size_t> canonical_ids;
		size_t attempts = 0;
		size_t selected_hits = 0;
		size_t shadow_checks = 0;
		size_t failed_closed = 0;
	};

	std::unordered_map<HDSim3D const*, IndividualEventMeshReuseValidationState>&
	IndividualEventMeshReuseValidationStates()
	{
		static std::unordered_map<HDSim3D const*,
			IndividualEventMeshReuseValidationState> states;
		return states;
	}

	#ifdef RICH_MPI
	double get_time()
	{
		return MPI_Wtime();
	}
	#else
	std::chrono::time_point<std::chrono::high_resolution_clock> get_time()
	{
		return std::chrono::high_resolution_clock::now();
	}
	#endif

	template <class T>
	void DisplayTime(T const& t1, T const& t2, std::string const& msg)
	{
		if(!RuntimeLogDetailed())
			return;
		#ifdef RICH_MPI
			int rank = -1;
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			if(rank == 0)
				std::cout<<msg<<" "<<t2 - t1<<" seconds"<<std::endl;
		#else
			std::cout<<msg<< std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()<<" mseconds"<<std::endl;
		#endif
	}

	double StepDiagnosticWallTime(void)
	{
		return std::chrono::duration<double>(
			std::chrono::steady_clock::now().time_since_epoch()).count();
	}

	double IndividualHydroWallTime()
	{
		#ifdef RICH_MPI
			return MPI_Wtime();
		#else
			return std::chrono::duration<double>(
				std::chrono::steady_clock::now().time_since_epoch()).count();
		#endif
	}

	// Per rank: the wall seconds of each phase, their total, the active and
	// canonical cell counts, the seconds spent inside MPI calls in each phase
	// (mpi_wait_profiler), and whether that profiler is on.
	constexpr size_t individual_hydro_phase_count = 13;
	constexpr size_t individual_hydro_total_index = individual_hydro_phase_count;
	constexpr size_t individual_hydro_active_index = individual_hydro_phase_count + 1;
	constexpr size_t individual_hydro_canonical_index = individual_hydro_phase_count + 2;
	constexpr size_t individual_hydro_mpi_index = individual_hydro_phase_count + 3;
	constexpr size_t individual_hydro_profiled_index =
		individual_hydro_mpi_index + individual_hydro_phase_count;
	constexpr size_t individual_hydro_record_count = individual_hydro_profiled_index + 1;
	using IndividualHydroPhaseRecord =
		std::array<double, individual_hydro_record_count>;

	void ReportIndividualHydroPhaseTiming(
		IndividualHydroPhaseRecord const& local_record)
	{
		static std::array<char const*, individual_hydro_phase_count> const names = {{
			"setup", "first_mesh", "first_gather_sync",
			"first_update_source_scatter", "generator_prediction", "event_mesh",
			"event_gather_sync", "cell_face", "flux", "extensive", "point_motion",
			"source", "final_update_scatter_reuse"
		}};
		int rank = 0;
		int rank_count = 1;
		#ifdef RICH_MPI
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
		#endif
		std::vector<double> gathered;
		if(rank == 0)
			gathered.resize(static_cast<size_t>(rank_count) *
				individual_hydro_record_count);
		#ifdef RICH_MPI
			MPI_Gather(local_record.data(),
				static_cast<int>(individual_hydro_record_count), MPI_DOUBLE,
				rank == 0 ? gathered.data() : nullptr,
				static_cast<int>(individual_hydro_record_count), MPI_DOUBLE,
				0, MPI_COMM_WORLD);
		#else
			gathered.assign(local_record.begin(), local_record.end());
		#endif
		if(rank != 0)
			return;

		auto statistic = [&](size_t field, double quantile)
		{
			std::vector<double> values(static_cast<size_t>(rank_count));
			for(int source_rank = 0; source_rank < rank_count; ++source_rank)
				values[static_cast<size_t>(source_rank)] =
					gathered[static_cast<size_t>(source_rank) *
						individual_hydro_record_count + field];
			std::sort(values.begin(), values.end());
			if(quantile == 0.5 && values.size() % 2 == 0)
				return 0.5 * (values[values.size() / 2 - 1] +
					values[values.size() / 2]);
			size_t const index = std::min(values.size() - 1,
				static_cast<size_t>(std::ceil(quantile * values.size())) - 1);
			return values[index];
		};
		double active_cells_global = 0;
		double canonical_cells_global = 0;
		for(int source_rank = 0; source_rank < rank_count; ++source_rank)
		{
			size_t const offset = static_cast<size_t>(source_rank) *
				individual_hydro_record_count;
			active_cells_global += gathered[offset + individual_hydro_active_index];
			canonical_cells_global +=
				gathered[offset + individual_hydro_canonical_index];
		}
		std::cout << "INDIVIDUAL_HYDRO_PHASE_TIMING"
			<< " ranks=" << rank_count
			<< " active_cells_global=" << active_cells_global
			<< " canonical_cells_global=" << canonical_cells_global;
		for(size_t phase = 0; phase < individual_hydro_phase_count; ++phase)
			std::cout << " " << names[phase] << "_median=" << statistic(phase, 0.5)
				<< " " << names[phase] << "_p95=" << statistic(phase, 0.95)
				<< " " << names[phase] << "_max=" << statistic(phase, 1.0);
		std::cout << " total_median=" << statistic(individual_hydro_total_index, 0.5)
			<< " total_p95=" << statistic(individual_hydro_total_index, 0.95)
			<< " total_max=" << statistic(individual_hydro_total_index, 1.0);
		// Busy time (wall minus MPI) per phase, when every rank profiled MPI.
		bool profiled = true;
		for(int source_rank = 0; source_rank < rank_count; ++source_rank)
			profiled = profiled && gathered[static_cast<size_t>(source_rank) *
				individual_hydro_record_count + individual_hydro_profiled_index] != 0;
		if(profiled)
			for(size_t phase = 0; phase < individual_hydro_phase_count; ++phase)
			{
				double busy_sum = 0;
				double busy_max = 0;
				double mpi_min = std::numeric_limits<double>::infinity();
				for(int source_rank = 0; source_rank < rank_count; ++source_rank)
				{
					size_t const offset = static_cast<size_t>(source_rank) *
						individual_hydro_record_count;
					double const mpi = gathered[offset + individual_hydro_mpi_index + phase];
					double const busy = gathered[offset + phase] - mpi;
					busy_sum += busy;
					busy_max = std::max(busy_max, busy);
					mpi_min = std::min(mpi_min, mpi);
				}
				std::cout << " " << names[phase] << "_busy_mean=" << busy_sum / rank_count
					<< " " << names[phase] << "_busy_max=" << busy_max
					<< " " << names[phase] << "_mpi_min=" << mpi_min;
			}
		std::cout << std::endl;
	}

	// Agrees a run-wide value across ranks on first use: every rank must pass
	// the same value (unset counts as the fallback), else every rank throws.
	double AgreedIndividualValue(double local, char const* name)
	{
		double extrema[2] = {local, -local};
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, extrema, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
		if(extrema[0] != -extrema[1])
			throw std::invalid_argument(std::string(name) + " differs across MPI ranks");
		return extrema[0];
	}

	// Individual-mode experiment switches, agreed across ranks in one
	// collective on first use (from the event mesh build, the hydro time-step
	// suggestion or the synchronized limits, which every rank reaches in the
	// same order):
	// - RICH_INDIVIDUAL_PARTIAL_BUILD_FRACTION overrides the scheduler's
	//   closure threshold (a rank whose partial target exceeds this fraction of
	//   its owned cells makes the build full; option default 0.5), so partial
	//   builds above the default can be measured.  A number in (0, 1].
	// - RICH_INDIVIDUAL_GUARD_FLOOR: apply (unset, 1, on, true, yes, apply),
	//   report, or off (0, off, false, no); see IndividualGuardFloor.
	// - RICH_INDIVIDUAL_CLOSURE_REEXPAND: strict boolean, default off; see the
	//   closure loop in build_event_mesh_impl.
	// - RICH_INDIVIDUAL_ADJACENCY_SEED_SHELLS: auto (default), 2 or 3 shells seeded
	//   from the adjacency cache before a partial build.
	// - RICH_INDIVIDUAL_PARTIAL_THRESHOLD_ADAPTIVE: strict boolean, default off;
	//   the per-rank closure threshold follows the measured build costs
	//   (UpdateIndividualPartialCostModel); an explicit
	//   RICH_INDIVIDUAL_PARTIAL_BUILD_FRACTION takes precedence.
	enum class IndividualGuardFloorSetting { off = 0, report = 1, apply = 2 };

	struct IndividualExperimentSettings
	{
		double partial_build_fraction;
		IndividualGuardFloorSetting guard_floor;
		bool closure_reexpand;
		int seed_shells;
		bool adaptive_threshold;
	};

	IndividualExperimentSettings const& AgreedIndividualExperimentSettings()
	{
		static IndividualExperimentSettings const settings = []()
		{
			char const* const value = std::getenv("RICH_INDIVIDUAL_PARTIAL_BUILD_FRACTION");
			double fraction = 0;
			if(value != nullptr && value[0] != '\0')
			{
				char* end = nullptr;
				fraction = std::strtod(value, &end);
				if(end == value || *end != '\0' || !std::isfinite(fraction) ||
					!(fraction > 0) || !(fraction <= 1))
					fraction = -1;
			}
			char const* const floor_value = std::getenv("RICH_INDIVIDUAL_GUARD_FLOOR");
			std::string const floor_text = floor_value == nullptr ? "" : floor_value;
			double floor_mode = -1;
			if(floor_text.empty() || floor_text == "1" || floor_text == "on" ||
			   floor_text == "true" || floor_text == "yes" || floor_text == "apply")
				floor_mode = 2;
			else if(floor_text == "report")
				floor_mode = 1;
			else if(floor_text == "0" || floor_text == "off" || floor_text == "false" ||
			   floor_text == "no")
				floor_mode = 0;
			StrictBooleanEnvironment const reexpand =
				ReadStrictBooleanEnvironment("RICH_INDIVIDUAL_CLOSURE_REEXPAND");
			double const reexpand_mode = !reexpand.valid ? -1.0 : (reexpand.value ? 1.0 : 0.0);
			char const* const shells_value =
				std::getenv("RICH_INDIVIDUAL_ADJACENCY_SEED_SHELLS");
			std::string const shells_text = shells_value == nullptr ? "" : shells_value;
			// 0: automatic (see build_event_mesh_impl).
			double const shells = shells_text.empty() || shells_text == "auto" ? 0.0 :
				(shells_text == "2" ? 2.0 : (shells_text == "3" ? 3.0 : -1.0));
			StrictBooleanEnvironment const adaptive =
				ReadStrictBooleanEnvironment("RICH_INDIVIDUAL_PARTIAL_THRESHOLD_ADAPTIVE");
			double const adaptive_mode = !adaptive.valid ? -1.0 : (adaptive.value ? 1.0 : 0.0);
			double extrema[10] = {fraction, -fraction, floor_mode, -floor_mode,
				reexpand_mode, -reexpand_mode, shells, -shells, adaptive_mode, -adaptive_mode};
#ifdef RICH_MPI
			MPI_Allreduce(MPI_IN_PLACE, extrema, 10, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
			if(extrema[8] != -extrema[9])
				throw std::invalid_argument(
					"RICH_INDIVIDUAL_PARTIAL_THRESHOLD_ADAPTIVE differs across MPI ranks");
			if(extrema[8] < 0)
				throw std::invalid_argument(
					"RICH_INDIVIDUAL_PARTIAL_THRESHOLD_ADAPTIVE must be a strict boolean");
			if(extrema[6] != -extrema[7])
				throw std::invalid_argument(
					"RICH_INDIVIDUAL_ADJACENCY_SEED_SHELLS differs across MPI ranks");
			if(extrema[6] < 0)
				throw std::invalid_argument(
					"RICH_INDIVIDUAL_ADJACENCY_SEED_SHELLS must be auto, 2 or 3");
			if(extrema[4] != -extrema[5])
				throw std::invalid_argument(
					"RICH_INDIVIDUAL_CLOSURE_REEXPAND differs across MPI ranks");
			if(extrema[4] < 0)
				throw std::invalid_argument(
					"RICH_INDIVIDUAL_CLOSURE_REEXPAND must be a strict boolean");
			if(extrema[0] != -extrema[1])
				throw std::invalid_argument(
					"RICH_INDIVIDUAL_PARTIAL_BUILD_FRACTION differs across MPI ranks");
			if(extrema[2] != -extrema[3])
				throw std::invalid_argument(
					"RICH_INDIVIDUAL_GUARD_FLOOR differs across MPI ranks");
			if(extrema[0] < 0)
				throw std::invalid_argument(
					"RICH_INDIVIDUAL_PARTIAL_BUILD_FRACTION must be a number in (0, 1]");
			if(extrema[2] < 0)
				throw std::invalid_argument(
					"RICH_INDIVIDUAL_GUARD_FLOOR must be off, report or apply");
			return IndividualExperimentSettings{extrema[0],
				static_cast<IndividualGuardFloorSetting>(static_cast<int>(extrema[2])),
				extrema[4] > 0, static_cast<int>(extrema[6]), extrema[8] > 0};
		}();
		return settings;
	}

	double IndividualPartialBuildFraction(double configured)
	{
		double const override_value =
			AgreedIndividualExperimentSettings().partial_build_fraction;
		return override_value > 0 ? override_value : configured;
	}

	struct IndividualMeshBuildRecord
	{
		char const* result = "none";
		char const* reason = "none";
		size_t initial_target = 0;
		size_t final_target = 0;
		size_t attempts = 0;
		// Why the partial mesh was rebuilt (collective, traced runs only):
		// some rank added target cells, or only depths changed; and how many
		// depth-only rounds were re-expanded on the same mesh instead.
		size_t rebuilds_for_additions = 0;
		size_t rebuilds_depth_only = 0;
		size_t reexpansions = 0;
		double partial_seconds = 0;
		double full_seconds = 0;
	};

	// Rank-0 record per individual event-mesh build (RICH_INDIVIDUAL_PERF_TRACE):
	// the result and its reason, the closure threshold, the per-rank target
	// before and after closure growth (maximum, and the largest fraction of a
	// rank's owned cells with that rank), the partial attempts, and the
	// maximum over ranks of the wall time: whole call, partial attempts, full
	// build.  Result, reason and attempts are decided collectively.
	void ReportIndividualMeshBuild(IndividualMeshBuildRecord const& record,
		char const* stage, double threshold_fraction, size_t canonical_count,
		size_t active_count, double seconds)
	{
		double const owned = std::max(1.0, static_cast<double>(canonical_count));
		double maxima[7] = {
			static_cast<double>(record.initial_target),
			static_cast<double>(record.final_target),
			static_cast<double>(record.initial_target) / owned,
			static_cast<double>(record.final_target) / owned,
			seconds, record.partial_seconds, record.full_seconds};
		double sums[3] = {
			static_cast<double>(record.final_target),
			static_cast<double>(active_count),
			static_cast<double>(canonical_count)};
		int rank = 0;
		struct
		{
			double value;
			int rank;
		} hottest = {maxima[3], 0};
#ifdef RICH_MPI
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		hottest.rank = rank;
		MPI_Allreduce(MPI_IN_PLACE, maxima, 7, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, sums, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &hottest, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
#endif
		if(rank != 0)
			return;
		string const stage_text(stage);
		char const* const mesh = stage_text.find("first-half") == string::npos ? "event" :
			(stage_text.find("shadow") == string::npos ? "first_half" : "first_half_shadow");
		string reason(record.reason);
		std::replace(reason.begin(), reason.end(), ' ', '_');
		std::cout << "INDIVIDUAL_MESH_BUILD mesh=" << mesh
			<< " result=" << record.result << " reason=" << reason
			<< " attempts=" << record.attempts
			<< " rebuilds_additions=" << record.rebuilds_for_additions
			<< " rebuilds_depth_only=" << record.rebuilds_depth_only
			<< " reexpansions=" << record.reexpansions
			<< " threshold=" << threshold_fraction
			<< " active_global=" << sums[1]
			<< " canonical_global=" << sums[2]
			<< " target_initial_max=" << maxima[0]
			<< " target_final_max=" << maxima[1]
			<< " target_final_sum=" << sums[0]
			<< " fraction_initial_max=" << maxima[2]
			<< " fraction_final_max=" << maxima[3]
			<< " fraction_final_max_rank=" << hottest.rank
			<< " seconds_max=" << maxima[4]
			<< " partial_seconds_max=" << maxima[5]
			<< " full_seconds_max=" << maxima[6]
			<< std::endl;
	}

	// Adaptive per-rank closure threshold.  After every event-mesh build: the
	// max over ranks of the partial-attempt and full-build times and of the
	// target fraction (one MPI_MAX), then exponentially weighted updates
	// (factor 0.9, about ten builds of memory) of a linear fit of the time per
	// partial attempt against the largest per-rank target fraction f, of the
	// attempts per partial build, and of the full-build time.  The threshold is
	// the f at which attempts x (a + b f) equals the full-build time, clamped
	// to [0.05, 1]; it moves with the measured costs both ways.  Not counted:
	// the cheaper downstream work on a partial mesh (~0.2 s per TDE event
	// measured), which makes the rule favour full builds.  Collective.
	template<class Model>
	void UpdateIndividualPartialCostModel(Model& model,
		IndividualMeshBuildRecord const& record, size_t canonical_count,
		double threshold, std::uint64_t event_tick)
	{
		double const owned = std::max(1.0, static_cast<double>(canonical_count));
		double maxima[3] = {record.partial_seconds, record.full_seconds,
			static_cast<double>(record.final_target) / owned};
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, maxima, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
		constexpr double decay = 0.9;
		bool const partial = std::strcmp(record.result, "partial") == 0;
		if(record.attempts > 0 && maxima[0] > 0)
		{
			double const x = maxima[2];
			double const y = maxima[0] / static_cast<double>(record.attempts);
			model.w = decay * model.w + 1;
			model.x = decay * model.x + x;
			model.xx = decay * model.xx + x * x;
			model.y = decay * model.y + y;
			model.xy = decay * model.xy + x * y;
			model.attempts = decay * model.attempts +
				static_cast<double>(record.attempts);
			model.attempts_w = decay * model.attempts_w + 1;
			++model.partial_samples;
		}
		if(!partial && maxima[1] > 0)
		{
			model.full = decay * model.full + maxima[1];
			model.full_w = decay * model.full_w + 1;
			++model.full_samples;
		}
		if(model.partial_samples < 4 || model.full_samples < 2)
			return;
		double const full_cost = model.full / model.full_w;
		double const attempts = model.attempts / model.attempts_w;
		double const denominator = model.w * model.xx - model.x * model.x;
		double const slope = denominator > 1e-12 * model.w * model.w ?
			(model.w * model.xy - model.x * model.y) / denominator : 0;
		double const intercept = (model.y - slope * model.x) / model.w;
		double fraction = 1.0;
		if(slope > 0)
			fraction = (full_cost / attempts - intercept) / slope;
		else if(attempts * intercept > full_cost)
			fraction = 0.05;
		fraction = std::min(1.0, std::max(0.05, fraction));
		double const previous = model.fraction > 0 ? model.fraction : threshold;
		model.fraction = fraction;
		int rank = 0;
#ifdef RICH_MPI
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
		if(rank == 0 && std::abs(fraction - previous) > 0.02)
		{
			std::ostringstream line;
			line << std::setprecision(6) << "INDIVIDUAL_PARTIAL_THRESHOLD event_tick="
				<< event_tick << " fraction=" << fraction << " previous=" << previous
				<< " per_attempt_intercept=" << intercept << " per_attempt_slope=" << slope
				<< " attempts=" << attempts << " full_cost=" << full_cost
				<< " partial_samples=" << model.partial_samples
				<< " full_samples=" << model.full_samples;
			std::cout << line.str() << std::endl;
		}
	}

	bool HasPositiveThermalMargin(Conserved3D const& state,
		double mass_threshold)
	{
		if (!(state.mass > 0.0) || !(state.mass > mass_threshold) ||
			!std::isfinite(state.mass) ||
			!std::isfinite(state.energy) ||
			!std::isfinite(abs(state.momentum)))
			return false;
		double const kinetic_energy =
			ScalarProd(state.momentum, state.momentum) /
			(2.0 * state.mass);
		double const internal_energy = state.energy - kinetic_energy;
		double const energy_scale =
			std::max(std::abs(state.energy), std::abs(kinetic_energy));
		return internal_energy > 1e-12 * energy_scale &&
			std::isfinite(internal_energy);
	}

	struct IndividualFaceGeometry
	{
		size_t neighbor;
		double area;
		Vector3D centroid;
		Vector3D normal;
	};

	struct IndividualCellGeometry
	{
		size_t global;
		double volume;
		Vector3D centroid;
		vector<IndividualFaceGeometry> faces;
	};

	size_t GlobalNeighbor(Tessellation3D const& tess,
		size_t local_neighbor,
		size_t global_cell_count)
	{
		auto const& map = tess.GetIndicesInAllPoints();
		auto const found = map.find(local_neighbor);
		if(found == map.end() || found->second >= global_cell_count)
			return std::numeric_limits<size_t>::max();
		return found->second;
	}

	IndividualCellGeometry CaptureCellGeometry(Tessellation3D const& tess,
		size_t local,
		size_t global,
		size_t global_cell_count)
	{
		IndividualCellGeometry result;
		result.global = global;
		result.volume = tess.GetVolume(local);
		result.centroid = tess.GetCellCM(local);
		vector<size_t> neighbors;
		tess.GetNeighbors(local, neighbors);
		face_vec const& faces = tess.GetCellFaces(local);
		if(neighbors.size() != faces.size())
			throw std::logic_error("Voronoi face/neighbor count mismatch during partial parity check");
		result.faces.reserve(faces.size());
		for(size_t i = 0; i < faces.size(); ++i)
		{
			Vector3D normal = tess.Normal(faces[i]);
			double const normal_size = abs(normal);
			if(!(normal_size > 0))
				throw std::logic_error("Zero Voronoi face normal during partial parity check");
			normal *= 1.0 / normal_size;
			if(neighbors[i] < tess.getMeshPoints().size())
			{
				Vector3D const outward = tess.GetMeshPoint(neighbors[i]) -
					tess.GetMeshPoint(local);
				if(ScalarProd(normal, outward) < 0)
					normal *= -1;
			}
			result.faces.push_back({GlobalNeighbor(tess, neighbors[i], global_cell_count),
				tess.GetArea(faces[i]), tess.FaceCM(faces[i]), normal});
		}
		return result;
	}

	vector<IndividualCellGeometry> CaptureSelectedGeometry(
		Tessellation3D const& tess,
		ActiveMeshView const& view,
		vector<size_t> const& active,
		size_t global_cell_count)
	{
		vector<IndividualCellGeometry> result;
		result.reserve(active.size());
		for(size_t global : active)
		{
			if(!view.containsGlobal(global))
				throw std::logic_error(
					"Selected cell missing during partial parity check");
			result.push_back(CaptureCellGeometry(tess, view.globalToLocal(global),
				global, global_cell_count));
		}
		return result;
	}

	bool GeometryClose(double left, double right, double relative = 2e-8)
	{
		return std::abs(left - right) <= relative *
			std::max({1.0, std::abs(left), std::abs(right)});
	}

	bool GeometryClose(Vector3D const& left, Vector3D const& right,
		double relative = 2e-8)
	{
		return GeometryClose(left.x, right.x, relative) &&
			GeometryClose(left.y, right.y, relative) &&
			GeometryClose(left.z, right.z, relative);
	}

	bool GeometryCloseAtScale(Vector3D const& left, Vector3D const& right,
		double scale, double relative = 2e-8)
	{
		double const safe_scale = std::max(1.0, std::abs(scale));
		return GeometryClose(left.x / safe_scale, right.x / safe_scale, relative) &&
			GeometryClose(left.y / safe_scale, right.y / safe_scale, relative) &&
			GeometryClose(left.z / safe_scale, right.z / safe_scale, relative);
	}

	bool ActiveGeometryMatches(vector<IndividualCellGeometry> const& partial,
		Tessellation3D const& full,
		ActiveMeshView const& full_view,
		size_t global_cell_count,
		bool report_mismatch = true)
	{
		for(IndividualCellGeometry const& expected : partial)
		{
			size_t const full_local = full_view.globalToLocal(expected.global);
			IndividualCellGeometry const candidate = CaptureCellGeometry(full,
				full_local, expected.global,
				global_cell_count);
			double const cell_width = full.GetWidth(full_local);
			if(!GeometryClose(expected.volume, candidate.volume))
			{
				if(report_mismatch)
					std::cout << "Individual partial parity mismatch: cell "
						<< expected.global << " volume partial=" << expected.volume
						<< " full=" << candidate.volume << std::endl;
				return false;
			}
			if(!GeometryCloseAtScale(expected.centroid, candidate.centroid,
				cell_width))
			{
				if(report_mismatch)
					std::cout << "Individual partial parity mismatch: cell "
						<< expected.global << " centroid partial=" << expected.centroid
						<< " full=" << candidate.centroid << std::endl;
				return false;
			}
			if(expected.faces.size() != candidate.faces.size())
			{
				if(report_mismatch)
					std::cout << "Individual partial parity mismatch: cell "
						<< expected.global << " face count partial="
						<< expected.faces.size() << " full=" << candidate.faces.size()
						<< std::endl;
				return false;
			}
			vector<unsigned char> used(candidate.faces.size(), 0);
			for(IndividualFaceGeometry const& expected_face : expected.faces)
			{
				size_t best = candidate.faces.size();
				double best_distance = std::numeric_limits<double>::max();
				for(size_t i = 0; i < candidate.faces.size(); ++i)
				{
					if(used[i] || candidate.faces[i].neighbor != expected_face.neighbor)
						continue;
					double const distance = abs(candidate.faces[i].centroid -
						expected_face.centroid);
					if(distance < best_distance)
					{
						best = i;
						best_distance = distance;
					}
				}
				if(best == candidate.faces.size())
				{
					if(report_mismatch)
						std::cout << "Individual partial parity mismatch: cell "
							<< expected.global << " has no full face for neighbor "
							<< expected_face.neighbor << std::endl;
					return false;
				}
				if(!GeometryClose(expected_face.area, candidate.faces[best].area) ||
				   !GeometryCloseAtScale(expected_face.centroid,
					candidate.faces[best].centroid, cell_width) ||
				   !GeometryClose(expected_face.normal, candidate.faces[best].normal))
				{
					if(report_mismatch)
						std::cout << "Individual partial parity mismatch: cell "
							<< expected.global << " neighbor " << expected_face.neighbor
							<< " face area partial=" << expected_face.area
							<< " full=" << candidate.faces[best].area
							<< " centroid partial=" << expected_face.centroid
							<< " full=" << candidate.faces[best].centroid
							<< " normal partial=" << expected_face.normal
							<< " full=" << candidate.faces[best].normal << std::endl;
					return false;
				}
				used[best] = 1;
			}
		}
		return true;
	}

#ifdef RICH_MPI
	// Identity of every mesh point for one event: (owning rank, canonical index
	// on that rank); points outside the box are (-2, face-independent sentinel).
	// Collective (SyncPartialBuildData).
	vector<std::pair<int, size_t> > MeshPointIdentities(Tessellation3D& tess,
		ActiveMeshView const& view, size_t canonical_count, int rank)
	{
		size_t const mesh_size = tess.getMeshPoints().size();
		vector<int> owner(mesh_size, -1);
		auto const& procs = tess.GetDuplicatedProcs();
		auto const& ghosts = tess.GetGhostIndeces();
		for(size_t peer = 0; peer < ghosts.size() && peer < procs.size(); ++peer)
			for(size_t const ghost : ghosts[peer])
				if(ghost < mesh_size)
					owner[ghost] = procs[peer];
		vector<size_t> canonical(canonical_count);
		std::iota(canonical.begin(), canonical.end(), 0);
		vector<size_t> mesh_index(tess.GetPointNo(), ActiveMeshView::invalidIndex());
		for(size_t local = 0; local < view.localSize() && local < mesh_index.size(); ++local)
			mesh_index[local] = view.localToGlobal(local);
		tess.SyncPartialBuildData(mesh_index, canonical);
		vector<std::pair<int, size_t> > identities(mesh_size,
			std::make_pair(-1, ActiveMeshView::invalidIndex()));
		for(size_t point = 0; point < mesh_size; ++point)
		{
			if(tess.IsPointOutsideBox(point))
				identities[point] = std::make_pair(-2, std::numeric_limits<size_t>::max());
			else if(point < mesh_index.size())
				identities[point] = std::make_pair(point < tess.GetPointNo() ? rank : owner[point],
					mesh_index[point]);
		}
		return identities;
	}

	struct IdentifiedCellGeometry
	{
		size_t global = 0;
		double volume = 0;
		double width = 0;
		Vector3D centroid;
		// (neighbour identity, area, face centroid), sorted by identity then area.
		vector<std::tuple<std::pair<int, size_t>, double, Vector3D> > faces;
	};

	vector<IdentifiedCellGeometry> CaptureIdentifiedGeometry(Tessellation3D& tess,
		ActiveMeshView const& view, vector<size_t> const& cells, size_t canonical_count, int rank)
	{
		vector<std::pair<int, size_t> > const identities =
			MeshPointIdentities(tess, view, canonical_count, rank);
		vector<IdentifiedCellGeometry> result;
		result.reserve(cells.size());
		vector<size_t> neighbors;
		for(size_t const global : cells)
		{
			if(!view.containsGlobal(global))
				throw std::logic_error("Re-expansion check: selected cell missing from the mesh");
			size_t const local = view.globalToLocal(global);
			IdentifiedCellGeometry cell;
			cell.global = global;
			cell.volume = tess.GetVolume(local);
			cell.width = tess.GetWidth(local);
			cell.centroid = tess.GetCellCM(local);
			tess.GetNeighbors(local, neighbors);
			face_vec const& faces = tess.GetCellFaces(local);
			for(size_t i = 0; i < faces.size() && i < neighbors.size(); ++i)
				cell.faces.emplace_back(neighbors[i] < identities.size() ? identities[neighbors[i]] :
					std::make_pair(-1, ActiveMeshView::invalidIndex()), tess.GetArea(faces[i]), tess.FaceCM(faces[i]));
			std::sort(cell.faces.begin(), cell.faces.end(), [](auto const& a, auto const& b)
				{return std::get<0>(a) != std::get<0>(b) ? std::get<0>(a) < std::get<0>(b) : std::get<1>(a) < std::get<1>(b);});
			result.push_back(std::move(cell));
		}
		return result;
	}

	// Collective: compares the kept and the rebuilt mesh cell by cell and prints
	// one rank-0 line; topology must match exactly, geometry is measured.
	void ReportReexpandComparison(vector<IdentifiedCellGeometry> const& kept,
		vector<IdentifiedCellGeometry> const& rebuilt, std::uint64_t event_tick)
	{
		double worst[4] = {0, 0, 0, 0}; // volume, area (relative), cell and face centroid (/width)
		unsigned long long counts[3] = {kept.size(), 0, 0}; // cells, topology mismatches, faces
		bool const same_cells = kept.size() == rebuilt.size();
		for(size_t c = 0; c < kept.size(); ++c)
		{
			IdentifiedCellGeometry const& a = kept[c];
			if(!same_cells || rebuilt[c].global != a.global || rebuilt[c].faces.size() != a.faces.size())
			{
				++counts[1];
				continue;
			}
			IdentifiedCellGeometry const& b = rebuilt[c];
			double const width = std::max(a.width, std::numeric_limits<double>::min());
			worst[0] = std::max(worst[0], std::abs(a.volume - b.volume) / std::max(std::abs(a.volume), 1e-300));
			worst[2] = std::max(worst[2], abs(a.centroid - b.centroid) / width);
			bool topology = true;
			for(size_t f = 0; f < a.faces.size(); ++f)
			{
				if(std::get<0>(a.faces[f]) != std::get<0>(b.faces[f]))
				{
					topology = false;
					break;
				}
				++counts[2];
				double const area = std::get<1>(a.faces[f]);
				worst[1] = std::max(worst[1], std::abs(area - std::get<1>(b.faces[f])) /
					std::max(std::abs(area), 1e-300 + 1e-12 * width * width));
				worst[3] = std::max(worst[3], abs(std::get<2>(a.faces[f]) - std::get<2>(b.faces[f])) / width);
			}
			if(!topology)
				++counts[1];
		}
		int rank = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Allreduce(MPI_IN_PLACE, counts, 3, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, worst, 4, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		if(rank == 0)
			std::cout << "INDIVIDUAL_REEXPAND_VERIFY event_tick=" << event_tick << " cells=" << counts[0]
				<< " faces=" << counts[2] << " topology_mismatch_cells=" << counts[1]
				<< std::setprecision(3) << " max_rel_volume=" << worst[0] << " max_rel_area=" << worst[1]
				<< " max_centroid_shift_widths=" << worst[2] << " max_face_centroid_shift_widths=" << worst[3]
				<< std::endl;
	}
#endif

	// Cells named in RICH_INDIVIDUAL_TRACE_CELL_IDS (comma-separated stable
	// IDs), sorted; empty when unset.  The value must agree on every rank: the
	// trace gathers to rank 0 at every event while it is non-empty.
	std::vector<std::size_t> const& IndividualTraceCellIds()
	{
		static std::vector<std::size_t> const ids = []()
		{
			std::vector<std::size_t> parsed;
			char const* const value = std::getenv("RICH_INDIVIDUAL_TRACE_CELL_IDS");
			if(value == nullptr || value[0] == '\0')
				return parsed;
			std::string const text(value);
			std::size_t begin = 0;
			while(begin <= text.size())
			{
				std::size_t const end = std::min(text.find(',', begin), text.size());
				std::string const token = text.substr(begin, end - begin);
				if(token.empty() || token.find_first_not_of("0123456789") != std::string::npos)
					throw std::invalid_argument(
						"RICH_INDIVIDUAL_TRACE_CELL_IDS must be a comma-separated list of cell IDs");
				parsed.push_back(static_cast<std::size_t>(std::stoull(token)));
				begin = end + 1;
			}
			std::sort(parsed.begin(), parsed.end());
			parsed.erase(std::unique(parsed.begin(), parsed.end()), parsed.end());
			return parsed;
		}();
		return ids;
	}

	// Collective: every rank passes its records (possibly empty); rank 0
	// prints them in rank order.
	void EmitIndividualCellTrace(std::string const& local_records)
	{
#ifdef RICH_MPI
		int rank = 0;
		int size = 1;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		int const length = static_cast<int>(local_records.size());
		std::vector<int> lengths(rank == 0 ? size : 0);
		MPI_Gather(&length, 1, MPI_INT, lengths.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
		std::vector<int> offsets;
		std::vector<char> gathered;
		if(rank == 0)
		{
			offsets.assign(size, 0);
			for(int peer = 1; peer < size; ++peer)
				offsets[peer] = offsets[peer - 1] + lengths[peer - 1];
			gathered.resize(static_cast<std::size_t>(offsets.back() + lengths.back()));
		}
		MPI_Gatherv(local_records.data(), length, MPI_CHAR, gathered.data(),
			lengths.data(), offsets.data(), MPI_CHAR, 0, MPI_COMM_WORLD);
		if(rank == 0 && !gathered.empty())
			std::cout.write(gathered.data(), static_cast<std::streamsize>(gathered.size()));
		if(rank == 0)
			std::cout << std::flush;
#else
		std::cout << local_records << std::flush;
#endif
	}
}

void InvalidateIndividualEventMeshReuseValidation(HDSim3D const& simulation)
{
	IndividualEventMeshReuseValidationState& state =
		IndividualEventMeshReuseValidationStates()[&simulation];
	++state.epoch;
	state.cached_epoch = std::numeric_limits<size_t>::max();
	state.canonical_ids.clear();
}

Tessellation3D& HDSim3D::getTessellation(void)
{
	return tess_;
}

vector<ComputationalCell3D>& HDSim3D::getCells(void)
{
	return cells_;
}

vector<Conserved3D>& HDSim3D::getExtensives(void)
{
	return extensive_;
}

const vector<Conserved3D>& HDSim3D::getExtensives(void) const
{
	return extensive_;
}

void HDSim3D::SetSphericalShellProjector(
	std::shared_ptr<SphericalShellProjector3D> projector,
	FluxCalculator3D const& perturbation_flux_calculator)
{
	spherical_shell_projector_ = std::move(projector);
	spherical_shell_geometry_ = spherical_shell_projector_ ?
		spherical_shell_projector_->GetShellGeometry() : nullptr;
	spherical_perturbation_flux_calculator_ =
		spherical_shell_projector_ ? &perturbation_flux_calculator : nullptr;
	RefreshSphericalShellGeometry("initial mesh", true);
}

void HDSim3D::RefreshSphericalShellGeometry(
	char const* update_name,
	bool always_report)
{
	if (!spherical_shell_geometry_)
		return;
	bool const mesh_changed = spherical_shell_geometry_->Update(tess_);
	if (!always_report && !mesh_changed)
		return;
#ifdef RICH_MPI
	int rank = -1;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank != 0)
		return;
#endif
	std::cout << "Spherical shell update (" << update_name << "): "
		<< spherical_shell_geometry_->GetShellRadii().size()
		<< " shells" << std::endl;
}

FluxCalculator3D const&
HDSim3D::GetSphericalPerturbationFluxCalculator(void) const
{
	if (!spherical_perturbation_flux_calculator_)
		throw UniversalError(
			"Spherical symmetry requires a perturbation flux calculator");
	return *spherical_perturbation_flux_calculator_;
}

FluxCalculator3D const& HDSim3D::GetFullStateFluxCalculator(void) const
{
	return spherical_shell_projector_ ?
		GetSphericalPerturbationFluxCalculator() : fc_;
}

void HDSim3D::ApplySphericalBackgroundCorrection(
	vector<ComputationalCell3D> const& stage_input_cells,
	vector<Conserved3D> const& stage_input_extensives,
	vector<Vector3D> const& face_velocities,
	vector<Vector3D> const& point_velocities,
	double time,
	double dt,
	bool source_before_extensive_update,
	vector<Conserved3D>& full_candidate)
{
	if (!spherical_shell_projector_)
		return;

	size_t const local_count = tess_.GetPointNo();
	vector<Conserved3D> background_extensives;
	spherical_shell_projector_->ProjectExtensives(tess_,
		stage_input_extensives, background_extensives);
	double maximum_background_deviation = 0;
	for (size_t i = 0; i < local_count; ++i) {
		double scale =
			std::abs(background_extensives[i].mass) +
			abs(background_extensives[i].momentum) +
			std::abs(background_extensives[i].energy) +
			std::abs(background_extensives[i].internal_energy);
		double deviation =
			std::abs(stage_input_extensives[i].mass -
				background_extensives[i].mass) +
			abs(stage_input_extensives[i].momentum -
				background_extensives[i].momentum) +
			std::abs(stage_input_extensives[i].energy -
				background_extensives[i].energy) +
			std::abs(stage_input_extensives[i].internal_energy -
				background_extensives[i].internal_energy);
		for (size_t tracer = 0; tracer < MAX_TRACERS; ++tracer) {
			scale += std::abs(background_extensives[i].tracers[tracer]);
			deviation += std::abs(stage_input_extensives[i].tracers[tracer] -
				background_extensives[i].tracers[tracer]);
		}
		maximum_background_deviation =
			std::max(maximum_background_deviation,
				deviation / std::max(scale, 1e-300));
	}
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &maximum_background_deviation, 1,
		MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
	bool const symmetric_to_roundoff =
		maximum_background_deviation <= 1e-12;
	vector<ComputationalCell3D> background_cells = stage_input_cells;
	background_extensives.resize(local_count);
	background_cells.resize(local_count);
	cu_(background_cells, eos_, tess_, background_extensives);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, background_extensives, true);
	MPI_exchange_data(tess_, background_cells, true);
#endif

	vector<Conserved3D> background_fluxes;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >
		background_face_values;
	fc_.Calculate(background_fluxes, tess_, face_velocities,
		background_cells, background_extensives, eos_, time, dt,
		background_face_values);

	vector<Conserved3D> background_candidate = background_extensives;
	if (source_before_extensive_update) {
		double const source_start = StepDiagnosticWallTime();
		source_(tess_, background_cells, background_fluxes, point_velocities,
			time, dt, background_candidate);
		last_source_step_timing_.second_seconds +=
			StepDiagnosticWallTime() - source_start;
		++last_source_step_timing_.calls;
		eu_(background_fluxes, tess_, dt, background_cells,
			background_candidate, time, face_velocities,
			point_velocities, background_face_values);
	}
	else {
		eu_(background_fluxes, tess_, dt, background_cells,
			background_candidate, time, face_velocities,
			point_velocities, background_face_values);
		double const source_start = StepDiagnosticWallTime();
		source_(tess_, background_cells, background_fluxes, point_velocities,
			time, dt, background_candidate);
		last_source_step_timing_.first_seconds +=
			StepDiagnosticWallTime() - source_start;
		++last_source_step_timing_.calls;
	}

	vector<Conserved3D> background_rate = background_candidate;
	for (size_t i = 0; i < local_count; ++i)
		background_rate[i] -= background_extensives[i];
	vector<Conserved3D> projected_background_rate;
	spherical_shell_projector_->ProjectRates(tess_,
		background_rate, projected_background_rate);

	// Evolve a nonspherical state with the ordinary 3-D operator and only
	// replace the angular error of the spherical-background contribution:
	// P L_S(PU) + L(U) - L_S(PU).
	if (!symmetric_to_roundoff)
		++spherical_perturbation_evaluation_count_;

	for (size_t i = 0; i < local_count; ++i) {
		if (symmetric_to_roundoff) {
			full_candidate[i] = background_extensives[i] +
				projected_background_rate[i];
		}
		else {
			Conserved3D const full_rate =
				full_candidate[i] - stage_input_extensives[i];
			full_candidate[i] = stage_input_extensives[i] +
				projected_background_rate[i] +
				(full_rate - background_rate[i]);
		}
		if (full_candidate[i].mass > 0)
			full_candidate[i].internal_energy =
				full_candidate[i].energy -
				ScalarProd(full_candidate[i].momentum,
					full_candidate[i].momentum) /
				(2.0 * full_candidate[i].mass);
	}
}

HDSim3D::HDSim3D(Tessellation3D& tess,
	vector<ComputationalCell3D>& cells,
	vector<Conserved3D>& extensives,
	const EquationOfState& eos,
	ProgressTracker &pt,
	const PointMotion3D& pm,
	TimeStepFunction3D& tsc,
	const FluxCalculator3D& fc,
	const CellUpdater3D& cu,
	const ExtensiveUpdater3D& eu,
	const SourceTerm3D& source,
	const pair<vector<string>, vector<string> >& tsn,
	bool SR
	#ifdef RICH_MPI
	, std::shared_ptr<CostCalculator3D> cost_calc
	#endif // RICH_MPI
	) :
	tess_(tess),
	eos_(eos), cells_(cells), extensive_(extensives), pm_(pm), tsc_(tsc), fc_(fc), cu_(cu), eu_(eu), source_(source), pt_(pt),
	special_relativity_(SR)
	#ifdef RICH_MPI
	, cost_calc_(cost_calc), exchange_chain_(MPI_COMM_WORLD)
	#endif // RICH_MPI
{
	const bool validity_check = tess.GetPointNo() <= cells.size();
	assert(validity_check);
	assert(tsn.second.size() <= MAX_STICKERS);
	assert(tsn.first.size() <= MAX_TRACERS);
	// sort tracers and stickers
	size_t N = tess.GetPointNo();
	vector<size_t> tindex = sort_index(tsn.first);
	vector<size_t> sindex = sort_index(tsn.second);
	ComputationalCell3D::tracerNames = VectorValues(tsn.first, tindex);
	ComputationalCell3D::stickerNames = VectorValues(tsn.second, sindex);
	for (size_t i = 0; i < N; ++i)
	{
		for (size_t j = 0; j < tindex.size(); ++j)
			cells_[i].tracers[j] = cells[i].tracers[tindex[j]];
		for (size_t j = 0; j < sindex.size(); ++j)
			cells_[i].stickers[j] = cells[i].stickers[sindex[j]];
	}

#ifdef RICH_MPI
	ComputationalCell3D cdummy;
	MPI_exchange_data(tess_, cells_, true);
#endif
	extensive_.resize(N);
	if (SR)
	{
		for (size_t i = 0; i < N; ++i)
			PrimitiveToConservedSR(cells_[i], tess.GetVolume(i), extensive_[i], eos_);
	}
	else
	{
		for (size_t i = 0; i < N; ++i)
			PrimitiveToConserved(cells_[i], tess.GetVolume(i), extensive_[i]);
	}
}

namespace
{
	void CalcFaceVelocities(Tessellation3D const& tess, vector<Vector3D> const& point_vel, vector<Vector3D>& res)
	{
		size_t N = tess.GetTotalFacesNumber();
		res.resize(N);
		for (size_t i = 0; i < N; ++i)
		{
			if (tess.BoundaryFace(i))
				res[i] = Vector3D();
			else
			{
				try
				{
					res[i] = tess.CalcFaceVelocity(i, point_vel[tess.GetFaceNeighbors(i).first], point_vel[tess.GetFaceNeighbors(i).second]);
				}
				catch (UniversalError & /*eo*/)
				{
					throw;
				}
			}
		}
	}

	void MovePoints(Tessellation3D& tess, std::vector<Vector3D> const& point_vel, double const dt)
	{
		size_t const N = tess.GetPointNo();
		std::vector<Vector3D>& points = tess.accessMeshPoints();
		for(size_t i = 0; i < N; ++i)
			points[i] += point_vel[i] * dt;
	}

	#ifdef RICH_MPI
		void UpdateTessellation(Tessellation3D& tess, const vector<Vector3D>& point_vel, double dt, ExchangeChain &chain, vector<Vector3D> &points, std::vector<Vector3D> const* orgpoints = nullptr)
	#else // RICH_MPI
		void UpdateTessellation(Tessellation3D& tess, const vector<Vector3D>& point_vel, double dt, vector<Vector3D> &points, std::vector<Vector3D> const* orgpoints = nullptr)
	#endif // RICH_MPI
	{
		MEMORY_PROFILE_SCOPE("tessellation rebuild");
		if (orgpoints == nullptr)
		{
			const vector<Vector3D> &mesh = tess.getMeshPoints();
			points.assign(mesh.begin(), mesh.end());
		}
		else
			points = *orgpoints;
		points.resize(tess.GetPointNo());
		if(orgpoints != nullptr)
		{
			size_t const N = points.size();
			for (size_t i = 0; i < N; ++i)
				points[i] += point_vel[i] * dt;
		}
		
		#ifdef RICH_MPI
		tess.BuildParallel(points);
		chain.Exchange(tess.GetSentProcs(), tess.GetSentPoints(), tess.GetSelfIndex());
		#else // RICH_MPI
		tess.Build(points);
		#endif // RICH_MPI
	}

	void ExtensiveAvg(vector<Conserved3D>& res, vector<Conserved3D> const& other)
	{
		assert(res.size() == other.size());
		size_t N = res.size();
		for (size_t i = 0; i < N; ++i)
		{
			res[i] += other[i];
			res[i] *= 0.5;
			double kinetic = 0.5 * ScalarProd(res[i].momentum, res[i].momentum) / res[i].mass;
    		res[i].internal_energy = res[i].energy - kinetic;
		}
	}

	std::pair<std::vector<size_t>, std::vector<size_t>>
	FindXBoundaryFaces(const Tessellation3D& tess)
	{
		auto box = tess.GetBoxCoordinates();
		Vector3D ll = box.first;
		Vector3D ur = box.second;
		size_t Norg = tess.GetPointNo();
		std::vector<size_t> left_faces, right_faces;
		size_t Nfaces = tess.GetTotalFacesNumber();
		for (size_t i = 0; i < Nfaces; ++i)
		{
			if (!tess.BoundaryFace(i))
				continue;
			Vector3D normal = normalize(tess.Normal(i));
			if (std::abs(normal.x) < 0.5)
				continue;
			size_t n0 = tess.GetFaceNeighbors(i).first;
			size_t n1 = tess.GetFaceNeighbors(i).second;
			size_t ghost = (n0 < Norg) ? n1 : n0;
			Vector3D ghost_point = tess.GetMeshPoint(ghost);
			if (ghost_point.x < ll.x)
				left_faces.push_back(i);
			else if (ghost_point.x > ur.x)
				right_faces.push_back(i);
		}
		return std::make_pair(left_faces, right_faces);
	}

	double SetBoundaryFaceVelocities(
		const std::vector<size_t>& face_indices,
		const Tessellation3D& tess,
		const std::vector<ComputationalCell3D>& cells,
		const EquationOfState& eos,
		const Hllc3D& hllc,
		const ComputationalCell3D* external_state,
		std::vector<Vector3D>& face_vel)
	{
		double weighted_vx = 0;
		double total_area = 0;
		size_t Norg = tess.GetPointNo();
		for (size_t fi : face_indices)
		{
			const std::pair<size_t, size_t>& neighbors = tess.GetFaceNeighbors(fi);
			size_t n0 = neighbors.first;
			size_t n1 = neighbors.second;
			size_t interior = (n0 < Norg) ? n0 : n1;
			Vector3D raw_normal = normalize(tess.Normal(fi));
			double sign = (n0 < Norg) ? 1.0 : -1.0;
			double nx_sign = (raw_normal.x * sign > 0) ? 1.0 : -1.0;
			Vector3D outward_normal(nx_sign, 0, 0);

			double contact_speed;
			if (external_state == nullptr)
			{
				const ComputationalCell3D& cell = cells[interior];
				ComputationalCell3D vacuum_state = cell;
				double vac_factor = 1e-10;
				vacuum_state.velocity = Vector3D(0, 0, 0);
				vacuum_state.density = cell.density * vac_factor;
				vacuum_state.pressure = cell.pressure * vac_factor * 0.01;
				vacuum_state.internal_energy = eos.dp2e(vacuum_state.density,
					vacuum_state.pressure, cell.tracers, ComputationalCell3D::tracerNames);
				std::pair<double, double> ustar_pstar = hllc.GetUstarPstar(
					cell, vacuum_state, eos, outward_normal);
				contact_speed = ustar_pstar.first;
			}
			else
			{
				std::pair<double, double> ustar_pstar = hllc.GetUstarPstar(
					cells[interior], *external_state, eos, outward_normal);
				contact_speed = ustar_pstar.first;
			}
			face_vel[fi] = contact_speed * outward_normal;
			double area = tess.GetArea(fi);
			weighted_vx += contact_speed * outward_normal.x * area;
			total_area += area;
		}
		return (total_area > 0) ? (weighted_vx / total_area) : 0.0;
	}

	double OverrideBoundaryFluxes(
		const std::vector<size_t>& face_indices,
		const Tessellation3D& tess,
		const std::vector<ComputationalCell3D>& cells,
		const EquationOfState& eos,
		const Hllc3D& hllc,
		const ComputationalCell3D* external_state,
		std::vector<Conserved3D>& fluxes,
		std::vector<Vector3D>& face_vel)
	{
		double weighted_vx = 0;
		double total_area = 0;
		size_t Norg = tess.GetPointNo();
		for (size_t fi : face_indices)
		{
			const std::pair<size_t, size_t>& neighbors = tess.GetFaceNeighbors(fi);
			size_t n0 = neighbors.first;
			size_t n1 = neighbors.second;
			size_t interior = (n0 < Norg) ? n0 : n1;
			Vector3D raw_normal = normalize(tess.Normal(fi));
			double sign = (n0 < Norg) ? 1.0 : -1.0;
			double nx_sign = (raw_normal.x * sign > 0) ? 1.0 : -1.0;
			Vector3D outward_normal(nx_sign, 0, 0);

			double contact_speed;
			if (external_state == nullptr)
			{
				const ComputationalCell3D& cell = cells[interior];
				ComputationalCell3D vacuum_state = cell;
				double vac_factor = 1e-10;
				vacuum_state.density = cell.density * vac_factor;
				vacuum_state.pressure = cell.pressure * vac_factor;
				vacuum_state.internal_energy = eos.dp2e(vacuum_state.density,
					vacuum_state.pressure, cell.tracers, ComputationalCell3D::tracerNames);
				std::pair<double, double> ustar_pstar = hllc.GetUstarPstar(
					cell, vacuum_state, eos, outward_normal);
				contact_speed = ustar_pstar.first;
				fluxes[fi] = Conserved3D();
			}
			else
			{
				std::pair<double, double> ustar_pstar = hllc.GetUstarPstar(
					cells[interior], *external_state, eos, outward_normal);
				contact_speed = ustar_pstar.first;
				Vector3D fv = contact_speed * outward_normal;
				RotateSolveBack3D(outward_normal, cells[interior], *external_state,
					fv, hllc, fluxes[fi], eos);
			}
			face_vel[fi] = contact_speed * outward_normal;
			double area = tess.GetArea(fi);
			weighted_vx += contact_speed * outward_normal.x * area;
			total_area += area;
		}
		return (total_area > 0) ? (weighted_vx / total_area) : 0.0;
	}

	void SetBoxAndRebuild(Tessellation3D& tess, const Vector3D& new_ll, const Vector3D& new_ur,
		std::vector<Vector3D>& points_scratch
	#ifdef RICH_MPI
		, ExchangeChain& chain
	#endif
	)
	{
		tess.SetBox(new_ll, new_ur);
		const vector<Vector3D>& mesh = tess.getMeshPoints();
		points_scratch.assign(mesh.begin(), mesh.end());
		points_scratch.resize(tess.GetPointNo());
	#ifdef RICH_MPI
		tess.BuildParallel(points_scratch);
		chain.Exchange(tess.GetSentProcs(), tess.GetSentPoints(),
			tess.GetSelfIndex());
	#else
		tess.Build(points_scratch);
	#endif
	}
}


void HDSim3D::timeAdvance2(void)
{
	last_source_step_timing_ = SourceStepTiming();
	last_mesh_build_timing_ = MeshBuildTiming();
	RefreshSphericalShellGeometry("timeAdvance2 entry");
	MEMORY_PROFILE_SCOPE("hydro timeAdvance2");
#ifdef RICH_MPI
	this->exchange_chain_.Reset(tess_.GetPointNo());
#endif // RICH_MPI
	MEMORY_DEBUG_PRINT("hydro: after MPI reset");
	const double time = pt_.getTime();
	vector<Vector3D> &point_vel = this->point_vel_scratch_;
	vector<Vector3D> &face_vel = this->face_vel_scratch_;
	vector<Conserved3D> &fluxes = this->fluxes_scratch_;
	vector<Conserved3D> &mid_extensives = this->mid_extensives_scratch_;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > &face_values = this->face_values_scratch_;
	point_vel.clear();
	face_vel.clear();
	face_vel_build_generation_ = kNoFaceVelocityGeneration;
	fluxes.clear();
	face_values.clear();
	pm_(tess_, cells_, time, point_vel);
#ifdef RICH_MPI
	Vector3D vdummy;
	MPI_exchange_data(tess_, point_vel, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	double dt = tsc_(tess_, cells_, eos_, face_vel, time);
	pm_.ApplyFix(tess_, cells_, time, dt, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	if (auto* cfl = dynamic_cast<CourantFriedrichsLewy*>(&tsc_))
		cfl->SetPointVelocities(&point_vel);
	else if (auto* cfl1d = dynamic_cast<CFL1D*>(&tsc_))
		cfl1d->SetPointVelocities(&point_vel);
	dt = tsc_(tess_, cells_, eos_, face_vel, time);
	MEMORY_DEBUG_PRINT("hydro: after CFL + face velocities");
	auto t1 = get_time();
	auto t2 = t1;
	GetFullStateFluxCalculator().Calculate(
		fluxes, tess_, face_vel, cells_, extensive_,
		eos_, time, dt, face_values);
	MEMORY_DEBUG_PRINT("hydro: after flux calc");
	mid_extensives = extensive_;
	eu_(fluxes, tess_, dt, cells_, mid_extensives, time,
		face_vel, point_vel, face_values);
	MEMORY_DEBUG_PRINT("hydro: after extensive update");
	t1 = get_time();
	double const first_source_start = StepDiagnosticWallTime();
	source_(tess_, cells_, fluxes, point_vel, time, dt,
		mid_extensives);
	last_source_step_timing_.first_seconds +=
		StepDiagnosticWallTime() - first_source_start;
	++last_source_step_timing_.calls;
	t2 = get_time();
	DisplayTime(t1, t2, "Source time ");
	ApplySphericalBackgroundCorrection(cells_, extensive_, face_vel,
		point_vel, time, dt, false, mid_extensives);
	MEMORY_DEBUG_PRINT("hydro: after source terms");
	// if (pt_.getCycle() % 10 == 0 && pm_.MovedPoints())
	// {
		// vector<Vector3D>& mesh = tess_.accessMeshPoints();
		// mesh.resize(tess_.GetPointNo());
		// vector<size_t> order = HilbertOrder3D(mesh);
		// size_t Nlocal = order.size();
		// ApplyPermutation(mesh, order);
		// mid_extensives.resize(Nlocal);
		// ApplyPermutation(mid_extensives, order);
		// extensive_.resize(Nlocal);
		// ApplyPermutation(extensive_, order);
		// cells_.resize(Nlocal);
		// ApplyPermutation(cells_, order);
		// point_vel.resize(Nlocal);
		// ApplyPermutation(point_vel, order);
// #ifdef RICH_MPI
		// tess_.PreparePoints(mesh, order);
// #endif
	// }
	Conserved3D edummy;
	ComputationalCell3D cdummy;
	if(pm_.MovedPoints())
	{
		MovePoints(tess_, point_vel, dt);
		t1 = get_time();
		{
			MeshBuildTimer mesh_build_timer(last_mesh_build_timing_);
		#ifdef RICH_MPI
			UpdateTessellation(tess_, point_vel, dt, this->exchange_chain_, this->tessellation_points_scratch_);
		#else // RICH_MPI
			UpdateTessellation(tess_, point_vel, dt, this->tessellation_points_scratch_);
		#endif // RICH_MPI
		}
		RefreshSphericalShellGeometry("timeAdvance2 mesh build", true);
		t2 = get_time();
		DisplayTime(t1, t2, "Voronoi build time ");
#ifdef RICH_MPI
		MPI_exchange_data(tess_, mid_extensives, false);
		MPI_exchange_data(tess_, extensive_, false);
		MPI_exchange_data(tess_, cells_, false);
		MPI_exchange_data(tess_, point_vel, false);
		MPI_exchange_data(tess_, point_vel, true);
#endif
	}
	MEMORY_DEBUG_PRINT("hydro: after Voronoi rebuild");
cu_(cells_, eos_, tess_, mid_extensives);
#ifdef RICH_MPI
MPI_exchange_data(tess_, cells_, true);
#endif
	MEMORY_DEBUG_PRINT("hydro: after cell update (1st half)");

	CalcFaceVelocities(tess_, point_vel, face_vel);
	face_vel_build_generation_ = tess_.GetBuildGeneration();
	vector<Conserved3D> const stage_two_input_extensives = mid_extensives;
	GetFullStateFluxCalculator().Calculate(
		fluxes, tess_, face_vel, cells_,
		mid_extensives, eos_, time + dt, dt, face_values);
	t1 = get_time();
	double const second_source_start = StepDiagnosticWallTime();
	source_(tess_, cells_, fluxes, point_vel, time + dt, dt,
		mid_extensives);
	last_source_step_timing_.second_seconds +=
		StepDiagnosticWallTime() - second_source_start;
	++last_source_step_timing_.calls;
	t2 = get_time();
	DisplayTime(t1, t2, "Second source time ");
	eu_(fluxes, tess_, dt, cells_, mid_extensives, time + dt,
		face_vel, point_vel, face_values);
	ApplySphericalBackgroundCorrection(cells_,
		stage_two_input_extensives, face_vel, point_vel,
		time + dt, dt, true, mid_extensives);
	ExtensiveAvg(extensive_, mid_extensives);
cu_(cells_, eos_, tess_, extensive_);
#ifdef RICH_MPI
MPI_exchange_data(tess_, cells_, true);
#endif
MEMORY_DEBUG_PRINT("hydro: after cell update (2nd half)");
}

void HDSim3D::timeAdvanceIndividual(const IndividualStepContext& context)
{
	last_source_step_timing_ = SourceStepTiming();
	last_mesh_build_timing_ = MeshBuildTiming();
	if(special_relativity_ || spherical_shell_projector_ != nullptr)
		throw std::logic_error("Individual timesteps support only Newtonian Cartesian 3D hydro");
	if(!eu_.SupportsIndividualTimeSteps())
		throw std::logic_error("Configured extensive updater does not support individual timesteps");
	if(!source_.SupportsIndividualTimeSteps())
		throw std::logic_error("Configured source term does not support individual timesteps");
	if(!GetFullStateFluxCalculator().SupportsIndividualTimeSteps())
		throw std::logic_error("Configured flux reconstruction does not support second-order individual timesteps");
	MEMORY_PROFILE_SCOPE("hydro timeAdvanceIndividual");
	// Agreed across ranks before any rank rejects it: the flag gates
	// collectives (the phase and mesh-build reports).
	static int const phase_trace_agreed = []()
	{
		StrictBooleanEnvironment const parsed =
			ReadStrictBooleanEnvironment("RICH_INDIVIDUAL_PERF_TRACE");
		return static_cast<int>(AgreedIndividualValue(
			!parsed.valid ? -1 : (parsed.value ? 1 : 0), "RICH_INDIVIDUAL_PERF_TRACE"));
	}();
	if(phase_trace_agreed < 0)
		throw std::invalid_argument(
			"RICH_INDIVIDUAL_PERF_TRACE must be a strict boolean");
	bool const phase_trace = phase_trace_agreed != 0;
	IndividualHydroPhaseRecord phase_record = {{}};
	double phase_start = phase_trace ? IndividualHydroWallTime() : 0;
	double phase_mpi_start = phase_trace ? mpi_wait_profiler::Seconds() : 0;
	auto finish_phase = [&](size_t phase)
	{
		if(!phase_trace)
			return;
		double const finish = IndividualHydroWallTime();
		double const mpi_finish = mpi_wait_profiler::Seconds();
		phase_record.at(phase) += finish - phase_start;
		phase_record.at(individual_hydro_mpi_index + phase) += mpi_finish - phase_mpi_start;
		phase_start = finish;
		phase_mpi_start = mpi_finish;
	};
	const size_t canonical_count = extensive_.size();
#ifdef RICH_MPI
	bool const entropy_fix_enabled = std::find(
		ComputationalCell3D::tracerNames.begin(),
		ComputationalCell3D::tracerNames.end(), string("Entropy")) !=
		ComputationalCell3D::tracerNames.end();
#endif
	if(context.active_mask.size() != canonical_count)
		throw std::logic_error(
			"Individual hydro context does not match the canonical owned-cell count");
	auto any_rank = [](bool value)
	{
#ifdef RICH_MPI
		int flag = value ? 1 : 0;
		MPI_Allreduce(MPI_IN_PLACE, &flag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
		return flag != 0;
#else
		return value;
#endif
	};
	auto all_ranks = [](bool value)
	{
#ifdef RICH_MPI
		int flag = value ? 1 : 0;
		MPI_Allreduce(MPI_IN_PLACE, &flag, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
		return flag != 0;
#else
		return value;
#endif
	};
	vector<Vector3D>& point_vel = point_vel_scratch_;
	vector<Vector3D>& face_vel = face_vel_scratch_;
	face_vel_build_generation_ = kNoFaceVelocityGeneration;
	vector<Conserved3D>& fluxes = fluxes_scratch_;
	auto& face_values = face_values_scratch_;
	if(context.point_velocities.size() != canonical_count)
		context.point_velocities.assign(canonical_count, Vector3D());
	if(context.cached_accelerations.size() != canonical_count)
		context.cached_accelerations.assign(canonical_count, Vector3D());
	if(context.gravity_half_kick_pending.size() != canonical_count)
		context.gravity_half_kick_pending.assign(canonical_count, 0);

	if(individual_points_.size() != canonical_count)
	{
		if(tess_.GetPointNo() != canonical_count)
			throw std::logic_error(
				"Cannot initialize canonical individual generator positions from a partial mesh");
		individual_points_.resize(canonical_count);
		for(size_t global = 0; global < canonical_count; ++global)
			individual_points_[global] = tess_.GetMeshPoint(global);
	}
	if(individual_centroids_.size() != canonical_count)
	{
		if(tess_.GetPointNo() != canonical_count)
			throw std::logic_error(
				"Cannot initialize canonical individual cell centroids from a partial mesh");
		individual_centroids_.resize(canonical_count);
		for(size_t global = 0; global < canonical_count; ++global)
			individual_centroids_[global] = tess_.GetCellCM(global);
	}
	vector<Vector3D> all_points = individual_points_;
	context.gravity_source_points = individual_centroids_;
	context.gravity_source_masses.resize(extensive_.size());
	context.gravity_source_ids.resize(extensive_.size());
	for(size_t global = 0; global < extensive_.size(); ++global)
	{
		context.gravity_source_masses[global] = extensive_[global].mass;
		context.gravity_source_ids[global] =
			static_cast<std::uint64_t>(cells_[global].ID);
	}

	bool const globally_all_active = all_ranks(
		context.active_indices.size() == canonical_count &&
		std::all_of(context.active_mask.begin(), context.active_mask.end(),
			[](unsigned char value) { return value != 0; }));
	bool const partial_requested =
		context.mesh_build_policy == IndividualMeshBuildPolicy::AutoPartial &&
		source_.SupportsPartialMesh() && spherical_shell_projector_ == nullptr &&
		!globally_all_active;
	if(individual_mesh_restore_pending_)
	{
		bool const globally_has_target = any_rank(!individual_mesh_target_ids_.empty());
		bool restore_failed = false;
		vector<size_t> restore_target;
		if(partial_requested && globally_has_target)
		{
			std::unordered_map<size_t, size_t> id_to_global;
			id_to_global.reserve(cells_.size());
			for(size_t global = 0; global < cells_.size(); ++global)
				id_to_global[cells_[global].ID] = global;
			restore_target.reserve(individual_mesh_target_ids_.size());
			for(size_t id : individual_mesh_target_ids_)
			{
				auto const found = id_to_global.find(id);
				if(found == id_to_global.end())
				{
					restore_failed = true;
					break;
				}
				restore_target.push_back(found->second);
			}

			if(!any_rank(restore_failed))
			{
				try
				{
					{
						MeshBuildTimer mesh_build_timer(last_mesh_build_timing_);
#ifdef RICH_MPI
						tess_.BuildPartiallyParallel(all_points, restore_target, true, true);
#else
						tess_.BuildPartially(all_points, restore_target);
#endif
					}
					ActiveMeshView restored_view(tess_, canonical_count);
					(void)restored_view;
				}
				catch(...)
				{
					restore_failed = true;
				}
			}
		}

		if(partial_requested && globally_has_target && any_rank(restore_failed))
		{
			{
				MeshBuildTimer mesh_build_timer(last_mesh_build_timing_);
				++last_mesh_build_timing_.full_builds;
#ifdef RICH_MPI
				tess_.BuildParallel(all_points, true, true);
#else
				tess_.Build(all_points);
#endif
			}
			individual_mesh_target_ids_.clear();
			std::cerr << "Individual partial Voronoi restart fallback: "
				"saved target closure could not be restored" << std::endl;
		}
		else if(!partial_requested)
			individual_mesh_target_ids_.clear();
		individual_mesh_restore_pending_ = false;
	}

	// Adjacency cache and two-shell seeding of the partial target.  Recording
	// is rank-local; seeding exchanges requests, so the switch is collective.
	static bool const adjacency_seed_flag =
		ReadDefaultOnBooleanEnvironment("RICH_INDIVIDUAL_ADJACENCY_SEED");
	bool const adjacency_cache_enabled = all_ranks(adjacency_seed_flag);
	bool const adjacency_seed_enabled =
		partial_requested && adjacency_cache_enabled;
	vector<size_t> seed;
	vector<size_t> seed_depth_two;
	vector<size_t> warm_target;
	if(partial_requested)
	{
	seed = context.active_indices;
	vector<unsigned char> seeded(canonical_count, 0);
	for(size_t global : seed)
		seeded.at(global) = 1;
	// Warm-start distributed closure from the last accepted target set.  Keep
	// this separate from the current reconstruction seed: cached geometric
	// support must not be promoted to a depth-one stencil cell.  Stable IDs make
	// the cache safe across local reordering and AMR; removed IDs are ignored.
	// The accepted target contains its own warm start, so this set can only
	// grow from one event to the next until a full build clears it; after a
	// large active set, events with a handful of active cells then rebuild
	// nearly the whole mesh as a "partial" target (7-9 s each, job 10199567).
	// With the adjacency seed supplying the two-cell shell directly, the warm
	// start is not needed for closure and is left out.
	if(partial_requested && !individual_mesh_target_ids_.empty() &&
	   !adjacency_seed_enabled)
	{
		std::unordered_map<size_t, size_t> current_index_by_id;
		current_index_by_id.reserve(canonical_count);
		for(size_t global = 0; global < canonical_count; ++global)
			current_index_by_id.emplace(cells_[global].ID, global);
		for(size_t id : individual_mesh_target_ids_)
		{
			auto const found = current_index_by_id.find(id);
			if(found != current_index_by_id.end() && !seeded[found->second])
				warm_target.push_back(found->second);
		}
	}
	// Reuse the previous face halo.  Synchronizing the previous activity mask is
	// essential in MPI: a locally passive cell adjacent to a remote active cell
	// must be a target so its reconstruction slope can be sent to the face owner.
	std::unique_ptr<ActiveMeshView> previous_view;
	bool previous_mapping_failed = false;
	try
	{
		previous_view.reset(new ActiveMeshView(tess_, canonical_count));
	}
	catch(...)
	{
		previous_mapping_failed = true;
	}
	if(!any_rank(previous_mapping_failed))
	{
		std::unique_ptr<IndividualStepContext> previous_context;
		bool previous_sync_failed = false;
		try
		{
			previous_context.reset(new IndividualStepContext(
				previous_view->remapContext(context, false)));
		}
		catch(...)
		{
			previous_sync_failed = true;
		}
		if(!any_rank(previous_sync_failed))
		{
			for(size_t face = 0; face < tess_.GetTotalFacesNumber(); ++face)
			{
				auto const neighbors = tess_.GetFaceNeighbors(face);
				if(!previous_context->isActive(neighbors.first) &&
				   !previous_context->isActive(neighbors.second))
					continue;
				for(size_t local : {neighbors.first, neighbors.second})
				{
					if(local >= tess_.GetPointNo())
						continue;
					size_t const global = previous_view->localToGlobal(local);
					if(global < canonical_count && !seeded[global])
					{
						seeded[global] = 1;
						seed.push_back(global);
					}
				}
			}
		}
	}
	if(adjacency_seed_enabled)
	{
		// Two-shell seed from the adjacency cache.  The closure loop after a
		// build expands active cells to depth one and those to depth two, and
		// rebuilds whenever it adds a cell; seeding both shells up front makes
		// the first build the last one in the common case.  Local neighbours
		// are resolved by stable ID; remote ones are requested from their
		// owner, one sparse exchange per depth.  The cache is a predictor
		// only: the closure check after the build still decides.
		if(individual_adjacency_.size() != canonical_count)
			individual_adjacency_.assign(canonical_count,
				IndividualAdjacencyRecord());
		std::unordered_map<size_t, size_t> index_by_id;
		index_by_id.reserve(canonical_count);
		for(size_t global = 0; global < canonical_count; ++global)
			index_by_id.emplace(cells_[global].ID, global);
		int rank = 0;
		int rank_count = 1;
#ifdef RICH_MPI
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
#endif
		vector<unsigned char> depth_two_seeded(canonical_count, 0);
		auto record_valid = [&](size_t global)
		{
			IndividualAdjacencyRecord const& record =
				individual_adjacency_[global];
			return record.cell_id == cells_[global].ID &&
				record.neighbor_ids.size() == record.neighbor_owners.size();
		};
		// Expand `sources` by one shell: local neighbours not yet seeded at
		// this depth or shallower go to `shell`, remote ones to their owner.
		auto expand = [&](vector<size_t> const& sources,
			vector<unsigned char>& mark, vector<size_t>& shell,
			vector<vector<size_t> >& remote)
		{
			for(size_t global : sources)
			{
				if(!record_valid(global))
					continue;
				IndividualAdjacencyRecord const& record =
					individual_adjacency_[global];
				for(size_t entry = 0; entry < record.neighbor_ids.size();
					++entry)
				{
					int const owner = record.neighbor_owners[entry];
					if(owner == rank)
					{
						auto const found =
							index_by_id.find(record.neighbor_ids[entry]);
						if(found == index_by_id.end() ||
						   seeded[found->second] || mark[found->second])
							continue;
						mark[found->second] = 1;
						shell.push_back(found->second);
					}
					else if(owner >= 0 && owner < rank_count)
						remote[static_cast<size_t>(owner)].push_back(
							record.neighbor_ids[entry]);
				}
			}
		};
		auto receive = [&](vector<vector<size_t> > const& incoming,
			vector<unsigned char>& mark, vector<size_t>& shell)
		{
			for(auto const& requests : incoming)
				for(size_t id : requests)
				{
					auto const found = index_by_id.find(id);
					if(found == index_by_id.end() ||
					   seeded[found->second] || mark[found->second])
						continue;
					mark[found->second] = 1;
					shell.push_back(found->second);
				}
		};
		// Depth one: face neighbours of the active cells.  `seeded` already
		// marks the active cells and the previous-halo seed.
		vector<size_t> depth_one;
		vector<vector<size_t> > remote_requests(
			static_cast<size_t>(rank_count));
		expand(context.active_indices, seeded, depth_one, remote_requests);
#ifdef RICH_MPI
		receive(MPI_Exchange_all_to_all(remote_requests, MPI_COMM_WORLD),
			seeded, depth_one);
#endif
		seed.insert(seed.end(), depth_one.begin(), depth_one.end());
		// Depth two: neighbours of every depth-one cell, the previous-halo
		// seed included since the last mesh knew it as depth one.
		vector<size_t> depth_one_sources;
		depth_one_sources.reserve(seed.size());
		for(size_t global : seed)
			if(!context.isActive(global))
				depth_one_sources.push_back(global);
		for(auto& requests : remote_requests)
			requests.clear();
		expand(depth_one_sources, depth_two_seeded, seed_depth_two,
			remote_requests);
#ifdef RICH_MPI
		receive(MPI_Exchange_all_to_all(remote_requests, MPI_COMM_WORLD),
			depth_two_seeded, seed_depth_two);
#endif
		// Optional third shell (RICH_INDIVIDUAL_ADJACENCY_SEED_SHELLS=3): the
		// neighbours the cache records for the depth-two cells, seeded as
		// depth two (supports, not expanded by the closure loop).  The cells
		// the closure loop adds after a build are neighbours that appeared since
		// the cache was recorded, typically former second neighbours; on the
		// TDE they were ~1% of the target yet cost two rebuilds per build.
		// Automatic: three shells under segmented Hilbert ownership, two
		// otherwise.  Measured on the TDE (256 ranks): the third shell cut
		// partial-build attempts but made builds full more often under one
		// range per rank (wall unchanged), and saved 4 % when the partition
		// spreads the active cells (per-rank fractions stay low).  The
		// partition is replicated, so every rank decides alike.
		int seed_shells = AgreedIndividualExperimentSettings().seed_shells;
		if(seed_shells == 0)
		{
			seed_shells = 2;
#ifdef RICH_MPI
			std::shared_ptr<HilbertLoadBalancer<Vector3D>> const hilbert =
				std::dynamic_pointer_cast<HilbertLoadBalancer<Vector3D>>(
					tess_.GetLoadBalancer());
			if(hilbert && !hilbert->positionalOwnership())
				seed_shells = 3;
#endif
		}
		if(seed_shells >= 3)
		{
			vector<size_t> const depth_two_sources = seed_depth_two;
			for(auto& requests : remote_requests)
				requests.clear();
			expand(depth_two_sources, depth_two_seeded, seed_depth_two,
				remote_requests);
#ifdef RICH_MPI
			receive(MPI_Exchange_all_to_all(remote_requests, MPI_COMM_WORLD),
				depth_two_seeded, seed_depth_two);
#endif
		}
	}
	}

	auto remember_partial_target = [&](vector<size_t> const& target)
	{
		individual_mesh_target_ids_.clear();
		individual_mesh_target_ids_.reserve(target.size());
		for(size_t global : target)
			individual_mesh_target_ids_.push_back(cells_.at(global).ID);
	};
	bool const mesh_trace = phase_trace;
	// Adaptive per-rank closure threshold: the fraction at which a partial
	// build is predicted to cost as much as a full one (see
	// UpdateIndividualPartialCostModel); the configured value until the model
	// has samples, and never with an explicit override.
	IndividualExperimentSettings const& experiment = AgreedIndividualExperimentSettings();
	bool const adaptive_threshold =
		experiment.adaptive_threshold && !(experiment.partial_build_fraction > 0);
	bool const mesh_timing = mesh_trace || adaptive_threshold;
	double const configured_build_fraction =
		IndividualPartialBuildFraction(context.partial_build_fraction);
	double const partial_build_fraction =
		adaptive_threshold && individual_partial_cost_.fraction > 0 ?
		individual_partial_cost_.fraction : configured_build_fraction;
	IndividualMeshBuildRecord mesh_record;
	auto build_event_mesh_impl = [&](vector<Vector3D> const& points,
		vector<size_t> const& initial_seed, vector<size_t> const& depth_two_seed,
		vector<size_t>& cached_target,
		char const* stage, bool force_full) -> ActiveMeshView
	{
		auto full_build = [&](char const* reason) -> ActiveMeshView
		{
			individual_mesh_target_ids_.clear();
			cached_target.clear();
			mesh_record.result = "full";
			mesh_record.reason = reason;
			double const full_start = mesh_timing ? IndividualHydroWallTime() : 0;
			auto const start = get_time();
			{
				MeshBuildTimer mesh_build_timer(last_mesh_build_timing_);
				++last_mesh_build_timing_.full_builds;
#ifdef RICH_MPI
				tess_.BuildParallel(points, true, true);
#else
				tess_.Build(points);
#endif
			}
			if(mesh_timing)
				mesh_record.full_seconds += IndividualHydroWallTime() - full_start;
			auto const finish = get_time();
			DisplayTime(start, finish, string("Individual full Voronoi build (") + reason + ") ");
			RefreshSphericalShellGeometry(stage, true);
			return ActiveMeshView(tess_, canonical_count);
		};

		if(force_full)
			return full_build("event mesh reuse shadow reference");
		if(!partial_requested)
			return full_build("policy or source requires full geometry");

		vector<unsigned char> included(canonical_count, 0);
		vector<unsigned char> depth(canonical_count, 255);
		vector<size_t> target;
		target.reserve(initial_seed.size() * 2 + 8);
		for(size_t global : context.active_indices)
		{
			included.at(global) = 1;
			depth[global] = 0;
			target.push_back(global);
		}
		for(size_t global : initial_seed)
			if(!included.at(global))
			{
				included[global] = 1;
				depth[global] = 1;
				target.push_back(global);
			}
		// Depth-two seeds carry their depth so the closure loop does not
		// expand them; a cell also seeded at depth one keeps depth one.
		for(size_t global : depth_two_seed)
			if(!included.at(global))
			{
				included[global] = 1;
				depth[global] = 2;
				target.push_back(global);
			}
		for(size_t global : cached_target)
			if(!included.at(global))
			{
				included[global] = 1;
				target.push_back(global);
			}

		// The closure threshold decides whether a full build is cheaper than
		// the partial one.  It is judged per rank on purpose: a partial build
		// whose target covers most of one rank's cells costs that rank about
		// as much as a full build (its ghost search dominates), and every rank
		// waits for it.  Judging the fraction on the whole mesh instead was
		// measured worse (job 10199567 against 10199440: 366 s against 146 s
		// over the same 24 events), so that variant stays behind a switch.
		static bool const per_rank_threshold = []()
		{
			StrictBooleanEnvironment const parsed = ReadStrictBooleanEnvironment(
				"RICH_INDIVIDUAL_PARTIAL_THRESHOLD_GLOBAL");
			if(!parsed.valid)
				throw std::invalid_argument(
					"RICH_INDIVIDUAL_PARTIAL_THRESHOLD_GLOBAL must be a strict boolean");
			return !parsed.value;
		}();
		size_t const full_threshold = static_cast<size_t>(
			partial_build_fraction * static_cast<double>(canonical_count));
		auto exceeds_threshold = [&](size_t local_target) -> bool
		{
			if(per_rank_threshold)
				return any_rank(local_target > full_threshold);
			unsigned long long counts[2] = {
				static_cast<unsigned long long>(local_target),
				static_cast<unsigned long long>(canonical_count)};
#ifdef RICH_MPI
			MPI_Allreduce(MPI_IN_PLACE, counts, 2, MPI_UNSIGNED_LONG_LONG,
				MPI_SUM, MPI_COMM_WORLD);
#endif
			return static_cast<double>(counts[0]) >
				partial_build_fraction * static_cast<double>(counts[1]);
		};
		mesh_record.initial_target = target.size();
		mesh_record.final_target = target.size();
		if(exceeds_threshold(target.size()))
			return full_build("closure threshold");

		for(;;)
		{
			bool partial_build_failed = false;
			string partial_build_error;
			++mesh_record.attempts;
			double const partial_start = mesh_timing ? IndividualHydroWallTime() : 0;
			try
			{
				auto const start = get_time();
				{
					MeshBuildTimer mesh_build_timer(last_mesh_build_timing_);
#ifdef RICH_MPI
					tess_.BuildPartiallyParallel(points, target, true, true);
#else
					tess_.BuildPartially(points, target);
#endif
				}
				auto const finish = get_time();
				DisplayTime(start, finish, "Individual partial Voronoi build ");
			}
			catch(MadVoro::Exception::MadVoroException const& error)
			{
				partial_build_failed = true;
				partial_build_error = error.getErrorMessage();
			}
			catch(UniversalError const& error)
			{
				partial_build_failed = true;
				partial_build_error = error.getErrorMessage();
			}
			catch(std::exception const& error)
			{
				partial_build_failed = true;
				partial_build_error = error.what();
			}
			catch(...)
			{
				partial_build_failed = true;
				partial_build_error = "non-standard exception";
			}
			if(mesh_timing)
				mesh_record.partial_seconds += IndividualHydroWallTime() - partial_start;
			if(any_rank(partial_build_failed))
			{
#ifdef RICH_MPI
				int rank = 0;
				MPI_Comm_rank(MPI_COMM_WORLD, &rank);
				int reporting_rank = partial_build_failed ?
					rank : std::numeric_limits<int>::max();
				MPI_Allreduce(MPI_IN_PLACE, &reporting_rank, 1, MPI_INT,
					MPI_MIN, MPI_COMM_WORLD);
				if(rank == reporting_rank)
#endif
					std::cerr << "Individual partial Voronoi build failed"
#ifdef RICH_MPI
						<< " on rank " << rank
#endif
						<< ": " << partial_build_error << std::endl;
				return full_build("partial build exception");
			}

			std::unique_ptr<ActiveMeshView> view_ptr;
			bool mapping_failed = false;
			try
			{
				view_ptr.reset(new ActiveMeshView(tess_, canonical_count));
			}
			catch(...)
			{
				mapping_failed = true;
			}
			if(any_rank(mapping_failed))
				return full_build("partial mapping invariant");
			ActiveMeshView& view = *view_ptr;
			vector<size_t> additions;
			vector<size_t> neighbors;
			bool closure_changed = false;
#ifdef RICH_MPI
			// Map each remote ghost to the canonical index on its owning rank.
			// A newly appearing cross-rank face must make that endpoint a target
			// on its owner; otherwise its cached centroid can be stale even when
			// the active cell geometry itself looks complete.
			int rank = 0;
			int rank_count = 1;
			MPI_Comm_rank(MPI_COMM_WORLD, &rank);
			MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
			vector<int> ghost_owner(tess_.getMeshPoints().size(), -1);
			auto const& duplicate_ranks = tess_.GetDuplicatedProcs();
			auto const& ghost_indices = tess_.GetGhostIndeces();
			bool ghost_mapping_valid =
				duplicate_ranks.size() == ghost_indices.size();
			for(size_t peer = 0; peer < ghost_indices.size(); ++peer)
			{
				if(peer >= duplicate_ranks.size())
				{
					ghost_mapping_valid = false;
					continue;
				}
				for(size_t ghost : ghost_indices[peer])
				{
					if(duplicate_ranks[peer] < 0 ||
					   duplicate_ranks[peer] >= rank_count ||
					   ghost >= ghost_owner.size() ||
					   (ghost_owner[ghost] >= 0 &&
					    ghost_owner[ghost] != duplicate_ranks[peer]))
					{
						ghost_mapping_valid = false;
						continue;
					}
					ghost_owner[ghost] = duplicate_ranks[peer];
				}
			}
			if(!all_ranks(ghost_mapping_valid))
				return full_build("remote ghost ownership invariant");

			vector<size_t> canonical_indices(canonical_count);
			std::iota(canonical_indices.begin(), canonical_indices.end(), 0);
			vector<size_t> mesh_owner_indices(
				tess_.GetPointNo(), ActiveMeshView::invalidIndex());
			for(size_t local = 0; local < view.localSize(); ++local)
				mesh_owner_indices[local] = view.localToGlobal(local);
			tess_.SyncPartialBuildData(mesh_owner_indices, canonical_indices);
			vector<unsigned char> canonical_target = included;
			vector<unsigned char> mesh_target(tess_.GetPointNo(), 0);
			for(size_t local = 0; local < view.localSize(); ++local)
				mesh_target[local] = included[view.localToGlobal(local)];
			tess_.SyncPartialBuildData(mesh_target, canonical_target);
			vector<vector<size_t> > remote_requests(
				static_cast<size_t>(rank_count));
#endif
			vector<size_t> frontier;
			frontier.reserve(target.size());
			for(size_t global : target)
				if(depth[global] < 2 && view.containsGlobal(global))
					frontier.push_back(global);
			vector<unsigned char> expanded_depth(canonical_count, 255);
#ifdef RICH_MPI
			bool const closure_reexpand =
				AgreedIndividualExperimentSettings().closure_reexpand;
#endif
			// Expansion rounds on this mesh.  A depth decrease requested by
			// another rank for a cell already built here needs only that cell's
			// neighbours on this mesh, not a new build: with
			// RICH_INDIVIDUAL_CLOSURE_REEXPAND the round repeats until no rank
			// changes a depth, and the mesh is rebuilt only if some rank added
			// a target cell.  Every continuation decision is collective.
#ifdef RICH_MPI
			vector<size_t> remote_depth_frontier;
#endif
			for(;;)
			{
			while(!frontier.empty())
			{
				size_t const global = frontier.back();
				frontier.pop_back();
				if(depth[global] >= 2 ||
				   expanded_depth[global] <= depth[global] ||
				   !view.containsGlobal(global))
					continue;
				expanded_depth[global] = depth[global];
				size_t const local = view.globalToLocal(global);
				if(local >= tess_.GetPointNo())
					continue;
				tess_.GetNeighbors(local, neighbors);
				for(size_t neighbor : neighbors)
				{
					size_t const neighbor_global = view.meshLocalToGlobal(neighbor);
					unsigned char const neighbor_depth =
						static_cast<unsigned char>(depth[global] + 1);
					if(neighbor_global >= canonical_count)
					{
#ifdef RICH_MPI
						if(neighbor < ghost_owner.size() &&
						   ghost_owner[neighbor] >= 0 &&
						   ghost_owner[neighbor] != rank &&
						   neighbor < mesh_owner_indices.size() &&
						   mesh_owner_indices[neighbor] !=
						       ActiveMeshView::invalidIndex())
						{
							size_t const owner = static_cast<size_t>(
								ghost_owner[neighbor]);
							remote_requests[owner].push_back(
								mesh_owner_indices[neighbor]);
							remote_requests[owner].push_back(
								static_cast<size_t>(neighbor_depth));
						}
#endif
						continue;
					}
					if(!included[neighbor_global])
					{
						included[neighbor_global] = 1;
						depth[neighbor_global] = neighbor_depth;
						additions.push_back(neighbor_global);
					}
					else if(neighbor_depth < depth[neighbor_global])
					{
						depth[neighbor_global] = neighbor_depth;
						if(neighbor_depth < 2 && view.containsGlobal(neighbor_global))
							frontier.push_back(neighbor_global);
					}
				}
			}

#ifdef RICH_MPI
			vector<vector<size_t> > const incoming_requests =
				MPI_Exchange_all_to_all(remote_requests, MPI_COMM_WORLD);
			bool remote_request_valid =
				incoming_requests.size() == static_cast<size_t>(rank_count);
			for(auto const& requests : incoming_requests)
			{
				if(requests.size() % 2 != 0)
				{
					remote_request_valid = false;
					continue;
				}
				for(size_t request = 0; request < requests.size(); request += 2)
				{
					size_t const global = requests[request];
					size_t const requested_depth = requests[request + 1];
					if(global >= canonical_count || requested_depth > 2)
					{
						remote_request_valid = false;
						continue;
					}
					unsigned char const new_depth =
						static_cast<unsigned char>(requested_depth);
					if(!included[global])
					{
						included[global] = 1;
						depth[global] = new_depth;
						additions.push_back(global);
						closure_changed = true;
					}
					else if(new_depth < depth[global])
					{
						depth[global] = new_depth;
						closure_changed = true;
						remote_depth_frontier.push_back(global);
					}
				}
			}
			if(!all_ranks(remote_request_valid))
				return full_build("remote closure request invariant");
			if(closure_reexpand && !any_rank(!additions.empty()) &&
			   any_rank(!remote_depth_frontier.empty()))
			{
				++mesh_record.reexpansions;
				for(size_t global : remote_depth_frontier)
					if(depth[global] < 2 && view.containsGlobal(global))
						frontier.push_back(global);
				remote_depth_frontier.clear();
				for(auto& requests : remote_requests)
					requests.clear();
				closure_changed = false;
				continue;
			}
#endif
			break;
			}
			closure_changed = closure_changed || !additions.empty();

			if(!any_rank(closure_changed))
			{
				bool closed = true;
				size_t closure_active = ActiveMeshView::invalidIndex();
				size_t closure_neighbor = ActiveMeshView::invalidIndex();
				size_t closure_neighbor_global = ActiveMeshView::invalidIndex();
				for(size_t global : context.active_indices)
				{
					if(!view.containsGlobal(global))
					{
						closed = false;
						closure_active = global;
						break;
					}
					size_t const local = view.globalToLocal(global);
					tess_.GetNeighbors(local, neighbors);
					for(size_t neighbor : neighbors)
					{
						size_t const neighbor_global = view.meshLocalToGlobal(neighbor);
						if(neighbor_global < canonical_count &&
						   neighbor >= tess_.GetPointNo())
						{
							closed = false;
							closure_active = global;
							closure_neighbor = neighbor;
							closure_neighbor_global = neighbor_global;
							break;
						}
#ifdef RICH_MPI
						if(neighbor_global >= canonical_count &&
						   neighbor < ghost_owner.size() &&
						   ghost_owner[neighbor] >= 0 &&
						   (neighbor >= mesh_target.size() ||
						    mesh_target[neighbor] == 0))
						{
							closed = false;
							closure_active = global;
							closure_neighbor = neighbor;
							closure_neighbor_global =
								neighbor < mesh_owner_indices.size() ?
								mesh_owner_indices[neighbor] :
								ActiveMeshView::invalidIndex();
							break;
						}
#endif
					}
					if(!closed)
						break;
				}
				bool const globally_closed = all_ranks(closed);
#ifdef RICH_MPI
				if(globally_closed && mesh_record.reexpansions > 0 && ReexpandVerifyRequested())
				{
					vector<size_t> checked;
					for(size_t global = 0; global < canonical_count; ++global)
						if(included[global] && depth[global] < 2 && view.containsGlobal(global))
							checked.push_back(global);
					vector<IdentifiedCellGeometry> const kept =
						CaptureIdentifiedGeometry(tess_, view, checked, canonical_count, rank);
					bool rebuild_failed = false;
					++mesh_record.attempts;
					try
					{
						MeshBuildTimer mesh_build_timer(last_mesh_build_timing_);
						tess_.BuildPartiallyParallel(points, target, true, true);
					}
					catch(...)
					{
						rebuild_failed = true;
					}
					if(any_rank(rebuild_failed))
						return full_build("re-expansion check rebuild failed");
					std::unique_ptr<ActiveMeshView> rebuilt_view;
					bool rebuilt_mapping_failed = false;
					try
					{
						rebuilt_view.reset(new ActiveMeshView(tess_, canonical_count));
					}
					catch(...)
					{
						rebuilt_mapping_failed = true;
					}
					if(any_rank(rebuilt_mapping_failed))
						return full_build("re-expansion check mapping invariant");
					bool missing = false;
					for(size_t const global : checked)
						missing = missing || !rebuilt_view->containsGlobal(global);
					if(any_rank(missing))
						return full_build("re-expansion check lost a checked cell");
					vector<IdentifiedCellGeometry> const rebuilt =
						CaptureIdentifiedGeometry(tess_, *rebuilt_view, checked, canonical_count, rank);
					ReportReexpandComparison(kept, rebuilt, context.event_tick);
					cached_target = target;
					remember_partial_target(target);
					mesh_record.result = "partial";
					mesh_record.reason = "closed after re-expansion check";
					return std::move(*rebuilt_view);
				}
#endif
				if(globally_closed)
				{
						if(!context.verify_partial_build &&
						   !VerifyPartialBuildRequested())
						{
							cached_target = target;
							remember_partial_target(target);
							mesh_record.result = "partial";
							mesh_record.reason = "closed";
							return std::move(*view_ptr);
					}

					// Active-face reconstruction also consumes slopes from depth-one
					// neighbors, so parity must cover both source layers.
					vector<size_t> reconstruction_cells;
					for(size_t global = 0; global < canonical_count; ++global)
						if(included[global] && depth[global] < 2)
							reconstruction_cells.push_back(global);
					vector<IndividualCellGeometry> const partial_geometry =
						CaptureSelectedGeometry(tess_, view, reconstruction_cells,
							canonical_count);
					double const reference_wall_start =
						mesh_timing ? IndividualHydroWallTime() : 0;
					auto const reference_start = get_time();
					{
						MeshBuildTimer mesh_build_timer(last_mesh_build_timing_);
						++last_mesh_build_timing_.full_builds;
#ifdef RICH_MPI
						tess_.BuildParallel(points, true, true);
#else
						tess_.Build(points);
#endif
					}
					if(mesh_timing)
						mesh_record.full_seconds +=
							IndividualHydroWallTime() - reference_wall_start;
					auto const reference_finish = get_time();
					DisplayTime(reference_start, reference_finish,
						"Individual full-reference Voronoi parity build ");
					std::unique_ptr<ActiveMeshView> full_view;
					bool full_mapping_failed = false;
					try
					{
						full_view.reset(new ActiveMeshView(tess_, canonical_count));
					}
					catch(...)
					{
						full_mapping_failed = true;
					}
					if(any_rank(full_mapping_failed))
						return full_build("full-reference mapping invariant");
					bool const geometry_matches = ActiveGeometryMatches(partial_geometry,
						tess_, *full_view, canonical_count);
						if(!all_ranks(geometry_matches))
						{
							std::cout << "Individual partial Voronoi fallback: debug parity mismatch"
								<< std::endl;
							cached_target.clear();
							individual_mesh_target_ids_.clear();
							mesh_record.result = "full";
							mesh_record.reason = "debug parity mismatch";
							return std::move(*full_view);
					}
					bool rebuild_failed = false;
					++mesh_record.attempts;
					double const rebuild_start = mesh_timing ? IndividualHydroWallTime() : 0;
					try
					{
						MeshBuildTimer mesh_build_timer(last_mesh_build_timing_);
#ifdef RICH_MPI
						tess_.BuildPartiallyParallel(points, target, true, true);
#else
						tess_.BuildPartially(points, target);
#endif
					}
					catch(...)
					{
						rebuild_failed = true;
					}
					if(mesh_timing)
						mesh_record.partial_seconds += IndividualHydroWallTime() - rebuild_start;
					if(any_rank(rebuild_failed))
						return full_build("partial rebuild after parity check failed");
					std::unique_ptr<ActiveMeshView> rebuilt_view;
					bool rebuilt_mapping_failed = false;
					try
					{
						rebuilt_view.reset(new ActiveMeshView(tess_, canonical_count));
					}
					catch(...)
					{
						rebuilt_mapping_failed = true;
					}
						if(any_rank(rebuilt_mapping_failed))
							return full_build("partial rebuild mapping invariant");
						cached_target = target;
						remember_partial_target(target);
						mesh_record.result = "partial";
						mesh_record.reason = "closed after parity check";
					return std::move(*rebuilt_view);
				}
				if(!closed)
				{
#ifdef RICH_MPI
					int rank = 0;
					MPI_Comm_rank(MPI_COMM_WORLD, &rank);
					std::cerr << "Individual partial closure incomplete on rank " << rank
						<< ": active=" << closure_active
						<< " neighbor_local=" << closure_neighbor
						<< " neighbor_global=" << closure_neighbor_global
						<< " target_size=" << target.size();
					if(closure_neighbor_global < included.size())
						std::cerr << " included="
							<< static_cast<int>(included[closure_neighbor_global]);
					if(closure_neighbor < tess_.getMeshPoints().size())
						std::cerr << " geometric_owner="
							<< tess_.GetOwner(tess_.GetMeshPoint(closure_neighbor));
					std::cerr << std::endl;
#endif
				}
				return full_build("incomplete active reconstruction closure");
			}

			bool const rebuild_for_additions =
				mesh_trace && any_rank(!additions.empty());
			target.insert(target.end(), additions.begin(), additions.end());
			mesh_record.final_target = target.size();
			if(exceeds_threshold(target.size()))
				return full_build("expanded closure threshold");
			if(mesh_trace)
			{
				if(rebuild_for_additions)
					++mesh_record.rebuilds_for_additions;
				else
					++mesh_record.rebuilds_depth_only;
			}
		}
	};
	auto build_event_mesh = [&](vector<Vector3D> const& points,
		vector<size_t> const& initial_seed, vector<size_t> const& depth_two_seed,
		vector<size_t>& cached_target,
		char const* stage, bool force_full) -> ActiveMeshView
	{
		mesh_record = IndividualMeshBuildRecord();
		double const start = mesh_trace ? IndividualHydroWallTime() : 0;
		ActiveMeshView view = build_event_mesh_impl(points, initial_seed,
			depth_two_seed, cached_target, stage, force_full);
		if(mesh_trace)
			ReportIndividualMeshBuild(mesh_record, stage, partial_build_fraction,
				canonical_count, context.active_indices.size(),
				IndividualHydroWallTime() - start);
		if(adaptive_threshold)
			UpdateIndividualPartialCostModel(individual_partial_cost_, mesh_record,
				canonical_count, partial_build_fraction, context.event_tick);
		return view;
	};

	StrictBooleanEnvironment const selected_flag =
		ReadStrictBooleanEnvironment("RICH_INDIVIDUAL_REUSE_EVENT_MESH");
	StrictBooleanEnvironment const shadow_flag =
		ReadStrictBooleanEnvironment("RICH_INDIVIDUAL_REUSE_EVENT_MESH_SHADOW");
	IndividualEventMeshReuseMode local_reuse_mode =
		IndividualEventMeshReuseMode::Off;
	if(!selected_flag.valid || !shadow_flag.valid)
		local_reuse_mode = IndividualEventMeshReuseMode::Invalid;
	else if(selected_flag.value && shadow_flag.value)
		local_reuse_mode = IndividualEventMeshReuseMode::Conflict;
	else if(selected_flag.value)
		local_reuse_mode = IndividualEventMeshReuseMode::Selected;
	else if(shadow_flag.value)
		local_reuse_mode = IndividualEventMeshReuseMode::Shadow;

	int reuse_mode_extrema[2] = {
		static_cast<int>(local_reuse_mode),
		-static_cast<int>(local_reuse_mode)};
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, reuse_mode_extrema, 2, MPI_INT, MPI_MAX,
		MPI_COMM_WORLD);
#endif
	int const minimum_reuse_mode = -reuse_mode_extrema[1];
	int const maximum_reuse_mode = reuse_mode_extrema[0];
	bool const reuse_mode_consistent =
		minimum_reuse_mode == maximum_reuse_mode;
	IndividualEventMeshReuseMode const reuse_mode = reuse_mode_consistent ?
		static_cast<IndividualEventMeshReuseMode>(minimum_reuse_mode) :
		IndividualEventMeshReuseMode::Invalid;
	bool const reuse_gate_requested = reuse_mode_consistent &&
		(reuse_mode == IndividualEventMeshReuseMode::Selected ||
		 reuse_mode == IndividualEventMeshReuseMode::Shadow);

	IndividualEventMeshReuseValidationState* reuse_state = nullptr;
	if(reuse_gate_requested)
	{
		reuse_state = &IndividualEventMeshReuseValidationStates()[this];
		++reuse_state->attempts;
	}
	bool local_cache_state_failure = false;
	bool local_mapping_failure = false;
	bool local_epoch_failure = false;
	unsigned long long local_id_mismatches = 0;
	unsigned long long local_coordinate_mismatches = 0;
	std::unique_ptr<ActiveMeshView> reusable_view;
	if(reuse_gate_requested)
	{
		local_cache_state_failure = !individual_event_mesh_reusable_ ||
			!individual_mesh_target_ids_.empty() ||
			tess_.GetPointNo() != canonical_count;
		local_epoch_failure = reuse_state->cached_epoch != reuse_state->epoch;
		if(reuse_state->canonical_ids.size() != canonical_count)
			local_id_mismatches = 1;
		else
			for(size_t global = 0; global < canonical_count; ++global)
				if(reuse_state->canonical_ids[global] != cells_[global].ID)
					++local_id_mismatches;
		if(!local_cache_state_failure)
		{
			try
			{
				reusable_view.reset(new ActiveMeshView(tess_, canonical_count));
				if(reusable_view->localSize() != canonical_count ||
				   reusable_view->globalSize() != canonical_count)
					local_mapping_failure = true;
				for(size_t local = 0;
					!local_mapping_failure && local < reusable_view->localSize(); ++local)
				{
					size_t const global = reusable_view->localToGlobal(local);
					if(global >= canonical_count ||
					   !reusable_view->containsGlobal(global) ||
					   reusable_view->globalToLocal(global) != local)
					{
						local_mapping_failure = true;
						break;
					}
					Vector3D const& cached = tess_.GetMeshPoint(local);
					Vector3D const& expected = all_points[global];
					if(cached.x != expected.x || cached.y != expected.y ||
					   cached.z != expected.z)
						++local_coordinate_mismatches;
				}
			}
			catch(...)
			{
				local_mapping_failure = true;
				reusable_view.reset();
			}
		}
	}
	bool const local_reusable_event_mesh = reuse_gate_requested &&
		!local_cache_state_failure && !local_mapping_failure &&
		!local_epoch_failure && local_id_mismatches == 0 &&
		local_coordinate_mismatches == 0 && reusable_view.get() != nullptr;
	// Eligibility and mode are collective: every rank either reuses the cached
	// full mesh or every rank follows the ordinary rebuilding path.
	bool const reuse_event_mesh = reuse_gate_requested &&
		!any_rank(!local_reusable_event_mesh);
	bool shadow_attempted = false;
	bool shadow_geometry_matches = false;
	// The mesh at the interval-start generator positions exists to serve the
	// first-half source phase; the fluxes use the event mesh only.  A
	// conservative force whose first half is a kick from cached
	// accelerations does not need it, so when every rank's source can run
	// from cache the build is skipped: it was half of every event's mesh
	// work.  A cell without a pending kick (first event, cleared cache,
	// centre-sink reset) makes every rank build as before, and any
	// event-mesh reuse experiment keeps the original path.
	static bool const first_half_from_cache_flag =
		ReadDefaultOnBooleanEnvironment("RICH_INDIVIDUAL_FIRST_HALF_FROM_CACHE");
	bool const first_half_from_cache =
		minimum_reuse_mode ==
			static_cast<int>(IndividualEventMeshReuseMode::Off) &&
		maximum_reuse_mode ==
			static_cast<int>(IndividualEventMeshReuseMode::Off) &&
		!any_rank(!first_half_from_cache_flag ||
			source_.IndividualFirstHalfNeedsGeometry(context));
	finish_phase(0);
	vector<ComputationalCell3D> local_cells;
	vector<Conserved3D> local_extensives;
	const double event_dt = context.event_time - context.previous_event_time;
	if(first_half_from_cache)
	{
		finish_phase(1);
		finish_phase(2);
		double const first_source_start = StepDiagnosticWallTime();
		source_.ApplyIndividualFirstHalfFromCache(cells_,
			context.point_velocities, context.previous_event_time, context,
			extensive_);
		last_source_step_timing_.first_seconds +=
			StepDiagnosticWallTime() - first_source_start;
		++last_source_step_timing_.calls;
		finish_phase(3);
	}
	else
	{
	ActiveMeshView first_view = [&]() -> ActiveMeshView
	{
		if(!reuse_event_mesh)
			return build_event_mesh(all_points, seed, seed_depth_two, warm_target,
				"timeAdvanceIndividual first-half mesh", false);
		if(reuse_mode == IndividualEventMeshReuseMode::Shadow)
		{
			shadow_attempted = true;
			vector<IndividualCellGeometry> cached_geometry;
			bool local_capture_failure = false;
			try
			{
				vector<size_t> canonical_indices(canonical_count);
				std::iota(canonical_indices.begin(), canonical_indices.end(), 0);
					cached_geometry = CaptureSelectedGeometry(tess_, *reusable_view,
					canonical_indices, canonical_count);
			}
			catch(...)
			{
				local_capture_failure = true;
			}
			bool const capture_failure = any_rank(local_capture_failure);
			ActiveMeshView rebuilt = build_event_mesh(all_points, seed, seed_depth_two, warm_target,
				"timeAdvanceIndividual first-half mesh shadow", true);
			if(!capture_failure)
			{
				bool local_geometry_matches = false;
				try
				{
					local_geometry_matches = ActiveGeometryMatches(cached_geometry,
						tess_, rebuilt, canonical_count, false);
				}
				catch(...)
				{
					local_geometry_matches = false;
				}
				shadow_geometry_matches = all_ranks(local_geometry_matches);
			}
			return rebuilt;
		}
		auto const start = get_time();
		ActiveMeshView result = std::move(*reusable_view);
		auto const finish = get_time();
		DisplayTime(start, finish,
			"Individual full Voronoi reuse (previous event mesh) ");
		return result;
	}();
	finish_phase(1);
	bool const gate_failed_closed = !reuse_mode_consistent ||
		reuse_mode == IndividualEventMeshReuseMode::Conflict ||
		reuse_mode == IndividualEventMeshReuseMode::Invalid ||
		(reuse_gate_requested && !reuse_event_mesh) ||
		(shadow_attempted && !shadow_geometry_matches);
	if(reuse_state != nullptr)
	{
		if(reuse_mode == IndividualEventMeshReuseMode::Selected && reuse_event_mesh)
			++reuse_state->selected_hits;
		if(shadow_attempted)
			++reuse_state->shadow_checks;
		if(gate_failed_closed)
			++reuse_state->failed_closed;
	}
	if(minimum_reuse_mode != static_cast<int>(IndividualEventMeshReuseMode::Off) ||
	   maximum_reuse_mode != static_cast<int>(IndividualEventMeshReuseMode::Off))
	{
		unsigned long long aggregate[6] = {
			local_reusable_event_mesh ? 1ULL : 0ULL,
			local_cache_state_failure ? 1ULL : 0ULL,
			local_mapping_failure ? 1ULL : 0ULL,
			local_epoch_failure ? 1ULL : 0ULL,
			local_id_mismatches,
			local_coordinate_mismatches};
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, aggregate, 6, MPI_UNSIGNED_LONG_LONG,
			MPI_SUM, MPI_COMM_WORLD);
		int rank = 0;
		int rank_count = 1;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
#else
		int const rank = 0;
		int const rank_count = 1;
#endif
		if(rank == 0)
		{
			char const* const mode_name = reuse_mode_consistent ?
				IndividualEventMeshReuseModeName(reuse_mode) : "inconsistent";
			std::cout << "INDIVIDUAL_EVENT_MESH_REUSE mode=" << mode_name
				<< " ranks=" << rank_count
				<< " eligible_ranks=" << aggregate[0]
				<< " cache_state_failures=" << aggregate[1]
				<< " mapping_failures=" << aggregate[2]
				<< " epoch_failures=" << aggregate[3]
				<< " id_mismatches=" << aggregate[4]
				<< " coordinate_mismatches=" << aggregate[5]
				<< " selected="
				<< (reuse_mode == IndividualEventMeshReuseMode::Selected &&
					reuse_event_mesh ? 1 : 0)
				<< " shadow_checked=" << (shadow_attempted ? 1 : 0)
				<< " shadow_match="
				<< (shadow_attempted ? (shadow_geometry_matches ? 1 : 0) : -1)
				<< " failed_closed=" << (gate_failed_closed ? 1 : 0);
			if(reuse_state != nullptr)
				std::cout << " attempts_total=" << reuse_state->attempts
					<< " selected_total=" << reuse_state->selected_hits
					<< " shadow_total=" << reuse_state->shadow_checks
					<< " failed_closed_total=" << reuse_state->failed_closed;
			std::cout << std::endl;
		}
	}
		for(size_t local = 0; local < first_view.localSize(); ++local)
		{
			size_t const global = first_view.localToGlobal(local);
			if(context.isActive(global))
				individual_centroids_[global] = tess_.GetCellCM(local);
		}
	context.gravity_source_points = individual_centroids_;
	IndividualStepContext first_context = first_view.remapContext(context);
	first_view.gatherOwnedInto(cells_, local_cells);
	first_view.gatherOwnedInto(extensive_, local_extensives);
	tess_.SyncPartialBuildData(local_cells, cells_);
	tess_.SyncPartialBuildData(local_extensives, extensive_);
	finish_phase(2);
	fluxes.assign(tess_.GetTotalFacesNumber(), Conserved3D());
	point_vel.clear();
	face_vel.clear();
	face_values.clear();
	// Each face interval is one explicit midpoint step.  The state, the
	// generator velocity and the geometry of a cell that closes its interval
	// at this event are all taken from the interval start: the primitive is
	// the one stamped at the last activation (the reconstruction extrapolates
	// it to the face midpoint), the face velocity is the one that moved the
	// generators, and primitives are recovered only after this event's fluxes
	// and sources have been applied.  Recovering active primitives here, or
	// recomputing their velocity before the fluxes, paired an end-of-interval
	// state or velocity with a start-of-interval stamp and geometry; the
	// mismatch removed mass that no face ever swept.  Here the first-half
	// source terms see the velocities currently moving the generators.
	first_view.gatherOwnedInto(context.point_velocities, point_vel);
	{
		vector<Vector3D> all_point_velocities = context.point_velocities;
		tess_.SyncPartialBuildData(point_vel, all_point_velocities);
	}
	double const first_source_start = StepDiagnosticWallTime();
	source_.ApplyIndividual(tess_, local_cells, fluxes, point_vel,
		context.previous_event_time, first_context, IndividualSourcePhase::FirstHalf,
		local_extensives);
	last_source_step_timing_.first_seconds +=
		StepDiagnosticWallTime() - first_source_start;
	++last_source_step_timing_.calls;
	for(size_t local : first_context.active_indices)
	{
		size_t const global = first_view.localToGlobal(local);
		context.cached_accelerations[global] = first_context.cached_accelerations[local];
		context.gravity_half_kick_pending[global] =
			first_context.gravity_half_kick_pending[local];
	}
	first_view.scatterOwned(local_cells, cells_);
	first_view.scatterOwned(local_extensives, extensive_);
	finish_phase(3);
	}

	auto const prediction_start = get_time();
	if(pm_.MovedPoints())
		for(size_t global = 0; global < all_points.size(); ++global)
		{
			all_points[global] += context.point_velocities[global] * event_dt;
			individual_centroids_[global] +=
				context.point_velocities[global] * event_dt;
		}
	context.gravity_source_points = individual_centroids_;
	auto const prediction_finish = get_time();
	DisplayTime(prediction_start, prediction_finish, "Individual generator prediction ");
	finish_phase(4);

	ActiveMeshView event_view = build_event_mesh(all_points, seed, seed_depth_two, warm_target,
		"timeAdvanceIndividual event mesh", false);
		for(size_t local = 0; local < event_view.localSize(); ++local)
		{
			size_t const global = event_view.localToGlobal(local);
			if(context.isActive(global))
				individual_centroids_[global] = tess_.GetCellCM(local);
		}
	context.gravity_source_points = individual_centroids_;
	finish_phase(5);
	IndividualStepContext local_context = event_view.remapContext(context);
	event_view.gatherOwnedInto(cells_, local_cells);
	event_view.gatherOwnedInto(extensive_, local_extensives);
	tess_.SyncPartialBuildData(local_cells, cells_);
	tess_.SyncPartialBuildData(local_extensives, extensive_);
	if(adjacency_cache_enabled)
	{
		// Record the face neighbours of every owned cell in this event mesh
		// by stable ID.  Local supports map through the view; remote ghosts
		// carry their owner's ID in the synced primitive copy.  A full build
		// refreshes every cell; a partial one refreshes its targets.
		if(individual_adjacency_.size() != canonical_count)
			individual_adjacency_.assign(canonical_count,
				IndividualAdjacencyRecord());
		size_t const mesh_size = tess_.getMeshPoints().size();
		vector<int> ghost_owner(mesh_size, -1);
		int rank = 0;
#ifdef RICH_MPI
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		{
			auto const& duplicate_ranks = tess_.GetDuplicatedProcs();
			auto const& ghost_indices = tess_.GetGhostIndeces();
			for(size_t peer = 0; peer < ghost_indices.size() &&
				peer < duplicate_ranks.size(); ++peer)
				for(size_t ghost : ghost_indices[peer])
					if(ghost < mesh_size)
						ghost_owner[ghost] = duplicate_ranks[peer];
		}
#endif
		vector<size_t> neighbors;
		for(size_t local = 0; local < event_view.localSize(); ++local)
		{
			size_t const global = event_view.localToGlobal(local);
			if(global >= canonical_count)
				continue;
			IndividualAdjacencyRecord& record = individual_adjacency_[global];
			record.cell_id = cells_[global].ID;
			record.neighbor_ids.clear();
			record.neighbor_owners.clear();
			tess_.GetNeighbors(local, neighbors);
			for(size_t neighbor : neighbors)
			{
				if(neighbor >= mesh_size || tess_.IsPointOutsideBox(neighbor))
					continue;
				size_t const neighbor_global =
					event_view.meshLocalToGlobal(neighbor);
				if(neighbor_global < canonical_count)
				{
					record.neighbor_ids.push_back(cells_[neighbor_global].ID);
					record.neighbor_owners.push_back(rank);
					continue;
				}
				int const owner = ghost_owner[neighbor];
				if(owner < 0 || owner == rank || neighbor >= local_cells.size())
					continue;
				record.neighbor_ids.push_back(local_cells[neighbor].ID);
				record.neighbor_owners.push_back(owner);
			}
		}
	}
	event_view.gatherOwnedInto(context.point_velocities, point_vel);
	vector<Vector3D> all_point_velocities = context.point_velocities;
	tess_.SyncPartialBuildData(point_vel, all_point_velocities);
	context.point_velocities = all_point_velocities;
	finish_phase(6);

	fluxes.assign(tess_.GetTotalFacesNumber(), Conserved3D());
	face_values.clear();
	// Slopes and cell centres must come from the same primitive on every rank.
	// Both are the frozen interval-start primitives here; recovery follows
	// the flux update below.
	tess_.SyncPartialBuildData(local_cells, cells_);
	CalcFaceVelocities(tess_, point_vel, face_vel);
	if(auto* cfl = dynamic_cast<CourantFriedrichsLewy*>(&tsc_))
		cfl->SetPointVelocities(&point_vel);
	else if(auto* cfl1d = dynamic_cast<CFL1D*>(&tsc_))
		cfl1d->SetPointVelocities(&point_vel);
	finish_phase(7);
	GetFullStateFluxCalculator().CalculateIndividual(fluxes, tess_, face_vel,
		local_cells, local_extensives, eos_, local_context, face_values);
	finish_phase(8);
	// Conserved-change accounting: the canonical extensives before and after
	// this event's flux update give every touched cell's net change, active or
	// passive, local or received.  suggestIndividualTimeSteps turns the mass
	// lost over an interval into a timestep limit; suggestIndividualChangeWakes
	// wakes a passive cell whose content changed by more than a set fraction.
	vector<Conserved3D>& pre_flux = individual_pre_flux_extensives_scratch_;
	pre_flux = extensive_;
	eu_.UpdateIndividual(fluxes, tess_, local_context, local_cells, local_extensives,
		context.event_time, face_vel, point_vel, face_values, &cells_, &extensive_);
	if(individual_conserved_change_.size() != canonical_count)
	{
		DrainIndividualChangeWakeSamples();
		individual_conserved_change_.assign(canonical_count,
			IndividualConservedChange());
		for(size_t global = 0; global < canonical_count; ++global)
		{
			individual_conserved_change_[global].mass_at_activation =
				pre_flux[global].mass;
			individual_conserved_change_[global].energy_at_activation =
				pre_flux[global].energy;
		}
	}
	// A woken cell that activates now is sampled before its own update is
	// added: the wake bounds the change it took while passive.
	{
		double const wake_ratio_limit = IndividualChangeWakeRatioLimit();
		for(size_t global = 0; global < canonical_count; ++global)
			if(context.isActive(global))
				individual_change_wake_accounting_.SampleAtActivation(
					individual_conserved_change_[global], cells_[global].ID, wake_ratio_limit);
	}
	for(size_t global = 0; global < canonical_count; ++global)
	{
		double const dm = extensive_[global].mass - pre_flux[global].mass;
		double const de = extensive_[global].energy - pre_flux[global].energy;
		double const dth = extensive_[global].internal_energy -
			pre_flux[global].internal_energy;
		if(dm == 0 && de == 0 && dth == 0)
			continue;
		IndividualConservedChange& change = individual_conserved_change_[global];
		change.mass_abs_change += std::abs(dm);
		change.mass_loss += std::max(0.0, -dm);
		change.thermal_loss += std::max(0.0, -dth);
		change.energy_abs_change += std::abs(de);
	}
	if(local_context.gravity_source_masses.size() != extensive_.size())
		throw std::logic_error(
			"Individual gravity source mass cache has the wrong canonical size");
	for(size_t global = 0; global < extensive_.size(); ++global)
		local_context.gravity_source_masses[global] = extensive_[global].mass;
	// Second-half source terms may depend on primitive fields; recover them
	// from the flux-updated extensive state before source evaluation.
#ifdef RICH_MPI
	if(entropy_fix_enabled)
		tess_.SyncPartialBuildData(local_extensives, extensive_);
#endif
	cu_.UpdateIndividual(local_cells, eos_, tess_, local_extensives, local_context);
	finish_phase(9);
	// The elapsed intervals are closed.  Choose the velocities that move the
	// generators of the cells activated here through their coming interval,
	// from the recovered primitives and the event-mesh geometry.  The same
	// velocities close that interval's fluxes at its end.
	{
		event_view.scatterOwned(local_cells, cells_);
		vector<Vector3D> next_point_velocities;
		pm_(tess_, local_cells, context.event_time, next_point_velocities);
		bool const event_mesh_is_partial = any_rank(
			event_view.localSize() != canonical_count);
		if(event_mesh_is_partial)
		{
			vector<Vector3D> all_next_velocities = context.point_velocities;
			pm_.ApplyFixIndividual(tess_, local_cells, cells_,
				context.event_time, event_dt, next_point_velocities,
				all_next_velocities);
		}
		else
			pm_.ApplyFix(tess_, local_cells, context.event_time, event_dt,
				next_point_velocities);
		// Cell trace: the velocity that moved each traced generator through the
		// interval closing here, captured before this event replaces it.
		std::vector<std::size_t> const& trace_ids = IndividualTraceCellIds();
		std::vector<std::pair<size_t, Vector3D>> traced_closed;
		if(!trace_ids.empty())
			for(size_t local = 0; local < event_view.localSize(); ++local)
			{
				size_t const global = event_view.localToGlobal(local);
				if(global < canonical_count && std::binary_search(trace_ids.begin(),
					trace_ids.end(), static_cast<std::size_t>(cells_[global].ID)))
					traced_closed.emplace_back(local, context.point_velocities.at(global));
			}
		// Only cells whose interval begins at this event replace their cached
		// velocity; passive closure cells keep the velocity of their own interval.
		for(size_t local : local_context.active_indices)
		{
			if(local >= next_point_velocities.size())
				throw std::logic_error(
					"Individual point-velocity update is outside the event mesh");
			context.point_velocities.at(event_view.localToGlobal(local)) =
				next_point_velocities[local];
		}
		event_view.gatherOwnedInto(context.point_velocities, point_vel);
		all_point_velocities = context.point_velocities;
		tess_.SyncPartialBuildData(point_vel, all_point_velocities);
		context.point_velocities = all_point_velocities;
		// Timestep limits are evaluated from these face velocities.
		CalcFaceVelocities(tess_, point_vel, face_vel);
		// Cell trace, geometry half: generator, centroid, the velocity that
		// moved it through the closed interval and the one installed now, and
		// every face neighbour's generator and installed velocity.  The limits
		// half is printed by suggestIndividualTimeSteps.
		if(!trace_ids.empty())
		{
			int trace_rank = 0;
			vector<int> trace_owner;
#ifdef RICH_MPI
			MPI_Comm_rank(MPI_COMM_WORLD, &trace_rank);
			if(!traced_closed.empty())
			{
				trace_owner.assign(tess_.getMeshPoints().size(), -1);
				auto const& duplicate_ranks = tess_.GetDuplicatedProcs();
				auto const& ghost_indices = tess_.GetGhostIndeces();
				for(size_t peer = 0; peer < ghost_indices.size() &&
					peer < duplicate_ranks.size(); ++peer)
					for(size_t ghost : ghost_indices[peer])
						if(ghost < trace_owner.size())
							trace_owner[ghost] = duplicate_ranks[peer];
			}
#endif
			auto const xyz = [](std::ostream& out, Vector3D const& value)
			{
				out << value.x << ',' << value.y << ',' << value.z;
			};
			std::ostringstream records;
			records << std::setprecision(12);
			for(auto const& traced : traced_closed)
			{
				size_t const local = traced.first;
				size_t const global = event_view.localToGlobal(local);
				Vector3D const& r = tess_.GetMeshPoint(local);
				Vector3D const& s = tess_.GetCellCM(local);
				double const width = tess_.GetWidth(local);
				Vector3D const& w = point_vel.at(local);
				records << "INDIVIDUAL_CELL_TRACE event_tick=" << context.event_tick
					<< " previous_event_tick=" << context.previous_event_tick
					<< " event_time=" << context.event_time
					<< " cell_id=" << cells_[global].ID << " rank=" << trace_rank
					<< " active=" << (context.isActive(global) ? 1 : 0)
					<< " bin=" << (global < context.cell_time_bins.size() ?
						static_cast<int>(context.cell_time_bins[global]) : -1)
					<< " cell_dt=" << context.cellTimeStep(global)
					<< " primitive_tick=" << (global < context.primitive_ticks.size() ?
						context.primitive_ticks[global] : 0)
					<< " r=";
				xyz(records, r);
				records << " centroid=";
				xyz(records, s);
				records << " width=" << width << " volume=" << tess_.GetVolume(local)
					<< " d_over_width=" << fastabs(s - r) / width << " v=";
				xyz(records, local_cells.at(local).velocity);
				records << " w_closed=";
				xyz(records, traced.second);
				records << " w_next=";
				xyz(records, w);
				records << '\n';
				for(size_t const face : tess_.GetCellFaces(local))
				{
					auto const& sides = tess_.GetFaceNeighbors(face);
					size_t const other = sides.first == local ? sides.second : sides.first;
					records << "INDIVIDUAL_CELL_TRACE_FACE event_tick=" << context.event_tick
						<< " cell_id=" << cells_[global].ID << " face_area=" << tess_.GetArea(face);
					if(tess_.BoundaryFace(face) || tess_.IsPointOutsideBox(other) ||
						other >= local_cells.size() || other >= point_vel.size())
					{
						records << " boundary=1\n";
						continue;
					}
					Vector3D const& r_other = tess_.GetMeshPoint(other);
					Vector3D const& w_other = point_vel[other];
					double const separation = fastabs(r_other - r);
					// Same-rank cells outside the partial target map to a
					// canonical index like owned ones; remote ghosts carry
					// their owner from the exchange metadata.
					size_t const other_global = event_view.meshLocalToGlobal(other);
					bool const other_local = other_global < canonical_count;
					int const owner = other_local ? trace_rank :
						(other < trace_owner.size() ? trace_owner[other] : -1);
					records << " neighbor_id=" << (other_local ?
							cells_[other_global].ID : local_cells[other].ID)
						<< " neighbor_owner=" << owner << " r_neighbor=";
					xyz(records, r_other);
					records << " w_neighbor=";
					xyz(records, w_other);
					records << " separation=" << separation
						<< " relative_speed=" << fastabs(w - w_other)
						<< " approach_speed=" << (separation > 0 ?
							ScalarProd(w - w_other, r_other - r) / separation : 0.0)
						<< '\n';
				}
			}
			EmitIndividualCellTrace(records.str());
		}
	}
	finish_phase(10);
	double const second_source_start = StepDiagnosticWallTime();
	source_.ApplyIndividual(tess_, local_cells, fluxes, point_vel,
		context.event_time, local_context, IndividualSourcePhase::SecondHalf,
		local_extensives);
	last_source_step_timing_.second_seconds +=
		StepDiagnosticWallTime() - second_source_start;
	++last_source_step_timing_.calls;
	for(size_t local = 0; local < event_view.localSize(); ++local)
	{
		size_t const global = event_view.localToGlobal(local);
		context.cached_accelerations[global] = local_context.cached_accelerations[local];
		context.gravity_half_kick_pending[global] =
			local_context.gravity_half_kick_pending[local];
	}
	finish_phase(11);
#ifdef RICH_MPI
	if(entropy_fix_enabled)
		tess_.SyncPartialBuildData(local_extensives, extensive_);
#endif
	cu_.UpdateIndividual(local_cells, eos_, tess_, local_extensives, local_context);
	event_view.scatterOwned(local_cells, cells_);
	event_view.scatterOwned(local_extensives, extensive_);
	individual_points_ = all_points;
	individual_event_mesh_reusable_ =
		event_view.localSize() == canonical_count &&
		individual_mesh_target_ids_.empty();
	if(reuse_state != nullptr)
	{
		if(individual_event_mesh_reusable_)
		{
			reuse_state->canonical_ids.resize(canonical_count);
			for(size_t global = 0; global < canonical_count; ++global)
				reuse_state->canonical_ids[global] = cells_[global].ID;
			reuse_state->cached_epoch = reuse_state->epoch;
		}
		else
		{
			reuse_state->canonical_ids.clear();
			reuse_state->cached_epoch = std::numeric_limits<size_t>::max();
		}
	}
	finish_phase(12);
	if(phase_trace)
	{
		phase_record[individual_hydro_total_index] = std::accumulate(
			phase_record.begin(),
			phase_record.begin() + individual_hydro_phase_count, 0.0);
		phase_record[individual_hydro_active_index] =
			static_cast<double>(context.active_indices.size());
		phase_record[individual_hydro_canonical_index] =
			static_cast<double>(canonical_count);
		phase_record[individual_hydro_profiled_index] =
			mpi_wait_profiler::Enabled() ? 1.0 : 0.0;
		ReportIndividualHydroPhaseTiming(phase_record);
	}
}

namespace
{
	double IndividualGuardFraction(char const* name, double fallback)
	{
		char const* const value = std::getenv(name);
		if(value == nullptr || value[0] == '\0')
			return fallback;
		char* end = nullptr;
		double const parsed = std::strtod(value, &end);
		if(end == value || !(parsed > 0) || !(parsed <= 1) || !std::isfinite(parsed))
			throw std::invalid_argument(std::string(name) +
				" must be a number in (0, 1]");
		return parsed;
	}

	// Fraction of a cell's mass one interval may remove (mass-loss limit).
	double IndividualMassLossFraction()
	{
		static double const value =
			IndividualGuardFraction("RICH_INDIVIDUAL_MASS_LOSS_FRACTION", 0.25);
		return value;
	}

	// Fraction of a cell's thermal energy one interval may remove.  The
	// radiation retries test the thermal energy, and a cell whose own-face
	// fluxes were built from its interval-start state can overdraw thermal
	// content that finer neighbours already removed.
	double IndividualThermalLossFraction()
	{
		static double const value =
			IndividualGuardFraction("RICH_INDIVIDUAL_THERMAL_LOSS_FRACTION", 0.5);
		return value;
	}

	// Relative conserved change that wakes a passive cell.
	double IndividualWakeChangeFraction()
	{
		static double const value =
			IndividualGuardFraction("RICH_INDIVIDUAL_WAKE_CHANGE_FRACTION", 0.25);
		return value;
	}

	// As IndividualGuardFraction, but zero is a legal value meaning "off".
	double IndividualOptionalFraction(char const* name, double fallback)
	{
		char const* const value = std::getenv(name);
		if(value == nullptr || value[0] == '\0')
			return fallback;
		char* end = nullptr;
		double const parsed = std::strtod(value, &end);
		if(end == value || !(parsed >= 0) || !(parsed <= 1) || !std::isfinite(parsed))
			throw std::invalid_argument(std::string(name) +
				" must be a number in [0, 1]");
		return parsed;
	}

	// Fraction of the distance to a neighbouring generator that a cell's own
	// generator may cover relative to that neighbour within one interval;
	// 0 = off.  Honouring the assigned bin does not bound this: a RoundCells
	// kick held for a long interval overshoots the centroid by (G - 1) times
	// the offset, G = chi c dt / R, and the next kicks drive neighbours into
	// each other within one legal interval (cell 50107, 2026-09-22: G ~ 7 in
	// bin 37, then a pair closing at 5.4 in bin 36).  At 0.2 the same restart
	// passed the abort point (early robustness gate, t >= 0.180) with 0.61x the
	// events and 0.94x the step wall of the aborting configuration over the
	// matched window t = 0.141161735..0.174667985 (jobs 10200856 vs 10200855).
	// That evidence limited the full relative speed |w_i - w_j|; since
	// 2026-09-24 only the closing speed limits, with the default 0.25.
	double IndividualMeshDriftFraction()
	{
		static double const value =
			IndividualOptionalFraction("RICH_INDIVIDUAL_MESH_DRIFT_FRACTION", 0.25);
		return value;
	}

	// Rank-0 aggregate per event; one representative cell at detailed level.
	void ReportIndividualConservedGuard(char const* kind,
		unsigned long long local_count, double local_extreme, size_t local_id,
		std::uint64_t event_tick)
	{
		unsigned long long count = local_count;
		int rank = 0;
#ifdef RICH_MPI
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
			MPI_COMM_WORLD);
#endif
		if(count == 0 || rank != 0)
			return;
		std::cout << "INDIVIDUAL_CONSERVED_GUARD event_tick=" << event_tick
			<< ' ' << kind << '=' << count;
		if(RuntimeLogDetailed() && local_count > 0)
			std::cout << std::setprecision(6) << " rank0_example_cell=" << local_id
				<< " ratio=" << local_extreme;
		std::cout << std::endl;
	}

	// Accuracy of the conserved-change wake: the change ratio a woken cell has
	// accumulated when it activates, against twice the wake fraction.  Censored
	// samples lost their accumulators before activating.  Collective.
	void ReportIndividualChangeWakeActivation(IndividualChangeWakeAccounting::Tally const& local,
		unsigned long long outstanding, double limit, std::uint64_t event_tick)
	{
		unsigned long long counts[7] = {local.issued, local.sampled, local.sampled_above,
			local.censored, local.censored_above, local.sampled_after_issue_above, outstanding};
		double largest[4] = {local.sampled_largest, local.censored_largest, local.issued_largest,
			local.sampled_after_issue_largest};
		int rank = 0;
#ifdef RICH_MPI
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Allreduce(MPI_IN_PLACE, counts, 7, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, largest, 4, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
		if((counts[0] == 0 && counts[1] == 0 && counts[3] == 0) || rank != 0)
			return;
		// issued/sampled/censored are per event; outstanding is the current count, so over a run
		// sum(issued) = sum(sampled) + sum(censored) + final outstanding.
		std::cout << "INDIVIDUAL_CHANGE_WAKE_ACTIVATION event_tick=" << event_tick
			<< " issued=" << counts[0] << " samples=" << counts[1] << " above_limit=" << counts[2]
			<< " censored=" << counts[3] << " censored_above_limit=" << counts[4]
			<< " after_issue_above_limit=" << counts[5] << " outstanding=" << counts[6]
			<< std::setprecision(6) << " largest_ratio=" << largest[0]
			<< " censored_largest_ratio=" << largest[1] << " issued_largest_ratio=" << largest[2]
			<< " after_issue_largest_ratio=" << largest[3] << " limit=" << limit;
		if(RuntimeLogDetailed() && local.sampled > 0)
			std::cout << " rank0_example_cell=" << local.sampled_largest_id
				<< " rank0_ratio=" << local.sampled_largest;
		std::cout << std::endl;
	}

	// Guard floor (RICH_INDIVIDUAL_GUARD_FLOOR, applied by default since
	// 2026-09-25: on the TDE the thermal-loss guard held near-vacuum cells at
	// the central sink's edge at 0.02-0.05 of the global step, and the floor
	// gave 4.4x the individual throughput with a state closer to the global
	// run's (guard-floor A/B/G from snap_full_54; the crash-50107 restart passed
	// with it, the drift guard never below the floor there).  The floor is the
	// smallest hydro/source limit (CFL and source term, not radiation's cap)
	// over the owned cells, each cached from the
	// cell's latest activation or synchronized evaluation: the hydro part of
	// the step a global step would take, except that inactive cells contribute
	// the limit of their interval start (not refreshed) and cells without an
	// entry on this rank (new AMR children, migrated cells, until activated)
	// contribute nothing; the report gives how many owned cells have an entry,
	// not their age.  "report" counts the mesh-drift,
	// mass-loss and thermal-loss limits below it; "apply" also raises them to
	// it.  Agreed across ranks with the other experiment switches.
	struct IndividualGuardFloorMode
	{
		bool apply;
		bool report;
	};

	IndividualGuardFloorMode IndividualGuardFloor()
	{
		IndividualGuardFloorSetting const setting =
			AgreedIndividualExperimentSettings().guard_floor;
		return IndividualGuardFloorMode{setting == IndividualGuardFloorSetting::apply,
			setting != IndividualGuardFloorSetting::off};
	}

	// Guard limits below the floor in one event (applied or not) and the
	// deepest one: kind 0 drift, 1 mass loss, 2 thermal loss.
	struct IndividualGuardFloorTally
	{
		unsigned long long below[3] = {0, 0, 0};
		double ratio = std::numeric_limits<double>::infinity();
		unsigned long long id = 0;
		double record[9] = {-1, -1, -1, 0, 0, 0, -1, -1, -1};

		void note(int kind, double guard_limit, double floor, double hydro_limit,
			size_t cell_id, Vector3D const& position, ComputationalCell3D const& cell)
		{
			++below[kind];
			double const deepest = guard_limit / floor;
			if(!(deepest < ratio))
				return;
			ratio = deepest;
			id = static_cast<unsigned long long>(cell_id);
			double const values[9] = {static_cast<double>(kind), guard_limit,
				hydro_limit, position.x, position.y, position.z,
				cell.density, fastabs(cell.velocity), cell.pressure};
			std::copy(values, values + 9, record);
		}
	};

	// Rank-0 record per event: the floor, whether it was applied, the cells
	// the floor's cache covers, the guard limits below it by kind, and the
	// deepest one (MINLOC of limit/floor, broadcast from its rank).
	void ReportIndividualGuardFloor(IndividualGuardFloorTally const& tally,
		double floor, bool applied, unsigned long long covered,
		unsigned long long owned, std::uint64_t event_tick)
	{
		unsigned long long counts[5] = {tally.below[0], tally.below[1], tally.below[2],
			covered, owned};
		unsigned long long id = tally.id;
		double record[9];
		std::copy(tally.record, tally.record + 9, record);
		int rank = 0;
#ifdef RICH_MPI
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		struct
		{
			double value;
			int rank;
		} deepest = {tally.ratio, rank};
		MPI_Allreduce(MPI_IN_PLACE, counts, 5, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
			MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &deepest, 1, MPI_DOUBLE_INT, MPI_MINLOC,
			MPI_COMM_WORLD);
		MPI_Bcast(record, 9, MPI_DOUBLE, deepest.rank, MPI_COMM_WORLD);
		MPI_Bcast(&id, 1, MPI_UNSIGNED_LONG_LONG, deepest.rank, MPI_COMM_WORLD);
		double const ratio = deepest.value;
#else
		double const ratio = tally.ratio;
#endif
		// Printed only for events with a guard limit below the floor (it runs
		// on every event now that the floor is applied by default).
		if(rank != 0 || counts[0] + counts[1] + counts[2] == 0)
			return;
		static char const* const names[3] = {"drift", "mass", "thermal"};
		int const kind = static_cast<int>(record[0]);
		std::ostringstream line;
		line << std::setprecision(6) << "INDIVIDUAL_GUARD_FLOOR event_tick=" << event_tick
			<< " floor=" << floor << " applied=" << (applied ? 1 : 0)
			<< " covered=" << counts[3] << " owned=" << counts[4]
			<< " below_drift=" << counts[0] << " below_mass=" << counts[1]
			<< " below_thermal=" << counts[2];
		if(counts[0] + counts[1] + counts[2] > 0 && kind >= 0 && kind < 3)
			line << " deepest_ratio=" << ratio
				<< " example_id=" << id
				<< " example_reason=" << names[kind]
				<< " example_guard_limit=" << record[1]
				<< " example_hydro_limit=" << record[2]
				<< " example_position=" << record[3] << ',' << record[4] << ',' << record[5]
				<< " example_density=" << record[6]
				<< " example_speed=" << record[7]
				<< " example_pressure=" << record[8];
		std::cout << line.str() << std::endl;
	}
}

void HDSim3D::suggestIndividualTimeSteps(const IndividualStepContext& context,
	vector<double>& time_step_limits) const
{
	ActiveMeshView const view(tess_, context.active_mask.size());
	IndividualStepContext const local_context = view.remapContext(context, false);
	vector<ComputationalCell3D> local_cells = view.gatherOwned(cells_);
	vector<ComputationalCell3D> all_cells = cells_;
	tess_.SyncPartialBuildData(local_cells, all_cells);
	vector<double> local_limits(tess_.GetPointNo(),
		std::numeric_limits<double>::infinity());
	tsc_.SuggestIndividualTimeSteps(tess_, local_cells, eos_, face_vel_scratch_,
		context.event_time, local_context, local_limits);
	if(individual_limit_reason_.size() != extensive_.size())
		individual_limit_reason_.assign(extensive_.size(), 0);
	for(size_t const local : local_context.active_indices)
	{
		size_t const global = view.localToGlobal(local);
		individual_limit_reason_.at(global) = 0;
		if(local_limits.at(local) < time_step_limits.at(global))
			individual_limit_reason_.at(global) = 1;
		time_step_limits.at(global) = std::min(
			time_step_limits.at(global), local_limits.at(local));
	}
	// Guard floor (see IndividualGuardFloor): the smallest cached hydro/source
	// limit over the owned cells.  One MPI_MIN per event, only when the floor
	// is applied or reported.
	IndividualGuardFloorMode const floor_mode = IndividualGuardFloor();
	double guard_floor = std::numeric_limits<double>::infinity();
	unsigned long long floor_covered = 0;
	if(floor_mode.report)
	{
		for(size_t const local : local_context.active_indices)
		{
			double const hydro_limit = local_limits.at(local);
			if(std::isfinite(hydro_limit) && hydro_limit > 0)
				individual_hydro_limit_by_id_[cells_.at(view.localToGlobal(local)).ID] =
					hydro_limit;
		}
		// Drop cells that no longer exist (AMR removal, migration).
		if(individual_hydro_limit_by_id_.size() > 2 * cells_.size() + 64)
		{
			std::unordered_map<size_t, double> kept;
			kept.reserve(cells_.size());
			for(ComputationalCell3D const& cell : cells_)
			{
				auto const found = individual_hydro_limit_by_id_.find(cell.ID);
				if(found != individual_hydro_limit_by_id_.end())
					kept.insert(*found);
			}
			individual_hydro_limit_by_id_.swap(kept);
		}
		for(ComputationalCell3D const& cell : cells_)
		{
			auto const found = individual_hydro_limit_by_id_.find(cell.ID);
			if(found != individual_hydro_limit_by_id_.end())
			{
				guard_floor = std::min(guard_floor, found->second);
				++floor_covered;
			}
		}
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, &guard_floor, 1, MPI_DOUBLE, MPI_MIN,
			MPI_COMM_WORLD);
#endif
	}
	// Kept for drivers that need the global-step reference between events
	// (GetIndividualGlobalStepReference); the same value on every rank.
	individual_global_step_reference_ =
		std::isfinite(guard_floor) && guard_floor > 0 ? guard_floor : 0;
	bool const apply_floor = floor_mode.apply && std::isfinite(guard_floor) &&
		guard_floor > 0;
	IndividualGuardFloorTally floor_tally;
	// A guard limit below the floor and binding: counted, and raised to the
	// floor when it is applied.
	auto floor_guard = [&](int kind, double& limit, size_t global)
	{
		if(!floor_mode.report || !std::isfinite(guard_floor) ||
		   !(limit < guard_floor) || !(limit < time_step_limits.at(global)))
			return;
		double hydro_limit = std::numeric_limits<double>::infinity();
		Vector3D position;
		if(view.containsGlobal(global))
		{
			size_t const local = view.globalToLocal(global);
			if(local < local_limits.size())
				hydro_limit = local_limits[local];
			position = tess_.GetMeshPoint(local);
		}
		floor_tally.note(kind, limit, guard_floor, hydro_limit, cells_[global].ID,
			position, cells_[global]);
		if(apply_floor)
			limit = guard_floor;
	};
	// The effective guard settings, once per run on rank 0: nothing else in a
	// log shows a guard that never binds.
	static bool guards_reported = false;
	if(!guards_reported)
	{
		guards_reported = true;
		int guards_rank = 0;
#ifdef RICH_MPI
		MPI_Comm_rank(MPI_COMM_WORLD, &guards_rank);
#endif
		if(guards_rank == 0)
			std::cout << "INDIVIDUAL_GUARDS mesh_drift_fraction=" << IndividualMeshDriftFraction()
				<< " mass_loss_fraction=" << IndividualMassLossFraction()
				<< " thermal_loss_fraction=" << IndividualThermalLossFraction()
				<< " wake_change_fraction=" << IndividualWakeChangeFraction() << std::endl;
	}
	// Mesh-deformation limit: a generator may not close more than a fraction
	// of the distance to a neighbouring generator within one interval.  The
	// CFL bounds gas motion across the faces that exist when an interval opens
	// and says nothing about the mesh deforming under its own point velocities.
	// Two neighbours approaching at the closing speed -(w_i - w_j).(x_i - x_j)/d
	// meet on the timescale d/closing, and can be driven through each other
	// within one long interval (the negative-mass abort of cell 50107).  A pair
	// moving apart or sliding past each other cannot cross, so only the
	// closing component limits (user's decision, 2026-09-24).  Tangential
	// sliding still rotates faces and can shrink a neighbour's cell or open a
	// new face within the interval; this guard does not bound that (the
	// mass-loss and conserved-change wake guards act on its consequences).
	double const drift_fraction = IndividualMeshDriftFraction();
	vector<Vector3D> const& local_point_velocities = point_vel_scratch_;
	unsigned long long drift_limited = 0;
	double drift_tightest = std::numeric_limits<double>::infinity();
	size_t drift_id = 0;
	if(drift_fraction > 0)
	for(size_t const local : local_context.active_indices)
	{
		if(local >= local_point_velocities.size())
			continue;
		Vector3D const& point_velocity = local_point_velocities[local];
		Vector3D const& point = tess_.GetMeshPoint(local);
		double limit = std::numeric_limits<double>::infinity();
		auto const& faces = tess_.GetCellFaces(local);
		for(size_t const face : faces)
		{
			if(tess_.BoundaryFace(face))
				continue;
			auto const& neighbors = tess_.GetFaceNeighbors(face);
			size_t const other = neighbors.first == local ?
				neighbors.second : neighbors.first;
			if(other >= local_point_velocities.size())
				continue;
			Vector3D const separation = point - tess_.GetMeshPoint(other);
			double const distance = fastabs(separation);
			if(!(distance > 0))
				continue;
			double const closing = -ScalarProd(
				point_velocity - local_point_velocities[other], separation) / distance;
			if(!(closing > 0))
				continue;
			limit = std::min(limit, drift_fraction * distance / closing);
		}
		// A degenerate pair (two generators almost on top of each other)
		// would otherwise drive the limit to zero and stall the scheduler.
		// Four bins below this cell's own CFL is as far as mesh drift may
		// tighten it; closer than that is the regulariser's problem.
		double const hydro_limit = local_limits.at(local);
		if(std::isfinite(hydro_limit) && hydro_limit > 0)
			limit = std::max(limit, 0.0625 * hydro_limit);
		size_t const global = view.localToGlobal(local);
		floor_guard(0, limit, global);
		if(limit < time_step_limits.at(global))
		{
			double const interval = context.cellTimeStep(global);
			time_step_limits.at(global) = limit;
			individual_limit_reason_.at(global) = 2;
			++drift_limited;
			double const ratio = interval > 0 ? limit / interval :
				std::numeric_limits<double>::infinity();
			if(ratio < drift_tightest)
			{
				drift_tightest = ratio;
				drift_id = cells_[global].ID;
			}
		}
	}
	ReportIndividualConservedGuard("mesh_drift_limited", drift_limited,
		drift_tightest, drift_id, context.event_tick);
	// Step audit: an interval must never outlast the limit in force when it
	// opened.  A limit that is computed but not honoured buys nothing, and the
	// crashes so far ran intervals several times longer than the state at
	// their own start allowed, so report that ratio directly.  The stored
	// limit is this function's own; limits added after it can only shrink the
	// interval further, so an overrun reported here is a real one.
	if(individual_limit_at_activation_.size() != extensive_.size())
		individual_limit_at_activation_.assign(extensive_.size(),
			std::numeric_limits<double>::infinity());
	unsigned long long overruns = 0;
	double worst_overrun = 0;
	size_t overrun_id = 0;
	for(size_t const global : context.active_indices)
	{
		if(global >= individual_limit_at_activation_.size())
			continue;
		double const opened_with = individual_limit_at_activation_[global];
		double const interval = context.cellTimeStep(global);
		if(!(opened_with > 0) || !std::isfinite(opened_with) || !(interval > 0))
			continue;
		double const ratio = interval / opened_with;
		if(!(ratio > 1.000001))
			continue;
		++overruns;
		if(ratio > worst_overrun)
		{
			worst_overrun = ratio;
			overrun_id = cells_[global].ID;
		}
	}
	ReportIndividualConservedGuard("step_overruns", overruns, worst_overrun,
		overrun_id, context.event_tick);
	// Mass-loss limit: no interval may remove more than a fraction of a cell's
	// mass.  The CFL bounds wave speeds, not the fraction of content a face
	// flux may carry; every negative-mass abort so far lost ~100% of a cell
	// with the CFL satisfied.  The rate is the mass lost over the interval
	// that just closed, from this cell's own faces and its neighbours' faces
	// alike, so it also covers mesh motion relative to the gas.
	double const loss_fraction = IndividualMassLossFraction();
	double const thermal_fraction = IndividualThermalLossFraction();
	unsigned long long limited = 0;
	unsigned long long thermal_limited = 0;
	double tightest_ratio = std::numeric_limits<double>::infinity();
	size_t tightest_id = 0;
	double thermal_tightest = std::numeric_limits<double>::infinity();
	size_t thermal_id = 0;
	// Cell trace, limits half (the geometry half is printed where the point
	// velocities are chosen): per traced active cell, the hydro/source limit,
	// the drift timescale min_j |r_i - r_j| / |w_i - w_j| whatever the drift
	// fraction (the drift limit is the fraction times it, floored at 1/16 of
	// the hydro limit), the mass- and thermal-loss limits before this
	// function resets their accumulators, and the limit this function leaves.
	struct TracedLimits
	{
		size_t global;
		double hydro;
		double drift_time;
		double mass;
		double thermal;
	};
	std::vector<std::size_t> const& trace_ids = IndividualTraceCellIds();
	std::vector<TracedLimits> traced_limits;
	if(!trace_ids.empty())
		for(size_t const local : local_context.active_indices)
		{
			size_t const global = view.localToGlobal(local);
			if(global >= cells_.size() || !std::binary_search(trace_ids.begin(),
				trace_ids.end(), static_cast<std::size_t>(cells_[global].ID)))
				continue;
			TracedLimits traced{global, local_limits.at(local),
				std::numeric_limits<double>::infinity(),
				std::numeric_limits<double>::infinity(),
				std::numeric_limits<double>::infinity()};
			if(local < local_point_velocities.size())
				for(size_t const face : tess_.GetCellFaces(local))
				{
					if(tess_.BoundaryFace(face))
						continue;
					auto const& sides = tess_.GetFaceNeighbors(face);
					size_t const other = sides.first == local ? sides.second : sides.first;
					if(other >= local_point_velocities.size())
						continue;
					double const drift = fastabs(local_point_velocities[local] -
						local_point_velocities[other]);
					double const distance = fastabs(tess_.GetMeshPoint(local) -
						tess_.GetMeshPoint(other));
					if(drift > 0 && distance > 0)
						traced.drift_time = std::min(traced.drift_time, distance / drift);
				}
			if(global < individual_conserved_change_.size())
			{
				IndividualConservedChange const& change = individual_conserved_change_[global];
				double const interval = context.cellTimeStep(global);
				if(change.mass_loss > 0 && interval > 0 && extensive_[global].mass > 0 &&
					std::isfinite(change.mass_loss))
					traced.mass = loss_fraction * extensive_[global].mass * interval /
						change.mass_loss;
				if(change.thermal_loss > 0 && interval > 0 &&
					extensive_[global].internal_energy > 0 &&
					std::isfinite(change.thermal_loss))
					traced.thermal = thermal_fraction * extensive_[global].internal_energy *
						interval / change.thermal_loss;
			}
			traced_limits.push_back(traced);
		}
	double const wake_ratio_limit = IndividualChangeWakeRatioLimit();
	if(individual_conserved_change_.size() == extensive_.size())
		for(size_t global : context.active_indices)
		{
			if(global >= individual_conserved_change_.size())
				continue;
			IndividualConservedChange& change = individual_conserved_change_[global];
			individual_change_wake_accounting_.SampleAtActivation(change, cells_[global].ID,
				wake_ratio_limit);
			double const interval = context.cellTimeStep(global);
			double const mass = extensive_[global].mass;
			if(change.mass_loss > 0 && interval > 0 && mass > 0 &&
				std::isfinite(change.mass_loss))
			{
				double limit = loss_fraction * mass * interval / change.mass_loss;
				floor_guard(1, limit, global);
				if(limit < time_step_limits.at(global))
				{
					time_step_limits.at(global) = limit;
					individual_limit_reason_.at(global) = 3;
					++limited;
					double const ratio = limit / interval;
					if(ratio < tightest_ratio)
					{
						tightest_ratio = ratio;
						tightest_id = cells_[global].ID;
					}
				}
			}
			double const thermal = extensive_[global].internal_energy;
			if(change.thermal_loss > 0 && interval > 0 && thermal > 0 &&
				std::isfinite(change.thermal_loss))
			{
				double limit =
					thermal_fraction * thermal * interval / change.thermal_loss;
				floor_guard(2, limit, global);
				if(limit < time_step_limits.at(global))
				{
					time_step_limits.at(global) = limit;
					individual_limit_reason_.at(global) = 4;
					++thermal_limited;
					double const ratio = limit / interval;
					if(ratio < thermal_tightest)
					{
						thermal_tightest = ratio;
						thermal_id = cells_[global].ID;
					}
				}
			}
			change.mass_at_activation = mass;
			change.energy_at_activation = extensive_[global].energy;
			change.mass_loss = 0;
			change.thermal_loss = 0;
			change.mass_abs_change = 0;
			change.energy_abs_change = 0;
		}
	ReportIndividualChangeWakeActivation(individual_change_wake_accounting_.GetTally(),
		IndividualChangeWakeAccounting::Outstanding(individual_conserved_change_),
		wake_ratio_limit, context.event_tick);
	individual_change_wake_accounting_.ClearTally();
	ReportIndividualConservedGuard("mass_limited", limited, tightest_ratio,
		tightest_id, context.event_tick);
	ReportIndividualConservedGuard("thermal_limited", thermal_limited,
		thermal_tightest, thermal_id, context.event_tick);
	if(floor_mode.report)
		ReportIndividualGuardFloor(floor_tally, guard_floor, apply_floor,
			floor_covered, static_cast<unsigned long long>(cells_.size()),
			context.event_tick);
	for(size_t const global : context.active_indices)
		if(global < individual_limit_at_activation_.size())
			individual_limit_at_activation_[global] =
				time_step_limits.at(global);
	if(!trace_ids.empty())
	{
		std::ostringstream records;
		records << std::setprecision(12);
		for(TracedLimits const& traced : traced_limits)
			records << "INDIVIDUAL_CELL_TRACE_LIMITS event_tick=" << context.event_tick
				<< " cell_id=" << cells_[traced.global].ID
				<< " closed_interval=" << context.cellTimeStep(traced.global)
				<< " hydro_limit=" << traced.hydro
				<< " drift_time=" << traced.drift_time
				<< " drift_fraction=" << drift_fraction
				<< " mass_limit=" << traced.mass
				<< " thermal_limit=" << traced.thermal
				<< " hydro_step_limit=" << time_step_limits.at(traced.global) << '\n';
		EmitIndividualCellTrace(records.str());
	}
}

bool HDSim3D::CellTimeStepLimitsStale(void) const
{
	if(dynamic_cast<CourantFriedrichsLewy const*>(&tsc_) == nullptr)
		return false;
	return face_vel_build_generation_ != tess_.GetBuildGeneration() ||
		cells_.size() < tess_.GetPointNo() ||
		face_vel_scratch_.size() < tess_.GetTotalFacesNumber();
}

bool HDSim3D::CollectCellTimeStepLimits(vector<double>& limits) const
{
	auto const* cfl = dynamic_cast<CourantFriedrichsLewy const*>(&tsc_);
	if(cfl == nullptr || CellTimeStepLimitsStale())
		return false;
	cfl->CellTimeSteps(tess_, cells_, eos_, face_vel_scratch_, limits);
	return true;
}

bool HDSim3D::RefreshIndividualAccelerations(
	vector<Vector3D>& accelerations) const
{
	size_t const N = tess_.GetPointNo();
	int usable = source_.UsesIndividualAccelerationCache() &&
		source_.SupportsIndividualAccelerationRefresh() &&
		cells_.size() == N && extensive_.size() == N ? 1 : 0;
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &usable, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
	accelerations.clear();
	if(usable == 0)
		return false;
	source_.RefreshIndividualAccelerations(tess_, cells_, extensive_,
		getTime(), accelerations);
	return true;
}

bool HDSim3D::SynchronizedTimeStepLimits(
	vector<Vector3D> const& point_velocities, vector<double>& limits,
	vector<Vector3D>& accelerations) const
{
	auto const* cfl = dynamic_cast<CourantFriedrichsLewy const*>(&tsc_);
	size_t const N = tess_.GetPointNo();
	int usable = cfl != nullptr && cells_.size() == N &&
		extensive_.size() == N && point_velocities.size() >= N ? 1 : 0;
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &usable, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
	if(usable == 0)
		return false;
	vector<Vector3D> mesh_velocities(point_velocities.begin(),
		point_velocities.begin() + static_cast<std::ptrdiff_t>(N));
#ifdef RICH_MPI
	// As the global step: ghosts carry their owners' velocities; boundary
	// faces move with zero velocity (CalcFaceVelocities).
	MPI_exchange_data(tess_, mesh_velocities, true);
#endif
	vector<Vector3D> face_velocities;
	CalcFaceVelocities(tess_, mesh_velocities, face_velocities);
	// As CourantFriedrichsLewy::SuggestIndividualTimeSteps: the wave-speed
	// rule and the source term's per-cell limits times the source factor.
	cfl->CellTimeSteps(tess_, cells_, eos_, face_velocities, limits, false);
	vector<double> source_limits(N, std::numeric_limits<double>::infinity());
	accelerations.clear();
	source_.SynchronizedIndividualLimits(tess_, cells_, extensive_, getTime(),
		source_limits, accelerations);
	for(size_t local = 0; local < N; ++local)
		limits[local] = std::min(limits[local],
			source_limits[local] * cfl->GetSourceCFL());
	// As suggestIndividualTimeSteps: a generator may close at most a fraction
	// of the distance to a face neighbour, with the same floor of four bins
	// below the cell's own limit.  New neighbours (at rest) change this bound.
	// Under RICH_INDIVIDUAL_GUARD_FLOOR it also stops at the smallest limit of
	// any cell, here exact since every cell's limit is at hand; these limits
	// also refresh the floor's cache.
	double const drift_fraction = IndividualMeshDriftFraction();
	IndividualGuardFloorMode const floor_mode = IndividualGuardFloor();
	if(floor_mode.report)
		for(size_t local = 0; local < N; ++local)
			if(std::isfinite(limits[local]) && limits[local] > 0)
				individual_hydro_limit_by_id_[cells_[local].ID] = limits[local];
	// Computed and counted whenever the floor is reported, applied only under
	// "apply", so both arms of an A/B run the same collectives.
	double guard_floor = 0;
	unsigned long long sync_below_floor = 0;
	if(floor_mode.report)
	{
		guard_floor = std::numeric_limits<double>::infinity();
		for(size_t local = 0; local < N; ++local)
			if(std::isfinite(limits[local]) && limits[local] > 0)
				guard_floor = std::min(guard_floor, limits[local]);
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, &guard_floor, 1, MPI_DOUBLE, MPI_MIN,
			MPI_COMM_WORLD);
#endif
		if(!std::isfinite(guard_floor))
			guard_floor = 0;
	}
	if(drift_fraction > 0)
		for(size_t local = 0; local < N; ++local)
		{
			Vector3D const& point_velocity = mesh_velocities[local];
			Vector3D const& point = tess_.GetMeshPoint(local);
			double limit = std::numeric_limits<double>::infinity();
			for(size_t const face : tess_.GetCellFaces(local))
			{
				if(tess_.BoundaryFace(face))
					continue;
				auto const& neighbors = tess_.GetFaceNeighbors(face);
				size_t const other = neighbors.first == local ?
					neighbors.second : neighbors.first;
				if(other >= mesh_velocities.size())
					continue;
				Vector3D const separation = point - tess_.GetMeshPoint(other);
				double const distance = fastabs(separation);
				if(!(distance > 0))
					continue;
				double const closing = -ScalarProd(
					point_velocity - mesh_velocities[other], separation) / distance;
				if(closing > 0)
					limit = std::min(limit, drift_fraction * distance / closing);
			}
			double const own_limit = limits[local];
			if(std::isfinite(own_limit) && own_limit > 0)
				limit = std::max(limit, 0.0625 * own_limit);
			if(limit < guard_floor && limit < limits[local])
			{
				++sync_below_floor;
				if(floor_mode.apply)
					limit = guard_floor;
			}
			limits[local] = std::min(limits[local], limit);
		}
	if(floor_mode.report)
	{
		unsigned long long below = sync_below_floor;
		int rank = 0;
#ifdef RICH_MPI
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Allreduce(MPI_IN_PLACE, &below, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,
			MPI_COMM_WORLD);
#endif
		if(rank == 0)
		{
			std::ostringstream line;
			line << std::setprecision(6) << "INDIVIDUAL_GUARD_FLOOR_SYNCHRONIZED floor="
				<< guard_floor << " applied=" << (floor_mode.apply ? 1 : 0)
				<< " below_drift=" << below << " owned_cells_rank0=" << N;
			std::cout << line.str() << std::endl;
		}
	}
	return true;
}

double HDSim3D::IndividualChangeWakeRatioLimit(void) const
{
	return 2 * IndividualWakeChangeFraction();
}

void HDSim3D::DrainIndividualChangeWakeSamples(void) const
{
	individual_change_wake_accounting_.Drain(individual_conserved_change_,
		IndividualChangeWakeRatioLimit());
}

void HDSim3D::suggestIndividualChangeWakes(const IndividualStepContext& context,
	vector<double>& change_ratios) const
{
	// Conserved-change wake: a passive cell whose mass or energy has moved by
	// more than a fraction of its activation value ends its interval at the
	// next event.  Signal-speed wakes cannot see this; a near-vacuum cell fed
	// by dense neighbours can change by an order of magnitude within one
	// interval while every wave-speed criterion is satisfied.  The scheduler
	// sets the deadline (finalizeChangeWakes): it used to be the last event
	// spacing, a difference of two floating-point times that rounded a
	// power-of-two spacing just short and so dropped a bin every event.
	if(individual_conserved_change_.size() != extensive_.size() ||
		change_ratios.size() != extensive_.size())
		return;
	double const change_fraction = IndividualWakeChangeFraction();
	unsigned long long woken = 0;
	double largest_ratio = 0;
	size_t largest_id = 0;
	for(size_t global = 0; global < extensive_.size(); ++global)
	{
		if(context.isActive(global))
			continue;
		double const ratio = IndividualConservedChangeRatio(individual_conserved_change_[global]);
		if(!(ratio > change_fraction))
			continue;
		change_ratios[global] = std::max(change_ratios[global], ratio);
		individual_change_wake_accounting_.Issue(individual_conserved_change_[global]);
		++woken;
		if(ratio > largest_ratio)
		{
			largest_ratio = ratio;
			largest_id = cells_[global].ID;
		}
	}
	ReportIndividualConservedGuard("change_wakes", woken, largest_ratio,
		largest_id, context.event_tick);
}

void HDSim3D::timeAdvanceLagrangian1D(
	const ComputationalCell3D* left_external,
	const ComputationalCell3D* right_external)
{
	last_source_step_timing_ = SourceStepTiming();
	last_mesh_build_timing_ = MeshBuildTiming();
	RefreshSphericalShellGeometry("timeAdvanceLagrangian1D entry");
	MEMORY_PROFILE_SCOPE("hydro timeAdvanceLagrangian1D");
#ifdef RICH_MPI
	this->exchange_chain_.Reset(tess_.GetPointNo());
#endif // RICH_MPI
	const double time = pt_.getTime();
	vector<Vector3D> &point_vel = this->point_vel_scratch_;
	vector<Vector3D> &face_vel = this->face_vel_scratch_;
	face_vel_build_generation_ = kNoFaceVelocityGeneration;
	vector<Conserved3D> &fluxes = this->fluxes_scratch_;
	vector<Conserved3D> &mid_extensives = this->mid_extensives_scratch_;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > &face_values = this->face_values_scratch_;
	point_vel.clear();
	face_vel.clear();
	fluxes.clear();
	face_values.clear();

	std::pair<Vector3D, Vector3D> orig_box = tess_.GetBoxCoordinates();
	Vector3D orig_ll = orig_box.first;
	Vector3D orig_ur = orig_box.second;
	Hllc3D hllc_local;

	std::pair<std::vector<size_t>, std::vector<size_t> > xfaces = FindXBoundaryFaces(tess_);
	std::vector<size_t>& left_faces = xfaces.first;
	std::vector<size_t>& right_faces = xfaces.second;

	// ---- Phase A: predictor at time t ----
	pm_(tess_, cells_, time, point_vel);
#ifdef RICH_MPI
	Vector3D vdummy;
	MPI_exchange_data(tess_, point_vel, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	SetBoundaryFaceVelocities(left_faces, tess_, cells_, eos_, hllc_local, left_external, face_vel);
	SetBoundaryFaceVelocities(right_faces, tess_, cells_, eos_, hllc_local, right_external, face_vel);
	double dt = tsc_(tess_, cells_, eos_, face_vel, time);
	pm_.ApplyFix(tess_, cells_, time, dt, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	SetBoundaryFaceVelocities(left_faces, tess_, cells_, eos_, hllc_local, left_external, face_vel);
	SetBoundaryFaceVelocities(right_faces, tess_, cells_, eos_, hllc_local, right_external, face_vel);
	if (auto* cfl = dynamic_cast<CourantFriedrichsLewy*>(&tsc_))
		cfl->SetPointVelocities(&point_vel);
	else if (auto* cfl1d = dynamic_cast<CFL1D*>(&tsc_))
		cfl1d->SetPointVelocities(&point_vel);
	dt = tsc_(tess_, cells_, eos_, face_vel, time);

	fc_.Calculate(fluxes, tess_, face_vel, cells_, extensive_, eos_, time, dt, face_values);

	double vx_left_A = OverrideBoundaryFluxes(
		left_faces, tess_, cells_, eos_, hllc_local, left_external, fluxes, face_vel);
	double vx_right_A = OverrideBoundaryFluxes(
		right_faces, tess_, cells_, eos_, hllc_local, right_external, fluxes, face_vel);
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &vx_left_A, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vx_right_A, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

	mid_extensives = extensive_;
	eu_(fluxes, tess_, dt, cells_, mid_extensives, time, face_vel, point_vel, face_values);
	auto t1 = get_time();
	double const first_source_start = StepDiagnosticWallTime();
	source_(tess_, cells_, fluxes, point_vel, time, dt, mid_extensives);
	last_source_step_timing_.first_seconds +=
		StepDiagnosticWallTime() - first_source_start;
	++last_source_step_timing_.calls;
	auto t2 = get_time();
	DisplayTime(t1, t2, "Source time ");

	// if (pt_.getCycle() % 10 == 0 && pm_.MovedPoints())
	// {
		// vector<Vector3D>& mesh = tess_.accessMeshPoints();
		// mesh.resize(tess_.GetPointNo());
		// vector<size_t> order = HilbertOrder3D(mesh);
		// size_t Nlocal = order.size();
		// ApplyPermutation(mesh, order);
		// mid_extensives.resize(Nlocal);
		// ApplyPermutation(mid_extensives, order);
		// extensive_.resize(Nlocal);
		// ApplyPermutation(extensive_, order);
		// cells_.resize(Nlocal);
		// ApplyPermutation(cells_, order);
		// point_vel.resize(Nlocal);
		// ApplyPermutation(point_vel, order);
// #ifdef RICH_MPI
		// tess_.PreparePoints(mesh, order);
// #endif
	// }

	if (pm_.MovedPoints())
		MovePoints(tess_, point_vel, dt);
	Vector3D new_ll = orig_ll;
	Vector3D new_ur = orig_ur;
	new_ll.x += vx_left_A * dt;
	new_ur.x += vx_right_A * dt;
#ifdef RICH_MPI
	{
		MeshBuildTimer mesh_build_timer(last_mesh_build_timing_);
		SetBoxAndRebuild(tess_, new_ll, new_ur, this->tessellation_points_scratch_, this->exchange_chain_);
	}
#else
	{
		MeshBuildTimer mesh_build_timer(last_mesh_build_timing_);
		SetBoxAndRebuild(tess_, new_ll, new_ur, this->tessellation_points_scratch_);
	}
#endif
	t1 = get_time();
	t2 = get_time();

#ifdef RICH_MPI
	MPI_exchange_data(tess_, mid_extensives, false);
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, cells_, false);
	MPI_exchange_data(tess_, point_vel, false);
	MPI_exchange_data(tess_, point_vel, true);
#endif

	cu_(cells_, eos_, tess_, mid_extensives);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif

	// ---- Phase B: corrector at time t + dt ----
	std::pair<std::vector<size_t>, std::vector<size_t> > xfaces_B = FindXBoundaryFaces(tess_);
	std::vector<size_t>& left_faces_B = xfaces_B.first;
	std::vector<size_t>& right_faces_B = xfaces_B.second;

	CalcFaceVelocities(tess_, point_vel, face_vel);
	SetBoundaryFaceVelocities(left_faces_B, tess_, cells_, eos_, hllc_local, left_external, face_vel);
	SetBoundaryFaceVelocities(right_faces_B, tess_, cells_, eos_, hllc_local, right_external, face_vel);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, time + dt, dt, face_values);

	double vx_left_B = OverrideBoundaryFluxes(
		left_faces_B, tess_, cells_, eos_, hllc_local, left_external, fluxes, face_vel);
	double vx_right_B = OverrideBoundaryFluxes(
		right_faces_B, tess_, cells_, eos_, hllc_local, right_external, fluxes, face_vel);
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &vx_left_B, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &vx_right_B, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

	t1 = get_time();
	double const second_source_start = StepDiagnosticWallTime();
	source_(tess_, cells_, fluxes, point_vel, time + dt, dt, mid_extensives);
	last_source_step_timing_.second_seconds +=
		StepDiagnosticWallTime() - second_source_start;
	++last_source_step_timing_.calls;
	t2 = get_time();
	DisplayTime(t1, t2, "Second source time ");
	eu_(fluxes, tess_, dt, cells_, mid_extensives, time + dt, face_vel, point_vel, face_values);
	ExtensiveAvg(extensive_, mid_extensives);

	Vector3D final_ll = orig_ll;
	Vector3D final_ur = orig_ur;
	final_ll.x += 0.5 * (vx_left_A + vx_left_B) * dt;
	final_ur.x += 0.5 * (vx_right_A + vx_right_B) * dt;
#ifdef RICH_MPI
	{
		MeshBuildTimer mesh_build_timer(last_mesh_build_timing_);
		SetBoxAndRebuild(tess_, final_ll, final_ur, this->tessellation_points_scratch_, this->exchange_chain_);
	}
#else
	{
		MeshBuildTimer mesh_build_timer(last_mesh_build_timing_);
		SetBoxAndRebuild(tess_, final_ll, final_ur, this->tessellation_points_scratch_);
	}
#endif

#ifdef RICH_MPI
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, cells_, false);
#endif

	cu_(cells_, eos_, tess_, extensive_);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
}

void HDSim3D::timeAdvance(void)
{
	RefreshSphericalShellGeometry("timeAdvance entry");
	MEMORY_PROFILE_SCOPE("hydro timeAdvance");
	MEMORY_DEBUG_PRINT("hydro1: start");
#ifdef RICH_MPI
	this->exchange_chain_.Reset(tess_.GetPointNo());
#endif // RICH_MPI
	const double time = pt_.getTime();

	vector<Vector3D> &point_vel = this->point_vel_scratch_;
	vector<Vector3D> &face_vel = this->face_vel_scratch_;
	face_vel_build_generation_ = kNoFaceVelocityGeneration;
	vector<Conserved3D> &fluxes = this->fluxes_scratch_;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > &face_values = this->face_values_scratch_;
	point_vel.clear();
	face_vel.clear();
	fluxes.clear();
	face_values.clear();
	pm_(tess_, cells_, time, point_vel);
#ifdef RICH_MPI
	Vector3D vdummy;
	MPI_exchange_data(tess_, point_vel, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	const double dt = tsc_(tess_, cells_, eos_, face_vel, time);
	pm_.ApplyFix(tess_, cells_, time, dt, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, extensive_, eos_, time, dt, face_values);
	source_(tess_, cells_, fluxes, point_vel, time, dt, extensive_);
	eu_(fluxes, tess_, dt, cells_, extensive_, time, face_vel, point_vel, face_values);
	if(pm_.MovedPoints())
	{
	MovePoints(tess_, point_vel, dt);
	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->exchange_chain_, this->tessellation_points_scratch_);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->tessellation_points_scratch_);
	#endif // RICH_MPI
	RefreshSphericalShellGeometry("timeAdvance mesh build", true);

	#ifdef RICH_MPI
	// Keep relevant points
	ComputationalCell3D cdummy;
	Conserved3D edummy;
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, cells_, false);
#endif
	}
	cu_(cells_, eos_, tess_, extensive_);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	MEMORY_DEBUG_PRINT("hydro1: end");
}


void HDSim3D::timeAdvance3(void)
{
	RefreshSphericalShellGeometry("timeAdvance3 entry");
	MEMORY_PROFILE_SCOPE("hydro timeAdvance3");
	MEMORY_DEBUG_PRINT("hydro3: start");
#ifdef RICH_MPI
	this->exchange_chain_.Reset(tess_.GetPointNo());
#endif // RICH_MPI
	const double time = pt_.getTime();

	vector<Vector3D> &point_vel = this->point_vel_scratch_;
	vector<Vector3D> &face_vel = this->face_vel_scratch_;
	face_vel_build_generation_ = kNoFaceVelocityGeneration;
	vector<Vector3D> &oldpoints = this->oldpoints_scratch_;
	vector<Conserved3D> &fluxes = this->fluxes_scratch_;
	vector<Conserved3D> &mid_extensives = this->mid_extensives_scratch_;
	vector<Conserved3D> &u1 = this->u1_scratch_;
	vector<Conserved3D> &u2 = this->u2_scratch_;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > &face_values = this->face_values_scratch_;
	point_vel.clear();
	face_vel.clear();
	fluxes.clear();
	face_values.clear();
	pm_(tess_, cells_, time, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	double dt = tsc_(tess_, cells_, eos_, face_vel, time);
	pm_.ApplyFix(tess_, cells_, time, dt, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	dt = tsc_(tess_, cells_, eos_, face_vel, time);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, extensive_, eos_, time, 0.5 * dt, face_values);
	mid_extensives = extensive_;
	eu_(fluxes, tess_, 0.5 * dt, cells_, mid_extensives, time, face_vel, point_vel, face_values);
	source_(tess_, cells_, fluxes, point_vel, time, 0.5 * dt, mid_extensives);

	// if (pt_.getCycle() % 10 == 0)
	// {
		// vector<Vector3D>& mesh = tess_.accessMeshPoints();
		// mesh.resize(tess_.GetPointNo());
		// vector<size_t> order = HilbertOrder3D(mesh);
		// size_t Nlocal = order.size();
		// ApplyPermutation(mesh, order);
		// mid_extensives.resize(Nlocal);
		// ApplyPermutation(mid_extensives, order);
		// extensive_.resize(Nlocal);
		// ApplyPermutation(extensive_, order);
		// cells_.resize(Nlocal);
		// ApplyPermutation(cells_, order);
		// point_vel.resize(Nlocal);
		// ApplyPermutation(point_vel, order);
	// }
	oldpoints = tess_.accessMeshPoints();
	oldpoints.resize(tess_.GetPointNo());
	MovePoints(tess_, point_vel, dt * 0.5);
	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, 0.5 * dt, this->exchange_chain_, this->tessellation_points_scratch_);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, 0.5 * dt, this->tessellation_points_scratch_);
	#endif // RICH_MPI
	RefreshSphericalShellGeometry("timeAdvance3 first mesh build", true);
#ifdef RICH_MPI
	// Keep relevant points
	MPI_exchange_data(tess_, mid_extensives, false);
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, cells_, false);
	MPI_exchange_data(tess_, point_vel, false);
	//MPI_exchange_data(tess_, du1, false);
	MPI_exchange_data(tess_, oldpoints, false);
	MPI_exchange_data(tess_, point_vel, true);
#endif
	u1 = mid_extensives;
	cu_(cells_, eos_, tess_, mid_extensives);


#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, time + 0.5 * dt, 2 * dt, face_values);
	//mid_extensives = extensive_;
	source_(tess_, cells_, fluxes, point_vel, time + 0.5 * dt, 2 * dt,  mid_extensives);
	eu_(fluxes, tess_, 2 * dt, cells_, mid_extensives, time + 0.5 * dt, face_vel, point_vel, face_values);
	mid_extensives = mid_extensives - 3 * (u1 - extensive_);

	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->exchange_chain_, this->tessellation_points_scratch_, &oldpoints);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->tessellation_points_scratch_, &oldpoints);
	#endif // RICH_MPI
	RefreshSphericalShellGeometry("timeAdvance3 second mesh build", true);
#ifdef RICH_MPI
	// Keep relevant points
	MPI_exchange_data(tess_, mid_extensives, false);
	MPI_exchange_data(tess_, u1, false);
	//MPI_exchange_data(tess_, du2, false);
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, cells_, false);
	MPI_exchange_data(tess_, point_vel, false);
	MPI_exchange_data(tess_, point_vel, true);
#endif
	u2 = mid_extensives;
	cu_(cells_, eos_, tess_, mid_extensives);


#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, time + dt, dt / 6, face_values);
	source_(tess_, cells_, fluxes, point_vel, time + dt, dt / 6,  mid_extensives);
	eu_(fluxes, tess_, dt / 6, cells_, mid_extensives, time + dt, face_vel, point_vel, face_values);
	extensive_ = mid_extensives - (1.0 / 3.0) * (2 * u2 + extensive_) + u1;
	cu_(cells_, eos_, tess_, extensive_);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	MEMORY_DEBUG_PRINT("hydro3: end");
}

void HDSim3D::timeAdvance33(void)
{
	RefreshSphericalShellGeometry("timeAdvance33 entry");
	MEMORY_PROFILE_SCOPE("hydro timeAdvance33");
	MEMORY_DEBUG_PRINT("hydro33: start");
#ifdef RICH_MPI
	this->exchange_chain_.Reset(tess_.GetPointNo());
#endif // RICH_MPI
	const double time = pt_.getTime();

	vector<Vector3D> &point_vel = this->point_vel_scratch_;
	vector<Vector3D> &face_vel = this->face_vel_scratch_;
	face_vel_build_generation_ = kNoFaceVelocityGeneration;
	vector<Vector3D> &oldpoints = this->oldpoints_scratch_;
	vector<Conserved3D> &fluxes = this->fluxes_scratch_;
	vector<Conserved3D> &mid_extensives = this->mid_extensives_scratch_;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > &face_values = this->face_values_scratch_;
	point_vel.clear();
	face_vel.clear();
	fluxes.clear();
	face_values.clear();
	pm_(tess_, cells_, time, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	double dt = tsc_(tess_, cells_, eos_, face_vel, time);
	pm_.ApplyFix(tess_, cells_, time, dt, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	dt = tsc_(tess_, cells_, eos_, face_vel, time);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, extensive_, eos_, time, dt, face_values);
	mid_extensives = extensive_;
	eu_(fluxes, tess_, dt, cells_, mid_extensives, time, face_vel, point_vel, face_values);
	source_(tess_, cells_, fluxes, point_vel, time, dt, mid_extensives);

	// if (pt_.getCycle() % 10 == 0)
	// {
		// vector<Vector3D>& mesh = tess_.accessMeshPoints();
		// mesh.resize(tess_.GetPointNo());
		// vector<size_t> order = HilbertOrder3D(mesh);
		// size_t Nlocal = order.size();
		// ApplyPermutation(mesh, order);
		// mid_extensives.resize(Nlocal);
		// ApplyPermutation(mid_extensives, order);
		// extensive_.resize(Nlocal);
		// ApplyPermutation(extensive_, order);
		// cells_.resize(Nlocal);
		// ApplyPermutation(cells_, order);
		// point_vel.resize(Nlocal);
		// ApplyPermutation(point_vel, order);
	// }
	oldpoints = tess_.accessMeshPoints();
	oldpoints.resize(tess_.GetPointNo());
	MovePoints(tess_, point_vel, dt);
	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->exchange_chain_, this->tessellation_points_scratch_);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->tessellation_points_scratch_);
	#endif // RICH_MPI
	RefreshSphericalShellGeometry("timeAdvance33 first mesh build", true);
#ifdef RICH_MPI
	// Keep relevant points
	MPI_exchange_data(tess_, mid_extensives, false);
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, cells_, false);
	MPI_exchange_data(tess_, point_vel, false);
	MPI_exchange_data(tess_, oldpoints, false);
	MPI_exchange_data(tess_, point_vel, true);
#endif
	cu_(cells_, eos_, tess_, mid_extensives);


#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, time + dt, dt, face_values);
	//mid_extensives = extensive_;
	source_(tess_, cells_, fluxes, point_vel, time + dt, dt, mid_extensives);
	eu_(fluxes, tess_, dt, cells_, mid_extensives, time + dt, face_vel, point_vel, face_values);
	mid_extensives = 0.25 * mid_extensives + 0.75 * extensive_;

	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, dt / 2, this->exchange_chain_, this->tessellation_points_scratch_, &oldpoints);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, dt / 2, this->tessellation_points_scratch_, &oldpoints);
	#endif // RICH_MPI
	RefreshSphericalShellGeometry("timeAdvance33 second mesh build", true);
#ifdef RICH_MPI
	// Keep relevant points
	MPI_exchange_data(tess_, mid_extensives, false);
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, oldpoints, false);
	MPI_exchange_data(tess_, cells_, false);
	MPI_exchange_data(tess_, point_vel, false);
	MPI_exchange_data(tess_, point_vel, true);
#endif
	cu_(cells_, eos_, tess_, mid_extensives);


#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, time + 0.5 * dt, dt, face_values);
	source_(tess_, cells_, fluxes, point_vel, time + 0.5 * dt, dt, mid_extensives);
	eu_(fluxes, tess_, dt, cells_, mid_extensives, time + 0.5 * dt, face_vel, point_vel, face_values);
	extensive_ = 0.33333333333333333333333 * (2 * mid_extensives + extensive_);

	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->exchange_chain_, this->tessellation_points_scratch_, &oldpoints);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->tessellation_points_scratch_, &oldpoints);
	#endif // RICH_MPI
	RefreshSphericalShellGeometry("timeAdvance33 third mesh build", true);

#ifdef RICH_MPI
	// Keep relevant points
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, cells_, false);
#endif

	cu_(cells_, eos_, tess_, extensive_);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	MEMORY_DEBUG_PRINT("hydro33: end");
}

void HDSim3D::timeAdvance32(void)
{
	RefreshSphericalShellGeometry("timeAdvance32 entry");
	MEMORY_PROFILE_SCOPE("hydro timeAdvance32");
	MEMORY_DEBUG_PRINT("hydro32: start");
#ifdef RICH_MPI
	this->exchange_chain_.Reset(tess_.GetPointNo());
#endif // RICH_MPI
	const double time = pt_.getTime();

	vector<Vector3D> &point_vel = this->point_vel_scratch_;
	vector<Vector3D> &face_vel = this->face_vel_scratch_;
	face_vel_build_generation_ = kNoFaceVelocityGeneration;
	vector<Conserved3D> &fluxes = this->fluxes_scratch_;
	vector<Conserved3D> &mid_extensives = this->mid_extensives_scratch_;
	vector<Conserved3D> &u1 = this->u1_scratch_;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > &face_values = this->face_values_scratch_;
	point_vel.clear();
	face_vel.clear();
	fluxes.clear();
	face_values.clear();
	pm_(tess_, cells_, time, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	double dt = tsc_(tess_, cells_, eos_, face_vel, time);
	pm_.ApplyFix(tess_, cells_, time, dt, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	dt = tsc_(tess_, cells_, eos_, face_vel, time);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, extensive_, eos_, time, 0.5 * dt, face_values);
	mid_extensives = extensive_;
	eu_(fluxes, tess_, dt, cells_, mid_extensives, time, face_vel, point_vel, face_values);
	source_(tess_, cells_, fluxes, point_vel, time, dt, mid_extensives);

	// if (pt_.getCycle() % 10 == 0)
	// {
		// vector<Vector3D>& mesh = tess_.accessMeshPoints();
		// mesh.resize(tess_.GetPointNo());
		// vector<size_t> order = HilbertOrder3D(mesh);
		// size_t Nlocal = order.size();
		// ApplyPermutation(mesh, order);
		// mid_extensives.resize(Nlocal);
		// ApplyPermutation(mid_extensives, order);
		// extensive_.resize(Nlocal);
		// ApplyPermutation(extensive_, order);
		// cells_.resize(Nlocal);
		// ApplyPermutation(cells_, order);
		// point_vel.resize(Nlocal);
		// ApplyPermutation(point_vel, order);
	// }
	MovePoints(tess_, point_vel, dt);

	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->exchange_chain_, this->tessellation_points_scratch_);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->tessellation_points_scratch_);
	#endif // RICH_MPI
	RefreshSphericalShellGeometry("timeAdvance32 mesh build", true);

	#ifdef RICH_MPI
	// Keep relevant points
	MPI_exchange_data(tess_, mid_extensives, false);
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, cells_, false);
	MPI_exchange_data(tess_, point_vel, false);
	MPI_exchange_data(tess_, point_vel, true);
#endif
	u1 = mid_extensives;
	cu_(cells_, eos_, tess_, mid_extensives);

#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, time + dt, dt, face_values);
	source_(tess_, cells_, fluxes, point_vel, time + dt, dt, mid_extensives);
	eu_(fluxes, tess_, dt, cells_, mid_extensives, time + dt, face_vel, point_vel, face_values);
	mid_extensives = 0.5 * (mid_extensives + extensive_);
	cu_(cells_, eos_, tess_, mid_extensives);

#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, time + dt, dt, face_values);
	source_(tess_, cells_, fluxes, point_vel, time + dt, dt, mid_extensives);
	eu_(fluxes, tess_, dt, cells_, mid_extensives, time + dt, face_vel, point_vel, face_values);
	//extensive_ = 0.333333333333333333*(extensive_ + u1 + mid_extensives);
	extensive_ = 0.333333333333333333 * (extensive_ + u1 + mid_extensives);
	cu_(cells_, eos_, tess_, extensive_);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	MEMORY_DEBUG_PRINT("hydro32: end");
}

void HDSim3D::timeAdvance4(void)
{
	RefreshSphericalShellGeometry("timeAdvance4 entry");
	MEMORY_PROFILE_SCOPE("hydro timeAdvance4");
	MEMORY_DEBUG_PRINT("hydro4: start");
#ifdef RICH_MPI
	this->exchange_chain_.Reset(tess_.GetPointNo());
#endif // RICH_MPI
	const double time = pt_.getTime();

	vector<Vector3D> &point_vel = this->point_vel_scratch_;
	vector<Vector3D> &face_vel = this->face_vel_scratch_;
	face_vel_build_generation_ = kNoFaceVelocityGeneration;
	vector<Vector3D> &oldpoints = this->oldpoints_scratch_;
	vector<Conserved3D> &fluxes = this->fluxes_scratch_;
	vector<Conserved3D> &mid_extensives = this->mid_extensives_scratch_;
	vector<Conserved3D> &du1 = this->u1_scratch_;
	vector<Conserved3D> &du2 = this->u2_scratch_;
	vector<Conserved3D> &du3 = this->u3_scratch_;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > &face_values = this->face_values_scratch_;
	point_vel.clear();
	face_vel.clear();
	fluxes.clear();
	face_values.clear();
	pm_(tess_, cells_, time, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	double dt = tsc_(tess_, cells_, eos_, face_vel, time);
	pm_.ApplyFix(tess_, cells_, time, dt, point_vel);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, point_vel, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	dt = tsc_(tess_, cells_, eos_, face_vel, time);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, extensive_, eos_, time, 0.5 * dt, face_values);
	mid_extensives = extensive_;
	eu_(fluxes, tess_, 0.5 * dt, cells_, mid_extensives, time, face_vel, point_vel, face_values);
	source_(tess_, cells_, fluxes, point_vel, time, 0.5 * dt, mid_extensives);

	// if (pt_.getCycle() % 10 == 0)
	// {
		// vector<Vector3D>& mesh = tess_.accessMeshPoints();
		// mesh.resize(tess_.GetPointNo());
		// vector<size_t> order = HilbertOrder3D(mesh);
		// size_t Nlocal = order.size();
		// ApplyPermutation(mesh, order);
		// mid_extensives.resize(Nlocal);
		// ApplyPermutation(mid_extensives, order);
		// extensive_.resize(Nlocal);
		// ApplyPermutation(extensive_, order);
		// cells_.resize(Nlocal);
		// ApplyPermutation(cells_, order);
		// point_vel.resize(Nlocal);
		// ApplyPermutation(point_vel, order);
	// }
	oldpoints = tess_.accessMeshPoints();
	oldpoints.resize(tess_.GetPointNo());
	MovePoints(tess_, point_vel, dt * 0.5);
	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, 0.5 * dt, this->exchange_chain_, this->tessellation_points_scratch_);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, 0.5 * dt, this->tessellation_points_scratch_);
	#endif // RICH_MPI
	RefreshSphericalShellGeometry("timeAdvance4 first mesh build", true);

#ifdef RICH_MPI
	// Keep relevant points
	MPI_exchange_data(tess_, mid_extensives, false);
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, cells_, false);
	MPI_exchange_data(tess_, point_vel, false);
	//MPI_exchange_data(tess_, du1, false);
	MPI_exchange_data(tess_, oldpoints, false);
	MPI_exchange_data(tess_, point_vel, true);
#endif
	cu_(cells_, eos_, tess_, mid_extensives);
	du1 = mid_extensives - extensive_;

#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif

	CalcFaceVelocities(tess_, point_vel, face_vel);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, time + 0.5 * dt, 0.5 * dt, face_values);
	//mid_extensives = extensive_;
	source_(tess_, cells_, fluxes, point_vel, time + 0.5 * dt, 0.5 * dt, mid_extensives);
	mid_extensives = mid_extensives - du1;
	eu_(fluxes, tess_, 0.5 * dt, cells_, mid_extensives, time + 0.5 * dt, face_vel, point_vel, face_values);
	cu_(cells_, eos_, tess_, mid_extensives);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	du2 = mid_extensives - extensive_;

	fc_.Calculate(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, time + 0.5 * dt, dt, face_values);
	source_(tess_, cells_, fluxes, point_vel, time + 0.5 * dt, dt, mid_extensives);
	mid_extensives = mid_extensives - du2;
	eu_(fluxes, tess_, dt, cells_, mid_extensives, time + 0.5 * dt, face_vel, point_vel, face_values);

	#ifdef RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->exchange_chain_, this->tessellation_points_scratch_, &oldpoints);
	#else // RICH_MPI
		UpdateTessellation(tess_, point_vel, dt, this->tessellation_points_scratch_, &oldpoints);
	#endif // RICH_MPI
	RefreshSphericalShellGeometry("timeAdvance4 second mesh build", true);

#ifdef RICH_MPI
	// Keep relevant points
	MPI_exchange_data(tess_, mid_extensives, false);
	MPI_exchange_data(tess_, du1, false);
	MPI_exchange_data(tess_, du2, false);
	//MPI_exchange_data(tess_, du3, false);
	MPI_exchange_data(tess_, extensive_, false);
	MPI_exchange_data(tess_, cells_, false);
	MPI_exchange_data(tess_, point_vel, false);
	MPI_exchange_data(tess_, point_vel, true);
#endif
	cu_(cells_, eos_, tess_, mid_extensives);
	du3 = mid_extensives - extensive_;

#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	CalcFaceVelocities(tess_, point_vel, face_vel);
	fc_.Calculate(fluxes, tess_, face_vel, cells_, mid_extensives, eos_, time + dt, dt / 6, face_values);
	source_(tess_, cells_, fluxes, point_vel, time + dt, dt / 6,  mid_extensives);
	mid_extensives = mid_extensives - du3;
	eu_(fluxes, tess_, dt / 6, cells_, mid_extensives, time + dt, face_vel, point_vel, face_values);
	extensive_ = mid_extensives + (1.0 / 6.0) * (2 * du1 + 4 * du2 + 2 * du3);
	cu_(cells_, eos_, tess_, extensive_);
#ifdef RICH_MPI
	MPI_exchange_data(tess_, cells_, true);
#endif
	MEMORY_DEBUG_PRINT("hydro4: end");
}

const Tessellation3D& HDSim3D::getTessellation(void) const
{
	return tess_;
}

const vector<ComputationalCell3D>& HDSim3D::getCells(void) const
{
	return cells_;
}

double HDSim3D::getTime(void) const
{
	return pt_.getTime();
}

size_t HDSim3D::getCycle(void) const
{
	return static_cast<size_t>(pt_.getCycle());
}
