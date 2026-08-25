#include "RadiationStep.hpp"
#include "misc/memory_debug.hpp"
#include "misc/universal_error.hpp"
#include "newtonian/three_dimensional/simulation/ActiveMeshView.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>
#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace {

bool individualPerformanceTraceEnabled()
{
    static bool const enabled = []()
    {
        char const* const value = std::getenv("RICH_INDIVIDUAL_PERF_TRACE");
        return value != nullptr && value[0] != '\0' &&
            std::strcmp(value, "0") != 0 &&
            std::strcmp(value, "false") != 0 &&
            std::strcmp(value, "off") != 0 &&
            std::strcmp(value, "no") != 0;
    }();
    return enabled;
}

double elapsedSeconds(std::chrono::steady_clock::time_point const start)
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
}

#ifdef RICH_MPI
void broadcast_step_failure(RadiationDriver const& matrix_builder,
                            std::string& reason,
                            size_t& cell_id)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int const has_local = matrix_builder.getLastStepFailureReason().empty() ? 0 : 1;
    int const has_cell_id =
        (has_local && matrix_builder.getLastStepFailureCellId() !=
            std::numeric_limits<size_t>::max()) ? 1 : 0;
    int const score = has_cell_id ? 2 : has_local;
    struct {
        int score;
        int rank;
    } local_pick{score, rank};
    struct {
        int score;
        int rank;
    } global_pick{0, std::numeric_limits<int>::max()};
    MPI_Allreduce(&local_pick, &global_pick, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
    int const source_rank = global_pick.score > 0
        ? global_pick.rank
        : std::numeric_limits<int>::max();
    if (source_rank == std::numeric_limits<int>::max()) {
        reason.clear();
        cell_id = std::numeric_limits<size_t>::max();
        return;
    }

    char reason_buf[2048] = {};
    std::uint64_t local_cell_id = static_cast<std::uint64_t>(matrix_builder.getLastStepFailureCellId());
    if (rank == source_rank) {
        std::string const& local_reason = matrix_builder.getLastStepFailureReason();
        std::strncpy(reason_buf, local_reason.c_str(), sizeof(reason_buf) - 1);
    }
    MPI_Bcast(reason_buf, static_cast<int>(sizeof(reason_buf)), MPI_CHAR, source_rank, MPI_COMM_WORLD);
    MPI_Bcast(&local_cell_id, 1, MPI_UINT64_T, source_rank, MPI_COMM_WORLD);

    reason = reason_buf;
    cell_id = static_cast<size_t>(local_cell_id);
}

std::vector<size_t> collect_step_failure_cells(
    RadiationDriver const& matrix_builder)
{
    int rank_count = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
    std::uint64_t local_cell_id = std::numeric_limits<std::uint64_t>::max();
    if(!matrix_builder.getLastStepFailureReason().empty() &&
       matrix_builder.getLastStepFailureIsCellLocal() &&
       matrix_builder.getLastStepFailureCellId() !=
           std::numeric_limits<size_t>::max())
        local_cell_id = static_cast<std::uint64_t>(
            matrix_builder.getLastStepFailureCellId());

    std::vector<std::uint64_t> gathered(
        static_cast<std::size_t>(rank_count),
        std::numeric_limits<std::uint64_t>::max());
    MPI_Allgather(&local_cell_id, 1, MPI_UINT64_T, gathered.data(), 1,
                  MPI_UINT64_T, MPI_COMM_WORLD);

    std::vector<size_t> result;
    result.reserve(gathered.size());
    for(std::uint64_t const cell_id : gathered)
        if(cell_id != std::numeric_limits<std::uint64_t>::max())
            result.push_back(static_cast<size_t>(cell_id));
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

bool has_collective_step_failure(RadiationDriver const& matrix_builder)
{
    int local_collective =
        !matrix_builder.getLastStepFailureReason().empty() &&
        !matrix_builder.getLastStepFailureIsCellLocal() &&
        !matrix_builder.getLastStepFailureIsRemote() ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &local_collective, 1, MPI_INT, MPI_MAX,
                  MPI_COMM_WORLD);
    return local_collective != 0;
}
#else
void broadcast_step_failure(RadiationDriver const& matrix_builder,
                            std::string& reason,
                            size_t& cell_id)
{
    reason = matrix_builder.getLastStepFailureReason();
    cell_id = matrix_builder.getLastStepFailureCellId();
}

std::vector<size_t> collect_step_failure_cells(
    RadiationDriver const& matrix_builder)
{
    if(matrix_builder.getLastStepFailureReason().empty() ||
       !matrix_builder.getLastStepFailureIsCellLocal() ||
       matrix_builder.getLastStepFailureCellId() ==
           std::numeric_limits<size_t>::max())
        return std::vector<size_t>();
    return std::vector<size_t>{matrix_builder.getLastStepFailureCellId()};
}

bool has_collective_step_failure(RadiationDriver const& matrix_builder)
{
    return !matrix_builder.getLastStepFailureReason().empty() &&
        !matrix_builder.getLastStepFailureIsCellLocal() &&
        !matrix_builder.getLastStepFailureIsRemote();
}
#endif

double passive_radiation_time_step_limit(
    ComputationalCell3D const& primitive,
    Conserved3D const& before,
    Conserved3D const& after,
    double maximum_radiation_energy_density,
    double event_dt)
{
    if(!(event_dt > 0) || !std::isfinite(event_dt) ||
       !(primitive.density > 0) || !std::isfinite(primitive.density) ||
       !(before.mass > 0) || !std::isfinite(before.mass))
        return std::numeric_limits<double>::max();

    double const inverse_volume = primitive.density / before.mass;
    double const before_Erad = before.Erad * inverse_volume;
    double const after_Erad = after.Erad * inverse_volume;
    double const epsilon = 64 * std::numeric_limits<double>::epsilon();
    double const total_reference = std::max(
        std::abs(before_Erad), std::abs(after_Erad));
    bool changed = std::abs(after_Erad - before_Erad) >
        epsilon * std::max(1.0, total_reference);
    double difference = std::abs(after_Erad - before_Erad) /
        (std::abs(after_Erad) +
         0.02 * maximum_radiation_energy_density +
         std::numeric_limits<double>::min());

    std::size_t const group_count = std::min(
        before.Eg.size(), after.Eg.size());
    double const group_floor = group_count == 0 ? 0 :
        (0.01 * maximum_radiation_energy_density +
         std::abs(after_Erad)) /
            static_cast<double>(group_count);
    for(std::size_t group = 0; group < group_count; ++group) {
        double const before_group = before.Eg[group] * inverse_volume;
        double const after_group = after.Eg[group] * inverse_volume;
        double const group_reference = std::max(
            std::abs(before_group), std::abs(after_group));
        double const delta = std::abs(after_group - before_group);
        changed = changed || delta >
            epsilon * std::max(1.0, group_reference);
        difference = std::max(difference,
            0.2 * delta /
                (std::abs(after_group) + group_floor +
                 std::numeric_limits<double>::min()));
    }
    if(!changed || !std::isfinite(difference))
        return std::numeric_limits<double>::max();

    // Extrapolate this event's accepted passive flux to the same 15% target
    // used by the active MG limiter.  Do not cap the estimate at two event
    // steps: commitEvent already compares it with the passive cell's actual
    // remaining interval.  A requested interval shorter than the event just
    // accepted cannot be applied retroactively, so wake at the next event.
    double const safe_dt = event_dt * 0.15 /
        std::max(difference, std::numeric_limits<double>::min());
    if(!std::isfinite(safe_dt))
        return std::numeric_limits<double>::max();
    return std::max(event_dt, safe_dt);
}

class RetryProbeBackoff
{
public:
    void restore(std::uint64_t const successful_candidates,
                 std::uint64_t const cooldown)
    {
        successful_candidates_ = static_cast<std::size_t>(std::min(
            successful_candidates,
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())));
        cooldown_ = static_cast<std::size_t>(std::max<std::uint64_t>(
            initial_cooldown,
            std::min<std::uint64_t>(maximum_cooldown, cooldown)));
    }

    void recordAcceptance(bool const was_probe)
    {
        if(was_probe)
            cooldown_ = initial_cooldown;
        ++successful_candidates_;
    }

    bool readyToProbe() const
    {return successful_candidates_ >= cooldown_;}

    void recordProbeScheduled()
    {successful_candidates_ = 0;}

    void recordFailure(bool const was_probe)
    {
        cooldown_ = was_probe ?
            std::min(maximum_cooldown, 2 * cooldown_) :
            initial_cooldown;
        successful_candidates_ = 0;
    }

    std::size_t cooldown() const
    {return cooldown_;}

    std::size_t successfulCandidates() const
    {return successful_candidates_;}

private:
    static constexpr std::size_t initial_cooldown = 8;
    static constexpr std::size_t maximum_cooldown = 256;
    std::size_t successful_candidates_ = 0;
    std::size_t cooldown_ = initial_cooldown;
};

bool identityOwnedMoveRequested()
{
    static bool const enabled = []()
    {
        char const* const value =
            std::getenv("RICH_INDIVIDUAL_IDENTITY_OWNED_MOVE");
        bool local_enabled = false;
        bool local_valid = true;
        if(value != nullptr && value[0] != '\0') {
            if(std::strcmp(value, "1") == 0 ||
               std::strcmp(value, "true") == 0 ||
               std::strcmp(value, "on") == 0 ||
               std::strcmp(value, "yes") == 0)
                local_enabled = true;
            else if(std::strcmp(value, "0") != 0 &&
                    std::strcmp(value, "false") != 0 &&
                    std::strcmp(value, "off") != 0 &&
                    std::strcmp(value, "no") != 0)
                local_valid = false;
        }
        unsigned int option_state = !local_valid ? 4u :
            (local_enabled ? 2u : 1u);
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &option_state, 1, MPI_UNSIGNED, MPI_BOR,
                      MPI_COMM_WORLD);
#endif
        if((option_state & 4u) != 0u)
            throw UniversalError(
                "RICH_INDIVIDUAL_IDENTITY_OWNED_MOVE must be unset or one "
                "of 0, 1, false, true, off, on, no, yes");
        if((option_state & 3u) == 3u)
            throw UniversalError(
                "RICH_INDIVIDUAL_IDENTITY_OWNED_MOVE differs between MPI "
                "ranks");
        return (option_state & 2u) != 0u;
    }();
    return enabled;
}

template<typename T>
class MovedOwnedStorageGuard
{
public:
    MovedOwnedStorageGuard(std::vector<T>& canonical,
                           std::vector<T>& local,
                           std::size_t const owned_size,
                           bool const active) :
        canonical_(canonical), local_(local), owned_size_(owned_size),
        active_(active)
    {}

    ~MovedOwnedStorageGuard()
    {
        if(active_)
            restore();
    }

    void commit()
    {
        if(!active_)
            return;
        restore();
        active_ = false;
    }

private:
    void restore()
    {
        local_.resize(owned_size_);
        canonical_ = std::move(local_);
    }

    std::vector<T>& canonical_;
    std::vector<T>& local_;
    std::size_t owned_size_;
    bool active_;
};

} // namespace
RadiationStep::RadiationStep(Tessellation3D &tess, std::vector<ComputationalCell3D> &cells,
                    std::vector<Conserved3D> &extensives,
                    ProgressTracker &pt,
                    #ifdef RICH_MPI
                        std::shared_ptr<CostCalculator3D> cost,
                    #endif // RICH_MPI
                    const RadiationDriver &matrix_builder, bool /*no_hydro*/) :
                    tess(tess), cells(cells), extensives(extensives), pt(pt), matrix_builder(matrix_builder)
                        , suggested_dt(std::numeric_limits<double>::max())
                    #ifdef RICH_MPI
                        , cost(cost)
                    #endif // RICH_MPI
{}

#ifdef RICH_MPI
    bool RadiationStep::allowRebalance(void)
    {
        return this->cost != nullptr;
    }

    std::string RadiationStep::getRequiredLB(void) const
    {
        if (!this->cost)
            return "";
        return "hydro";
    }

    std::vector<double> RadiationStep::getLoadBalanceWeights(void)
    {
        if (this->cost)
            return this->cost->CalculateCost(this->tess, this->cells);
        return std::vector<double>(this->tess.GetPointNo(), 1.0);
    }
#endif // RICH_MPI

void RadiationStep::step(double dt)
{
	int total_iters = 0;
	double const CG_eps = 1e-11;
	size_t const N = this->tess.GetPointNo();

#ifdef DEBUG
	if(N == 0) std::clog<<"Zero cells in RadiationTimeStep"<<std::endl;
#endif

	int rank = 0;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

	double total_elapsed_time = 0;
	double dt_try = dt;
	double dt_try_ceiling = dt;
	RetryProbeBackoff retry_probe_backoff;
	bool candidate_is_retry_probe = false;
	size_t reduce_counter = 0;
	int max_iter_done = 0;


	this->matrix_builder.prestep(this->tess, this->cells);
	MEMORY_DEBUG_PRINT("radiation: after prestep");
	while(total_elapsed_time < dt * 0.9999999)
	{
		dt_try = std::min(dt_try, dt - total_elapsed_time);

		double const candidate_time = this->pt.getTime() + total_elapsed_time;
		bool step_success = this->matrix_builder.step(CG_eps, total_iters,
			this->tess, this->cells, this->extensives, dt_try,
			candidate_time);
		MEMORY_DEBUG_PRINT("radiation: after solver step");

		max_iter_done = std::max(max_iter_done, total_iters);
		
		if(not step_success)
		{
			bool const rejected_retry_probe = candidate_is_retry_probe;
			candidate_is_retry_probe = false;
			retry_probe_backoff.recordFailure(rejected_retry_probe);
			reduce_counter++;
			dt_try *= 0.5;
			dt_try_ceiling = std::min(dt_try_ceiling, dt_try);
			std::string reason;
				size_t cell_id = std::numeric_limits<size_t>::max();
			broadcast_step_failure(this->matrix_builder, reason, cell_id);
			if(rank == 0) {
				std::ostringstream msg;
				msg << "Reducing dt, new dt " << dt_try;
				if (!reason.empty()) {
					msg << " (" << reason;
					if (cell_id != std::numeric_limits<size_t>::max())
						msg << ", example cell ID " << cell_id;
					msg << ")";
				}
				if(rejected_retry_probe)
					msg << " [retry probe rejected; next probe after "
					    << retry_probe_backoff.cooldown()
					    << " accepted candidates]";
				std::clog << msg.str() << std::endl;
			}
			
			double const next_elapsed_time = total_elapsed_time + dt_try;
			double const accepted_time = this->pt.getTime() + total_elapsed_time;
			double const next_candidate_end_time =
				this->pt.getTime() + next_elapsed_time;
			if(!(dt_try > 0) || !std::isfinite(dt_try) ||
			   !(next_elapsed_time > total_elapsed_time) ||
			   !(next_candidate_end_time > accepted_time)) {
				if (rank == 0) {
					std::ostringstream msg;
					msg << "Radiation step failed: candidate cannot advance after halving"
					    << " (target dt " << dt
					    << ", completed " << total_elapsed_time
					    << ", candidate dt " << dt_try << ")";
					if (!reason.empty()) {
						msg << ", last failure: " << reason;
						if (cell_id != std::numeric_limits<size_t>::max())
							msg << ", example cell ID " << cell_id;
					}
					std::clog << msg.str() << std::endl;
				}
				UniversalError eo(
					"radiation candidate cannot advance after halving");
				eo.addEntry("target dt", dt);
				eo.addEntry("completed elapsed time", total_elapsed_time);
				eo.addEntry("candidate dt", dt_try);
				eo.addEntry("accepted time", accepted_time);
				eo.addEntry("next candidate end time", next_candidate_end_time);
				if (!reason.empty()) {
					eo.addEntry("last radiation failure", reason);
					if (cell_id != std::numeric_limits<size_t>::max())
						eo.addEntry("example cell ID", cell_id);
				}
				throw eo;
			}
		}
		else {
			total_elapsed_time += dt_try;
			retry_probe_backoff.recordAcceptance(candidate_is_retry_probe);
			candidate_is_retry_probe = false;
			// A rejected candidate can force a very small fractional step.  Hold
			// it before probing a factor-two larger interval.  Failed recovery
			// probes exponentially extend the cooldown; a successful probe resets
			// it so transient restrictions still recover quickly.
			double const old_ceiling = dt_try_ceiling;
			double next_ceiling = old_ceiling;
			if(retry_probe_backoff.readyToProbe() && old_ceiling < dt)
				next_ceiling = std::min(dt, 2.0 * old_ceiling);
			double const next_dt = std::min(
				dt - total_elapsed_time,
				std::min(2.0 * dt_try, next_ceiling));
			if(next_dt > old_ceiling) {
				dt_try_ceiling = next_ceiling;
				retry_probe_backoff.recordProbeScheduled();
				candidate_is_retry_probe = true;
			}
			dt_try = next_dt;
		}
	}

	this->suggested_dt = this->matrix_builder.calculate_dt(dt, this->tess, this->cells);

	this->matrix_builder.poststep();
	MEMORY_DEBUG_PRINT("radiation: after poststep");

#ifdef RICH_MPI
	MPI_exchange_data(this->tess, this->cells, true);
#endif
	
    
	// double grow_factor = 1.25;
	// if(max_iter_done > 200)
	// 	grow_factor = 1.02;
	// else
	// 	if(max_iter_done > 125)
	// 		grow_factor = 1.05;

	// new_dt = std::min(new_dt, dt*grow_factor) * std::pow(0.5, std::max(static_cast<double>(reduce_counter), 0.0));
	// if(max_iter_done > 300)
	// 	new_dt = dt * 0.9;
	// return this->radiation_dt_;
}

bool RadiationStep::supportsIndividualTimeSteps(void) const
{
    return matrix_builder.supportsIndividualTimeSteps();
}

void RadiationStep::afterIndividualAMR(void) noexcept
{
    matrix_builder.releaseIndividualTopologyStorage();
}

void RadiationStep::onIndividualForceAllActiveLatch(void) noexcept
{
    matrix_builder.releaseIndividualTopologyStorage();
}

void RadiationStep::beforeIndividualRebalance(void) noexcept
{
    matrix_builder.releaseIndividualTopologyStorage();
}

void RadiationStep::stepIndividual(IndividualStepContext const& context)
{
    if(!matrix_builder.supportsIndividualTimeSteps())
        throw std::runtime_error("radiation driver does not support individual timesteps");
#ifndef RICH_MPI
    if(context.active_indices.empty())
        return;
#endif

    bool const trace_performance = individualPerformanceTraceEnabled();
    last_individual_performance.clear();
    auto phase_start = std::chrono::steady_clock::now();
    int rank = 0;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    std::size_t const canonical_owned_size = cells.size();
    ActiveMeshView const mesh_view(tess, canonical_owned_size);
    IndividualStepContext const local_context =
        mesh_view.remapContext(context, false);
    bool owned_mapping_identity =
        mesh_view.localSize() == canonical_owned_size;
    for(std::size_t local = 0;
        owned_mapping_identity && local < mesh_view.localSize(); ++local)
        owned_mapping_identity = mesh_view.localToGlobal(local) == local;
    bool local_all_active =
        context.active_mask.size() == canonical_owned_size &&
        context.active_indices.size() == canonical_owned_size;
    for(std::size_t owned = 0;
        local_all_active && owned < canonical_owned_size; ++owned)
        local_all_active = context.active_mask[owned] != 0;
    bool use_identity_owned_move = false;
    bool const identity_owned_move_requested = identityOwnedMoveRequested();
    if(identity_owned_move_requested) {
        int collective_eligible =
            owned_mapping_identity && local_all_active ? 1 : 0;
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &collective_eligible, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
#endif
        use_identity_owned_move = collective_eligible != 0;
    }
    if((trace_performance || identity_owned_move_requested) && rank == 0)
        std::clog << "RADIATION_IDENTITY_OWNED_MOVE requested="
                  << (identity_owned_move_requested ? 1 : 0)
                  << " mapping_identity=" << (owned_mapping_identity ? 1 : 0)
                  << " all_active=" << (local_all_active ? 1 : 0)
                  << " selected=" << (use_identity_owned_move ? 1 : 0)
                  << " owned_cells=" << canonical_owned_size << std::endl;

    std::vector<ComputationalCell3D> local_cells;
    std::vector<Conserved3D> local_extensives;
    std::vector<ComputationalCell3D> all_cells;
    std::vector<Conserved3D> all_extensives;
    if(use_identity_owned_move) {
        all_cells = cells;
        all_extensives = extensives;
        local_cells = std::move(cells);
        local_extensives = std::move(extensives);
    }
    else {
        local_cells = mesh_view.gatherOwned(cells);
        local_extensives = mesh_view.gatherOwned(extensives);
        all_cells = cells;
        all_extensives = extensives;
    }
    MovedOwnedStorageGuard<ComputationalCell3D> cells_move_guard(
        cells, local_cells, canonical_owned_size, use_identity_owned_move);
    MovedOwnedStorageGuard<Conserved3D> extensives_move_guard(
        extensives, local_extensives, canonical_owned_size,
        use_identity_owned_move);
    if(trace_performance)
        last_individual_performance["radiation-gather-remap"] =
            elapsedSeconds(phase_start);
    phase_start = std::chrono::steady_clock::now();
    tess.SyncPartialBuildData(local_cells, all_cells);
    tess.SyncPartialBuildData(local_extensives, all_extensives);
    if(trace_performance)
        last_individual_performance["radiation-ghost-sync"] =
            elapsedSeconds(phase_start);

    double const tolerance = 1e-11;
    double completed_fraction = 0;
    IndividualRadiationDefectAccounting* const defect_accounting =
        context.radiation_defect_accounting;
    double candidate_fraction_ceiling = 1;
    RetryProbeBackoff retry_probe_backoff;
    if(defect_accounting != nullptr &&
       (defect_accounting->accepted_dirichlet_candidates > 0 ||
        defect_accounting->defect_rejections > 0)) {
        candidate_fraction_ceiling =
            defect_accounting->cooldown_fraction_ceiling;
        retry_probe_backoff.restore(
            defect_accounting->cooldown_accepted_candidates,
            defect_accounting->cooldown_required_candidates);
    }
    double candidate_fraction = candidate_fraction_ceiling;
    std::uint64_t observed_defect_rejections =
        defect_accounting == nullptr ? 0 :
        defect_accounting->defect_rejections;
    std::uint64_t pending_defect_retry_substeps = 0;
    bool defect_retry_active = false;
    bool candidate_is_retry_probe = false;
    double event_minimum_candidate_fraction = 1;
    std::size_t rejected_candidates = 0;
    std::size_t rejected_retry_probes = 0;
    std::size_t maximum_retry_probe_cooldown = retry_probe_backoff.cooldown();
    std::string representative_reason;
    std::size_t representative_cell = std::numeric_limits<std::size_t>::max();
    std::map<std::size_t, double> cell_retry_fractions;
    bool retry_limiter_requires_all_active = false;
    int maximum_iterations = 0;
    std::uint64_t global_active_cells =
        static_cast<std::uint64_t>(local_context.active_indices.size());
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &global_active_cells, 1, MPI_UINT64_T,
                  MPI_SUM, MPI_COMM_WORLD);
#endif

    phase_start = std::chrono::steady_clock::now();
    matrix_builder.prestepIndividual(tess, local_cells, local_context);
    if(trace_performance)
        last_individual_performance["radiation-prestep"] =
            elapsedSeconds(phase_start);
    phase_start = std::chrono::steady_clock::now();
    while(completed_fraction < 1 - 1e-13) {
        candidate_fraction = std::min(candidate_fraction, 1 - completed_fraction);
        event_minimum_candidate_fraction = std::min(
            event_minimum_candidate_fraction, candidate_fraction);
        smallest_individual_candidate_fraction = std::min(
            smallest_individual_candidate_fraction, candidate_fraction);
        int iterations = 0;
        double const candidate_end_time = context.previous_event_time +
            (completed_fraction + candidate_fraction) *
            (context.event_time - context.previous_event_time);
        bool const accepted = matrix_builder.stepIndividual(
            tolerance, iterations, tess, local_cells, local_extensives,
            local_context, candidate_fraction, candidate_end_time,
            use_identity_owned_move ? &all_cells : &cells,
            use_identity_owned_move ? &all_extensives : &extensives,
            &mesh_view.localToGlobalMapping());
        if(defect_accounting != nullptr &&
           defect_accounting->defect_rejections >
               observed_defect_rejections) {
            observed_defect_rejections =
                defect_accounting->defect_rejections;
            defect_retry_active = true;
        }
        maximum_iterations = std::max(maximum_iterations, iterations);
        if(accepted) {
            if(defect_retry_active)
                ++pending_defect_retry_substeps;
            completed_fraction += candidate_fraction;
            retry_probe_backoff.recordAcceptance(candidate_is_retry_probe);
            candidate_is_retry_probe = false;
            double const old_ceiling = candidate_fraction_ceiling;
            double next_ceiling = old_ceiling;
            if(retry_probe_backoff.readyToProbe() && old_ceiling < 1)
                next_ceiling = std::min(1.0, 2 * old_ceiling);
            double const next_fraction = std::min(
                1 - completed_fraction,
                std::min(2 * candidate_fraction,
                         next_ceiling));
            if(next_fraction > old_ceiling) {
                candidate_fraction_ceiling = next_ceiling;
                retry_probe_backoff.recordProbeScheduled();
                candidate_is_retry_probe = true;
            }
            candidate_fraction = next_fraction;
            continue;
        }

        bool const rejected_retry_probe = candidate_is_retry_probe;
        candidate_is_retry_probe = false;
        retry_probe_backoff.recordFailure(rejected_retry_probe);
        if(rejected_retry_probe) {
            ++rejected_retry_probes;
            maximum_retry_probe_cooldown = std::max(
                maximum_retry_probe_cooldown,
                retry_probe_backoff.cooldown());
        }
        ++rejected_candidates;
        broadcast_step_failure(
            matrix_builder, representative_reason, representative_cell);
        std::vector<std::size_t> const failed_cells =
            collect_step_failure_cells(matrix_builder);
        bool const collective_failure =
            has_collective_step_failure(matrix_builder);
        if(rejected_candidates == 1 && rank == 0) {
            std::clog << "INDIVIDUAL_RADIATION_REJECTION"
                      << " completed_fraction=" << completed_fraction
                      << " rejected_fraction=" << candidate_fraction
                      << " previous_event_tick="
                      << context.previous_event_tick
                      << " event_tick=" << context.event_tick
                      << " previous_event_time="
                      << context.previous_event_time
                      << " event_time=" << context.event_time
                      << " outer_dt="
                      << (context.event_time -
                          context.previous_event_time)
                      << " candidate_end_time=" << candidate_end_time
                      << " global_active_cells=" << global_active_cells;
            if(!representative_reason.empty())
                std::clog << " reason=" << representative_reason;
            if(representative_cell != std::numeric_limits<std::size_t>::max())
                std::clog << " cell_id=" << representative_cell;
            std::clog << " solver_iterations=" << iterations << std::endl;
        }
        candidate_fraction *= 0.5;
        // A rejected candidate is transactionally rolled back.  Hold the
        // accepted substeps at or below its halved interval for a short
        // cooldown instead of probing the same known-bad larger interval
        // after every success.  Sustained acceptance then permits one
        // conservative doubling probe so a transient restriction does not
        // force the entire event to use its smallest interval.
        candidate_fraction_ceiling = std::min(
            candidate_fraction_ceiling, candidate_fraction);
        if(collective_failure || failed_cells.empty())
            retry_limiter_requires_all_active = true;
        else
            for(std::size_t const failed_cell : failed_cells) {
                auto const inserted = cell_retry_fractions.emplace(
                    failed_cell, candidate_fraction);
                if(!inserted.second)
                    inserted.first->second = std::min(
                        inserted.first->second, candidate_fraction);
            }
        double const next_completed_fraction =
            completed_fraction + candidate_fraction;
        double const accepted_time = context.previous_event_time +
            completed_fraction *
            (context.event_time - context.previous_event_time);
        double const next_candidate_end_time = context.previous_event_time +
            next_completed_fraction *
            (context.event_time - context.previous_event_time);
        if(!(candidate_fraction > 0) ||
           !(next_completed_fraction > completed_fraction) ||
           !(next_candidate_end_time > accepted_time)) {
            UniversalError error(
                "individual radiation candidate cannot advance after halving");
            error.addEntry("rejected candidates", rejected_candidates);
            error.addEntry("completed fraction", completed_fraction);
            error.addEntry("candidate fraction", candidate_fraction);
            error.addEntry("previous event time", context.previous_event_time);
            error.addEntry("event time", context.event_time);
            error.addEntry("accepted time", accepted_time);
            error.addEntry("next candidate end time", next_candidate_end_time);
            if(!representative_reason.empty())
                error.addEntry("representative failure", representative_reason);
            if(representative_cell != std::numeric_limits<std::size_t>::max())
                error.addEntry("representative cell ID", representative_cell);
            throw error;
        }
    }

    std::vector<double> local_suggested_dt(
        tess.GetPointNo(), std::numeric_limits<double>::max());
    matrix_builder.calculateIndividualTimeSteps(
        local_context, tess, local_cells, local_suggested_dt,
        use_identity_owned_move ? &all_cells : &cells,
        &mesh_view.localToGlobalMapping());
    if(retry_limiter_requires_all_active)
        for(std::size_t local : local_context.active_indices)
            local_suggested_dt.at(local) = std::min(
                local_suggested_dt.at(local),
                event_minimum_candidate_fraction *
                    local_context.cellTimeStep(local));
    else if(!cell_retry_fractions.empty())
        for(std::size_t local : local_context.active_indices) {
            auto const retry = cell_retry_fractions.find(
                local_cells.at(local).ID);
            if(retry != cell_retry_fractions.end())
                local_suggested_dt.at(local) = std::min(
                    local_suggested_dt.at(local),
                    retry->second * local_context.cellTimeStep(local));
        }
    if(trace_performance)
        last_individual_performance["radiation-driver"] =
            elapsedSeconds(phase_start);
    phase_start = std::chrono::steady_clock::now();
    matrix_builder.poststepIndividual();
    cumulative_individual_rejected_candidates += rejected_candidates;
    if(use_identity_owned_move) {
        cells_move_guard.commit();
        extensives_move_guard.commit();
    }
    else {
        mesh_view.scatterMeshOwned(local_cells, cells);
        mesh_view.scatterMeshOwned(local_extensives, extensives);
    }
    suggested_individual_dt.assign(cells.size(), std::numeric_limits<double>::max());
    for(std::size_t local = 0; local < mesh_view.localSize(); ++local)
        suggested_individual_dt[mesh_view.localToGlobal(local)] =
            local_suggested_dt[local];
    if(trace_performance)
        last_individual_performance["radiation-commit-scatter"] =
            elapsedSeconds(phase_start);
    phase_start = std::chrono::steady_clock::now();

    // Diffusion may conservatively update an inactive recipient without
    // refreshing its primitive state.  Limit that cell from its accepted
    // radiation-flux rate so the scheduler wakes it only when that
    // limit is shorter than the passive cell's remaining interval.
    double const event_dt = context.event_time - context.previous_event_time;
    double maximum_radiation_energy_density =
        std::numeric_limits<double>::min();
    if(all_cells.size() == all_extensives.size())
        for(std::size_t global = 0; global < all_extensives.size(); ++global)
            if(all_extensives[global].mass > 0 &&
               std::isfinite(all_extensives[global].mass) &&
               all_cells[global].density > 0 &&
               std::isfinite(all_cells[global].density))
                maximum_radiation_energy_density = std::max(
                    maximum_radiation_energy_density,
                    std::abs(all_extensives[global].Erad) *
                        all_cells[global].density /
                        all_extensives[global].mass);
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &maximum_radiation_energy_density, 1,
                  MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
    if(all_cells.size() == extensives.size() &&
       all_extensives.size() == extensives.size())
        for(std::size_t global = 0; global < extensives.size(); ++global) {
            if(context.isActive(global))
                continue;
            double const passive_limit = passive_radiation_time_step_limit(
                all_cells[global], all_extensives[global],
                extensives[global], maximum_radiation_energy_density,
                event_dt);
            suggested_individual_dt[global] = std::min(
                suggested_individual_dt[global], passive_limit);
        }

    if(rejected_candidates > 0 && rank == 0) {
        std::clog << "Individual radiation retries: " << rejected_candidates;
        if(!representative_reason.empty()) {
            std::clog << ", representative reason: " << representative_reason;
            if(representative_cell != std::numeric_limits<std::size_t>::max())
                std::clog << ", cell ID " << representative_cell;
        }
        std::clog << ", maximum solver iterations " << maximum_iterations
                  << ", next-step limiter scope "
                  << (retry_limiter_requires_all_active ?
                      "all_active" : "failed_cells")
                  << ", limited cells " << cell_retry_fractions.size()
                  << ", minimum fraction "
                  << event_minimum_candidate_fraction
                  << ", failed recovery probes " << rejected_retry_probes
                  << ", maximum probe cooldown "
                  << maximum_retry_probe_cooldown
                  << std::endl;
    }
    if(trace_performance)
        last_individual_performance["radiation-limit-update"] =
            elapsedSeconds(phase_start);
    if(defect_accounting != nullptr &&
       (defect_accounting->accepted_dirichlet_candidates > 0 ||
        defect_accounting->defect_rejections > 0)) {
        defect_accounting->cooldown_accepted_candidates =
            retry_probe_backoff.successfulCandidates();
        defect_accounting->cooldown_required_candidates =
            retry_probe_backoff.cooldown();
        defect_accounting->cooldown_fraction_ceiling =
            candidate_fraction_ceiling;
        defect_accounting->defect_retry_substeps +=
            pending_defect_retry_substeps;
    }
}

void RadiationStep::suggestIndividualTimeSteps(
    IndividualStepContext const& context,
    std::vector<double>& time_step_limits) const
{
    std::size_t const common_size = std::min(
        suggested_individual_dt.size(), time_step_limits.size());
    for(std::size_t cell = 0; cell < common_size; ++cell)
        if(suggested_individual_dt[cell] <
           std::numeric_limits<double>::max())
            time_step_limits[cell] = std::min(
                time_step_limits[cell], suggested_individual_dt[cell]);
    for(std::size_t cell : context.active_indices)
        if(cell >= suggested_individual_dt.size())
            time_step_limits.at(cell) = std::min(
                time_step_limits.at(cell), suggested_dt);
}

double RadiationStep::suggestTimeStep(void) const
{
    return this->suggested_dt;
}

std::string RadiationStep::getName(void) const
{
    return "radiation";
}
