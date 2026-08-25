#include "Simulation.hpp"
#include "misc/universal_error.hpp"
#include "misc/memory_debug.hpp"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#ifdef RICH_MPI
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace
{
void requireIndividualTimeStepSupport(PhysicsStep const& physics_step)
{
    if(physics_step.supportsIndividualTimeSteps())
        return;

    std::string message = "Physics step '" + physics_step.getName() +
        "' does not support individual timesteps";
    std::string const reason = physics_step.individualTimeStepUnsupportedReason();
    if(!reason.empty())
        message += ": " + reason;
    throw std::invalid_argument(message);
}

#ifdef RICH_MPI
struct IndividualRebalanceRuntimeOptions
{
    bool enabled = false;
    bool trace = false;
    bool before_first_event = false;
    double threshold = 1.25;
    double immediate_amr_threshold = 1.5;
    std::size_t cooldown_events = 8;
    double amortization_factor = 2;
};

struct IndividualForceAllActiveRuntimeOptions
{
    bool enabled = false;
    bool latch_enabled = false;
    std::size_t minimum_bin = 63;
};

bool parseEnvironmentToggle(char const* const name, bool const fallback,
                             bool& valid)
{
    char const* const value = std::getenv(name);
    if(value == nullptr || value[0] == '\0')
        return fallback;
    if(std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
       std::strcmp(value, "on") == 0 || std::strcmp(value, "yes") == 0)
        return true;
    if(std::strcmp(value, "0") == 0 || std::strcmp(value, "false") == 0 ||
       std::strcmp(value, "off") == 0 || std::strcmp(value, "no") == 0)
        return false;
    valid = false;
    return fallback;
}

double parseEnvironmentDouble(char const* const name, double const fallback,
                              double const minimum, bool& valid)
{
    char const* const value = std::getenv(name);
    if(value == nullptr || value[0] == '\0')
        return fallback;
    errno = 0;
    char* end = nullptr;
    double const parsed = std::strtod(value, &end);
    if(errno != 0 || end == value || *end != '\0' ||
       !std::isfinite(parsed) || parsed < minimum)
    {
        valid = false;
        return fallback;
    }
    return parsed;
}

std::size_t parseEnvironmentSize(char const* const name,
                                 std::size_t const fallback, bool& valid)
{
    char const* const value = std::getenv(name);
    if(value == nullptr || value[0] == '\0')
        return fallback;
    errno = 0;
    char* end = nullptr;
    unsigned long long const parsed = std::strtoull(value, &end, 10);
    if(value[0] == '-' || errno != 0 || end == value || *end != '\0' ||
       parsed == 0 ||
       parsed > static_cast<unsigned long long>(
           std::numeric_limits<std::size_t>::max()))
    {
        valid = false;
        return fallback;
    }
    return static_cast<std::size_t>(parsed);
}

IndividualRebalanceRuntimeOptions const& individualRebalanceRuntimeOptions()
{
    static IndividualRebalanceRuntimeOptions const options = []()
    {
        IndividualRebalanceRuntimeOptions result;
        bool locally_valid = true;
        result.enabled = parseEnvironmentToggle(
            "RICH_INDIVIDUAL_AUTO_REBALANCE", false, locally_valid);
        result.trace = parseEnvironmentToggle(
            "RICH_INDIVIDUAL_PERF_TRACE", false, locally_valid);
        result.before_first_event = parseEnvironmentToggle(
            "RICH_INDIVIDUAL_REBALANCE_BEFORE_FIRST_EVENT", false,
            locally_valid);
        result.threshold = parseEnvironmentDouble(
            "RICH_INDIVIDUAL_REBALANCE_THRESHOLD", 1.25, 1.0,
            locally_valid);
        result.immediate_amr_threshold = parseEnvironmentDouble(
            "RICH_INDIVIDUAL_REBALANCE_AMR_THRESHOLD", 1.5,
            result.threshold, locally_valid);
        result.cooldown_events = parseEnvironmentSize(
            "RICH_INDIVIDUAL_REBALANCE_COOLDOWN", 8, locally_valid);
        result.amortization_factor = parseEnvironmentDouble(
            "RICH_INDIVIDUAL_REBALANCE_AMORTIZATION", 2.0, 0.0,
            locally_valid);

        int local_mask = (result.enabled ? 1 : 0) |
            (result.trace ? 2 : 0) |
            (result.before_first_event ? 4 : 0);
        int minimum_mask = local_mask;
        int maximum_mask = local_mask;
        double minimum_values[3] = {result.threshold,
                                    result.immediate_amr_threshold,
                                    result.amortization_factor};
        double maximum_values[3] = {result.threshold,
                                    result.immediate_amr_threshold,
                                    result.amortization_factor};
        unsigned long long minimum_cooldown = result.cooldown_events;
        unsigned long long maximum_cooldown = result.cooldown_events;
        int collective_valid = locally_valid ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &collective_valid, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &minimum_mask, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &maximum_mask, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, minimum_values, 3, MPI_DOUBLE, MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, maximum_values, 3, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &minimum_cooldown, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &maximum_cooldown, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        if(collective_valid == 0 || minimum_mask != maximum_mask ||
           !std::equal(std::begin(minimum_values), std::end(minimum_values),
                       std::begin(maximum_values)) ||
           minimum_cooldown != maximum_cooldown)
            throw std::invalid_argument(
                "Invalid or inconsistent individual rebalance environment options");
        return result;
    }();
    return options;
}

IndividualForceAllActiveRuntimeOptions const&
individualForceAllActiveRuntimeOptions()
{
    static IndividualForceAllActiveRuntimeOptions const options = []()
    {
        IndividualForceAllActiveRuntimeOptions result;
        bool locally_valid = true;
        char const* const raw = std::getenv(
            "RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_MIN_BIN");
        result.enabled = raw != nullptr && raw[0] != '\0';
        result.latch_enabled = parseEnvironmentToggle(
            "RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_LATCH", false,
            locally_valid);
        result.minimum_bin = parseEnvironmentSize(
            "RICH_INDIVIDUAL_FORCE_ALL_ACTIVE_MIN_BIN", 63,
            locally_valid);
        if(result.minimum_bin >= 64)
            locally_valid = false;
        if(result.latch_enabled && !result.enabled)
            locally_valid = false;

        int const local_mask = (result.enabled ? 1 : 0) |
            (result.latch_enabled ? 2 : 0);
        int minimum_mask = local_mask;
        int maximum_mask = local_mask;
        unsigned long long minimum_bin = result.minimum_bin;
        unsigned long long maximum_bin = result.minimum_bin;
        int collective_valid = locally_valid ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &collective_valid, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &minimum_mask, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &maximum_mask, 1, MPI_INT, MPI_MAX,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &minimum_bin, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &maximum_bin, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        if(collective_valid == 0 || minimum_mask != maximum_mask ||
           minimum_bin != maximum_bin)
            throw std::invalid_argument(
                "Invalid or inconsistent forced-active event environment option");
        return result;
    }();
    return options;
}

void reportIndividualRankDistribution(int const rank, int const rank_count,
                                      std::size_t const cycle,
                                      std::string const& phase,
                                      double const local_value,
                                      char const* const unit)
{
    std::vector<double> values(rank == 0 ? rank_count : 0);
    MPI_Gather(&local_value, 1, MPI_DOUBLE,
               rank == 0 ? values.data() : nullptr, 1, MPI_DOUBLE, 0,
               MPI_COMM_WORLD);
    if(rank != 0)
        return;
    auto const minimum_iterator =
        std::min_element(values.begin(), values.end());
    auto const maximum_iterator =
        std::max_element(values.begin(), values.end());
    int const minimum_rank =
        static_cast<int>(std::distance(values.begin(), minimum_iterator));
    int const maximum_rank =
        static_cast<int>(std::distance(values.begin(), maximum_iterator));
    std::sort(values.begin(), values.end());
    std::size_t const median_index = (values.size() - 1) / 2;
    std::size_t const p95_index = static_cast<std::size_t>(
        std::ceil(0.95 * static_cast<double>(values.size()))) - 1;
    double const mean = std::accumulate(values.begin(), values.end(), 0.0) /
        static_cast<double>(values.size());
    std::clog << std::setprecision(17)
              << "INDIVIDUAL_PERF cycle=" << cycle
              << " phase=" << phase
               << " unit=" << unit
               << " min=" << values.front()
               << " min_rank=" << minimum_rank
               << " median=" << values[median_index]
               << " mean=" << mean
               << " p95=" << values[p95_index]
               << " max=" << values.back()
               << " max_rank=" << maximum_rank << std::endl;
}

double peakResidentSetKiB()
{
    struct rusage usage;
    return getrusage(RUSAGE_SELF, &usage) == 0 ?
        static_cast<double>(usage.ru_maxrss) : -1.0;
}

double currentResidentSetKiB()
{
    std::ifstream statm("/proc/self/statm");
    unsigned long long virtual_pages = 0;
    unsigned long long resident_pages = 0;
    long const page_size = sysconf(_SC_PAGESIZE);
    if(page_size <= 0 || !(statm >> virtual_pages >> resident_pages))
        return -1.0;
    return static_cast<double>(resident_pages) *
        static_cast<double>(page_size) / 1024.0;
}
#endif
}

Simulation::Simulation(Tessellation3D &tess_, const std::vector<ComputationalCell3D> &cells_, EquationOfState &eos_, bool new_start) :
     tess(tess_), cells(cells_), extensives(cells_.size()), eos(eos_), Max_ID(0),
     wallclockTime(0), initializedFromRestart(!new_start)
#ifdef RICH_MPI
     , currentBox(tess_.GetBoxCoordinates())
#endif // RICH_MPI
{
    #ifdef RICH_MPI
        this->currentLoad = nullptr;
        MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
        MPI_Comm_size(MPI_COMM_WORLD, &this->size);
    #else // RICH_MPI
        this->rank = 0;
        this->size = 1;
    #endif // RICH_MPI

    if(new_start)
    {
        this->initializeCellIDs();
    }
    else
    {
        this->recomputeMaxID();
    }

#ifdef RICH_MPI
    ComputationalCell3D cdummy;
    MPI_exchange_data(this->tess, this->cells, true, 1, &cdummy);
#endif

    size_t N = this->tess.GetPointNo();
    for(size_t i = 0; i < N; ++i)
    {
        PrimitiveToConserved(this->cells[i], this->tess.GetVolume(i), this->extensives[i]);
    }
}

void Simulation::initializeCellIDs(void)
{
    size_t N = this->cells.size();
    size_t nstart = 0;
#ifdef RICH_MPI
    std::vector<size_t> nrecv(static_cast<size_t>(this->size), 0);
    size_t nsend = N;
    MPI_Allgather(&nsend, 1, MPI_UNSIGNED_LONG_LONG, &nrecv[0], 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    for (int i = 0; i < this->rank; ++i)
        nstart += nrecv[static_cast<size_t>(i)];
#endif
    for (size_t i = 0; i < N; ++i)
        this->cells[i].ID = nstart + i;
    this->Max_ID = nstart + N - 1;
#ifdef RICH_MPI
    for (int i = this->rank + 1; i < this->size; ++i)
        this->Max_ID += nrecv[static_cast<size_t>(i)];
#endif
}

void Simulation::recomputeMaxID(void)
{
    size_t N = this->cells.size();
    size_t maxid = 0;
    for (size_t i = 0; i < N; ++i)
        maxid = std::max(maxid, this->cells[i].ID);
#ifdef RICH_MPI
    MPI_Allreduce(&maxid, &this->Max_ID, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
#else
    this->Max_ID = maxid;
#endif
}

size_t &Simulation::GetMaxID(void)
{
    return this->Max_ID;
}

const size_t &Simulation::GetMaxID(void) const
{
    return this->Max_ID;
}

void Simulation::addPhysics(std::shared_ptr<PhysicsStep> physicsStep)
{
    if(this->timeIntegrationMode == TimeIntegrationMode::Individual)
        requireIndividualTimeStepSupport(*physicsStep);
    this->physics.push_back(physicsStep);
}

void Simulation::EnableIndividualTimeSteps(IndividualTimeStepOptions options)
{
	if(this->individualScheduler && this->individualScheduler->initialized())
		throw std::logic_error("Individual timesteps are already active");
	for(const std::shared_ptr<PhysicsStep> &physicsStep : this->physics)
		requireIndividualTimeStepSupport(*physicsStep);
#ifdef RICH_MPI
	// Expose canonical owned-cell state immediately.  Callers may initialize or
	// customize bins before the first event; ghost copies must never enter the
	// stable-ID scheduler.
	this->cells.resize(this->tess.GetPointNo());
	this->extensives.resize(this->tess.GetPointNo());
#endif
	this->individualScheduler = std::make_unique<IndividualTimeStepScheduler>(options);
    this->timeIntegrationMode = TimeIntegrationMode::Individual;
}

void Simulation::RequestSynchronizedIndividualEvent(void)
{
    if(this->timeIntegrationMode != TimeIntegrationMode::Individual ||
       !this->individualScheduler)
        throw std::logic_error(
            "A synchronized event requires individual-timestep mode");
    if(this->individualEventInProgress)
        throw std::logic_error(
            "Cannot request a synchronized event during an individual event");
    this->individualSynchronizedEventRequested = true;
}

bool Simulation::IndividualStateSynchronized(void) const
{
    bool synchronized =
        this->timeIntegrationMode == TimeIntegrationMode::Individual &&
        !this->individualEventInProgress &&
        this->individualScheduler && this->individualScheduler->initialized() &&
        this->extensives.size() == this->cells.size() &&
        this->cells.size() == this->individualScheduler->states().size() &&
        this->tess.GetPointNo() == this->cells.size();
    if(synchronized)
    {
        std::uint64_t const current_tick =
            this->individualScheduler->currentTick();
        for(CellTimeState const& state : this->individualScheduler->states())
            if(state.begin_tick != current_tick ||
               state.last_primitive_tick != current_tick)
            {
                synchronized = false;
                break;
            }
    }
#ifdef RICH_MPI
    int synchronized_on_every_rank = synchronized ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &synchronized_on_every_rank, 1, MPI_INT,
                  MPI_MIN, MPI_COMM_WORLD);
    synchronized = synchronized_on_every_rank != 0;
#endif
    return synchronized;
}

double Simulation::GetTime(void) const
{
    return this->tracker.getTime();
}

size_t Simulation::GetCycle(void) const
{
    return this->tracker.getCycle();
}

void Simulation::SetCycle(size_t cycle)
{
    this->tracker.cycle = cycle;
    // ReadSimulation restores the cycle on an already constructed Simulation.
    // Remember that provenance so an opt-in pre-event rebalance can
    // distinguish restored committed state from a fresh start.
    this->initializedFromRestart = true;
}

void Simulation::SetTime(double t)
{
    this->tracker.time = t;
    if(this->individualScheduler)
        this->individualScheduler->resetTimeOrigin(t);
}

void Simulation::SetTimeStep(double dt)
{
    this->tsc->SetTimeStep(dt);
}

double Simulation::GetTimeStep(void) const
{
    if(this->timeIntegrationMode == TimeIntegrationMode::Individual &&
       this->individualScheduler && this->individualScheduler->initialized())
        return this->individualScheduler->nextEventTimeStep();
    return this->tsc->GetTimeStep();
}

#ifdef RICH_MPI
    void Simulation::buildDataTransfer(void)
    {
        MPI_exchange_data(this->tess, this->extensives, false);
        MPI_exchange_data(this->tess, this->cells, false);
        const bool individual =
            this->timeIntegrationMode == TimeIntegrationMode::Individual &&
            this->individualScheduler && this->individualScheduler->initialized();
        if(individual)
            MPI_exchange_data(this->tess, this->individualScheduler->states(), false);
        else
        {
            ComputationalCell3D cdummy;
            MPI_exchange_data(this->tess, this->cells, true, 1, &cdummy);
        }
        for(MigrationBuffer &buff : this->migrationBuffers)
        {
            buff.transfer();
        }
        if(individual)
        {
            // Individual state is canonical and owned-only between events.
            this->cells.resize(this->tess.GetPointNo());
            this->extensives.resize(this->tess.GetPointNo());
            this->individualScheduler->rebuildIndex(this->cells);
        }
    }

    void Simulation::buildDataTransfer(const ExchangeChain &chain)
    {
        if (chain.GetNorg() == 0)
            return;
        for(MigrationBuffer &buff : this->migrationBuffers)
        {
            buff.transferChain(chain);
        }
    }
#endif // RICH_MPI

double Simulation::GetWallclockTime(void) const
{
    return this->wallclockTime;
}

void Simulation::SetWallclockTime(double t)
{
    this->wallclockTime = t;
}

void Simulation::step(void)
{
    if(this->timeIntegrationMode == TimeIntegrationMode::Individual)
    {
        this->stepIndividual();
        return;
    }

    MEMORY_DEBUG_PRINT("Simulation::step START cycle=" + std::to_string(this->tracker.getCycle()));
    this->lastPhysicsTimes.clear();
    auto stepWallStart = std::chrono::high_resolution_clock::now();
    double next_time_step = std::numeric_limits<double>::max();
    // double dt = std::numeric_limits<double>::max();
    #ifdef RICH_MPI
        if(this->rank == 0)
    #endif // RICH_MPI
    {
        std::cout << "\nCycle " << this->tracker.getCycle() << " at time " << this->tracker.getTime() << std::endl;
    }

    for(std::shared_ptr<PhysicsStep> physics : this->physics)
    {
        std::string name = physics->getName();
        if(this->rank == 0) std::cout << "Running physics: " << name << std::endl;

        bool didRebalance = false;
        #ifdef RICH_MPI
            // if(this->tracker.getCycle() == this->lastRebalanceCycle + 2)
            // {
            //     physics->dumpCost(this->tracker.getCycle());
            // }

            std::string LB = physics->getRequiredLB();
            bool firstTime = false;

            if(this->currentLB != LB)
            {
                if(this->rank == 0) std::cout << "Changing load balance to " << LB << " (from " << this->currentLB << ")" << std::endl;
                auto it = this->loads.find(LB);
                if(it != this->loads.cend())
                {
                    if(this->rank == 0) std::cout << "Load balance restored" << std::endl;
                    this->setCurrentLoadBalance(LB);
                }
                else
                {
                    if(this->rank == 0) std::cout << "Load balance generated for first time" << std::endl;
                    // std::vector<double> weights = physics->getLoadBalanceWeights();
                    // std::vector<Vector3D> points = this->tess.getMeshPoints();
                    // points.resize(this->tess.GetPointNo());
                    // this->tess.BuildParallel(points, weights, true);
                    firstTime = true;
                    // this->buildDataTransfer();
                }
            }

            bool forceRebalance = this->forceRebalanceSteps > 0 && this->tracker.getCycle() < this->forceRebalanceSteps;
            double rebalanceTime = 0;

            if(physics->allowRebalance() || forceRebalance)
            {
                if(this->rank == 0) std::cout << "allowRebalance=true, computing weights..." << std::endl;
                std::vector<double> weights = physics->getLoadBalanceWeights();
                if(this->rank == 0) std::cout << "Weights computed (" << weights.size() << "), checking ShouldRebalance..." << std::endl;
                bool shouldRebalance = this->tess.ShouldRebalance(weights);
                if(this->rank == 0)
                {
                    std::cout << "Should Rebalance: " << shouldRebalance << std::endl;
                }
                if(shouldRebalance)
                {
                    if(this->rank == 0) std::cout << "Doing rebalance on LB " << LB << std::endl;
                    auto rebalanceStart = std::chrono::high_resolution_clock::now();

                    didRebalance = true;
                    this->lastRebalanceCycle = this->tracker.getCycle();
                    physics->beforeLB();
                    this->tess.Rebalance(weights);
                    if(this->rank == 0)
                    {
                        std::cout << "Did rebalanced" << std::endl;
                        // auto lb = this->tess.GetLoadBalancer();
                        // if (lb) lb->printInfo();
                    }                
                    this->buildDataTransfer();
                    physics->afterLB();

                    MPI_Barrier(MPI_COMM_WORLD);
                    rebalanceTime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - rebalanceStart).count();
                    if(this->rank == 0) std::cout << "Rebalance time: " << rebalanceTime << "s" << std::endl;
                }
                else
                {
                    if(this->rank == 0) std::cout << LB << " is already rebalanced" << std::endl;
                }
            }

            std::shared_ptr<LoadBalancer<Vector3D>> load = this->tess.GetLoadBalancer();
            this->loads[LB] = load;
            this->currentLoad = load;
            this->currentLB = LB;
        #endif // RICH_MPI

        double dt = this->tsc->GetTimeStep();
        if(this->rank == 0) std::cout << "Running " << name << " with dt " << dt << std::endl;
        std::cout.flush();
        double dt_before = dt;

        MEMORY_DEBUG_PRINT("Before " + name);
        #ifdef RICH_MPI
            MPI_Barrier(MPI_COMM_WORLD);
        #endif // RICH_MPI
        auto start = std::chrono::high_resolution_clock::now();

        if(not didRebalance and this->tracker.getCycle() > 100)
        {
            vtune_start();
        }
        physics->step(dt);
        vtune_stop();

        double dt_actual = this->tsc->GetTimeStep();
        if(this->rank == 0 && dt_actual != dt_before)
            std::cout << "Hydro dt actually used: " << dt_actual << " (requested: " << dt_before << ")" << std::endl;

        double localTime = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count();

        #ifdef RICH_MPI
            MPI_Barrier(MPI_COMM_WORLD);
        #endif // RICH_MPI

        MEMORY_DEBUG_PRINT("After " + name);
        auto end = std::chrono::high_resolution_clock::now();
        double physicsTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

        #ifdef RICH_MPI
        double totalPhysicsTime = physicsTime + rebalanceTime;
        if(this->rank == 0) std::cout << "Physics " << name << " time: " << totalPhysicsTime << " (step=" << physicsTime << "s, rebalance=" << rebalanceTime << "s)" << std::endl;
        this->lastPhysicsTimes[name] = totalPhysicsTime;
        this->lastLocalPhysicsTimes[name] = localTime;
        #else
        if(this->rank == 0) std::cout << "Physics " << name << " time: " << physicsTime << std::endl;
        this->lastPhysicsTimes[name] = physicsTime;
        this->lastLocalPhysicsTimes[name] = localTime;
        #endif

        double dt_suggest = physics->suggestTimeStep();
        next_time_step = std::min(next_time_step, dt_suggest);
        // if(this->rank == 0) std::cout << "Suggested " << next_time_step << ", dt_suggest " << dt_suggest << std::endl;
        
        #ifdef RICH_MPI
            this->buildDataTransfer(physics->GetExchangeChain());
            
            if(firstTime)
            {
                physics->beforeLB();
                std::vector<double> weights = physics->getLoadBalanceWeights();
                this->tess.Rebalance(weights);
                if(this->rank == 0)
                {
                    std::cout << "Rebalanced first time - load balance:" << std::endl;
                    auto lb = this->tess.GetLoadBalancer();
                    if (lb) lb->printInfo();
                }            
                this->buildDataTransfer();
                physics->afterLB();
            }
        #endif // RICH_MPI

        #ifdef RICH_MPI
            std::pair<Vector3D, Vector3D> newBox = this->tess.GetBoxCoordinates();
            if(newBox != this->currentBox)
            {
                this->currentBox = newBox;
                for(auto &entry : this->loads)
                {
                    entry.second->changeBox(this->currentBox);
                }
            }
        #endif // RICH_MPI

        if(this->rank == 0)
        {
            std::cout << name << " suggested " << dt_suggest << " for dt " << std::endl;
            std::cout << std::endl;
        }
    }
    
    this->tracker.updateCycle();
    double dt_used = this->tsc->GetTimeStep();
    if(this->rank == 0)
        std::cout << "Advancing time by dt=" << dt_used << ", next suggested dt=" << next_time_step << std::endl;
    this->tracker.updateTime(dt_used);

    this->tsc->SetTimeStep(next_time_step);

    double stepWallSec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - stepWallStart).count();
    this->wallclockTime += stepWallSec;
}

#ifdef RICH_MPI
std::shared_ptr<PhysicsStep> Simulation::findIndividualBalanceStep(void) const
{
    for(std::shared_ptr<PhysicsStep> const& physicsStep : this->physics)
        if(physicsStep->allowRebalance())
            return physicsStep;
    return std::shared_ptr<PhysicsStep>();
}

Simulation::IndividualRebalanceResult
Simulation::rebalanceCommittedIndividualState(
    std::shared_ptr<PhysicsStep> const& balanceStep,
    bool const forceRebalance, double const threshold)
{
    if(!balanceStep || !balanceStep->allowRebalance())
        throw std::logic_error(
            "Individual load balancing has no supporting physics step");
    if(!this->individualScheduler ||
       !this->individualScheduler->initialized())
        throw std::logic_error(
            "Individual load balancing requires initialized scheduler state");

    IndividualRebalanceResult result;
    result.currentRssBeforeKiB = currentResidentSetKiB();
    const auto balance_start = std::chrono::high_resolution_clock::now();
    this->lastIndividualBalanceCheckCycle = this->tracker.getCycle();

    int local_state_valid =
        this->cells.size() == this->tess.GetPointNo() &&
        this->extensives.size() == this->cells.size() &&
        this->individualScheduler->states().size() == this->cells.size() ?
        1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &local_state_valid, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    if(local_state_valid == 0)
        throw std::logic_error(
            "Individual load balancing requires aligned committed state");

    unsigned long long local_owned_cells =
        static_cast<unsigned long long>(this->cells.size());
    result.totalOwnedCells = local_owned_cells;
    MPI_Allreduce(MPI_IN_PLACE, &result.totalOwnedCells, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

    std::vector<std::size_t> previous_owned_ids;
    previous_owned_ids.reserve(this->cells.size());
    for(ComputationalCell3D const& cell : this->cells)
        previous_owned_ids.push_back(cell.ID);
    std::sort(previous_owned_ids.begin(), previous_owned_ids.end());
    {
        std::vector<Vector3D> points = this->tess.getAllPoints();
        int local_points_valid = points.size() >= this->cells.size() ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &local_points_valid, 1, MPI_INT, MPI_MIN,
                      MPI_COMM_WORLD);
        if(local_points_valid == 0)
            throw std::logic_error(
                "Individual load balancing is missing canonical generator positions");
        points.resize(this->cells.size());
        this->tess.BuildParallel(points, true /* no rebalance */,
                                 true /* no exchange */);
    }

    balanceStep->beforeLB();
    std::vector<double> weights = balanceStep->getLoadBalanceWeights();
    int local_weights_valid = weights.size() == this->tess.GetPointNo() ?
        1 : 0;
    double local_weight = 0;
    if(local_weights_valid != 0)
        for(double const weight : weights)
        {
            if(!std::isfinite(weight) || weight <= 0)
            {
                local_weights_valid = 0;
                break;
            }
            local_weight += weight;
        }
    MPI_Allreduce(MPI_IN_PLACE, &local_weights_valid, 1, MPI_INT, MPI_MIN,
                  MPI_COMM_WORLD);
    if(local_weights_valid == 0)
    {
        balanceStep->afterLB();
        throw std::logic_error(
            "Individual load-balance weights must align with owned cells "
            "and be finite and positive");
    }

    double total_weight = local_weight;
    double maximum_weight = local_weight;
    MPI_Allreduce(MPI_IN_PLACE, &total_weight, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maximum_weight, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    double const mean_weight = total_weight / static_cast<double>(this->size);
    result.weightSkew = mean_weight > 0 ? maximum_weight / mean_weight : 1.0;
    int local_should_rebalance =
        (forceRebalance ||
         (result.weightSkew > threshold &&
          this->tess.ShouldRebalance(weights))) ? 1 : 0;
    int minimum_should_rebalance = local_should_rebalance;
    int maximum_should_rebalance = local_should_rebalance;
    MPI_Allreduce(MPI_IN_PLACE, &minimum_should_rebalance, 1, MPI_INT,
                  MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maximum_should_rebalance, 1, MPI_INT,
                  MPI_MAX, MPI_COMM_WORLD);
    if(minimum_should_rebalance != maximum_should_rebalance)
    {
        balanceStep->afterLB();
        throw std::logic_error(
            "Individual load-balance decision differs across MPI ranks");
    }
    result.applied = maximum_should_rebalance != 0;
    bool owned_count_preserved = true;

    if(result.applied)
    {
        // Release every package's old ownership-sized scratch before the
        // tessellation and registered state buffers migrate.  Calling only the
        // selected balanceStep's beforeLB() is insufficient: radiation is
        // normally not the package that supplies the hydro balance weights.
        for(std::shared_ptr<PhysicsStep> const& physicsStep : this->physics)
            physicsStep->beforeIndividualRebalance();
        this->tess.Rebalance(weights);
        std::vector<double>().swap(weights);
        // This migrates primitives, extensives, scheduler state, and every
        // registered migration buffer by stable cell ID, then rebuilds the
        // scheduler index against the new ownership.
        this->buildDataTransfer();
        for(ComputationalCell3D const& cell : this->cells)
            if(!std::binary_search(previous_owned_ids.begin(),
                                   previous_owned_ids.end(), cell.ID))
                ++result.migratedCells;
        MPI_Allreduce(MPI_IN_PLACE, &result.migratedCells, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        unsigned long long total_owned_after =
            static_cast<unsigned long long>(this->cells.size());
        MPI_Allreduce(MPI_IN_PLACE, &total_owned_after, 1,
                      MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        owned_count_preserved = total_owned_after == result.totalOwnedCells;
        if(owned_count_preserved)
        {
            this->lastRebalanceCycle = this->tracker.getCycle();
            ++this->individualOwnershipEpoch;
            const std::string load_name = balanceStep->getRequiredLB();
            this->currentLoad = this->tess.GetLoadBalancer();
            this->loads[load_name] = this->currentLoad;
            this->currentLB = load_name;
        }
    }
	std::vector<double>().swap(weights);
	std::vector<std::size_t>().swap(previous_owned_ids);

    // The selected physics step owns topology-dependent mesh caches.
    balanceStep->afterLB();
    if(!owned_count_preserved)
        throw std::logic_error(
            "Individual load balancing changed global owned-cell count");
    MPI_Barrier(MPI_COMM_WORLD);
    result.localSeconds = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - balance_start).count();
    result.maximumSeconds = result.localSeconds;
    MPI_Allreduce(MPI_IN_PLACE, &result.maximumSeconds, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
    if(result.applied)
        this->lastIndividualRebalanceSeconds = result.maximumSeconds;
    result.currentRssAfterKiB = currentResidentSetKiB();
    return result;
}

void Simulation::maybeRebalanceBeforeFirstIndividualEvent(void)
{
    if(this->preFirstIndividualRebalanceChecked)
        return;
    this->preFirstIndividualRebalanceChecked = true;

    IndividualRebalanceRuntimeOptions const& options =
        individualRebalanceRuntimeOptions();
    auto const trace_decision = [&](char const* const reason,
                                    bool const supported,
                                    double const owned_cell_skew,
                                    bool const requested)
    {
        if(options.trace && this->rank == 0)
            std::clog << std::setprecision(17)
                      << "INDIVIDUAL_PRE_FIRST_EVENT_REBALANCE_DECISION"
                      << " cycle=" << this->tracker.getCycle()
                      << " restart="
                      << (this->initializedFromRestart ? 1 : 0)
                      << " enabled="
                      << (options.before_first_event ? 1 : 0)
                      << " auto_rebalance=" << (options.enabled ? 1 : 0)
                      << " supported=" << (supported ? 1 : 0)
                      << " owned_max_mean=" << owned_cell_skew
                      << " threshold=" << options.threshold
                      << " requested=" << (requested ? 1 : 0)
                      << " reason=" << reason
                      << " ownership_epoch="
                      << this->individualOwnershipEpoch
                      << std::endl;
    };

    if(!options.before_first_event)
    {
        trace_decision("disabled", false, 1.0, false);
        return;
    }

    int const local_state = (this->initializedFromRestart ? 1 : 0) |
        (this->individualScheduler &&
         this->individualScheduler->initialized() ? 2 : 0);
    unsigned int state_mask =
        1u << static_cast<unsigned int>(local_state);
    MPI_Allreduce(MPI_IN_PLACE, &state_mask, 1, MPI_UNSIGNED, MPI_BOR,
                  MPI_COMM_WORLD);
    if((state_mask & (state_mask - 1u)) != 0u)
        throw std::logic_error(
            "Pre-first-event rebalance state differs across MPI ranks");
    if(!this->initializedFromRestart)
    {
        trace_decision("fresh_start", false, 1.0, false);
        return;
    }
    if(!this->individualScheduler ||
       !this->individualScheduler->initialized())
    {
        trace_decision("scheduler_uninitialized", false, 1.0, false);
        return;
    }
    if(this->individualEventInProgress)
        throw std::logic_error(
            "Pre-first-event rebalance cannot run during an event");
    if(!options.enabled)
    {
        trace_decision("auto_rebalance_disabled", false, 1.0, false);
        return;
    }

    std::shared_ptr<PhysicsStep> const balance_step =
        this->findIndividualBalanceStep();
    unsigned int support_mask =
        1u << static_cast<unsigned int>(balance_step ? 1 : 0);
    MPI_Allreduce(MPI_IN_PLACE, &support_mask, 1, MPI_UNSIGNED, MPI_BOR,
                  MPI_COMM_WORLD);
    if((support_mask & (support_mask - 1u)) != 0u)
        throw std::logic_error(
            "Pre-first-event rebalance support differs across MPI ranks");
    if(!balance_step)
    {
        trace_decision("unsupported", false, 1.0, false);
        return;
    }

    unsigned long long local_owned_cells =
        static_cast<unsigned long long>(this->cells.size());
    unsigned long long total_owned_cells = local_owned_cells;
    unsigned long long maximum_owned_cells = local_owned_cells;
    MPI_Allreduce(MPI_IN_PLACE, &total_owned_cells, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maximum_owned_cells, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    double const mean_owned_cells = static_cast<double>(total_owned_cells) /
        static_cast<double>(this->size);
    double const owned_cell_skew = mean_owned_cells > 0 ?
        static_cast<double>(maximum_owned_cells) / mean_owned_cells : 1.0;
    bool const requested = owned_cell_skew > options.threshold;
    trace_decision(requested ? "requested" : "balanced", true,
                   owned_cell_skew, requested);
    if(!requested)
        return;

    IndividualRebalanceResult const result =
        this->rebalanceCommittedIndividualState(
            balance_step, false, options.threshold);
    this->lastPhysicsTimes["individual-pre-first-event-load-balance"] =
        result.maximumSeconds;
    this->lastLocalPhysicsTimes["individual-pre-first-event-load-balance"] =
        result.localSeconds;
    if(this->rank == 0)
        std::clog << std::setprecision(17)
                  << "INDIVIDUAL_PRE_FIRST_EVENT_REBALANCE_RESULT"
                  << " cycle=" << this->tracker.getCycle()
                  << " applied=" << (result.applied ? 1 : 0)
                  << " seconds_max=" << result.maximumSeconds
                  << " weight_max_mean=" << result.weightSkew
                  << " migrated_cells=" << result.migratedCells
                  << " total_owned_cells=" << result.totalOwnedCells
                  << " ownership_epoch=" << this->individualOwnershipEpoch
                  << " current_rss_before_kib="
                  << result.currentRssBeforeKiB
                  << " current_rss_after_kib="
                  << result.currentRssAfterKiB
                  << " peak_rss_kib=" << peakResidentSetKiB()
                  << std::endl;
}
#endif

void Simulation::stepIndividual(void)
{
    if(!this->individualScheduler)
        throw std::logic_error("Individual timestep mode has no scheduler");
    if(!this->individualScheduler->initialized())
    {
        for(const std::shared_ptr<PhysicsStep> &physicsStep : this->physics)
            if(!physicsStep->supportsIndividualTimeSteps())
                throw std::invalid_argument("Physics step '" + physicsStep->getName() +
                                            "' does not support individual timesteps");
#ifdef RICH_MPI
        // Canonical scheduler state is owned-cell only.  Ghost primitives are
        // reconstructed collectively by each physics step from the current
        // tessellation exchange map.
        this->cells.resize(this->tess.GetPointNo());
        this->extensives.resize(this->tess.GetPointNo());
        double initial_dt_min = this->tsc->GetTimeStep();
        double initial_dt_max = initial_dt_min;
        MPI_Allreduce(MPI_IN_PLACE, &initial_dt_min, 1, MPI_DOUBLE,
                      MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &initial_dt_max, 1, MPI_DOUBLE,
                      MPI_MAX, MPI_COMM_WORLD);
        if(initial_dt_min != initial_dt_max)
            throw std::logic_error(
                "MPI individual timestep initialization requires one shared base timestep");
        this->tsc->SetTimeStep(initial_dt_min);
#endif
        this->individualScheduler->initialize(this->cells,
                                              this->tracker.getTime(),
                                              this->tsc->GetTimeStep());
    }

    this->lastPhysicsTimes.clear();
    this->lastLocalPhysicsTimes.clear();
    const auto stepWallStart = std::chrono::high_resolution_clock::now();
#ifdef RICH_MPI
    // Restart state is already committed.  Any ownership migration must
    // finish before prepareEvent creates an in-flight event context.
    this->maybeRebalanceBeforeFirstIndividualEvent();
#endif

    this->individualEventInProgress = true;
    struct EventProgressReset
    {
        bool& flag;
        ~EventProgressReset() { flag = false; }
    } event_progress_reset{this->individualEventInProgress};

    std::uint64_t event_tick = this->individualScheduler->nextEventTick();
    bool force_all_active_latched =
        this->individualScheduler->forceAllActiveLatched();
    bool force_all_active_once = this->individualSynchronizedEventRequested;
#ifdef RICH_MPI
    std::uint64_t const local_current_tick =
        this->individualScheduler->currentTick();
    std::uint64_t event_collective[7] = {
        event_tick, force_all_active_latched ? 1u : 0u,
        force_all_active_latched ? 0u : 1u, local_current_tick,
        std::numeric_limits<std::uint64_t>::max() - local_current_tick,
        force_all_active_once ? 1u : 0u,
        force_all_active_once ? 0u : 1u};
    MPI_Allreduce(MPI_IN_PLACE, event_collective, 7, MPI_UINT64_T,
                  MPI_MIN, MPI_COMM_WORLD);
    event_tick = event_collective[0];
    if(event_collective[1] == 0 && event_collective[2] == 0)
        throw std::logic_error(
            "Inconsistent force-all-active latch state across MPI ranks");
    std::uint64_t const maximum_current_tick =
        std::numeric_limits<std::uint64_t>::max() - event_collective[4];
    if(event_collective[3] != maximum_current_tick)
        throw std::logic_error(
            "Inconsistent individual scheduler clock across MPI ranks");
    if(event_collective[5] == 0 && event_collective[6] == 0)
        throw std::logic_error(
            "Inconsistent synchronized-event request across MPI ranks");
    force_all_active_latched = event_collective[1] != 0;
    force_all_active_once = event_collective[5] != 0;
#endif
    if(event_tick == std::numeric_limits<std::uint64_t>::max())
        throw std::logic_error("Individual timestep scheduler has no cells on any rank");
    IndividualStepContext context =
        this->individualScheduler->prepareEvent(this->cells, event_tick);
#ifndef RICH_MPI
    if((force_all_active_latched || force_all_active_once) &&
       context.active_indices.size() != this->cells.size())
        context = this->individualScheduler->prepareEvent(
            this->cells, event_tick, true);
#endif
#ifdef RICH_MPI
    IndividualForceAllActiveRuntimeOptions const& force_options =
        individualForceAllActiveRuntimeOptions();
    if(force_options.enabled || force_all_active_latched ||
       force_all_active_once)
    {
        std::uint64_t closure_counts[2] = {
            static_cast<std::uint64_t>(this->cells.size()),
            static_cast<std::uint64_t>(context.active_indices.size())};
        std::uint64_t maximum_active_interval_ticks = 0;
        std::vector<CellTimeState> const& states =
            this->individualScheduler->states();
        for(std::size_t const active : context.active_indices)
        {
            CellTimeState const& state = states.at(active);
            if(state.begin_tick >= event_tick)
                throw std::logic_error(
                    "Active individual cell has a non-advancing event interval");
            maximum_active_interval_ticks = std::max(
                maximum_active_interval_ticks,
                event_tick - state.begin_tick);
        }
        MPI_Allreduce(MPI_IN_PLACE, closure_counts, 2, MPI_UINT64_T, MPI_SUM,
                      MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &maximum_active_interval_ticks, 1,
                      MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
        std::uint64_t const force_threshold_ticks =
            force_options.enabled ?
            (std::uint64_t(1) << force_options.minimum_bin) : 0;
        bool const threshold_reached = force_options.enabled &&
            maximum_active_interval_ticks >= force_threshold_ticks;
        bool const latch_now = force_all_active_latched ||
            (force_options.latch_enabled && threshold_reached);
        if(latch_now && !force_all_active_latched)
        {
            this->individualScheduler->setForceAllActiveLatched(true);
            auto const release_start =
                std::chrono::high_resolution_clock::now();
            for(std::shared_ptr<PhysicsStep> const& physicsStep : this->physics)
                physicsStep->onIndividualForceAllActiveLatch();
            double release_seconds = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() -
                release_start).count();
            MPI_Allreduce(MPI_IN_PLACE, &release_seconds, 1, MPI_DOUBLE,
                          MPI_MAX, MPI_COMM_WORLD);
            if(this->rank == 0)
                std::clog << std::setprecision(17)
                          << "INDIVIDUAL_FORCE_ALL_ACTIVE_LATCH_SET"
                          << " cycle=" << this->tracker.getCycle()
                          << " previous_event_tick="
                          << this->individualScheduler->currentTick()
                          << " event_tick=" << event_tick
                          << " maximum_active_interval_ticks="
                          << maximum_active_interval_ticks
                          << " maximum_active_interval_dt="
                          << (this->individualScheduler->timeQuantum() *
                              static_cast<double>(
                                  maximum_active_interval_ticks))
                          << " minimum_bin=" << force_options.minimum_bin
                          << " threshold_ticks=" << force_threshold_ticks
                          << " global_active=" << closure_counts[1]
                          << " global_owned_cells=" << closure_counts[0]
                          << " topology_release_seconds_max="
                          << release_seconds
                          << std::endl;
        }
        bool const force_all_active = closure_counts[1] < closure_counts[0] &&
            (latch_now || threshold_reached || force_all_active_once);
        if(force_all_active)
        {
            std::uint64_t const global_promoted_cells =
                closure_counts[0] - closure_counts[1];
            context = this->individualScheduler->prepareEvent(
                this->cells, event_tick, true);
            if(this->rank == 0)
                std::clog << std::setprecision(17)
                          << "INDIVIDUAL_FORCE_ALL_ACTIVE"
                          << " reason=" << (force_all_active_once ?
                              "requested" : (latch_now ? "latched" : "threshold"))
                          << " latched=" << (latch_now ? 1 : 0)
                          << " cycle=" << this->tracker.getCycle()
                          << " previous_event_tick="
                          << this->individualScheduler->currentTick()
                          << " event_tick=" << event_tick
                          << " event_interval_ticks="
                          << (event_tick -
                              this->individualScheduler->currentTick())
                          << " maximum_active_interval_ticks="
                          << maximum_active_interval_ticks
                          << " maximum_active_interval_dt="
                          << (this->individualScheduler->timeQuantum() *
                              static_cast<double>(
                                  maximum_active_interval_ticks))
                          << " minimum_bin=" << force_options.minimum_bin
                          << " threshold_enabled="
                          << (force_options.enabled ? 1 : 0)
                          << " threshold_ticks=" << force_threshold_ticks
                          << " global_active_before=" << closure_counts[1]
                          << " global_owned_cells=" << closure_counts[0]
                          << " global_promoted_cells="
                          << global_promoted_cells << std::endl;
        }
    }
#endif
#ifdef RICH_MPI
    IndividualRebalanceRuntimeOptions const& balance_options =
        individualRebalanceRuntimeOptions();
#endif
    std::vector<double> timeStepLimits(
        this->individualScheduler->states().size(),
                                      std::numeric_limits<double>::infinity());

    if(this->rank == 0)
        std::cout << "\nIndividual cycle " << this->tracker.getCycle()
                  << " from time " << context.previous_event_time
                  << " to " << context.event_time
                  << " with " << context.active_indices.size()
                  << " active cells" << std::endl;

    for(const std::shared_ptr<PhysicsStep> &physicsStep : this->physics)
    {
        const std::string name = physicsStep->getName();
        MEMORY_DEBUG_PRINT("Before individual " + name);
        const auto start = std::chrono::high_resolution_clock::now();
        physicsStep->stepIndividual(context);
        const double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count();
        this->lastPhysicsTimes[name] = elapsed;
        this->lastLocalPhysicsTimes[name] = elapsed;
        for(auto const& counter :
            physicsStep->getIndividualPerformanceCounters())
            this->lastLocalPhysicsTimes[counter.first] = counter.second;
        MEMORY_DEBUG_PRINT("After individual " + name);
        if(this->rank == 0)
            std::cout << "Individual physics " << name << " time: "
                      << elapsed << std::endl;
    }

    if(this->individualPostPhysics)
    {
        const auto start = std::chrono::high_resolution_clock::now();
        this->individualPostPhysics(context);
        const double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count();
        this->lastPhysicsTimes["individual-post-physics"] = elapsed;
        this->lastLocalPhysicsTimes["individual-post-physics"] = elapsed;
    }

    for(const std::shared_ptr<PhysicsStep> &physicsStep : this->physics)
    {
        const std::string name = physicsStep->getName();
        const auto start = std::chrono::high_resolution_clock::now();
        physicsStep->suggestIndividualTimeSteps(context, timeStepLimits);
        const double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count();
        this->lastPhysicsTimes[name] += elapsed;
        this->lastLocalPhysicsTimes[name] += elapsed;
    }

    this->individualScheduler->commitEvent(context, this->tess,
                                           this->cells, timeStepLimits);
    this->individualSynchronizedEventRequested = false;
    this->tracker.time = context.event_time;
    bool individual_amr_applied = false;
    if(this->individualAMR)
    {
        const auto amr_start = std::chrono::high_resolution_clock::now();
        IndividualAMRChangeSet const changes = this->individualAMR(context);
        this->individualScheduler->applyAMRChangeSet(this->cells, changes,
                                                      &this->tess);
        individual_amr_applied = !changes.empty();
        const double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - amr_start).count();
        this->lastPhysicsTimes["individual-amr"] = elapsed;
        this->lastLocalPhysicsTimes["individual-amr"] = elapsed;
        if(this->rank == 0)
            std::cout << "Individual AMR time: " << elapsed << std::endl;
    }
#ifdef RICH_MPI
    // AMR change sets are rank-local, but the rebalance decision below enters
    // collectives.  Make the "AMR happened" predicate collective first so a
    // rank with no local changes cannot skip a rebalance requested by another
    // rank and deadlock the job.
    int any_individual_amr_applied = individual_amr_applied ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &any_individual_amr_applied, 1, MPI_INT,
                  MPI_MAX, MPI_COMM_WORLD);
    individual_amr_applied = any_individual_amr_applied != 0;

#endif

    if(individual_amr_applied)
    {
        auto const topology_release_start =
            std::chrono::high_resolution_clock::now();
        for(const std::shared_ptr<PhysicsStep>& physicsStep : this->physics)
            physicsStep->afterIndividualAMR();
        double const local_topology_release_seconds =
            std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() -
                topology_release_start).count();
        double maximum_topology_release_seconds =
            local_topology_release_seconds;
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &maximum_topology_release_seconds, 1,
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
        this->lastLocalPhysicsTimes["individual-topology-release"] =
            local_topology_release_seconds;
        this->lastPhysicsTimes["individual-topology-release"] =
            maximum_topology_release_seconds;
    }

#ifdef RICH_MPI
    bool const forced_balance = this->forceRebalanceSteps > 0 &&
        this->tracker.getCycle() < this->forceRebalanceSteps &&
        this->lastRebalanceCycle != this->tracker.getCycle();
    unsigned long long local_owned_cells =
        static_cast<unsigned long long>(this->cells.size());
    unsigned long long total_owned_cells = local_owned_cells;
    unsigned long long maximum_owned_cells = local_owned_cells;
    MPI_Allreduce(MPI_IN_PLACE, &total_owned_cells, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &maximum_owned_cells, 1,
                  MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    double const mean_owned_cells = static_cast<double>(total_owned_cells) /
        static_cast<double>(this->size);
    double const owned_cell_skew = mean_owned_cells > 0 ?
        static_cast<double>(maximum_owned_cells) / mean_owned_cells : 1.0;

    double local_event_seconds = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - stepWallStart).count();
    double maximum_event_seconds = local_event_seconds;
    double total_event_seconds = local_event_seconds;
    MPI_Allreduce(MPI_IN_PLACE, &maximum_event_seconds, 1, MPI_DOUBLE,
                  MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &total_event_seconds, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);
    double const mean_event_seconds = total_event_seconds /
        static_cast<double>(this->size);
    double const predicted_horizon_savings =
        static_cast<double>(balance_options.cooldown_events) *
        std::max(0.0, maximum_event_seconds - mean_event_seconds);
    bool const amortized = this->lastIndividualRebalanceSeconds <= 0 ||
        predicted_horizon_savings > balance_options.amortization_factor *
            this->lastIndividualRebalanceSeconds;
    bool const cooldown_complete =
        this->lastIndividualBalanceCheckCycle ==
            std::numeric_limits<size_t>::max() ||
        (this->tracker.getCycle() >= this->lastIndividualBalanceCheckCycle &&
         this->tracker.getCycle() - this->lastIndividualBalanceCheckCycle >=
             balance_options.cooldown_events);
    bool const immediate_after_amr = individual_amr_applied &&
        owned_cell_skew > balance_options.immediate_amr_threshold;
    bool const automatic_balance = balance_options.enabled &&
        owned_cell_skew > balance_options.threshold &&
        (immediate_after_amr || (cooldown_complete && amortized));
    bool const request_balance = forced_balance || automatic_balance;

    if(balance_options.trace && this->rank == 0)
        std::clog << std::setprecision(17)
                  << "INDIVIDUAL_LOAD_BALANCE_DECISION"
                  << " cycle=" << this->tracker.getCycle()
                  << " enabled=" << (balance_options.enabled ? 1 : 0)
                  << " forced=" << (forced_balance ? 1 : 0)
                  << " amr=" << (individual_amr_applied ? 1 : 0)
                  << " owned_max_mean=" << owned_cell_skew
                  << " threshold=" << balance_options.threshold
                  << " cooldown_complete=" << (cooldown_complete ? 1 : 0)
                  << " predicted_savings_seconds="
                  << predicted_horizon_savings
                  << " previous_migration_seconds="
                  << this->lastIndividualRebalanceSeconds
                  << " requested=" << (request_balance ? 1 : 0)
                  << " ownership_epoch=" << this->individualOwnershipEpoch
                  << std::endl;

    if(request_balance)
    {
        std::shared_ptr<PhysicsStep> const balance_step =
            this->findIndividualBalanceStep();
        if(!balance_step)
            throw std::logic_error(
                "Individual load balancing has no supporting physics step");
        IndividualRebalanceResult const result =
            this->rebalanceCommittedIndividualState(
                balance_step, forced_balance, balance_options.threshold);
        this->lastPhysicsTimes["individual-load-balance"] =
            result.maximumSeconds;
        this->lastLocalPhysicsTimes["individual-load-balance"] =
            result.localSeconds;
        if(this->rank == 0)
            std::cout << "Individual load balance time: "
                      << result.maximumSeconds
                      << " s, applied=" << (result.applied ? 1 : 0)
                      << ", weight max/mean=" << result.weightSkew
                      << ", migrated cells=" << result.migratedCells
                      << ", ownership epoch="
                      << this->individualOwnershipEpoch << std::endl;
    }

    if(balance_options.trace) {
        double const traced_step_seconds = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - stepWallStart).count();
        for(auto const& phase : this->lastLocalPhysicsTimes)
            reportIndividualRankDistribution(
                this->rank, this->size, this->tracker.getCycle(), phase.first,
                phase.second, "seconds");
        reportIndividualRankDistribution(
            this->rank, this->size, this->tracker.getCycle(), "event-wall",
            traced_step_seconds, "seconds");
        reportIndividualRankDistribution(
            this->rank, this->size, this->tracker.getCycle(), "current-rss",
            currentResidentSetKiB(), "KiB");
        reportIndividualRankDistribution(
            this->rank, this->size, this->tracker.getCycle(), "peak-rss",
            peakResidentSetKiB(), "KiB");
    }
#endif
    this->tracker.updateCycle();
    std::uint64_t next_tick = this->individualScheduler->nextEventTick();
#ifdef RICH_MPI
    MPI_Allreduce(MPI_IN_PLACE, &next_tick, 1, MPI_UINT64_T,
                  MPI_MIN, MPI_COMM_WORLD);
#endif
    const double next_dt = next_tick == std::numeric_limits<std::uint64_t>::max()
        ? std::numeric_limits<double>::infinity()
        : this->individualScheduler->timeQuantum() *
          static_cast<double>(next_tick - this->individualScheduler->currentTick());
    this->tsc->SetTimeStep(next_dt);
    this->wallclockTime += std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - stepWallStart).count();
}

#ifdef RICH_MPI
void Simulation::storeLoadBalance(const std::string &name, std::shared_ptr<LoadBalancer<Vector3D>> lb)
{
    this->loads[name] = lb;
}

void Simulation::PresetLoadBalance(const std::string &name)
{
    this->currentLoad = this->tess.GetLoadBalancer();
    this->loads[name] = this->currentLoad;
    this->currentLB = name;
}

void Simulation::setCurrentLoadBalance(const std::string &name)
{
    if(name.empty())
    {
        return;
    }
    auto it = this->loads.find(name);
    if(it == this->loads.end())
    {
        UniversalError eo("setCurrentLoadBalance: unknown load balance");
        eo.addEntry("Name", name);
        throw eo;
    }

    size_t Ntotal = this->tess.GetPointNo();
    #ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &Ntotal, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    #endif // RICH_MPI

    if(Ntotal == 0)
    {
        this->tess.PresetLoadBalancer(it->second);
    }
    else
    {
        this->tess.SetLoadBalancer(it->second);
        this->buildDataTransfer();
    }

    std::shared_ptr<LoadBalancer<Vector3D>> load = this->tess.GetLoadBalancer();
    this->loads[name] = load;
    this->currentLoad = load;
    this->currentLB = name;

    if(this->rank == 0)
    {
        std::cout << "Changed load balance" << std::endl;
        //this->currentLoad->printInfo();
    }
}

std::vector<std::pair<std::string, std::shared_ptr<LoadBalancer<Vector3D>>>> Simulation::GetLoads(void) const
{
    return std::vector<std::pair<std::string, std::shared_ptr<LoadBalancer<Vector3D>>>>(this->loads.begin(), this->loads.end());
}
#endif // RICH_MPI
