#ifndef PHYSICS_STEP_HPP
#define PHYSICS_STEP_HPP

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "mpi/ExchangeChain.hpp"
#include "newtonian/three_dimensional/simulation/IndividualTimeStep.hpp"
#include "newtonian/three_dimensional/simulation/StepDiagnostics.hpp"

class PhysicsStep
{
public:
    virtual ~PhysicsStep(){};

    virtual void step(double) = 0;

    virtual double suggestTimeStep(void) const = 0;

    virtual bool supportsIndividualTimeSteps(void) const
    {
        return false;
    }

    virtual std::string individualTimeStepUnsupportedReason(void) const
    {
        return std::string();
    }

    virtual void stepIndividual(const IndividualStepContext &)
    {
        throw std::runtime_error("Physics step does not support individual timesteps");
    }

    virtual void suggestIndividualTimeSteps(const IndividualStepContext &context,
                                            std::vector<double> &time_step_limits) const
    {
        const double limit = suggestTimeStep();
        for(std::size_t index : context.active_indices)
            time_step_limits.at(index) = std::min(time_step_limits.at(index), limit);
    }

    virtual void suggestIndividualWakeDeadlines(
        const IndividualStepContext &,
        std::vector<double> &) const
    {}

    // Conserved-change wakes: for each passive cell that must end its
    // interval at the next scheduled event, raise change_ratios[index] to the
    // measured change ratio (entries stay 0 otherwise).
    virtual void suggestIndividualChangeWakes(
        const IndividualStepContext &,
        std::vector<double> &) const
    {}

    // The scheduler has just been initialized for an individual phase whose
    // preceding global steps took global_time_step each.
    virtual void onIndividualSchedulerStart(double /*global_time_step*/)
    {}

    virtual bool getIndividualGeneratorPoints(
        std::vector<Vector3D>&) const
    {
        return false;
    }

    virtual bool getIndividualCellCentroids(
        std::vector<Vector3D>&) const
    {
        return false;
    }

    virtual bool contributesIndividualHydrodynamicSignal(void) const
    {
        return false;
    }

    // Cadence diagnostic: per canonical cell, which of this step's own limits
    // set its latest timestep suggestion (step-defined codes, 0 = none).
    virtual bool getIndividualLimitReasons(
        std::vector<unsigned char>&) const
    {
        return false;
    }

    // Per-cell timestep limits of the current full mesh, one per owned cell,
    // for judging the potential gain of individual timesteps while stepping
    // globally.  A step that only knows a global limit returns false and is
    // applied as a cap through suggestTimeStep() instead.
    virtual bool collectCellTimeStepLimits(std::vector<double>&) const
    {
        return false;
    }

    // Per-cell stability limits of a synchronized individual state whose mesh
    // changed outside an event (a box growth), before it advances: one per
    // owned cell, in mesh order, with `point_velocities` the velocities each
    // generator moves with through its next interval.  A step that keeps the
    // individual acceleration cache refreshes it into `accelerations` (one
    // per owned cell; empty otherwise).  Collective under MPI for a step that
    // overrides it; the default supplies none.
    virtual bool synchronizedCellTimeStepLimits(
        std::vector<Vector3D> const& /*point_velocities*/,
        std::vector<double>& /*limits*/,
        std::vector<Vector3D>& /*accelerations*/) const
    {
        return false;
    }

    // The individual acceleration of every owned cell on the current full
    // mesh (owned cells in mesh order), for refreshing the acceleration cache
    // after a topology change.  Collective under MPI for a step that
    // overrides it; the default supplies none.
    virtual bool refreshIndividualAccelerations(
        std::vector<Vector3D>& /*accelerations*/) const
    {
        return false;
    }

    // True when the limits are the step's own evaluation carried to the
    // current cells by ID (radiation), not evaluated on the current state.
    virtual bool cellTimeStepLimitsCached(void) const
    {
        return false;
    }

    // True when the step can supply per-cell limits but its cached inputs no
    // longer belong to the current mesh (it was rebuilt after they were set),
    // so collectCellTimeStepLimits fails.  The gain bound is then skipped and
    // retried, not formed with this step as a uniform cap.
    virtual bool cellTimeStepLimitsStale(void) const
    {
        return false;
    }

    // Cells the last collectCellTimeStepLimits call could not resolve and
    // left without a limit.
    virtual std::size_t cellTimeStepLimitFallbacks(void) const
    {
        return 0;
    }

    // Smallest limit of the step's own evaluation on this rank, including
    // cells collectCellTimeStepLimits could not resolve here (a cached step's
    // migrated cells), so that the reduced minimum is the step's global one.
    virtual double cellTimeStepLimitMinimum(void) const
    {
        return std::numeric_limits<double>::infinity();
    }


    // Invalidate topology-dependent state after a committed AMR change.
    virtual void afterIndividualAMR(void)
    {}

    // Drop storage that belongs only to the pre-latch partial-active layout.
    // The all-active latch is permanent, so this storage cannot be reused.
    virtual void onIndividualForceAllActiveLatch(void) noexcept
    {}

    // Drop topology-sized scratch storage before a committed ownership change.
    // This is distinct from beforeLB(): every physics package sees this hook,
    // while only the selected load-balancing package owns beforeLB()/afterLB().
    virtual void beforeIndividualRebalance(void) noexcept
    {}

    virtual std::string getName(void) const = 0;

    virtual std::map<std::string, double>
    getIndividualPerformanceCounters(void) const
    {
        return {};
    }

    virtual SourceStepTiming getSourceStepTiming(void) const
    {
        return SourceStepTiming();
    }

    virtual MeshBuildTiming getMeshBuildTiming(void) const
    {
        return MeshBuildTiming();
    }

    // Under MPI, a retry is collective: every rank must report each rejection
    // in the same order and count because the installed reporter reduces it.
    // Rank-local diagnostics must not use this interface.
    void setStepRetryReporter(
        std::function<void(StepRetryRecord const&)> reporter)
    {
        retry_reporter_ = std::move(reporter);
    }

#ifdef RICH_MPI
    virtual bool allowRebalance(void) = 0;

    virtual std::string getRequiredLB(void) const = 0;

    virtual std::vector<double> getLoadBalanceWeights(void) = 0;

    virtual void beforeLB(void)
    {}

    virtual void afterLB(void)
    {}

    // a physics is required to exchange points, as long it loggs the changes in an 'ExchangeChain'
    virtual ExchangeChain GetExchangeChain(void)
    {
        return ExchangeChain();
    }

    virtual void dumpCost(size_t /*cycle*/) const {}
#endif // RICH_MPI

protected:
    void reportStepRetry(StepRetryRecord const& retry) const
    {
        if(retry_reporter_)
            retry_reporter_(retry);
    }

private:
    std::function<void(StepRetryRecord const&)> retry_reporter_;
};

#endif// PHYSICS_STEP_HPP
