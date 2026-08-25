#ifndef PHYSICS_STEP_HPP
#define PHYSICS_STEP_HPP

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "mpi/ExchangeChain.hpp"
#include "newtonian/three_dimensional/simulation/IndividualTimeStep.hpp"

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

    virtual bool getIndividualGeneratorPoints(
        std::vector<Vector3D>&) const
    {
        return false;
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
};

#endif// PHYSICS_STEP_HPP
