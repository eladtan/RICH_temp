#ifndef PHYSICS_STEP_HPP
#define PHYSICS_STEP_HPP

#include <string>
#include <vector>
#include "mpi/ExchangeChain.hpp"

class PhysicsStep
{
public:
    virtual ~PhysicsStep(){};

    virtual void step(double) = 0;

    virtual double suggestTimeStep(void) const = 0;

    virtual std::string getName(void) const = 0;

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
