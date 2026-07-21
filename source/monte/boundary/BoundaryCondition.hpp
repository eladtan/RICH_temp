#ifndef BOUNDARY_CONDITION_HPP
#define BOUNDARY_CONDITION_HPP

#include "monte/MonteCarloParticle.hpp"
#include "monte/MonteCarloParticleStatus.hpp"

template<typename T, typename Grid>
class BoundaryCondition
{
public:
    BoundaryCondition(const Grid &grid);

    virtual ~BoundaryCondition() = default;

    virtual MonteCarloParticleStatus apply(MonteCarloParticle<T, Grid> &particle) = 0;

    virtual std::vector<MonteCarloParticle<T, Grid>> generateNewBoundaryParticles(double fullDt) = 0;

protected:
    const Grid &grid;
};

template<typename T, typename Grid>
BoundaryCondition<T, Grid>::BoundaryCondition(const Grid &grid)
    : grid(grid)
{}

#endif // BOUNDARY_CONDITION_HPP