#ifndef BOUNDARY_CONDITION_HPP
#define BOUNDARY_CONDITION_HPP

#include "monte/MonteCarloParticleStatus.hpp"

template<typename Grid>
class BoundaryCondition
{
public:
    BoundaryCondition(const Grid &grid);

    virtual ~BoundaryCondition() = default;

    virtual MonteCarloParticleStatus reachedBoundary(MonteCarloParticle<Grid> &particle) = 0;
};

template<typename Grid>
BoundaryCondition<Grid>::BoundaryCondition(const Grid &grid)
    : grid(grid)
{}

#endif // BOUNDARY_CONDITION_HPP