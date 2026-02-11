#ifndef MONTE_CARLO_FUNCTIONALITY_HPP
#define MONTE_CARLO_FUNCTIONALITY_HPP

#include "MonteCarloParticleStatus.hpp"

template<typename T, typename Grid>
struct MonteCarloFunctionality
{
public:
    using MCParticle = MonteCarloParticle<T, Grid>;

    MonteCarloParticleStatus change = MonteCarloParticleStatus::NO_CELL_MOVE;
    size_t nextCellIndex = std::numeric_limits<size_t>::max();
    std::vector<MCParticle> particlesToAdd;
};

#endif // MONTE_CARLO_FUNCTIONALITY_HPP