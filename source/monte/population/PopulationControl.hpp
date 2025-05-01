#ifndef POPULATION_CONTROL_HPP
#define POPULATION_CONTROL_HPP

#include "monte/MonteCarloParticle.hpp"

template<typename T, typename Grid>
class PopulationControl
{
public:
    using MCParticle = MonteCarloParticle<T, Grid>;

    PopulationControl(const Grid &grid);

    virtual ~PopulationControl() = default;

    virtual std::vector<MCParticle> activate(const std::vector<MCParticle> &particles) = 0;

protected:
    const Grid &grid;
};

template<typename T, typename Grid>
PopulationControl<T, Grid>::PopulationControl(const Grid &grid)
    : grid(grid)
{}


#endif // POPULATION_CONTROL_HPP