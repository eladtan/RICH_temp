#ifndef NO_MONTECARLO_PHYSICS_HPP
#define NO_MONTECARLO_PHYSICS_HPP

#include "MonteCarloPhysics.hpp"

template<typename T, typename Grid>
class NoMonteCarloPhysics : public MonteCarloPhysics<T, Grid>
{
public:
    using MCParticle = MonteCarloParticle<T, Grid>;
    using Functionality = MonteCarloFunctionality<T, Grid>;

    NoMonteCarloPhysics(Grid &grid, const std::shared_ptr<BoundaryCondition<T, Grid>> &boundary);

    std::vector<MCParticle> preStep(double fullDt) override;

    Functionality step(MCParticle &particle) override;

    void postStep(const std::vector<MCParticle> &particles) override;
};

template<typename T, typename Grid>
NoMonteCarloPhysics<T, Grid>::NoMonteCarloPhysics(Grid &grid, const std::shared_ptr<BoundaryCondition<T, Grid>> &boundary):
    MonteCarloPhysics<T, Grid>(grid, boundary)
{}

template<typename T, typename Grid>
typename NoMonteCarloPhysics<T, Grid>::Functionality NoMonteCarloPhysics<T, Grid>::step(MCParticle &particle)
{
    Functionality functionality;

    auto [faceIntersect, timeIntersect, nextCellIndex] = this->getIntersectionDetails(particle);
    assert(timeIntersect > 0);

    dt_t timeLeft = particle.timeLeft;
    dt_t dt = std::numeric_limits<dt_t>::infinity();

    // std::cout << "particle " << particle.id << ", time left is " << timeLeft << ", time intersect is " << timeIntersect << " (face " << faceIntersect << ")" << std::endl;
    if(timeLeft < timeIntersect)
    {
        functionality.change = MonteCarloParticleStatus::DONE;
        dt = timeLeft;
    }
    else
    {
        functionality.change = MonteCarloParticleStatus::CELL_MOVE;
        functionality.nextCellIndex = nextCellIndex;
        dt = timeIntersect;
    }

    particle.timeLeft -= dt;
    particle.location += particle.velocity * dt;
    return functionality;
}

template<typename T, typename Grid>
std::vector<MonteCarloParticle<T, Grid>> NoMonteCarloPhysics<T, Grid>::preStep(double fullDt)
{
    return std::vector<MonteCarloParticle<T, Grid>>();
}

template<typename T, typename Grid>
void NoMonteCarloPhysics<T, Grid>::postStep(const std::vector<MCParticle> &particles)
{}

#endif // NO_MONTECARLO_PHYSICS_HPP