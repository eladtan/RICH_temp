#ifndef COMB_POPULATION_CONTROL_HPP
#define COMB_POPULATION_CONTROL_HPP

#include <random>
#include "PopulationControl.hpp"

template<typename T, typename Grid>
class CombPopulationControl : public PopulationControl<T, Grid>
{
public:
    CombPopulationControl(const Grid &grid, size_t n);

    std::vector<MonteCarloParticle> activate(const std::vector<MonteCarloParticle> &particles) = 0;

private:
    size_t n;
    boost::random::mt19937_64 gen;
};

template<typename T, typename Grid>
CombPopulationControl<T, Grid>::CombPopulationControl(const Grid &grid, size_t n)
    : PopulationControl<T, Grid>(grid), n(n)
{}

template<typename T, typename Grid>
std::vector<MonteCarloParticle<T, Grid>> activate(const std::vector<MonteCarloParticle<T, Grid>> &particles)
{
    using MCParticle = MonteCarloParticle<T, Grid>;

    std::vector<MCParticle> result;

    size_t Ncells = this->grid.GetPointNo();
    std::vector<double> weights(Ncells);
    std::vector<std::vector<MCParticle*>> particlesInCells(Ncells);

    for(const MCParticle &particle : particles)
    {
        weights[particle.cellIndex] += particle.energy;
        particlesInCells[particle.cellIndex].push_back(&particle);
    }

    rank_t rank = 0;
    #ifndef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif // RICH_MPI

    std::mt19937_64 gen((873 * rank) + particles.size());
    std::uniform_real_distribution<double> dist(0, 1);

    for(size_t i = 0; i < Ncells; i++)
    {
        std::shuffle(particlesInCells[i].begin(), particlesInCells[i].end(), gen);
    
        double new_energy = weights[i] / this->n;

        double r = dist(gen);
        size_t comb_index = 0;
        double cum_sum_w = 0;
        for(MCParticle *particle : particlesInCells[i])
        {
            while((cum_sum_w + particle->weight) > (comb_index + r) * new_energy)
            {
                ++comb_index;
                result.emplace_back(*particle);
                result.back().id = std::numeric_limits<size_t>::max(); // todo ID
                result.back().weight = new_energy;
            }
            cum_sum_w += particle->weight;
        }
    }
    return result;
}

#endif // COMB_POPULATION_CONTROL_HPP