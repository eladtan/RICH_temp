#ifndef COMB_POPULATION_CONTROL_HPP
#define COMB_POPULATION_CONTROL_HPP

#include <random>
#include <boost/random/mersenne_twister.hpp>
#include "PopulationControl.hpp"

template<typename T, typename Grid>
class CombPopulationControl : public PopulationControl<T, Grid>
{
public:
    CombPopulationControl(const Grid &grid, size_t n);

    std::vector<MonteCarloParticle<T, Grid>> activate(const std::vector<MonteCarloParticle<T, Grid>> &particles) override;

private:
    size_t n;
    boost::random::mt19937_64 gen;
};

template<typename T, typename Grid>
CombPopulationControl<T, Grid>::CombPopulationControl(const Grid &grid, size_t n)
    : PopulationControl<T, Grid>(grid), n(n)
{}

template<typename T, typename Grid>
std::vector<MonteCarloParticle<T, Grid>> CombPopulationControl<T, Grid>::activate(const std::vector<MonteCarloParticle<T, Grid>> &particles)
{
    using MCParticle = MonteCarloParticle<T, Grid>;

    std::vector<MCParticle> result;

    size_t Ncells = this->grid.GetPointNo();
    std::vector<double> weights(Ncells, 0);
    std::vector<std::vector<const MCParticle*>> particlesInCells(Ncells);

    for(const MCParticle &particle : particles)
    {
        assert(particle.cellIndex < Ncells);
        weights[particle.cellIndex] += particle.weight;
        particlesInCells[particle.cellIndex].push_back(&particle);
    }

    #ifdef RICH_MPI
        rank_t rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::mt19937_64 gen((873 * rank) + particles.size());
    #else // RICH_MPI
        std::mt19937_64 gen(particles.size());
    #endif // RICH_MPI

    std::uniform_real_distribution<double> dist(0, 1);

    for(size_t i = 0; i < Ncells; i++)
    {
        if(particlesInCells[i].size() <= this->n)
        {
            for(const MCParticle *particle : particlesInCells[i])
            {
                result.push_back(*particle);
            }
            continue;
        }
        std::shuffle(particlesInCells[i].begin(), particlesInCells[i].end(), gen);
    
        double new_energy = weights[i] / this->n;

        double r = dist(gen);
        size_t comb_index = 0;
        double cum_sum_w = 0;
        for(const MCParticle *particle : particlesInCells[i])
        {
            while((cum_sum_w + particle->weight) > (comb_index + r) * new_energy)
            {
                ++comb_index;
                result.emplace_back(*particle);
                result.back().cellIndex = i;
                result.back().timeLeft = 0;
                result.back().weight = new_energy;
                result.back().initialWeight = new_energy;
            }
            cum_sum_w += particle->weight;
        }
    }
    return result;
}

#endif // COMB_POPULATION_CONTROL_HPP