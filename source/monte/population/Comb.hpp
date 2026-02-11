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

    #ifdef RICH_MPI
        rank_t rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::mt19937_64 gen((873 * rank) + particles.size());
    #else // RICH_MPI
        std::mt19937_64 gen(particles.size());
    #endif // RICH_MPI

    std::vector<MCParticle> result;

    size_t Ncells = this->grid.GetPointNo();
    std::vector<double> weights(Ncells, 0);
    std::vector<std::vector<const MCParticle*>> particlesInCells(Ncells);

    #ifdef MONTECARLO_DEBUG
    // check no duplications
    boost::container::flat_map<std::pair<rank_t, size_t>, size_t> particlesMap;
    for(size_t i = 0; i < particles.size(); i++)
    {
        const MCParticle &particle = particles[i];
        if(particle.id == std::numeric_limits<size_t>::max())
        {
            continue;
        }
        std::pair<rank_t, size_t> particleKey = {particle.rank, particle.id};
        if(particlesMap.find(particleKey) != particlesMap.end())
        {
            UniversalError eo("Comb Population Control: Particle with the same ID is being added to the same rank twice");
            eo.addEntry("Particle", particle);
            #ifdef RICH_MPI
                eo.addEntry("Rank", rank);
            #endif // RICH_MPI
            eo.addEntry("Index 1", particlesMap[particleKey]);
            eo.addEntry("Index 2", i);
            throw eo;
        }
        particlesMap[particleKey] = i;
    }
    #endif // MONTECARLO_DEBUG

    for(const MCParticle &particle : particles)
    {
        assert(particle.cellIndex < Ncells);
        weights[particle.cellIndex] += particle.weight;
        particlesInCells[particle.cellIndex].push_back(&particle);
    }

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

        std::sort(particlesInCells[i].begin(), particlesInCells[i].end(), [](const MCParticle *p1, const MCParticle *p2){return p1->rank < p2->rank or (p1->rank == p2->rank and p1->id < p2->id);});

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
                result.push_back(*particle);
                result.back().id = std::numeric_limits<size_t>::max(); // reset id, it will be set later
                #ifdef RICH_MPI
                result.back().rank = std::numeric_limits<rank_t>::max(); // reset rank, it will be set later
                #endif // RICH_MPI
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