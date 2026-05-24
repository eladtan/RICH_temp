#ifndef COMB_POPULATION_CONTROL_HPP
#define COMB_POPULATION_CONTROL_HPP

#include <cmath>
#include <limits>
#include <random>
#include <boost/random/mersenne_twister.hpp>
#include "PopulationControl.hpp"
#include "misc/universal_error.hpp"

template<typename T, typename Grid>
class CombPopulationControl : public PopulationControl<T, Grid>
{
public:
    CombPopulationControl(const Grid &grid, size_t Nmin = 20, double totalParticlesFactor = 2.0);

    std::vector<MonteCarloParticle<T, Grid>> activate(const std::vector<MonteCarloParticle<T, Grid>> &particles) override;

private:
    size_t Nmin;
    double totalParticlesFactor;
    boost::random::mt19937_64 gen;
};

template<typename T, typename Grid>
CombPopulationControl<T, Grid>::CombPopulationControl(const Grid &grid, size_t Nmin, double totalParticlesFactor)
    : PopulationControl<T, Grid>(grid), Nmin(Nmin), totalParticlesFactor(totalParticlesFactor)
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
    size_t Ntotal = Ncells;
    #ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &Ntotal, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    #endif // RICH_MPI
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

    double totalWeight = 0.0;
    for(const MCParticle &particle : particles)
    {
        if(this->grid.IsPointOutsideBox(particle.location))
        {
            UniversalError eo("Comb Population Control: input particle is outside the box");
            eo.addEntry("Particle", particle);
            eo.addEntry("Cell count", Ncells);
            if(particle.cellIndex < Ncells)
            {
                eo.addEntry("Cell center", this->grid.GetMeshPoint(particle.cellIndex));
                eo.addEntry("Inside declared cell", this->grid.IsPointInCell(particle.location, particle.cellIndex));
            }
            throw eo;
        }
        if(!std::isfinite(particle.weight))
        {
            UniversalError eo("Comb Population Control: particle weight is not finite");
            eo.addEntry("Particle weight", particle.weight);
            eo.addEntry("Particle cell index", particle.cellIndex);
            throw eo;
        }
        if(particle.weight < 0.0)
        {
            UniversalError eo("Comb Population Control: particle weight is negative");
            eo.addEntry("Particle weight", particle.weight);
            eo.addEntry("Particle cell index", particle.cellIndex);
            throw eo;
        }
        if(particle.weight == 0.0)
            continue;
        assert(particle.cellIndex < Ncells);
        weights[particle.cellIndex] += particle.weight;
        totalWeight += particle.weight;
        particlesInCells[particle.cellIndex].push_back(&particle);
    }

    #ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &totalWeight, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    #endif // RICH_MPI
    if(!std::isfinite(totalWeight))
    {
        UniversalError eo("Comb Population Control: total particle weight is not finite");
        eo.addEntry("Total weight", totalWeight);
        eo.addEntry("Local particle count", particles.size());
        throw eo;
    }
    if(totalWeight == 0.0)
        return result;

    Ntotal = static_cast<size_t>(Ntotal * this->Nmin * this->totalParticlesFactor);

    std::uniform_real_distribution<double> dist(0, 1);

    for(size_t i = 0; i < Ncells; i++)
    {
        size_t NinCell = std::min(this->Nmin * 20, std::max(this->Nmin, static_cast<size_t>(Ntotal * weights[i] / totalWeight)));
        if(particlesInCells[i].size() <= NinCell)
        {
            double weight_ideal = weights[i] / NinCell;
            for(const MCParticle *particle : particlesInCells[i])
            {   
                if(particle->weight > 2 * weight_ideal)
                {
                    MCParticle particleCpy = *particle;
                    size_t Nsplit = static_cast<size_t>(std::ceil(particle->weight / weight_ideal));
                    double weight_split = particle->weight / Nsplit;
                    particleCpy.weight = weight_split;
                    particleCpy.id = std::numeric_limits<size_t>::max(); // reset id, it will be set later
                    #ifdef RICH_MPI
                        particleCpy.rank = std::numeric_limits<rank_t>::max(); // reset rank, it will be set later
                    #endif // RICH_MPI
                    particleCpy.steps = 0;
                    // add Nsplit copies
                    for(size_t j = 0; j < Nsplit; j++)
                    {
                        result.push_back(particleCpy);
                    }
                }
                else
                {
                    result.push_back(*particle);
                }
            }
            continue;
        }

        #ifdef RICH_MPI
            std::sort(particlesInCells[i].begin(), particlesInCells[i].end(), [](const MCParticle *p1, const MCParticle *p2){return (p1->rank) < (p2->rank) or ((p1->rank) == (p2->rank) and p1->id < p2->id);});
        #else // RICH_MPI
            std::sort(particlesInCells[i].begin(), particlesInCells[i].end(), [](const MCParticle *p1, const MCParticle *p2){return p1->id < p2->id;});
        #endif // RICH_MPI

        std::shuffle(particlesInCells[i].begin(), particlesInCells[i].end(), gen);
    
        double new_energy = weights[i] / NinCell;

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

    for(auto &p : result)
    {
        if(this->grid.IsPointOutsideBox(p.location))
        {
            const size_t Ncells = this->grid.GetPointNo();
            if(p.cellIndex >= Ncells)
            {
                UniversalError eo("Comb Population Control: outside particle has invalid cell index");
                eo.addEntry("Particle cell index", p.cellIndex);
                eo.addEntry("Cell count", Ncells);
                throw eo;
            }
            const Vector3D original = p.location;
            const Vector3D direction = this->grid.GetMeshPoint(p.cellIndex) - original;
            double t = 1e-6;
            while(this->grid.IsPointOutsideBox(p.location) && t < 1.0)
            {
                p.location = original + t * direction;
                t *= 2;
            }
            if(this->grid.IsPointOutsideBox(p.location))
                p.location = this->grid.GetMeshPoint(p.cellIndex);
        }
    }

    return result;
}

#endif // COMB_POPULATION_CONTROL_HPP
