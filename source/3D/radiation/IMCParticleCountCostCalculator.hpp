#ifndef IMC_PARTICLE_COUNT_COST_CALCULATOR_HPP
#define IMC_PARTICLE_COUNT_COST_CALCULATOR_HPP

#ifdef RICH_MPI

#include "newtonian/three_dimensional/CostCalculator3D.hpp"
#include "3D/monte/MonteCarloManager3D.hpp"
#include <algorithm>

class IMCParticleCountCostCalculator : public CostCalculator3D
{
public:
    explicit IMCParticleCountCostCalculator(const std::shared_ptr<MonteCarloManager3D> &manager)
        : manager(manager)
    {}

    std::vector<double> CalculateCost(const Tessellation3D &tess,
                                      const vector<ComputationalCell3D> &cells) const override
    {
        size_t N = tess.GetPointNo();
        double start = static_cast<double>(manager->GetStartParticleCount());
        double end = static_cast<double>(manager->GetEndParticleCount());
        double avgPerCell = (start + end) / (2.0 * std::max<size_t>(N, 1));
        return std::vector<double>(N, std::max(avgPerCell, 1.0));
    }

private:
    const std::shared_ptr<MonteCarloManager3D> manager;
};

#endif // RICH_MPI

#endif // IMC_PARTICLE_COUNT_COST_CALCULATOR_HPP
