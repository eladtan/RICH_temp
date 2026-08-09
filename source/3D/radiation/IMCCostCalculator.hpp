#ifndef IMC_COST_CALCULATOR_HPP
#define IMC_COST_CALCULATOR_HPP

#ifdef RICH_MPI

#include "newtonian/three_dimensional/CostCalculator3D.hpp"
#include "3D/monte/MonteCarloManager3D.hpp"
#include <cassert>

class IMCCostCalculator : public CostCalculator3D
{
public:
    explicit IMCCostCalculator(const std::shared_ptr<MonteCarloManager3D> &manager,
                               double counterScale = 0.005,
                               double particleScale = 10.0)
        : manager(manager), counterScale(counterScale), particleScale(particleScale)
    {}

    std::vector<double> CalculateCost(const Tessellation3D &tess,
                                      const vector<ComputationalCell3D> &cells) const override
    {
        size_t N = tess.GetPointNo();
        std::vector<double> weights(N, 1.0);
        const std::vector<size_t> &counters = manager->GetCellsStepsCounters();
        const std::vector<size_t> &particleCounts = manager->GetBeginningParticleCount();
        assert(counters.size() == N);
        assert(particleCounts.size() == N);
        for (size_t j = 0; j < N; j++)
            weights[j] += static_cast<double>(counters[j]) * counterScale
                        + static_cast<double>(particleCounts[j]) * particleScale;
        return weights;
    }

private:
    const std::shared_ptr<MonteCarloManager3D> manager;
    double counterScale;
    double particleScale;
};

#endif // RICH_MPI

#endif // IMC_COST_CALCULATOR_HPP
