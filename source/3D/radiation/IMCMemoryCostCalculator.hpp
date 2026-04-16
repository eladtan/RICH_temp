#ifndef IMC_MEMORY_COST_CALCULATOR_HPP
#define IMC_MEMORY_COST_CALCULATOR_HPP

#ifdef RICH_MPI

#include "newtonian/three_dimensional/CostCalculator3D.hpp"
#include "3D/monte/MonteCarloManager3D.hpp"
#include <algorithm>

class IMCMemoryCostCalculator : public CostCalculator3D
{
public:
    explicit IMCMemoryCostCalculator(const std::shared_ptr<MonteCarloManager3D> &manager)
        : manager(manager)
    {}

    std::vector<double> CalculateCost(const Tessellation3D &tess,
                                      const vector<ComputationalCell3D> &cells) const override
    {
        size_t N = tess.GetPointNo();
        double memPerCell = static_cast<double>(manager->GetHandlerMemoryBytes())
                          / static_cast<double>(std::max<size_t>(N, 1));
        return std::vector<double>(N, std::max(memPerCell, 1.0));
    }

private:
    const std::shared_ptr<MonteCarloManager3D> manager;
};

#endif // RICH_MPI

#endif // IMC_MEMORY_COST_CALCULATOR_HPP
