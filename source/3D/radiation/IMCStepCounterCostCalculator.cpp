#ifdef RICH_MPI

#include "IMCStepCounterCostCalculator.hpp"
#include <utility>

IMCStepCounterCostCalculator::IMCStepCounterCostCalculator(
    std::unordered_map<size_t, double> costByCellID,
    Parameters params)
    : costByCellID_(std::move(costByCellID)), params_(params)
{}

std::vector<double> IMCStepCounterCostCalculator::CalculateCost(
    const Tessellation3D& tess,
    const vector<ComputationalCell3D>& cells) const
{
    size_t N = tess.GetPointNo();
    std::vector<double> weights(N, params_.missingCellCost);

    for (size_t i = 0; i < N; ++i) {
        auto it = costByCellID_.find(cells[i].ID);
        if (it != costByCellID_.end())
            weights[i] = std::max(it->second, params_.floorCost);
    }

    return weights;
}

#endif // RICH_MPI
